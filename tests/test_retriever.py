"""retriever.py の検証（OpenAI APIを一切呼ばずに検証できる範囲）。

ベクトル検索は _embed_query をモックして「正規化済み内積＝コサイン類似度」の
計算部分だけを検証し、BM25・RRFは完全に自作ロジックなのでAPIなしで
そのまま検証できる。
"""

import numpy as np
import pytest

import retriever


def _make_chunk(cid, text, page=1, section="s"):
    return {"id": cid, "text": text, "page": page, "section": section, "figures": []}


@pytest.fixture(autouse=True)
def reset_index():
    """テスト間でモジュールレベルのシングルトン_indexを汚染しないようにする。"""
    yield
    retriever._index.chunks = None
    retriever._index.vectors = None
    retriever._index.postings = None


def _load_bm25_only(chunks):
    retriever._index.chunks = chunks
    retriever._index._build_bm25_index()


class TestTokenize:
    def test_bigram_tokenization(self):
        assert retriever._tokenize("あいう") == ["あい", "いう"]

    def test_single_char_returns_itself(self):
        assert retriever._tokenize("a") == ["a"]

    def test_empty_string_returns_empty_list(self):
        assert retriever._tokenize("") == []

    def test_whitespace_is_removed_before_tokenizing(self):
        assert retriever._tokenize("あ い") == retriever._tokenize("あい")


class TestBM25:
    def test_relevant_doc_ranks_first(self):
        chunks = [
            _make_chunk("c0", "中小企業の人材確保に関する支援策"),
            _make_chunk("c1", "全く関係のない天気予報の話題です"),
            _make_chunk("c2", "海外旅行の観光案内パンフレット"),
        ]
        _load_bm25_only(chunks)

        results = retriever.search_bm25("中小企業の人材確保", k=3)

        assert results[0]["id"] == "c0"
        # 理論上の最大値（tf→∞で収束する上限）で正規化するため、実際のtfでは1.0未満になる。
        # 「実績最大値」で正規化していた旧実装は、無関係な質問でも上位が機械的に1.0になり
        # MIN_SCORE_THRESHOLDによる拒否が働かなくなる不具合があった（評価実行で発見・修正）。
        assert 0.0 < results[0]["score"] < 1.0
        assert results[1]["score"] == 0.0
        assert results[2]["score"] == 0.0

    def test_no_match_returns_zero_scores_without_error(self):
        chunks = [_make_chunk("c0", "あいうえお"), _make_chunk("c1", "かきくけこ")]
        _load_bm25_only(chunks)

        results = retriever.search_bm25("存在しない語句XYZ", k=2)

        assert all(r["score"] == 0.0 for r in results)

    def test_returns_at_most_k_results(self):
        chunks = [_make_chunk(f"c{i}", f"テスト文書{i}") for i in range(10)]
        _load_bm25_only(chunks)

        results = retriever.search_bm25("テスト文書", k=3)

        assert len(results) == 3


class TestSearchVector:
    def test_uses_normalized_dot_product_as_cosine_similarity(self, monkeypatch):
        chunks = [_make_chunk("c0", "text0"), _make_chunk("c1", "text1")]
        retriever._index.chunks = chunks
        retriever._index.vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        retriever._index._build_bm25_index()  # ensure_loadedがスキップされるよう最低限埋める

        monkeypatch.setattr(retriever, "_embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))

        results = retriever.search_vector("クエリ", k=2)

        assert results[0]["id"] == "c0"
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[1]["score"] == pytest.approx(0.0)


def _scored(chunk, score):
    return {**chunk, "score": score}


class TestSearchHybrid:
    def test_rrf_rank_order_favors_doc_ranked_first_in_both_lists(self, monkeypatch):
        chunks = [_make_chunk(f"c{i}", f"text{i}") for i in range(3)]
        retriever._index.chunks = chunks

        # c0が両方式で1位、c1とc2は方式ごとに順位が割れるケースを模擬する
        monkeypatch.setattr(
            retriever, "search_vector", lambda q, k: [_scored(chunks[0], 0.5), _scored(chunks[1], 0.4), _scored(chunks[2], 0.3)]
        )
        monkeypatch.setattr(
            retriever, "search_bm25", lambda q, k: [_scored(chunks[0], 0.5), _scored(chunks[2], 0.4), _scored(chunks[1], 0.3)]
        )

        results = retriever.search_hybrid("クエリ", k=3)

        assert results[0]["id"] == "c0"  # 順位（誰が1位か）はRRFで決まる

    def test_hybrid_score_uses_vector_cosine_not_rrf_rank_score(self, monkeypatch):
        """評価実行で発見した不具合の再発防止テスト:

        RRFの順位スコアをそのまま信頼度として使うと、質問が的外れでも
        候補プールの誰かが機械的に上位スコアになり、MIN_SCORE_THRESHOLDによる
        拒否が働かなくなる（範囲外質問Q09で実際に発生）。信頼度には
        ベクトルのコサイン類似度をそのまま使うべきで、RRFで加工してはいけない。
        """
        chunks = [_make_chunk("c0", "text0")]
        retriever._index.chunks = chunks
        monkeypatch.setattr(retriever, "search_vector", lambda q, k: [_scored(chunks[0], 0.12)])
        monkeypatch.setattr(retriever, "search_bm25", lambda q, k: [_scored(chunks[0], 0.9)])

        results = retriever.search_hybrid("クエリ", k=1)

        # RRFの順位スコアなら大きな値になりうるが、ここではベクトル側の
        # コサイン類似度（0.12、閾値0.30未満）がそのまま信頼度として返るべき
        assert results[0]["score"] == pytest.approx(0.12)

    def test_hybrid_falls_back_to_bm25_score_when_not_in_vector_pool(self, monkeypatch):
        chunks = [_make_chunk("c0", "text0")]
        retriever._index.chunks = chunks
        monkeypatch.setattr(retriever, "search_vector", lambda q, k: [])  # ベクトル候補プールに入らなかった
        monkeypatch.setattr(retriever, "search_bm25", lambda q, k: [_scored(chunks[0], 0.7)])

        results = retriever.search_hybrid("クエリ", k=1)

        assert results[0]["score"] == pytest.approx(0.7)
