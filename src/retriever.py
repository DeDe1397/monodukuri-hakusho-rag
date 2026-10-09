"""ベクトル検索・BM25・ハイブリッド検索を自作する。

外部ベクトルDB（Chroma/FAISS等）を使わない判断について：
対象は白書1冊・数百ページで、チャンク数はたかだか数千件規模。
numpyの行列積は数千×1536次元でもミリ秒オーダーで終わるため、
インデックス構築の複雑さやプロセス管理コストを払ってまで
外部DBを導入する理由がない。この判断自体をレポートの論点にする。

3方式のスコアはスケールがバラバラ（コサイン類似度は[-1,1]、BM25は
理論上無制限、RRFは1/(60+順位)の総和で小さい値）なので、
generator.py の MIN_SCORE_THRESHOLD で横並びに比較できるよう0〜1に正規化する。
vectorはコサイン類似度そのもの、BM25は「そのクエリの理論上の最大値
（Σ idf(t)×(k1+1)）」で割った値を使う。hybridはRRFで順位を統合するが、
返すスコアにはRRFの値ではなく実際の関連度を表すコサイン類似度（無ければ
BM25側のスコア）を使う。RRFはクエリの良し悪しに関わらず候補の誰かが
機械的に高スコアになるため、そのままでは MIN_SCORE_THRESHOLD による
拒否判定に使えない（評価実行で発見・修正。詳細はsearch_hybrid()参照）。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from openai import OpenAI

from config import (
    BM25_B,
    BM25_K1,
    CHUNKS_PATH,
    EMBED_MODEL,
    OPENAI_API_KEY,
    RRF_K,
    TOP_K,
)
from embedder import load_index


class _Index:
    """ベクトル・チャンク・BM25転置インデックスを一度だけ読み込んで保持する。"""

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list[dict] | None = None
        self.postings: dict[str, dict[int, int]] | None = None
        self.doc_lens: list[int] | None = None
        self.avg_doc_len: float = 0.0
        self.idf: dict[str, float] | None = None

    def ensure_loaded(self):
        """未読込なら、ディスクからベクトル・チャンクを読み込みBM25索引を構築する。

        既に読込済み（self.chunksがNone以外）なら即returnする。この遅延初期化
        ガードのおかげで、同一プロセス内で検索関数を何度呼んでも、ディスクからの
        再読込・BM25索引の再構築は最初の1回しか発生しない
        （Streamlitがボタンのたびにapp.pyを再実行しても、このオブジェクトは
        プロセスが生きている限りメモリに残り続けるため影響を受けない）。
        """
        if self.chunks is not None:
            return
        self.vectors, self.chunks = load_index(str(CHUNKS_PATH))
        self._build_bm25_index()

    def _build_bm25_index(self):
        """全チャンクを走査し、BM25用の転置索引（postings）・文書長・idfを構築する。

        postings[語] = {チャンクID: そのチャンク内での出現回数} という辞書。
        検索時にクエリの語ごとにこの辞書を引くだけで済むようにする、本の
        巻末索引に相当するもの（詳細はnotes参照）。埋め込みAPIと違い純粋な
        文字列処理なので無料・高速で、ディスクには保存せず毎回作り直す。
        """
        postings: dict[str, dict[int, int]] = {}
        doc_lens = []
        for doc_id, chunk in enumerate(self.chunks):
            tokens = _tokenize(chunk["text"])
            doc_lens.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, {})[doc_id] = tf
        n_docs = len(self.chunks)
        idf = {
            term: math.log((n_docs - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5) + 1)
            for term, doc_ids in postings.items()
        }
        self.postings = postings
        self.doc_lens = doc_lens
        self.avg_doc_len = sum(doc_lens) / n_docs if n_docs else 0.0
        self.idf = idf


_index = _Index()


def _tokenize(text: str) -> list[str]:
    """文字bi-gram（2文字の重なりあり部分文字列）のリストに分割する。

    日本語は形態素解析器なしでは分かち書きできないが、辞書付きの解析器を
    依存に足すとDocker環境・Streamlit Cloudの両方で構築が重くなる。
    文字bi-gramなら依存ゼロで「てにをは」の揺れにも頑健に部分一致できる。
    """
    normalized = re.sub(r"\s+", "", text)
    if len(normalized) < 2:
        return [normalized] if normalized else []
    return [normalized[i : i + 2] for i in range(len(normalized) - 1)]


def _embed_query(query: str) -> np.ndarray:
    """クエリ文字列をチャンクと同じ埋め込みモデルでベクトル化し、L2正規化して返す。

    チャンク側（embedder.embed_texts）と同じ前処理（正規化）を通すことで、
    後段の内積計算がそのままコサイン類似度になるようにする。
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _to_result(chunk: dict, score: float) -> dict:
    """チャンク辞書とスコアを、検索結果として返す共通の形に整形する。"""
    return {
        "id": chunk["id"],
        "text": chunk["text"],
        "page": chunk["page"],
        "section": chunk["section"],
        "figures": chunk.get("figures", []),
        "score": score,
    }


def search_vector(query: str, k: int = TOP_K) -> list[dict]:
    """クエリを埋め込み、正規化済み行列との内積を一括計算して上位k件を返す。"""
    _index.ensure_loaded()
    query_vec = _embed_query(query)
    # 全チャンクが正規化済みなので、内積がそのままコサイン類似度になる。
    # ライブラリのANNインデックスを使わず、全件との行列積を一度に計算する。
    similarities = _index.vectors @ query_vec
    top_idx = np.argsort(-similarities)[:k]
    return [_to_result(_index.chunks[i], float(similarities[i])) for i in top_idx]


def _bm25_scores(query: str) -> tuple[np.ndarray, float]:
    """(全チャンクの生BM25スコア, このクエリの理論上の最大スコア) を返す。

    理論上の最大値 = Σ idf(t)*(k1+1)（tf→∞でスコアが収束する上限）。
    実績（コーパス中で実際に付いた最高得点）で割ると、無関係な質問でも
    そのクエリの中で一番マシだった文書が機械的に1.0になってしまい、
    MIN_SCORE_THRESHOLDによる拒否が事実上働かなくなる（評価実行で発見した
    不具合。範囲外質問Q09で確認: 実績正規化だとbm25/hybridの最高スコアが
    1.0近辺に張り付き、コード側の閾値ゲートが一度も発動しなかった）。
    理論上の最大値で割れば、クエリ語をほとんど含まない文書しかヒットしない
    場合はスコアが低いまま残るため、閾値が意味を持つ。
    """
    _index.ensure_loaded()
    n_docs = len(_index.chunks)
    scores = np.zeros(n_docs, dtype=np.float64)
    query_terms = set(_tokenize(query))
    theoretical_max = 0.0
    for term in query_terms:
        doc_ids = _index.postings.get(term)
        if not doc_ids:
            continue
        idf = _index.idf[term]
        theoretical_max += idf * (BM25_K1 + 1)
        for doc_id, tf in doc_ids.items():
            doc_len = _index.doc_lens[doc_id]
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / _index.avg_doc_len)
            scores[doc_id] += idf * (tf * (BM25_K1 + 1)) / denom
    return scores, theoretical_max


def search_bm25(query: str, k: int = TOP_K) -> list[dict]:
    """自作BM25（文字bi-gramトークン化, k1=1.2, b=0.75）で上位k件を返す。"""
    scores, theoretical_max = _bm25_scores(query)
    top_idx = np.argsort(-scores)[:k]
    return [
        _to_result(_index.chunks[i], float(scores[i] / theoretical_max) if theoretical_max > 0 else 0.0)
        for i in top_idx
    ]


def search_hybrid(query: str, k: int = TOP_K) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) でベクトル検索とBM25の順位を統合する。

    順位の決定にはRRF（score = Σ 1/(RRF_K + rank)）を使うが、結果に付与する
    "score"（=MIN_SCORE_THRESHOLDでの閾値判定に使われる信頼度）にはRRFの値を
    使わない。RRFは「何位につけたか」しか見ておらず、質問がどれだけ的外れでも
    候補プールの中の誰かが機械的に1位になってしまうため、無関係な質問でも
    スコアが高止まりし、閾値ゲートが実質的に機能しなくなることが評価実行で
    分かった（範囲外質問Q09で、hybridの最高スコアが0.9台になり拒否できなかった）。
    そこで信頼度には、実際の関連度を反映するベクトルのコサイン類似度
    （無ければBM25の理論上限正規化スコア）を使う。
    """
    _index.ensure_loaded()
    pool = max(k * 4, 20)  # 統合前の候補プールは最終k件より広く取る
    vector_ranked = search_vector(query, k=pool)
    bm25_ranked = search_bm25(query, k=pool)

    rrf_scores: dict[str, float] = {}
    for rank, r in enumerate(vector_ranked):
        rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, r in enumerate(bm25_ranked):
        rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

    # 信頼度スコア: ベクトルのコサイン類似度を優先し、無ければBM25側を使う
    confidence: dict[str, float] = {r["id"]: r["score"] for r in bm25_ranked}
    confidence.update({r["id"]: r["score"] for r in vector_ranked})

    chunk_by_id = {c["id"]: c for c in _index.chunks}
    ranked_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:k]
    return [_to_result(chunk_by_id[cid], confidence.get(cid, 0.0)) for cid in ranked_ids]


SEARCH_FUNCTIONS = {
    "vector": search_vector,
    "bm25": search_bm25,
    "hybrid": search_hybrid,
}


def search(query: str, mode: str, k: int = TOP_K) -> list[dict]:
    """SEARCH_MODE設定値から検索関数を切り替えるための薄いディスパッチャ。"""
    if mode not in SEARCH_FUNCTIONS:
        raise ValueError(f"unknown search mode: {mode}")
    return SEARCH_FUNCTIONS[mode](query, k)
