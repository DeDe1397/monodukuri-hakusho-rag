"""chunker.py の階層分割ロジックの検証。

実PDFに依存しない合成データで、分割の優先順位（節→段落→文→文字数）と
overlapが文字数分割にしか使われないことをピンポイントで確認する。
"""

from chunker import _extract_figure_captions, split_into_chunks


def test_short_section_stays_as_single_chunk():
    pages = [{"text": "短い段落。", "page": 1, "section": "第1章 テスト"}]
    chunks = split_into_chunks(pages, size=600, overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "短い段落。"
    assert chunks[0]["page"] == 1
    assert chunks[0]["section"] == "第1章 テスト"


def test_long_section_splits_by_paragraph_without_overlap():
    a, b, c = "あ" * 30, "い" * 30, "う" * 30
    pages = [{"text": f"{a}\n\n{b}\n\n{c}", "page": 1, "section": "s"}]

    chunks = split_into_chunks(pages, size=50, overlap=10)

    assert [c["text"] for c in chunks] == [a, b, c]
    # 段落単位のパッキングにoverlapを混ぜていないことを確認
    assert not any(c1["text"].endswith(c2["text"][:10]) for c1, c2 in zip(chunks, chunks[1:]))


def test_oversized_paragraph_splits_by_sentence_not_mid_sentence():
    s1 = "あ" * 10 + "。"
    s2 = "い" * 10 + "。"
    pages = [{"text": s1 + s2, "page": 1, "section": "s"}]  # 1段落・句点2つ、段落全体はsizeを超える

    chunks = split_into_chunks(pages, size=20, overlap=5)

    assert [c["text"] for c in chunks] == [s1, s2]
    for c in chunks:
        assert c["text"].endswith("。")  # 文の途中で切れていない


def test_oversized_sentence_falls_back_to_char_split_with_overlap():
    text = "".join(str(i % 10) for i in range(40))  # 句点なしの40文字（数値の羅列を模擬）
    pages = [{"text": text, "page": 1, "section": "s"}]

    chunks = split_into_chunks(pages, size=10, overlap=3)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 10
    # 文字数分割の隣接チャンク同士はoverlap文字だけ重複している
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["text"][-3:] == nxt["text"][:3]


def test_page_attribution_follows_source_paragraph():
    pages = [
        {"text": "あ" * 40, "page": 1, "section": "s"},
        {"text": "い" * 40, "page": 2, "section": "s"},
    ]
    chunks = split_into_chunks(pages, size=50, overlap=10)

    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2


def test_section_change_forces_new_chunk_group():
    pages = [
        {"text": "短い1。", "page": 1, "section": "第1節"},
        {"text": "短い2。", "page": 2, "section": "第2節"},
    ]
    chunks = split_into_chunks(pages, size=600, overlap=100)

    assert len(chunks) == 2
    assert chunks[0]["section"] == "第1節"
    assert chunks[1]["section"] == "第2節"


def test_chunk_ids_are_unique_and_sequential():
    pages = [{"text": f"段落{i}。", "page": 1, "section": f"s{i}"} for i in range(5)]
    chunks = split_into_chunks(pages, size=600, overlap=100)

    assert [c["id"] for c in chunks] == [f"chunk_{i:05d}" for i in range(5)]


def test_figure_caption_extraction():
    text = "約34.6兆円の黒字を計上した（図110-7）。 図110-7 第一次所得収支の推移 8.2 7.8 8.6"
    assert _extract_figure_captions(text) == ["図110-7 第一次所得収支の推移"]


def test_no_figure_caption_returns_empty_list():
    assert _extract_figure_captions("図表のない普通の文章です。") == []


def test_real_chunks_never_exceed_configured_size(real_chunks):
    from config import CHUNK_SIZE

    over = [c for c in real_chunks if len(c["text"]) > CHUNK_SIZE]
    assert over == []


def test_real_chunks_have_required_fields(real_chunks):
    for key in ("id", "text", "page", "section", "figures"):
        assert key in real_chunks[0]
