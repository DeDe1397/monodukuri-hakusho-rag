"""pdf_loader.py の検証。

実PDFのレイアウトを目視・座標調査した上で確定した「正解」（既知の見出し・
ページ番号・段落境界）と突き合わせる、実データ回帰テスト。
"""

from pdf_loader import _is_mostly_numeric


def test_all_pages_loaded(real_pages):
    assert len(real_pages) == 340


def test_page_dict_shape(real_pages):
    for key in ("page", "text", "images", "section", "is_figure_heavy", "tables"):
        assert key in real_pages[0]


def test_chart_pages_are_not_misdetected_as_tables(real_pages):
    # ページ14はグラフ(業況判断DI)であり罫線の表組みではない。
    # find_tables()の素朴な適用はここを誤検出するため、非空率フィルタで弾けているか確認する。
    assert real_pages[14]["tables"] == []


def test_genuine_table_extracted_as_markdown(real_pages):
    # pdfインデックス244（印刷ページ232）「生成AI活用の主な用途」は罫線付きの本物の表組み
    page = real_pages[244]
    assert page["tables"]
    md = page["tables"][0]
    assert md.startswith("|")
    assert "|---|" in md or "| --- " in md
    assert "設計デザイン" in md or "コーディング" in md


def test_footer_page_number_extracted_correctly(real_pages):
    # 実PDFで目視確認済みの対応関係（pdfインデックス→印刷ページ番号）
    assert real_pages[14]["page"] == 2
    assert real_pages[254]["page"] == 242
    assert real_pages[256]["page"] == 244


def test_part_divider_pages_fallback_to_physical_index(real_pages):
    # 区切りページには印刷ページ番号がないため、pdfインデックス+1にフォールバックする
    assert real_pages[13]["page"] == 14
    assert real_pages[255]["page"] == 256


def test_no_page_has_completely_empty_text(real_pages):
    empty = [p["page"] for p in real_pages if len(p["text"]) == 0]
    assert empty == []


def test_section_labels_track_known_chapters(real_pages):
    assert real_pages[14]["section"] == "第1部　ものづくり基盤技術の現状と課題 / 第1章　業況"
    assert real_pages[22]["section"] == (
        "第1部　ものづくり基盤技術の現状と課題 / 第1章　業況 / 第2節　生産・出荷・在庫の状況"
    )


def test_section_state_carries_across_pages_missing_title(real_pages):
    # 章の2ページ目以降はタイトルが見出しボックスから欠落するが、直前の値を維持する
    assert real_pages[17]["section"] == real_pages[14]["section"]


def test_part_boundary_resets_chapter_and_section(real_pages):
    part1_last_section = real_pages[254]["section"]
    part2_first_section = real_pages[256]["section"]
    assert part1_last_section.startswith("第1部")
    assert part2_first_section.startswith("第2部")
    assert "第1章" in part2_first_section  # 部が変わると章番号がリセットされる


def test_decoration_and_margin_blocks_excluded_from_body(real_pages):
    # 右マージンの縦書き飾り見出し（製造業の業績動向第１節業況第1章、など）が
    # 本文末尾に混入していないことを確認する。
    text = real_pages[17]["text"]
    assert "業　況" not in text
    assert "第1 章" not in text
    assert "資料：日本銀行「全国企業短期経済観測調査」（2024年4月）" in text


def test_cover_repeat_and_toc_label_excluded(real_pages):
    assert "ものづくり基盤技術の振興施策" not in real_pages[0]["text"]


def test_paragraphs_are_separated_by_blank_line(real_pages):
    # 段落先頭の全角スペースを検出して\n\nで区切っている（chunker.pyの前提）
    assert "\n\n" in real_pages[20]["text"]


def test_chart_axis_numbers_are_dropped_from_body_text(real_pages):
    # p8「図110-7 第一次所得収支の推移」のグラフ軸目盛り（8.2, 7.8, 8.6...）は
    # 意味の薄い数値の羅列として本文から除外される。一方、本文の文章内にある
    # 意味のある数値（34.6兆円の黒字、など）は残ることを確認する。
    text = real_pages[20]["text"]
    assert "34.6兆円の黒字" in text
    assert "8.2 7.8 8.6" not in text


class TestIsMostlyNumeric:
    def test_chart_tick_values_detected_as_numeric(self):
        assert _is_mostly_numeric("74.0 49.2 27.0 37.0 38.8 18.7 11.4 1.6 7.9 1.2")

    def test_prose_sentence_not_detected_as_numeric(self):
        assert not _is_mostly_numeric("一方で、従業員数300人以下の企業においては、賃上げを実施した。")

    def test_prose_with_a_few_numbers_not_detected_as_numeric(self):
        assert not _is_mostly_numeric("2023年における賃上げの状況をみていく。9割超の企業が実施した。")


def test_chart_noise_no_longer_forces_sentence_level_fallback_split(real_pages):
    # 回帰テスト: 「一方で、従業員数300人以下の企業は...」という1文（p80）が、
    # 直後のグラフ「図243-4」の目盛り・凡例（全角スペースなしで続く）と連結され、
    # 903字（CHUNK_SIZE=600超）に膨らんでchunker.pyの文字数フォールバック分割を
    # 誘発し、前後の文脈から切り離された孤立チャンクを生んでいた不具合の再発防止。
    # 純粋な軸目盛りの数値の羅列は除外されるが、凡例ラベルや備考など文字混じりの
    # 断片は残るため、完全にノイズゼロにはならない。重要なのは「CHUNK_SIZEを
    # 超えてsentence-level分割を誘発しないこと」なので、そこだけを検証する。
    from config import CHUNK_SIZE

    page = next(p for p in real_pages if p["page"] == 80 and "300人以下の企業" in p["text"])
    paragraphs = [p for p in page["text"].split("\n\n") if "300人以下の企業" in p]
    assert paragraphs
    assert len(paragraphs[0]) < CHUNK_SIZE
    assert "74.0" not in paragraphs[0]  # 純粋な軸目盛りの数値は除外されている
