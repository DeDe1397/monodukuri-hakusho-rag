"""PDF → ページ単位のテキスト・章節ラベル・図表画像を抽出する。

対象PDF（ものづくり白書）を実機のブロック座標レベルで調べた結果、
単純な get_text() の文字列連結では章節ラベルもページ番号も正しく拾えないことが
分かった。以下の癖に合わせて実装している。

1. ページ番号はページ下部（y > 0.9*高さ）に孤立した数字だけのブロックとして
   存在する（左右の余白どちらに出るかはページによって変わる）。
2. 章・節見出しは「第1 章　業　況」のように本文とは別の独立ブロックとして
   ページのどこかに置かれるが、順序は一定しない。見開きページのレイアウトを
   ミラーしているらしく、右側ページでは「業　況／第1 章」のようにタイトルと
   マーカーの順序が逆転する。さらに右マージンには同じ章・節名を1文字ずつ
   縦に並べた飾り見出しブロックが重複して存在する（例:
   "製\n造\n業\nの\n業\n績\n動\n向\n第\n１\n節"）。
   これらはすべて改行・空白を取り除いた「コンパクト文字列」に対して
   「第+数字+部/章/節+タイトル」と「タイトル+第+数字+部/章/節」の両方向で
   正規表現を試すことで、順序に関わらず統一的に検出できる。
   本文の段落ブロックは数百文字あるのに対し見出し・飾りブロックは
   高々30字程度なので、文字数上限で本文中の偽陽性を排除している。
3. 章の2ページ目以降は見出しボックスに章番号だけが残りタイトルが欠落する。
   タイトルが空の更新は無視し、直前に確定したタイトルを引き継ぐ。
4. 部（第1部/第2部）の区切りページだけは本文の大部分がタイトル文字列で、
   マーカーが別ブロックとして離れた位置にある。区切りページはページ全体が
   短文（150字未満）という特徴があるため、短いページに限定した別ロジックで拾う。
5. 表紙の反復文言・目次見出し・脚注番号などの装飾ブロックは、本文ブロックの
   右端座標（x1）が本文カラムの最大値（実測で約539pt）を超える位置に出る。
   x1 > 540 を本文除外の判定に使う（実データで本文段落が超えないことを確認済み）。
6. 図表はほぼベクター描画のグラフで、ラスター画像ではなく大量の数値が
   テキストとして混入する。グラフの目盛り・凡例のブロックは全角スペースで
   始まらないため、何も対策しないと直前の本文段落にそのまま連結されてしまう
   （実データで発見：意味のある1文がグラフの数値と結合して900字超になり、
   chunker.pyの文字数フォールバック分割で前後の文脈から切り離された孤立
   チャンクを生み、検索でヒットしなくなる不具合があった）。
   文字の大半が数字・記号のブロック（`_is_mostly_numeric`）は本文から除外する。
   「別の段落として分離するだけ」も試したが、chunker.pyの貪欲パッキングが
   すぐ隣の文章段落と同じチャンクに再結合してしまい、埋め込みベクトルが
   数値ノイズで薄まる問題が残った（notes/session_log.md参照）。図表のキャプションは
   `figures`で、本物の表組みは`tables`で別途保持しているため、軸目盛りの
   生数値を落としても実質的な情報損失は小さいと判断した。
7. 表（罫線で区切られた表組み）は PyMuPDF の find_tables() で検出できるが、
   このPDFはグラフの目盛り・凡例も罫線なしの格子状に並ぶため、素朴に使うと
   グラフを表と誤検出する（実データで確認済み：誤検出はセルの大半が空）。
   セルの非空率が低い検出結果を捨てることで、本物の表組み（価格交渉の実施状況、
   生成AI活用事例の一覧など）だけを拾い、Markdown表としてページ本文に追加する。
   表の領域に重なる本文ブロックは、セル内容が生テキストとして二重に混ざらないよう
   通常の段落抽出から除外する。
"""

from __future__ import annotations

import re

import fitz

FOOTER_Y_RATIO = 0.9
CONTENT_MAX_X1 = 540  # これを超えるブロックは表紙反復・目次見出し・飾り見出しなど本文外
HEADING_CANDIDATE_MAX_LEN = 30  # 見出し・飾りブロックの上限文字数（本文段落は数百字あるため十分な余裕）
SHORT_PAGE_CHARS = 150  # これ未満は部区切り・表紙など「本文でない」ページとみなす目安

# find_tables()はグラフの目盛り・凡例も表と誤検出するため、セルの非空率が
# 低いものは捨てる。実データで真の表組みは非空率0.7〜1.0、グラフの誤検出は
# 中央値0.44程度に分布していたため、間を取って0.6をしきい値にした。
TABLE_MIN_FILL_RATIO = 0.6
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2

NUMERIC_BLOCK_MIN_RATIO = 0.5  # この割合以上が数字・記号ならグラフの目盛り等とみなす
_NUMERIC_CHARS = set("0123456789０１２３４５６７８９.-－%％,，")

_FORWARD_RE = re.compile(r"^第([0-9０-９]+)(部|章|節)(.+)$")
_BACKWARD_RE = re.compile(r"^(.+?)第([0-9０-９]+)(部|章|節)$")
_PART_MARKER_RE = re.compile(r"第([0-9０-９]+)\s*部")
_FOOTER_NUM_RE = re.compile(r"[0-9]+")


def _compact(text: str) -> str:
    """空白・改行・全角スペースをすべて除去した比較用文字列を作る。"""
    return re.sub(r"[\s　]+", "", text)


def _extract_footer_page(blocks: list[tuple], page_height: float) -> int | None:
    """ページ下部の孤立した数字ブロックを印刷ページ番号として取り出す。"""
    for x0, y0, x1, y1, text, *_ in blocks:
        if y0 < page_height * FOOTER_Y_RATIO:
            continue
        stripped = text.strip()
        if _FOOTER_NUM_RE.fullmatch(stripped):
            return int(stripped)
    return None


def _match_heading(text: str) -> tuple[str, str, str] | None:
    """ブロック本文から (単位, 番号, タイトル) を前方一致・後方一致の両方向で試す。"""
    compact = _compact(text)
    if not compact or len(compact) > HEADING_CANDIDATE_MAX_LEN:
        return None
    m = _FORWARD_RE.match(compact)
    if m:
        number, unit, title = m.group(1), m.group(2), m.group(3)
        return unit, number, title
    m = _BACKWARD_RE.match(compact)
    if m:
        title, number, unit = m.group(1), m.group(2), m.group(3)
        return unit, number, title
    return None


def _is_mostly_numeric(text: str) -> bool:
    """グラフの目盛り・凡例（数字の羅列）かどうかを、文字構成比で判定する。

    「74.0 49.2 27.0 ...」のようなブロックを、直前の文章段落に連結させない
    ために使う。全角スペースの有無だけでは判定できない（グラフの数値ブロックも
    段落先頭のような字下げは無いが、それは「新しい文章段落」ではないため）。
    """
    compact = re.sub(r"[\s　]+", "", text)
    if not compact:
        return False
    numeric_count = sum(1 for ch in compact if ch in _NUMERIC_CHARS)
    return numeric_count / len(compact) > NUMERIC_BLOCK_MIN_RATIO


def _is_body_block(x1: float, y0: float, page_height: float) -> bool:
    """このブロックが本文として扱ってよい位置にあるかを判定する。

    フッターのページ番号（y0がページ下部）と、表紙反復・目次見出し・
    飾り見出しなど本文カラム外（x1が本文カラムの最大値を超える）の
    どちらでもなければ True。
    """
    if y0 > page_height * FOOTER_Y_RATIO:
        return False  # フッターのページ番号
    if x1 > CONTENT_MAX_X1:
        return False  # 表紙反復・目次見出し・飾り見出しなどの本文外ブロック
    return True


def _extract_part_divider(full_text: str) -> str | None:
    """部の区切りページ（タイトル→マーカーが別ブロックで離れて置かれる）からタイトルを拾う。"""
    if len(full_text) >= SHORT_PAGE_CHARS:
        return None
    m = _PART_MARKER_RE.search(full_text)
    if not m:
        return None
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
    title_lines = [ln for ln in lines if not _PART_MARKER_RE.search(ln)]
    title = "".join(title_lines)
    return f"第{m.group(1)}部　{title}" if title else f"第{m.group(1)}部"


def _extract_images(doc: fitz.Document, page: fitz.Page) -> list[bytes]:
    """ページに埋め込まれたラスター画像を、生のバイト列のリストとして取り出す。

    対象PDFの図表はほとんどがベクター描画（線・図形）なので、実際に画像として
    抽出できるのは写真等が使われている一部のページのみ。破損画像は個別に
    スキップし、ページ全体の抽出は継続する。
    """
    images = []
    for xref, *_ in page.get_images(full=True):
        try:
            images.append(doc.extract_image(xref)["image"])
        except Exception:
            # 埋め込み画像の抽出に失敗しても本文抽出は継続する（破損画像対策）
            continue
    return images


def _table_fill_ratio(rows: list[list]) -> float:
    """find_tables()が検出したセルのうち、空でないセルの割合を返す。

    本物の表組みはセルがほぼ埋まっている（0.7〜1.0）一方、グラフの目盛り・
    凡例を誤検出したものはセルの大半が空になる（実データで中央値0.44程度）。
    この比率でグラフの誤検出を弾く（TABLE_MIN_FILL_RATIO参照）。
    """
    total = sum(len(row) for row in rows)
    if not total:
        return 0.0
    nonempty = sum(1 for row in rows for cell in row if cell not in (None, ""))
    return nonempty / total


def _extract_tables(page: fitz.Page) -> list[dict]:
    """本物の表組みだけをMarkdown化して (markdown, bbox) のリストで返す。"""
    results = []
    for table in page.find_tables().tables:
        if table.row_count < TABLE_MIN_ROWS or table.col_count < TABLE_MIN_COLS:
            continue
        if _table_fill_ratio(table.extract()) < TABLE_MIN_FILL_RATIO:
            continue  # グラフの目盛り・凡例の誤検出（セルの大半が空）を除外
        results.append({"markdown": table.to_markdown(), "bbox": table.bbox})
    return results


def _bbox_center_in(x0: float, y0: float, x1: float, y1: float, box) -> bool:
    """ブロックの矩形(x0,y0,x1,y1)の中心点が、box(x0,y0,x1,y1)の範囲内にあるかを判定する。

    表組みの領域と重なる本文ブロックを検出するために使う。矩形同士の重なり
    判定ではなく中心点だけで判定するのは、表の外周をわずかにはみ出した
    ブロック（セル境界の誤差など）を過剰に除外しないための単純化。
    """
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = box
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def load_pdf(path: str) -> list[dict]:
    """PDFを読み込み、ページ単位の辞書リストを返す（chunker.split_into_chunksへの入力）。

    各要素: {"page": int, "text": str, "images": list[bytes], "section": str | None,
             "is_figure_heavy": bool, "tables": list[str]}

    - page: 印刷されているページ番号（フッターで見つからない場合はpdf内の物理位置+1）
    - text: 見出し・フッター・装飾・グラフの生数値を除去し、表はMarkdown化した本文
    - section: 「第1部　.../ 第1章　.../ 第1節　...」のように、その時点で
      確定している部・章・節を連結した文字列。**このページ単位でしか section を
      追跡していないため、1ページの中で節の境界をまたぐ場合、そのページ全体が
      最後に検出した節のラベルになる**（前半が別の節に属していても区別できない、
      未修正の既知の限界）。
    - tables: このページで検出した本物の表組みをMarkdown化した文字列のリスト
    - is_figure_heavy: 本文が短く（150字未満）、図表中心のページと推定されるか
    """
    doc = fitz.open(path)
    pages: list[dict] = []

    current_part: str | None = None
    current_chapter: str | None = None
    current_section: str | None = None

    for i in range(doc.page_count):
        page = doc[i]
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        full_text = page.get_text()

        part_title = _extract_part_divider(full_text)
        if part_title:
            current_part = part_title
            current_chapter = None
            current_section = None

        tables = _extract_tables(page)
        table_bboxes = [t["bbox"] for t in tables]

        # このPDFはブロック＝行（折り返し1行ごとに別ブロック）であり、段落の区切りは
        # ブロック境界ではなく「段落先頭の全角スペース」で示される（実データで確認済み）。
        # そのため先頭が全角スペースのブロックだけを新しい段落の開始とみなし、
        # それ以外は直前の段落に半角スペース区切りで連結する（グラフの数値の羅列も
        # 同じ理由でスペース区切りにしておくと "8""6""4" が "864" に潰れず読める）。
        paragraph_groups: list[list[str]] = []
        for x0, y0, x1, y1, text, *_ in blocks:
            stripped = text.strip()
            if not stripped:
                continue

            heading = _match_heading(stripped)
            if heading:
                unit, number, title = heading
                if title:  # タイトル欠落の更新（章2ページ目以降）は前の値を維持
                    if unit == "章":
                        current_chapter = f"第{number}章　{title}"
                        current_section = None
                    elif unit == "節":
                        current_section = f"第{number}節　{title}"
                continue  # 見出し・飾りブロックは section に転記済みなので本文からは除く

            if not _is_body_block(x1, y0, page_height):
                continue

            if any(_bbox_center_in(x0, y0, x1, y1, box) for box in table_bboxes):
                continue  # 表組みの領域は下でMarkdown表として追加するので生テキストとしては拾わない

            if _is_mostly_numeric(stripped):
                # グラフの目盛り・凡例（意味の薄い数値）は本文から除外する。
                # 「別の段落として分離するだけ」ではchunker.pyの貪欲パッキングで
                # 結局すぐ隣の文章段落と同じチャンクに再結合されてしまい、
                # 埋め込みベクトルが数値ノイズで薄まる問題が残ることが実験で分かった
                # （notes/session_log.md参照）。図表のキャプションは`figures`で、
                # 本物の表組みは`tables`で別途保持しているため、軸目盛りの生数値を
                # 落としても実質的な情報損失は小さいと判断した。
                continue

            normalized = re.sub(r"\s+", " ", stripped)
            if text.startswith("　") or not paragraph_groups:
                paragraph_groups.append([normalized])
            else:
                paragraph_groups[-1].append(normalized)

        for t in tables:
            paragraph_groups.append([t["markdown"]])

        text = "\n\n".join(" ".join(group) for group in paragraph_groups)

        printed_page = _extract_footer_page(blocks, page_height)
        section = " / ".join(s for s in (current_part, current_chapter, current_section) if s)

        pages.append(
            {
                "page": printed_page if printed_page is not None else i + 1,
                "text": text,
                "images": _extract_images(doc, page),
                "section": section or None,
                "is_figure_heavy": len(text) < SHORT_PAGE_CHARS and not part_title,
                "tables": [t["markdown"] for t in tables],
            }
        )

    doc.close()
    return pages
