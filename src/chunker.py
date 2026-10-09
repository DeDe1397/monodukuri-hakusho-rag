"""自作の階層的チャンカー。

固定長分割にしなかった理由: 単純にN文字ごとに切ると、Q07のような
「複数章にまたがる相談型の質問」に対して、章の途中で文脈が寸断された
チャンクしか検索に引っかからず、経営指導員が読んでも話の筋が追えない
回答になってしまう。そこで次の優先順位で分割する。

1. 章・節（pdf_loader.py が既に確定させた section）が変わる境界を
   最優先の区切りとする。1つの節がそのまま size に収まるなら分割しない。
2. 節の中身が size を超える場合のみ、段落（pdf_loader.py が
   段落先頭の全角スペースを見て \n\n で区切った単位）で貪欲に詰め直す。
3. 1段落だけで size を超える場合（グラフの数値が延々と続くページなど）
   のみ、文単位（句点「。」）にさらに分解して詰め直す。
4. それでも1文が size を超える場合（句点が無い数値の羅列など）に限り、
   最後の手段として文字数で機械的に切り、overlap 分だけ前のチャンクと
   重複させる。

overlap は手順4の文字数分割にしか使わない。段落・文の単位は
それ自体で意味のまとまりを保っているため、そこに重複を挟むと
同じ内容が何度も埋め込まれてコストと検索ノイズが増えるだけと判断した。

分割の全段階で単位を (text, page) のタプルとして扱い、貪欲に詰めた
チャンクには「詰め始めに使った最初の単位のページ番号」を持たせる。
文字列を後から結合し直してページを逆引きするより、単位と出所ページを
最初から一緒に運ぶ方が単純で間違えにくい。
"""

from __future__ import annotations

import re

Unit = tuple[str, int]  # (text, page)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=。)")


def _greedy_pack(units: list[Unit], size: int, sep: str, split_fn) -> list[Unit]:
    """unitsを前から貪欲に詰め、sizeを超える手前で区切る。

    sep は詰めた単位同士をつなぐ区切り文字（段落なら"\n\n"、文なら句点で
    既に区切りがついているので""）。1ユニット単独でsizeを超える場合は
    split_fn でさらに細かく分割してから詰める。
    結果チャンクのページ番号は、その詰め込みの最初の単位のページを使う。
    """
    chunks: list[Unit] = []
    buf_text = ""
    buf_page: int | None = None

    def flush():
        nonlocal buf_text, buf_page
        if buf_text:
            chunks.append((buf_text, buf_page))
        buf_text, buf_page = "", None

    for text, page in units:
        if len(text) > size:
            flush()
            chunks.extend(split_fn(text, page))
            continue
        candidate = buf_text + sep + text if buf_text else text
        if buf_text and len(candidate) > size:
            flush()
            buf_text, buf_page = text, page
        else:
            buf_text = candidate
            buf_page = buf_page if buf_page is not None else page
    flush()
    return chunks


def _split_by_char(text: str, page: int, size: int, overlap: int) -> list[Unit]:
    """最後の手段: 文字数で機械的に切り、overlap分だけ前のチャンクと重複させる。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append((text[start:end], page))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _split_by_sentence(paragraph: str, page: int, size: int, overlap: int) -> list[Unit]:
    """句点「。」を優先境界として文単位で詰め直す。文自体が長い場合のみ文字数分割に落ちる。"""
    sentences = [(s, page) for s in _SENTENCE_SPLIT_RE.split(paragraph) if s]
    return _greedy_pack(sentences, size, "", lambda t, p: _split_by_char(t, p, size, overlap))


def _split_section(paragraphs: list[Unit], size: int, overlap: int) -> list[Unit]:
    """段落単位で詰め直す。段落自体が長い場合のみ文単位分割に落ちる。"""
    return _greedy_pack(paragraphs, size, "\n\n", lambda t, p: _split_by_sentence(t, p, size, overlap))


_FIGURE_CAPTION_RE = re.compile(
    r"図([0-9０-９]{2,4}-[0-9０-９]{1,2})\s+"
    r"((?:(?!\s*(?:[0-9０-９－\-]|図[0-9０-９]|資料[：:]|備考[：:]|出所[：:]))\S+\s*){1,12})"
)


def _extract_figure_captions(text: str) -> list[str]:
    """チャンク本文から「図110-7 第一次所得収支の推移」のようなキャプションを拾う。

    図表はほぼベクター描画のグラフでラスター画像を持たないため、画像認識では
    なくテキスト抽出時に残るキャプション文字列を頼りにする（Phase 8）。
    Q04のような「図表の数値を問う質問」は、本文の数値の羅列だけでは
    埋め込みが薄くなりがちなので、キャプションをチャンクのメタデータとして
    残しておき、UIでの出典表示や将来のリランキングに使えるようにしている。
    """
    captions = []
    for m in _FIGURE_CAPTION_RE.finditer(text):
        captions.append(f"図{m.group(1)} {m.group(2).strip()}")
    return captions


def _group_pages_by_section(pages: list[dict]) -> list[dict]:
    """sectionが同じ連続ページを1つのまとまりにする。"""
    runs: list[dict] = []
    sentinel = object()
    current = sentinel
    for p in pages:
        if not p["text"]:
            continue
        if p["section"] != current:
            runs.append({"section": p["section"], "pages": []})
            current = p["section"]
        runs[-1]["pages"].append(p)
    return runs


def split_into_chunks(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """ページ辞書のリストを、章節境界を優先したチャンクのリストに変換する（retriever.pyへの入力）。

    返す各チャンク: {"id": str, "text": str, "page": int, "section": str | None,
                     "figures": list[str]}
    page は、そのチャンク本文の先頭段落が実際に載っていたページ番号。
    節が複数ページにまたがっても出典が追えるようにするため、ページ単位ではなく
    段落単位でページを紐付けている。figures はチャンク内に含まれる図表キャプション
    （_extract_figure_captions参照）。
    """
    chunks: list[dict] = []

    for run in _group_pages_by_section(pages):
        section = run["section"]

        paragraphs: list[Unit] = [
            (para.strip(), p["page"])
            for p in run["pages"]
            for para in p["text"].split("\n\n")
            if para.strip()
        ]
        if not paragraphs:
            continue

        total_len = sum(len(t) for t, _ in paragraphs)
        if total_len <= size:
            text = "\n\n".join(t for t, _ in paragraphs)
            chunks.append({"text": text, "page": paragraphs[0][1], "section": section})
        else:
            for text, page in _split_section(paragraphs, size, overlap):
                chunks.append({"text": text, "page": page, "section": section})

    for i, chunk in enumerate(chunks):
        chunk["id"] = f"chunk_{i:05d}"
        chunk["figures"] = _extract_figure_captions(chunk["text"])
    return chunks
