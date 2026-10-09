"""PDF読込 → チャンク分割 → 埋め込み生成 までを一括実行する。

data/processed/ 配下は再生成可能な中間ファイルとして .gitignore しているため、
セットアップ後に一度だけ実行する（OPENAI_API_KEY が必須）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunker import split_into_chunks
from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, EMBED_BATCH_SIZE, RAW_PDF_PATH
from embedder import embed_texts, save_index
from pdf_loader import load_pdf


def main() -> None:
    """pdf_loader→chunker→embedderを順に実行し、data/processed/にインデックスを保存する。

    app.pyの「インデックスを構築する」ボタン（_build_index_in_app）も、
    起動元が違うだけで中身はこれと同じ3ステップ。
    """
    print(f"[1/3] PDF読込: {RAW_PDF_PATH}")
    pages = load_pdf(str(RAW_PDF_PATH))
    print(f"  -> {len(pages)} ページ")

    print(f"[2/3] チャンク分割 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = split_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    lengths = [len(c["text"]) for c in chunks]
    print(f"  -> {len(chunks)} チャンク "
          f"(最小{min(lengths)} / 中央値{sorted(lengths)[len(lengths)//2]} / 最大{max(lengths)}字)")

    print(f"[3/3] 埋め込み生成 (batch={EMBED_BATCH_SIZE})")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    save_index(vectors, chunks, str(CHUNKS_PATH))
    print(f"保存先: {CHUNKS_PATH.parent}")


if __name__ == "__main__":
    main()
