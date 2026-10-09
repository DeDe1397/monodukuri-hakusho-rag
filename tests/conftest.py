import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


@pytest.fixture(scope="session")
def real_pages():
    """実PDFを1回だけ読み込み、テスト全体で使い回す（load_pdfは数秒かかるため）。"""
    from config import RAW_PDF_PATH
    from pdf_loader import load_pdf

    return load_pdf(str(RAW_PDF_PATH))


@pytest.fixture(scope="session")
def real_chunks(real_pages):
    from chunker import split_into_chunks
    from config import CHUNK_OVERLAP, CHUNK_SIZE

    return split_into_chunks(real_pages, CHUNK_SIZE, CHUNK_OVERLAP)
