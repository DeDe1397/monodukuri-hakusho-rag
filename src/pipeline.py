"""検索と生成を束ねる薄い層。ロジックはretriever.py/generator.py側に置き、ここには書かない。"""

from __future__ import annotations

from config import SEARCH_MODE, TOP_K
from generator import generate, generate_memo
from retriever import search


def answer_question(query: str, mode: str = SEARCH_MODE, k: int = TOP_K) -> dict:
    """質問文を検索→生成のパイプラインに通し、generator.generateの戻り値をそのまま返す。"""
    contexts = search(query, mode, k)
    return generate(query, contexts, mode)


def draft_memo(consultation: str, mode: str = SEARCH_MODE, k: int = TOP_K) -> dict:
    """相談内容を検索→面談メモ生成のパイプラインに通し、generator.generate_memoの戻り値をそのまま返す。"""
    contexts = search(consultation, mode, k)
    return generate_memo(consultation, contexts, mode)
