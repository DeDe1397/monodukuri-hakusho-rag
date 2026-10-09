"""プロンプト組み立て・LLM呼び出し・出典付与。

MIN_SCORE_THRESHOLD 未満のときにLLMを呼ばず定型文を返す設計は、
「根拠が薄い質問には答えない」をプロンプトの指示だけに頼らず、
コードのif文で構造的に保証するためのもの（Q09: 範囲外質問対策）。
if文自体は100%決定的に働くが、それはスコアが正しく「関連度が低い」ことを
表している場合に限る。実際に評価を回したところ、BM25/RRFのスコアを
クエリごとの実績最大値・順位だけで正規化していた旧実装では、無関係な
質問でも候補の誰かが機械的に高スコアになり、この閾値ゲートが一度も
発動しないことが分かった（retriever.py側で修正済み）。それでも意味的に
近いが業種の異なる質問（例:「観光業の人材不足対策」）はベクトル検索の
コサイン類似度が0.4程度まで上がることがあり、閾値だけで完全に防げる
わけではない。そのためこの仕組みは唯一の防御ではなく、プロンプト内の
「わからなければ答えるな」という指示と合わせた多層防御として位置づけている。

MIN_SCORE_THRESHOLDは方式（vector/bm25/hybrid）ごとに値が異なる辞書になっている。
正規化後も分布の形は方式ごとに違う（評価用10問の実測で確認済み。根拠はconfig.py
のコメント参照）ため、同じ閾値を共有すると方式によって「ゲートが効きすぎる」
「効かなすぎる」という偏りが出る。そのためgenerate/generate_memoはmode引数を
受け取り、その方式用の閾値だけを見て判定する。
"""

from __future__ import annotations

import time

from openai import OpenAI

from config import GEN_MODEL, MIN_SCORE_THRESHOLD, OPENAI_API_KEY, SEARCH_MODE

_MAX_RETRIES = 3

_NOT_FOUND_MESSAGE = "資料内に該当する記述が見つかりませんでした。"
_MEMO_NOT_FOUND_MESSAGE = "関連する記述が十分に見つかりませんでした。相談内容をもう少し具体的にしてください。"

_SYSTEM_PROMPT = """あなたは商工会議所の経営指導員を支援するアシスタントです。
「令和5年度 ものづくり基盤技術の振興施策（ものづくり白書）」から抜粋された
以下の文脈のみに基づいて、質問に回答してください。

厳守事項:
- 与えられた文脈に書かれている内容のみを根拠として回答すること。文脈にない知識で補わない。
- 文脈に根拠となる記述が見当たらない場合は、推測せず「資料内に該当する記述が見つかりませんでした」と答えること。
- 回答中の各主張の直後に、出典として (第X章 第Y節, pXX) の形式でページ番号と章節を併記すること。
- 数値は文脈中に明示されている値をそのまま使うこと。概算・四捨五入・単位換算などの加工をしないこと。
"""

_MEMO_SYSTEM_PROMPT = """あなたは商工会議所の経営指導員が、中小製造業との面談後にメモを
作成するのを支援するアシスタントです。「令和5年度 ものづくり基盤技術の振興施策
（ものづくり白書）」から抜粋された以下の文脈のみに基づいて、次の見出し構成で
面談メモの下書きを作成してください。

## 相談概要
入力された相談内容を1〜2文で要約する。

## 白書における関連動向・課題認識
文脈から読み取れる関連する記述を、出典 (第X章 第Y節, pXX) を付けて箇条書きにする。

## 提案できる支援策
文脈に明記された支援策があれば出典付きで提案する。文脈に無ければ書かない。

## 要確認事項
文脈だけでは根拠が薄い、または相談内容に対応する記述が文脈に見当たらない論点を列挙する。

厳守事項:
- 文脈にない事実・数値・制度名を作らない。
- 根拠が薄い項目は無理に埋めず「要確認事項」に回す。
- 各主張には出典（章節・ページ）を付ける。
"""


def _format_contexts(contexts: list[dict]) -> str:
    """検索結果のリストを、チャンクごとに出典（章節・ページ）を付けたテキストブロックに整形する。

    build_prompt/build_memo_prompt共通の下請け関数。LLMに渡す前に、
    どのチャンクがどこ由来かを明示しておくことで、出典付き回答を書きやすくする。
    """
    blocks = []
    for c in contexts:
        section = c["section"] or "(章節不明)"
        blocks.append(f"[出典: {section}, p{c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, contexts: list[dict]) -> list[dict]:
    """検索結果を出典付きの文脈ブロックに整形し、system/userメッセージを組み立てる。"""
    user_content = f"# 文脈\n{_format_contexts(contexts)}\n\n# 質問\n{query}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_memo_prompt(consultation: str, contexts: list[dict]) -> list[dict]:
    """相談内容と検索結果から、面談メモ下書き用のsystem/userメッセージを組み立てる。"""
    user_content = f"# 文脈\n{_format_contexts(contexts)}\n\n# 相談内容\n{consultation}"
    return [
        {"role": "system", "content": _MEMO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _call_llm(messages: list[dict]) -> str:
    """OpenAIのChat Completions APIを呼び、生成された回答文字列を返す。

    ライブラリ（openai）を実際に呼び出しているのはこの関数だけで、
    build_prompt等の他の関数は全部このAPI呼び出しを組み立てる自作コード。
    一時的なAPIエラーは指数バックオフで最大_MAX_RETRIES回リトライする。
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"LLM呼び出しが{_MAX_RETRIES}回失敗しました") from last_error


def generate(query: str, contexts: list[dict], mode: str = SEARCH_MODE) -> dict:
    """検索結果から回答を生成する。スコアが閾値未満ならLLMを呼ばず定型文を返す。

    戻り値: {"answer": str, "contexts": list[dict], "used_llm": bool}
    テキストだけを返すのではなく辞書にしているのは、contexts（app.pyでの
    参照チャンク表示に使う）とused_llm（eval/run_eval.pyで範囲外質問を
    正しく拒否できたか自動判定するのに使う）を、回答文とは別に呼び出し元へ
    引き渡す必要があるため。modeは閾値を方式別に切り替えるために使う
    （MIN_SCORE_THRESHOLDの説明を参照）。
    """
    top_score = max((c["score"] for c in contexts), default=0.0)

    if top_score < MIN_SCORE_THRESHOLD[mode]:
        return {"answer": _NOT_FOUND_MESSAGE, "contexts": contexts, "used_llm": False}

    messages = build_prompt(query, contexts)
    answer = _call_llm(messages)
    return {"answer": answer, "contexts": contexts, "used_llm": True}


def generate_memo(consultation: str, contexts: list[dict], mode: str = SEARCH_MODE) -> dict:
    """相談内容から面談メモ下書きを生成する。閾値未満ならLLMを呼ばず定型文を返す。"""
    top_score = max((c["score"] for c in contexts), default=0.0)

    if top_score < MIN_SCORE_THRESHOLD[mode]:
        return {"answer": _MEMO_NOT_FOUND_MESSAGE, "contexts": contexts, "used_llm": False}

    messages = build_memo_prompt(consultation, contexts)
    answer = _call_llm(messages)
    return {"answer": answer, "contexts": contexts, "used_llm": True}
