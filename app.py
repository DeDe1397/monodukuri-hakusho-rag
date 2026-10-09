"""Streamlit エントリポイント。

チャット画面が基本機能、メモ下書き画面は拡張機能。
検索方式とtop-kをサイドバーで切り替えられるのは、
「vector/bm25/hybridで結果がこう変わる」をその場で見せられるようにするため
（README「検索方式を切り替えて比較できる」という設計方針に対応）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from config import APP_PASSWORD, CHUNKS_PATH, TOP_K
from pipeline import answer_question, draft_memo

st.set_page_config(page_title="ものづくり白書 相談チャットボット", layout="wide")


DISCLAIMER = (
    "回答はAIが白書の記述をもとに生成したもので、誤りや見落としを含む可能性があります。"
    "相談者への案内や書類作成など重要な判断の前には、必ず参照チャンクの出典ページ"
    "（ものづくり白書の原本）で内容を確認してください。"
)


def _check_password() -> bool:
    """公開デプロイでは誰でもOpenAI APIキー経由の呼び出し（＝課金）ができてしまう。
    閲覧者ごとにGoogleアカウント等を管理したくないため、個別認証ではなく
    共有パスワード1個で入口を絞る簡易な方式にしている。

    APP_PASSWORD（.envまたはStreamlit CloudのSecrets）を設定していない場合は
    ローカル開発を妨げないよう認証をスキップする。公開時は必ず設定する運用。
    """
    if not APP_PASSWORD:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("ものづくり白書 相談チャットボット")
    password = st.text_input("パスワードを入力してください", type="password")
    if not password:
        return False
    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.error("パスワードが違います。")
    return False


def _index_missing() -> bool:
    """chunks.jsonがディスクに無ければTrue（初回起動・再デプロイ直後など）。"""
    return not CHUNKS_PATH.exists()


def _build_index_in_app() -> None:
    """Streamlit Cloudにはターミナルがないため、scripts/build_index.pyと同じ処理を
    アプリのボタンから1回だけ実行できるようにする（PDFはリポジトリに同梱済み）。"""
    from chunker import split_into_chunks
    from config import CHUNK_OVERLAP, CHUNK_SIZE, RAW_PDF_PATH
    from embedder import embed_texts, save_index
    from pdf_loader import load_pdf

    with st.status("インデックスを構築中...", expanded=True) as status:
        status.write("PDFを読み込み中...")
        pages = load_pdf(str(RAW_PDF_PATH))
        status.write(f"{len(pages)}ページを読み込みました。チャンク分割中...")
        chunks = split_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        status.write(f"{len(chunks)}チャンクに分割しました。埋め込み生成中（数分かかります）...")
        vectors = embed_texts([c["text"] for c in chunks])
        save_index(vectors, chunks, str(CHUNKS_PATH))
        status.update(label="インデックス構築が完了しました。", state="complete")
    st.rerun()


def _render_sources(contexts: list[dict]) -> None:
    """検索結果（ページ・スコア・本文抜粋・含まれる図表）を展開表示する。

    LLMが回答文の中に自己申告で書く出典とは別に、実際に検索でヒットした
    生のチャンクをそのまま見せることで、回答の根拠を検証できるようにする。
    チャットタブ・メモ下書きタブの両方から呼ばれる共通部品。
    """
    with st.expander(f"参照チャンク（{len(contexts)}件）"):
        for c in contexts:
            st.markdown(f"**p{c['page']}　{c['section'] or '(章節不明)'}**　score={c['score']:.3f}")
            if c.get("figures"):
                st.caption("含まれる図表: " + " / ".join(c["figures"]))
            st.caption(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))
            st.divider()


def render_chat_tab() -> None:
    """チャットタブを描画する。

    st.session_state.messagesは画面に会話履歴を表示するためだけに使っており、
    pipeline.answer_question()には毎回「今回の質問文だけ」を渡している
    （過去のやり取りは検索・生成のどちらにも反映されない設計。会話の文脈を
    踏まえた検索をしたい場合はクエリリライトが必要になるが、現状は未実装）。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("contexts"):
                _render_sources(msg["contexts"])

    query = st.chat_input("ものづくり白書について質問してください")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("検索・回答生成中..."):
            result = answer_question(query, mode=st.session_state.search_mode, k=st.session_state.top_k)
        st.write(result["answer"])
        _render_sources(result["contexts"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "contexts": result["contexts"]}
    )


def render_memo_tab() -> None:
    """メモ下書きタブを描画する（拡張機能）。

    相談内容の入力文をそのまま検索クエリとして使う。入力は「従業員30名の
    金属加工業。人手不足とDX投資について...」のような複合的な長文になりやすく、
    文字bi-gramのBM25は苦手な傾向がある（Q07と同じ限界。サイドバーの既定値が
    hybridなので通常は問題になりにくい）。
    """
    st.caption(
        "相談内容を入力すると、白書の関連箇所を検索して面談メモの下書きを作成します。"
        "根拠が薄い項目は生成せず「要確認事項」として明示されます。"
    )
    consultation = st.text_area(
        "相談内容",
        placeholder="例）従業員30名の金属加工業。人手不足とDX投資について相談を受けた。",
        height=120,
    )
    if not st.button("メモ下書きを作成する", disabled=not consultation):
        return

    with st.spinner("検索・下書き生成中..."):
        result = draft_memo(consultation, mode=st.session_state.search_mode, k=st.session_state.top_k)
    st.markdown(result["answer"])
    _render_sources(result["contexts"])


def main() -> None:
    """アプリのエントリポイント。サイドバー（検索設定）とタブ構成を組み立てる。

    検索方式・top-kはst.session_stateに保存し、チャット・メモ下書き両タブが
    共通で参照する。Streamlitはボタン操作のたびにこのファイル全体を
    再実行するが、pipeline/retriever等のインポート済みモジュールの状態
    （読み込み済みインデックス等）はプロセスが生きている限り保持される。
    """
    if not _check_password():
        return

    st.title("ものづくり白書 相談チャットボット")
    st.caption("令和5年度 ものづくり基盤技術の振興施策（ものづくり白書）に基づく回答生成")

    with st.sidebar:
        st.header("検索設定")
        st.session_state.search_mode = st.selectbox("検索方式", ["hybrid", "vector", "bm25"], index=0)
        st.session_state.top_k = st.slider("top-k", min_value=1, max_value=10, value=TOP_K)

    if _index_missing():
        st.warning("検索インデックスが未構築です。")
        st.caption(
            "ローカルでは `python scripts/build_index.py` を実行してください。"
            "Streamlit Cloud はターミナルを持たないため、下のボタンから同じ処理を"
            "アプリ内で1回だけ実行できます（OPENAI_API_KEY がSecretsに必要・数分かかります）。"
        )
        if st.button("インデックスを構築する"):
            _build_index_in_app()
        return

    st.info(DISCLAIMER)

    tab_chat, tab_memo = st.tabs(["チャット", "メモ下書き"])
    with tab_chat:
        render_chat_tab()
    with tab_memo:
        render_memo_tab()


if __name__ == "__main__":
    main()
