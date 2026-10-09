"""ハイパーパラメータの一元管理。

チャンクサイズや検索方式を比較実験する上で、数値をコード中に
直書きすると「どの設定でその結果が出たか」が追えなくなる。すべてここに集約し、
eval/run_eval.py から上書きして比較できるようにする。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _secret(key: str, default: str | None = None) -> str | None:
    """環境変数優先、なければ Streamlit Cloud の st.secrets を見る。

    ローカルDockerでは.envを、Streamlit Cloudではダッシュボードで設定した
    Secretsを使うため、どちらの環境でも同じconfig.pyで動くようにしている。

    secrets.toml が実在するか事前にファイルの有無を見てから st.secrets に
    触れる。存在しない状態で st.secrets にアクセスすると、Streamlitが
    「No secrets found」という警告をアプリ内に描画してしまい、それが
    app.py の st.set_page_config() より先に走って
    StreamlitSetPageConfigMustBeFirstCommandError を起こすため
    （ローカルDocker環境で.envだけを使う場合はsecrets.tomlを置かないので、
    ここで確実に踏む問題だった）。
    """
    value = os.getenv(key)
    if value:
        return value

    secrets_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        ROOT_DIR / ".streamlit" / "secrets.toml",
    ]
    if not any(p.exists() for p in secrets_paths):
        return default

    try:
        import streamlit as st

        return st.secrets.get(key, default)
    except Exception:
        return default

# --- パス -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PDF_PATH = ROOT_DIR / "data" / "raw" / "令和５年度ものづくり基盤技術の振興施策.pdf"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"

# --- チャンク分割 -------------------------------------------------------
# 600字/100字重複は「見出しで区切った後の段落が概ね収まる長さ」を
# 白書の実物を確認して逆算した値。固定長分割ではなく階層分割の
# 最終フォールバックとしてのみ使われる（src/chunker.py 参照）。
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- 検索 ---------------------------------------------------------------
TOP_K = 5
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")  # "vector" | "bm25" | "hybrid"

# BM25 パラメータ（Robertson & Sparck Jones の推奨値をそのまま採用）
BM25_K1 = 1.2
BM25_B = 0.75

# RRF (Reciprocal Rank Fusion) の定数。値が大きいほど上位順位間の差が緩和される。
# 原論文 (Cormack et al., 2009) の実験で最も安定した k=60 を採用。
RRF_K = 60

# 検索結果の最高スコアがこれ未満の場合はLLMを呼ばず「該当なし」を返す。
# vector/bm25/hybridでスコアのスケールが元々バラバラ（コサイン類似度・BM25生スコア・
# RRFスコア）なので、retriever.py側でどの方式も0〜1に正規化してから返している
# （vector=コサイン類似度そのもの、bm25=クエリの理論上の最大値で正規化、
# hybrid=RRFで順位統合しつつ信頼度にはコサイン類似度を使う）。
#
# 方式ごとに閾値を分けている理由: 正規化後も分布の形はvector/hybridとbm25で
# 異なる（評価用10問の実測で確認。再現・再計測は eval/analyze_scores.py、
# 実測値そのものは eval/results/score_distribution.csv を参照）。
# 10問（範囲外1問）のみなのでPR曲線やAUROCのような統計的な閾値最適化はできず、
# ここでやっているのはスコアを並べて隙間を見る手動のキャリブレーションに留まる。
# - vector/hybrid: 範囲内質問の最低スコアが0.43、範囲外質問（観光業）が0.47で、
#   範囲内より範囲外が僅かに高いケースがあるため、閾値だけでは完全に分離できない
#   （0.30はこの2つの数値より十分低く設定し、ゲートは「露骨に無関係な質問」だけを
#   落とす網として機能させ、意味的に近い範囲外質問の排除はプロンプト側に委ねる）。
# - bm25: 範囲内質問の最低スコアが0.17、範囲外質問が0.13で、この2つの間に
#   明確な隙間がある。0.30のままだと範囲内質問の過半数（9問中6問）までゲートで
#   落ちてしまっていた（検索自体は成功していたのに回答できていなかった）ため、
#   0.15（0.13と0.17の中間）に下げて、範囲外質問は排除しつつ範囲内質問を
#   通すようにした。
MIN_SCORE_THRESHOLD = {"vector": 0.30, "bm25": 0.15, "hybrid": 0.30}

# --- モデル ---------------------------------------------------------------
EMBED_MODEL = _secret("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL = _secret("GEN_MODEL", "gpt-4o-mini")
EMBED_BATCH_SIZE = 100

OPENAI_API_KEY = _secret("OPENAI_API_KEY")

# 公開デプロイ用の簡易パスワード（任意）。Streamlit CloudをPublicで公開する場合、
# URLを知っていれば誰でもOpenAI API呼び出し（課金）ができてしまうため、
# app.pyの入口で1つの共有パスワードを掛ける。未設定ならローカル開発を
# 妨げないよう認証をスキップする（app.py の _check_password() 参照）。
APP_PASSWORD = _secret("APP_PASSWORD")
