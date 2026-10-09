"""埋め込みベクトルの生成・保存・読込。

OpenAI API 呼び出し自体はライブラリ（openai）に任せるが、
「バッチ化」「L2正規化」「保存形式」は評価の中心なので自作する。
"""

from __future__ import annotations

import json
import time

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from config import EMBED_BATCH_SIZE, EMBED_MODEL, OPENAI_API_KEY

_MAX_RETRIES = 3


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """各ベクトルの長さ（L2ノルム）が1になるように正規化する。

    保存時に正規化しておけば、検索時は内積を取るだけでコサイン類似度になる
    （||a||=||b||=1 のとき a・b = cosθ）。毎回ノルムで割る手間を検索時から無くすための前計算。
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # ゼロベクトル（空文字列などの異常入力）除算対策
    return vectors / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """テキスト群を埋め込みベクトルに変換する。100件程度のバッチでAPIに投げる。

    逐次呼び出しにしないのは、チャンク数が数百件規模（実測723件）になり、
    1件ずつ呼ぶとAPI往復のオーバーヘッド（内容の大小によらずほぼ一定にかかる
    通信時間）が呼び出し回数分積み重なってしまうため。100件ずつまとめる
    ことで、往復回数を1/100に減らせる（実測: 723件が約8回の呼び出しで済む）。
    バッチサイズを723件まるごとではなく100件に留めているのは、1回の
    リクエストが大きすぎるとAPI側のペイロード上限に触れたり、途中で
    失敗した場合のやり直し範囲が大きくなったりするリスクがあるため。
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_vectors: list[list[float]] = []

    for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="embedding"):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.embeddings.create(model=EMBED_MODEL, input=batch)
                all_vectors.extend(item.embedding for item in response.data)
                break
            except Exception as e:  # API側の一時的なエラーを指数バックオフでリトライ
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"embedding failed after {_MAX_RETRIES} retries") from last_error

    vectors = np.array(all_vectors, dtype=np.float32)
    return _l2_normalize(vectors)


def save_index(vectors: np.ndarray, chunks: list[dict], path: str) -> None:
    """ベクトルとチャンク本文を別ファイルに保存する。

    path はチャンクJSONの保存先として使い、ベクトルは同じディレクトリの
    vectors.npy に保存する（config.VECTORS_PATH と対応させる呼び出し側の責務）。
    """
    from pathlib import Path

    chunks_path = Path(path)
    vectors_path = chunks_path.parent / "vectors.npy"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(vectors_path, vectors)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(path: str) -> tuple[np.ndarray, list[dict]]:
    """save_index で保存したベクトルとチャンクを読み込む。"""
    from pathlib import Path

    chunks_path = Path(path)
    vectors_path = chunks_path.parent / "vectors.npy"

    vectors = np.load(vectors_path)
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    return vectors, chunks
