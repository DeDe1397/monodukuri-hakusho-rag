FROM python:3.12-slim

# Streamlit Cloud は 3.12 系を前提にするため、開発コンテナも同じメジャーバージョンに固定する。
# 3.13 以降を避けるのはデプロイ時の互換性事故を防ぐための判断。

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv はロックファイルからの再現可能なインストールが速く、pip単体より devcontainer 起動が短縮できるため採用。
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Streamlit のデフォルトポート
EXPOSE 8501
