"""MIN_SCORE_THRESHOLD（方式ごとの閾値）を決めるために使った、スコア分布の実測スクリプト。

run_eval.pyは検索→生成までの全体（LLM呼び出しあり）を実行するが、閾値を
決めたい場面ではLLMの回答そのものは不要で、「検索結果の最高スコアが
方式ごとにどんな値になるか」だけが知りたい。そのためLLMを呼ばず検索だけを
実行する（API費用・時間を節約するため、generator.generate()は呼ばない）。

使い方: 範囲内質問の最低スコアと、範囲外質問のスコアを方式ごとに比べ、
「範囲外質問より高く、範囲内質問の最低より低い」位置に閾値を置けるかを見る。
10問（範囲外1問）という少数のデータなので、PR曲線やAUROCのような統計的な
手法を適用するには足りない。ここでやっているのは「スコアを並べて隙間を見る」
という手動のキャリブレーションであり、厳密な閾値最適化ではない
（詳細はREADME「bm25の論点一致率が22%と極端に低かった」の節を参照）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_questions import EVAL_QUESTIONS
from retriever import search

MODES = ["vector", "bm25", "hybrid"]
TOP_K = 5
OUT_PATH = Path(__file__).resolve().parent / "results" / "score_distribution.csv"


def _measure_one(question: dict, mode: str) -> dict:
    """1問・1検索方式について、検索結果の最高スコアと期待ページの命中を記録する。

    top_scoreはgenerator.generate()が閾値ゲートの判定に使う値と同じ計算
    （search()が返すcontextsの中の最高score）。LLMは呼ばないのでここでは
    computeしない。
    """
    contexts = search(question["question"], mode, TOP_K)
    top_score = max((c["score"] for c in contexts), default=0.0)
    retrieved_pages = {c["page"] for c in contexts}

    if question["out_of_scope"]:
        source_hit = ""
    else:
        source_hit = bool(set(question["expected_source"]) & retrieved_pages)

    return {
        "mode": mode,
        "id": question["id"],
        "out_of_scope": question["out_of_scope"],
        "top_score": round(top_score, 4),
        "source_hit": source_hit,
    }


def main() -> None:
    """10問×3方式の最高スコアを測定し、CSVに保存しつつ方式ごとに見やすく表示する。"""
    rows = [_measure_one(q, mode) for mode in MODES for q in EVAL_QUESTIONS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for mode in MODES:
        print(f"--- {mode} ---")
        mode_rows = [r for r in rows if r["mode"] == mode]
        in_scope_scores = [r["top_score"] for r in mode_rows if not r["out_of_scope"]]
        oos_scores = [r["top_score"] for r in mode_rows if r["out_of_scope"]]
        for r in mode_rows:
            print(f"  {r['id']}  oos={r['out_of_scope']!s:5}  top_score={r['top_score']:.4f}  source_hit={r['source_hit']}")
        if in_scope_scores:
            print(f"  範囲内 最低={min(in_scope_scores):.4f} 最高={max(in_scope_scores):.4f}")
        if oos_scores:
            print(f"  範囲外 スコア={oos_scores}")

    print(f"\n保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
