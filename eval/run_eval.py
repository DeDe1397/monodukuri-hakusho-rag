"""検索方式（vector/bm25/hybrid）ごとに評価質問を実行し、比較表をCSV/Markdownで出力する。

自動判定できるのは「期待ページが検索結果に含まれたか」（source_hit・source_rank）と
「範囲外質問を正しく拒否できたか（used_llm=False）」の2つだけ。
expected_points（論点を満たしたか）とmust_not（幻覚がないか）は
文章の意味を読む必要があり自動判定が信頼できないため、判定欄を
空欄のまま出力し、レポート作成時に目視で埋める前提にしている。

source_hitは「正解が上位5件に入っているか」の2値（Hit Rate）。これだけでは
「1位で当たったか5位ギリギリで当たったか」が分からず、例えば検索成功率が同じ
89%の方式どうしでも、順位の良さが違う場合に区別できない。そのためsource_rank
（正解が見つかった順位、見つからなければNone）も記録し、MRR（Mean Reciprocal
Rank = 1/順位 の平均）を集計する。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_questions import EVAL_QUESTIONS
from pipeline import answer_question

MODES = ["vector", "bm25", "hybrid"]
OUT_DIR = Path(__file__).resolve().parent / "results"


def _source_rank(expected_source: list[int], retrieved_pages: list[int]) -> int | None:
    """expected_sourceのどれかが最初に出現する順位（1始まり）。無ければNone。

    retrieved_pagesは検索結果の順序そのもの（上位が先頭）なので、
    set化せずリストのまま先頭から見ていくことで順位が分かる。
    """
    expected = set(expected_source)
    for i, page in enumerate(retrieved_pages):
        if page in expected:
            return i + 1
    return None


def _run_one(question: dict, mode: str) -> dict:
    """1問・1検索方式を実行し、CSV1行分の辞書を返す。

    自動判定するのは source_hit/source_rank（expected_sourceのページが検索結果に
    含まれたか・何位だったか）と rejected_correctly（out_of_scope質問でLLMを
    呼ばずに拒否できたか）。expected_points_met・must_not_violatedは意味の判定が
    必要なため、空文字列のまま出力し人間の目視判定に委ねる。
    """
    result = answer_question(question["question"], mode=mode)
    retrieved_pages = [c["page"] for c in result["contexts"]]

    if question["out_of_scope"]:
        source_hit = None
        source_rank = None
        rejected_correctly = result["used_llm"] is False
    else:
        rank = _source_rank(question["expected_source"], retrieved_pages)
        source_hit = rank is not None
        source_rank = rank
        rejected_correctly = None

    return {
        "mode": mode,
        "id": question["id"],
        "question": question["question"],
        "answer": result["answer"],
        "retrieved_pages": ",".join(str(p) for p in retrieved_pages),
        "expected_source": ",".join(str(p) for p in question["expected_source"]),
        "source_hit": source_hit,
        "source_rank": source_rank,
        "out_of_scope": question["out_of_scope"],
        "rejected_correctly": rejected_correctly,
        "expected_points": " / ".join(question["expected_points"]),
        "expected_points_met": "",  # 手動判定欄
        "must_not": " / ".join(question["must_not"]),
        "must_not_violated": "",  # 手動判定欄
    }


def _write_detail_csv(rows: list[dict], path: Path) -> None:
    """質問×検索方式ごとの詳細結果を1行1件のCSVに書き出す。"""
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict[str, dict]:
    """詳細行を検索方式ごとに集計し、検索成功率・MRR・範囲外質問の正答拒否率を算出する。

    検索成功率（Hit Rate）は「上位5件に正解が入っているか」の2値なので、
    1位で当たったか5位ギリギリで当たったかを区別できない。MRR（1/順位 の平均、
    見つからなければ0）は順位の良さまで反映するため、Hit Rateが同じ方式間でも
    順位の質で差が出ることがある（実測ではbm25/hybridがvectorより
    Hit Rateは低いのにMRRは高かった。README参照）。
    """
    summary: dict[str, dict] = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        scoped = [r for r in mode_rows if not r["out_of_scope"]]
        out_of_scope = [r for r in mode_rows if r["out_of_scope"]]

        n_scoped = len(scoped) or 1
        n_oos = len(out_of_scope) or 1
        mrr = sum((1 / r["source_rank"]) if r["source_rank"] else 0 for r in scoped) / n_scoped
        summary[mode] = {
            "検索成功率": f"{sum(1 for r in scoped if r['source_hit']) / n_scoped:.0%}",
            "MRR": f"{mrr:.2f}",
            "範囲外質問の正答拒否率": f"{sum(1 for r in out_of_scope if r['rejected_correctly']) / n_oos:.0%}",
            "対象質問数": len(scoped),
            "範囲外質問数": len(out_of_scope),
        }
    return summary


def _write_summary_markdown(summary: dict[str, dict], path: Path) -> None:
    """検索方式×指標の比較表を、レポートにそのまま貼れるMarkdown表として出力する。"""
    metrics = list(next(iter(summary.values())).keys())
    lines = ["| 検索方式 | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for mode, values in summary.items():
        lines.append(f"| {mode} | " + " | ".join(str(values[m]) for m in metrics) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """EVAL_QUESTIONS全問を3検索方式で実行し、詳細CSVと比較表（Markdown/CSV）を出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in MODES:
        print(f"=== {mode} ===")
        for question in EVAL_QUESTIONS:
            row = _run_one(question, mode)
            rows.append(row)
            print(
                f"  {row['id']}: source_hit={row['source_hit']} source_rank={row['source_rank']} "
                f"rejected_correctly={row['rejected_correctly']}"
            )

    _write_detail_csv(rows, OUT_DIR / "eval_detail.csv")
    summary = _summarize(rows)
    _write_summary_markdown(summary, OUT_DIR / "eval_summary.md")

    with open(OUT_DIR / "eval_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        metrics = list(next(iter(summary.values())).keys())
        writer = csv.writer(f)
        writer.writerow(["検索方式", *metrics])
        for mode, values in summary.items():
            writer.writerow([mode, *[values[m] for m in metrics]])

    print(f"\n詳細: {OUT_DIR / 'eval_detail.csv'}")
    print(f"比較表: {OUT_DIR / 'eval_summary.md'} / {OUT_DIR / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
