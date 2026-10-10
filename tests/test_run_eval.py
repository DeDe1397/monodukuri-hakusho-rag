"""run_eval.py の集計ロジックの検証。LLM呼び出し(pipeline.answer_question)はモックする。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import run_eval  # noqa: E402
from eval_questions import EVAL_QUESTIONS  # noqa: E402


def _fake_answer_question(query, mode="hybrid", k=5):
    matching_question = next((q for q in EVAL_QUESTIONS if q["question"] == query), None)
    if matching_question and matching_question["out_of_scope"]:
        return {"answer": "資料内に該当する記述が見つかりませんでした。", "contexts": [], "used_llm": False}
    page = matching_question["expected_source"][0] if matching_question["expected_source"] else 999
    return {
        "answer": "ダミー回答",
        "contexts": [{"id": "c0", "text": "x", "page": page, "section": "s", "score": 0.9}],
        "used_llm": True,
    }


def test_run_one_detects_source_hit(monkeypatch):
    monkeypatch.setattr(run_eval, "answer_question", _fake_answer_question)
    q01 = next(q for q in EVAL_QUESTIONS if q["id"] == "Q01")

    row = run_eval._run_one(q01, "hybrid")

    assert row["source_hit"] is True
    assert row["source_rank"] == 1  # ダミーの検索結果は1件だけ＝常に1位
    assert row["rejected_correctly"] is None  # out_of_scopeでない質問では判定対象外


class TestSourceRank:
    """Hit Rateだけでは区別できない「何位で当たったか」を見るMRR用の下請け関数。"""

    def test_found_at_first_position(self):
        assert run_eval._source_rank([80], [80, 5, 99]) == 1

    def test_found_lower_in_the_list(self):
        assert run_eval._source_rank([80], [18, 5, 80, 273]) == 3

    def test_not_found_returns_none(self):
        assert run_eval._source_rank([80], [18, 5, 99]) is None

    def test_multiple_expected_pages_uses_earliest_match(self):
        # expected_sourceが複数ある質問（Q04等）では、どちらか早く出た方の順位を使う
        assert run_eval._source_rank([7, 8], [18, 8, 7]) == 2


def test_run_one_detects_correct_rejection_for_out_of_scope(monkeypatch):
    monkeypatch.setattr(run_eval, "answer_question", _fake_answer_question)
    q09 = next(q for q in EVAL_QUESTIONS if q["id"] == "Q09")
    assert q09["out_of_scope"] is True

    row = run_eval._run_one(q09, "hybrid")

    assert row["rejected_correctly"] is True
    assert row["source_hit"] is None


def test_summarize_computes_rates_per_mode(monkeypatch):
    monkeypatch.setattr(run_eval, "answer_question", _fake_answer_question)
    rows = [run_eval._run_one(q, "hybrid") for q in EVAL_QUESTIONS]

    summary = run_eval._summarize(rows)

    assert summary["hybrid"]["検索成功率"] == "100%"  # ダミーは常にexpected_source[0]を返すため
    assert summary["hybrid"]["MRR"] == "1.00"  # 検索結果1件だけ＝常に1位なのでMRRも満点
    assert summary["hybrid"]["範囲外質問の正答拒否率"] == "100%"


def test_all_ten_questions_have_required_fields():
    assert len(EVAL_QUESTIONS) == 10
    for q in EVAL_QUESTIONS:
        for key in ("id", "question", "expected_source", "expected_points", "must_not", "out_of_scope"):
            assert key in q
