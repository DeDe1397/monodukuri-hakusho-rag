"""generator.py の検証。LLM呼び出し自体はモックし、閾値ゲートとプロンプト整形だけを見る。"""

import generator


def _ctx(score, page=1, section="第1章", text="本文"):
    return {"id": "c0", "text": text, "page": page, "section": section, "score": score}


class TestThresholdGate:
    def test_below_threshold_skips_llm(self, monkeypatch):
        called = []
        monkeypatch.setattr(generator, "_call_llm", lambda messages: called.append(1) or "呼ばれた")

        result = generator.generate("質問", [_ctx(0.1)])

        assert result["used_llm"] is False
        assert "見つかりませんでした" in result["answer"]
        assert called == []  # LLMが呼ばれていないことを確認

    def test_above_threshold_calls_llm(self, monkeypatch):
        monkeypatch.setattr(generator, "_call_llm", lambda messages: "LLMからの回答")

        result = generator.generate("質問", [_ctx(0.9)])

        assert result["used_llm"] is True
        assert result["answer"] == "LLMからの回答"

    def test_empty_contexts_treated_as_score_zero(self, monkeypatch):
        monkeypatch.setattr(generator, "_call_llm", lambda messages: "呼ばれるべきではない")

        result = generator.generate("質問", [])

        assert result["used_llm"] is False

    def test_memo_below_threshold_uses_memo_specific_message(self, monkeypatch):
        monkeypatch.setattr(generator, "_call_llm", lambda messages: "呼ばれるべきではない")

        result = generator.generate_memo("相談内容", [_ctx(0.1)])

        assert result["used_llm"] is False
        assert result["answer"] == generator._MEMO_NOT_FOUND_MESSAGE

    def test_bm25_uses_lower_threshold_than_vector_and_hybrid(self, monkeypatch):
        """0.20はvector/hybridの閾値(0.30)未満だがbm25の閾値(0.15)以上。
        同じスコアでも方式によってゲートの判定が変わることを確認する
        （評価実行でbm25だけ閾値が高すぎて範囲内質問の大半を落としていた
        不具合の修正。詳細はconfig.pyのMIN_SCORE_THRESHOLDコメント参照）。
        """
        monkeypatch.setattr(generator, "_call_llm", lambda messages: "LLMからの回答")

        bm25_result = generator.generate("質問", [_ctx(0.20)], mode="bm25")
        vector_result = generator.generate("質問", [_ctx(0.20)], mode="vector")
        hybrid_result = generator.generate("質問", [_ctx(0.20)], mode="hybrid")

        assert bm25_result["used_llm"] is True
        assert vector_result["used_llm"] is False
        assert hybrid_result["used_llm"] is False


class TestBuildPrompt:
    def test_includes_page_and_section_citation(self):
        messages = generator.build_prompt("質問", [_ctx(0.9, page=42, section="第3章 テスト")])
        user_content = messages[1]["content"]

        assert "p42" in user_content
        assert "第3章 テスト" in user_content
        assert messages[0]["role"] == "system"

    def test_missing_section_shown_as_unknown(self):
        messages = generator.build_prompt("質問", [_ctx(0.9, section=None)])
        assert "(章節不明)" in messages[1]["content"]

    def test_memo_prompt_uses_memo_system_prompt(self):
        messages = generator.build_memo_prompt("相談内容", [_ctx(0.9)])
        assert messages[0]["content"] == generator._MEMO_SYSTEM_PROMPT
        assert "相談内容" in messages[1]["content"]
