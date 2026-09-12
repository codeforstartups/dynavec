"""Comprehensive test suite for dynavec.eval (LLM Judge, Faithfulness, Relevance, Runner, and Dashboard)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from dynavec.dashboard import _make_handler
from dynavec.eval import (
    AnswerRelevanceResult,
    CustomJudge,
    EvalRunner,
    EvalSummary,
    FaithfulnessResult,
    GeminiJudge,
    MockJudge,
    OpenAIJudge,
    RAGEvalResult,
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_rag,
    extract_json,
)
from dynavec.exceptions import MissingDependencyError
from dynavec.telemetry import TelemetryRecorder, aggregate, aggregate_eval

# ===========================================================================
# 1. JSON Extraction Robustness Tests
# ===========================================================================


class TestExtractJson:
    def test_raw_json(self) -> None:
        data = extract_json('{"score": 0.95, "reasoning": "great"}')
        assert data == {"score": 0.95, "reasoning": "great"}

    def test_markdown_json_fence(self) -> None:
        text = 'Here is the result:\n```json\n{\n  "score": 1.0,\n  "reasoning": "perfect"\n}\n```'
        data = extract_json(text)
        assert data == {"score": 1.0, "reasoning": "perfect"}

    def test_markdown_code_fence_no_lang(self) -> None:
        text = '```\n{\n  "score": 0.8\n}\n```'
        data = extract_json(text)
        assert data == {"score": 0.8}

    def test_embedded_json_in_text(self) -> None:
        text = 'Analysis completed. Result: {"claims": [{"claim": "A", "supported": true}]} End of report.'
        data = extract_json(text)
        assert len(data["claims"]) == 1
        assert data["claims"][0]["supported"] is True

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Empty response"):
            extract_json("   ")

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Could not parse valid JSON"):
            extract_json("This is purely plain text without any braces.")


# ===========================================================================
# 2. Pluggable Judge Implementation Tests
# ===========================================================================


class TestJudges:
    def test_mock_judge_sequential_and_default(self) -> None:
        resp1 = {"claims": [{"claim": "c1", "supported": True}], "reasoning": "ok"}
        resp2 = {"score": 0.9, "reasoning": "relevant"}
        judge = MockJudge(responses=[resp1, resp2], default_response={"score": 0.5})

        assert judge.judge_structured("prompt 1") == resp1
        assert judge.judge_structured("prompt 2") == resp2
        assert judge.judge_structured("prompt 3") == {"score": 0.5}
        assert len(judge.call_history) == 3

    def test_custom_judge(self) -> None:
        def fn(p: str) -> str:
            return json.dumps({"score": 0.85, "reasoning": f"Evaluated {len(p)} chars"})

        judge = CustomJudge(fn)
        res = judge.judge_structured("hello world")
        assert res["score"] == 0.85

    def test_custom_judge_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a callable"):
            CustomJudge("not a function")  # type: ignore[arg-type]

    def test_openai_judge_missing_dep_raises(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                OpenAIJudge()
            assert "OpenAIJudge" in str(exc_info.value)
            assert "openai" in str(exc_info.value)

    def test_gemini_judge_missing_dep_raises(self) -> None:
        with patch.dict("sys.modules", {"google.generativeai": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                GeminiJudge()
            assert "GeminiJudge" in str(exc_info.value)
            assert "google-generativeai" in str(exc_info.value)

    def test_openai_judge_mocked_execution(self) -> None:
        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 0.99, "reasoning": "perfect"}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_openai_module.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            judge = OpenAIJudge(model="gpt-4o-mini", api_key="sk-test")
            result = judge.judge_structured("evaluate this")
            assert result["score"] == 0.99
            mock_client.chat.completions.create.assert_called_once()

    def test_gemini_judge_mocked_execution(self) -> None:
        mock_google = MagicMock()
        mock_genai_module = MagicMock()
        mock_google.generativeai = mock_genai_module
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"score": 0.92, "reasoning": "good"}'
        mock_model.generate_content.return_value = mock_resp
        mock_genai_module.GenerativeModel.return_value = mock_model

        with patch.dict(
            "sys.modules", {"google": mock_google, "google.generativeai": mock_genai_module}
        ):
            judge = GeminiJudge(model="gemini-1.5-flash", api_key="test-key")
            result = judge.judge_structured("test prompt")
            assert result["score"] == 0.92
            mock_genai_module.configure.assert_called_once_with(api_key="test-key")


# ===========================================================================
# 3. Faithfulness Metric Evaluation Tests
# ===========================================================================


class TestFaithfulnessMetric:
    def test_perfect_faithfulness(self) -> None:
        judge_output = {
            "claims": [
                {
                    "claim": "Dynavec uses S3 Vectors for ANN.",
                    "supported": True,
                    "reasoning": "Direct match.",
                },
                {
                    "claim": "Dynavec stores docs in DynamoDB.",
                    "supported": True,
                    "reasoning": "Direct match.",
                },
            ],
            "reasoning": "All claims grounded.",
        }
        judge = MockJudge(default_response=judge_output)

        res = evaluate_faithfulness(
            query="How does dynavec work?",
            context=["Dynavec uses S3 Vectors for ANN and DynamoDB for document storage."],
            answer="Dynavec uses S3 Vectors for ANN. Dynavec stores docs in DynamoDB.",
            judge=judge,
        )

        assert res.score == 1.0
        assert res.supported_count == 2
        assert res.total_count == 2
        assert len(res.claims) == 2
        assert res.reasoning == "All claims grounded."

    def test_partial_hallucination(self) -> None:
        judge_output = {
            "claims": [
                {"claim": "Dynavec uses S3 Vectors.", "supported": True, "reasoning": "Supported."},
                {
                    "claim": "Dynavec was created in 1995.",
                    "supported": False,
                    "reasoning": "Not mentioned.",
                },
            ],
            "reasoning": "Half of the claims are hallucinated.",
        }
        judge = MockJudge(default_response=judge_output)

        res = evaluate_faithfulness(
            query="What is dynavec?",
            context="Dynavec uses S3 Vectors for search.",
            answer="Dynavec uses S3 Vectors and was created in 1995.",
            judge=judge,
        )

        assert res.score == 0.5
        assert res.supported_count == 1
        assert res.total_count == 2
        assert res.claims[1].supported is False

    def test_complete_hallucination(self) -> None:
        judge_output = {
            "claims": [
                {"claim": "Paris is in Germany.", "supported": False, "reasoning": "Contradicted."},
            ],
            "reasoning": "Entire answer is false.",
        }
        judge = MockJudge(default_response=judge_output)

        res = evaluate_faithfulness(
            query="Where is Paris?",
            context="Paris is the capital of France.",
            answer="Paris is in Germany.",
            judge=judge,
        )

        assert res.score == 0.0
        assert res.supported_count == 0
        assert res.total_count == 1

    def test_empty_answer(self) -> None:
        judge = MockJudge()
        res = evaluate_faithfulness(
            query="Any query",
            context=["Context chunk"],
            answer="",
            judge=judge,
        )
        assert res.score == 0.0
        assert len(res.claims) == 0
        assert "Empty answer" in res.reasoning

    def test_empty_context(self) -> None:
        judge = MockJudge()
        res = evaluate_faithfulness(
            query="Any query",
            context=[],
            answer="Dynavec is fast.",
            judge=judge,
        )
        assert res.score == 0.0
        assert len(res.claims) == 1
        assert res.claims[0].supported is False
        assert "Empty context" in res.reasoning


# ===========================================================================
# 4. Answer Relevance Metric Evaluation Tests
# ===========================================================================


class TestAnswerRelevanceMetric:
    def test_high_relevance(self) -> None:
        judge = MockJudge(
            default_response={"score": 0.95, "reasoning": "Direct and concise answer."}
        )
        res = evaluate_answer_relevance(
            query="How much does dynavec cost?",
            answer="Dynavec is serverless and costs around $3/mo for base DynamoDB + S3 Vectors.",
            judge=judge,
        )
        assert res.score == 0.95
        assert res.reasoning == "Direct and concise answer."

    def test_low_relevance_off_topic(self) -> None:
        judge = MockJudge(
            default_response={
                "score": 0.1,
                "reasoning": "Answer discusses weather instead of pricing.",
            }
        )
        res = evaluate_answer_relevance(
            query="How much does dynavec cost?",
            answer="The weather in Seattle is rainy today.",
            judge=judge,
        )
        assert res.score == 0.1

    def test_empty_query_or_answer(self) -> None:
        judge = MockJudge()
        assert evaluate_answer_relevance(query="", answer="test", judge=judge).score == 0.0
        assert evaluate_answer_relevance(query="test", answer="", judge=judge).score == 0.0


# ===========================================================================
# 5. Combined evaluate_rag & EvalRunner Tests
# ===========================================================================


class TestEvalRunner:
    def test_evaluate_rag_combined(self) -> None:
        f_resp = {
            "claims": [{"claim": "Fact 1", "supported": True}],
            "reasoning": "Grounded",
        }
        r_resp = {"score": 0.9, "reasoning": "Relevant"}
        judge = MockJudge(responses=[f_resp, r_resp])

        rag_res = evaluate_rag(
            query="What is dynavec?",
            context=["Dynavec is a vector DB."],
            answer="Dynavec is a vector DB.",
            judge=judge,
        )

        assert isinstance(rag_res, RAGEvalResult)
        assert rag_res.faithfulness is not None
        assert rag_res.faithfulness.score == 1.0
        assert rag_res.answer_relevance is not None
        assert rag_res.answer_relevance.score == 0.9
        assert rag_res.latency_ms >= 0.0

        d = rag_res.to_dict()
        assert d["query"] == "What is dynavec?"
        assert d["faithfulness"]["score"] == 1.0

    def test_eval_runner_batch_dataset(self) -> None:
        # Sample 1: Faithful (1.0) & Relevant (0.9) -> Pass
        # Sample 2: Partial (0.5) & Relevant (0.8) -> Fail (since pass_threshold=0.7)
        resp1_f = {"claims": [{"claim": "c1", "supported": True}]}
        resp1_r = {"score": 0.9}
        resp2_f = {
            "claims": [{"claim": "c2", "supported": True}, {"claim": "c3", "supported": False}]
        }
        resp2_r = {"score": 0.8}

        judge = MockJudge(responses=[resp1_f, resp1_r, resp2_f, resp2_r])
        runner = EvalRunner(judge=judge, pass_threshold=0.7)

        dataset = [
            {
                "query": "Q1",
                "context": ["C1"],
                "answer": "A1",
            },
            ("Q2", ["C2"], "A2"),
        ]

        summary = runner.run(dataset)
        assert isinstance(summary, EvalSummary)
        assert summary.total_samples == 2
        assert summary.mean_faithfulness == pytest.approx(0.75)
        assert summary.mean_relevance == pytest.approx(0.85)
        assert summary.pass_rate == 0.5  # 1 of 2 passed
        assert summary.pass_threshold == 0.7
        assert len(summary.results) == 2

        summary_dict = summary.to_dict()
        assert summary_dict["total_samples"] == 2
        assert len(summary_dict["results"]) == 2


# ===========================================================================
# 6. Telemetry & Dashboard API Integration Tests
# ===========================================================================


class TestTelemetryAndDashboardIntegration:
    def test_telemetry_event_eval_fields(self) -> None:
        rec = TelemetryRecorder()
        ev = rec.new_event(
            op="search",
            namespace="default",
            latency_ms=42.0,
            eval_faithfulness=0.95,
            eval_relevance=0.88,
        )
        rec.record(ev)

        assert ev.eval_faithfulness == 0.95
        assert ev.eval_relevance == 0.88
        d = ev.to_dict()
        assert d["eval_faithfulness"] == 0.95
        assert d["eval_relevance"] == 0.88

    def test_aggregate_includes_eval_means(self) -> None:
        rec = TelemetryRecorder()
        rec.record(rec.new_event("search", eval_faithfulness=1.0, eval_relevance=0.9))
        rec.record(rec.new_event("search", eval_faithfulness=0.8, eval_relevance=0.7))
        rec.record(rec.new_event("search"))  # unevaluated event

        stats = aggregate(rec.snapshot())
        assert stats["eval_faithfulness_mean"] == 0.9
        assert stats["eval_relevance_mean"] == 0.8
        assert stats["eval_count"] == 2

    def test_aggregate_eval_function(self) -> None:
        rec = TelemetryRecorder()
        rec.record(rec.new_event("search", eval_faithfulness=1.0, eval_relevance=0.9))  # pass
        rec.record(rec.new_event("search", eval_faithfulness=0.5, eval_relevance=0.8))  # fail
        rec.record(rec.new_event("upsert"))  # ignored by eval aggregation

        summary = aggregate_eval(rec.snapshot())
        assert summary["total_evals"] == 2
        assert summary["mean_faithfulness"] == 0.75
        assert summary["mean_relevance"] == 0.85
        assert summary["pass_rate"] == 0.5

    def test_dashboard_api_eval_endpoints(self) -> None:
        rec = TelemetryRecorder()
        rec.record(
            rec.new_event(
                "search",
                namespace="kb",
                latency_ms=50.0,
                eval_faithfulness=0.92,
                eval_relevance=0.85,
            )
        )

        handler_cls = _make_handler(rec)
        handler = handler_cls.__new__(handler_cls)
        handler.wfile = MagicMock()
        sent_data = {}

        def mock_send(code, body, ctype="application/json"):
            sent_data["code"] = code
            sent_data["body"] = json.loads(body)
            sent_data["ctype"] = ctype

        handler._send = mock_send

        # Test GET /api/eval/summary
        handler.path = "/api/eval/summary"
        handler.do_GET()
        assert sent_data["code"] == 200
        assert sent_data["body"]["total_evals"] == 1
        assert sent_data["body"]["mean_faithfulness"] == 0.92

        # Test GET /api/eval/runs
        handler.path = "/api/eval/runs"
        handler.do_GET()
        assert sent_data["code"] == 200
        assert len(sent_data["body"]) == 1
        assert sent_data["body"][0]["eval_faithfulness"] == 0.92
        assert sent_data["body"][0]["eval_relevance"] == 0.85


# ===========================================================================
# 7. Visual Charting Tests
# ===========================================================================


class TestPlotEvalSummary:
    def test_plot_eval_summary_missing_dep_raises(self) -> None:
        from dynavec.eval import EvalSummary, plot_eval_summary

        summary = EvalSummary(
            total_samples=1,
            mean_faithfulness=0.9,
            mean_relevance=0.8,
            pass_rate=1.0,
            pass_threshold=0.7,
            p95_latency_ms=10.0,
            results=[
                RAGEvalResult(
                    query="Q",
                    context=["C"],
                    answer="A",
                    faithfulness=FaithfulnessResult(score=0.9),
                    answer_relevance=AnswerRelevanceResult(score=0.8),
                )
            ],
        )

        with patch.dict("sys.modules", {"matplotlib": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                plot_eval_summary(summary)
            assert "matplotlib" in str(exc_info.value)
            assert "benchmark" in str(exc_info.value)

    def test_plot_eval_summary_mocked_generation(self) -> None:
        from dynavec.eval import EvalSummary, plot_eval_summary

        r1 = RAGEvalResult(
            query="Q1",
            context=["C1"],
            answer="A1",
            faithfulness=FaithfulnessResult(score=0.9, supported_count=1, total_count=1),
            answer_relevance=AnswerRelevanceResult(score=0.85),
            latency_ms=10.0,
        )

        summary = EvalSummary(
            total_samples=1,
            mean_faithfulness=0.9,
            mean_relevance=0.85,
            pass_rate=1.0,
            pass_threshold=0.7,
            p95_latency_ms=10.0,
            results=[r1],
        )

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_matplotlib = MagicMock()
        mock_matplotlib.pyplot = mock_plt

        with patch.dict(
            "sys.modules", {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}
        ):
            saved_path = plot_eval_summary(summary, output_path="out.png")
            assert saved_path == "out.png"
            mock_plt.savefig.assert_called_once_with("out.png", bbox_inches="tight")
            mock_plt.close.assert_called_once_with(mock_fig)

    def test_plot_eval_summary_empty_results_raises(self) -> None:
        from dynavec.eval import EvalSummary, plot_eval_summary

        empty_summary = EvalSummary(
            total_samples=0,
            mean_faithfulness=0.0,
            mean_relevance=0.0,
            pass_rate=0.0,
            pass_threshold=0.7,
            p95_latency_ms=0.0,
            results=[],
        )

        mock_plt = MagicMock()
        with patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
            with pytest.raises(ValueError, match="Cannot plot empty evaluation results"):
                plot_eval_summary(empty_summary)
