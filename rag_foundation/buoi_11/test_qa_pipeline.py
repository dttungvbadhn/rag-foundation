import unittest

from qa_pipeline import (NO_EVIDENCE_ANSWER, QAConfig, SYSTEM_PROMPT, answer_question,
                         format_context, generation_error_message)


class FakeModels:
    def __init__(self): self.calls = []
    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"text": "Câu trả lời [S1], bổ sung [R1]."})()


class FakeGemini:
    def __init__(self): self.models = FakeModels()


class QuotaModels:
    def generate_content(self, **_kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: 42s secret-payload")


class TransientModels:
    def __init__(self): self.calls = 0
    def generate_content(self, **_kwargs):
        self.calls += 1
        if self.calls < 3:
            error = type("ServerError", (Exception,), {})
            raise error("503 temporary")
        return type("Response", (), {"text": "Đã trả lời [S1]."})()


EVIDENCE = {
    "seeds": [{"element_id": "a", "id": "A", "text": "Điều 1 quy định X.", "labels": ["Chunk"], "score": .9}],
    "related": [{"element_id": "b", "id": "B", "text": "Điều 2 bổ sung Y.", "labels": ["Chunk"],
                 "seed_element_id": "a", "hop": 1, "relationship_path": ["CAN_CU"]}],
}


class QATests(unittest.TestCase):
    def test_prompt_describes_schema_and_legal_structure(self):
        for text in ("CAN_CU", "THAY_THE", "HOP_NHAT", "Điều", "Khoản", "không suy đoán"):
            self.assertIn(text, SYSTEM_PROMPT)

    def test_formats_seed_and_related_context(self):
        context, citations, truncated = format_context(EVIDENCE)
        self.assertIn("[S1]", context); self.assertIn("[R1]", context)
        self.assertEqual(["S1", "R1"], [item["citation"] for item in citations])
        self.assertFalse(truncated)

    def test_empty_evidence_does_not_call_gemini(self):
        client = FakeGemini()
        result = answer_question("x?", retrieval={"seeds": [], "related": []},
                                 qa_config=QAConfig("key"), gemini_client=client)
        self.assertEqual("insufficient_evidence", result["status"])
        self.assertEqual(NO_EVIDENCE_ANSWER, result["answer"])
        self.assertEqual([], client.models.calls)

    def test_calls_gemini_once_with_grounded_prompt(self):
        client = FakeGemini()
        result = answer_question("X là gì?", retrieval=EVIDENCE,
                                 qa_config=QAConfig("key"), gemini_client=client)
        self.assertEqual("answered", result["status"])
        self.assertEqual(1, result["generation_call_count"])
        self.assertEqual(1, len(client.models.calls))
        self.assertIn("<context>", client.models.calls[0]["contents"])
        self.assertEqual("gemini-flash-latest", client.models.calls[0]["model"])

    def test_missing_key_preserves_evidence_without_api_call(self):
        result = answer_question("x?", retrieval=EVIDENCE, qa_config=QAConfig(""))
        self.assertEqual("generation_unavailable", result["status"])
        self.assertEqual(0, result["generation_call_count"])

    def test_quota_error_preserves_evidence_and_is_sanitized(self):
        client = type("Client", (), {"models": QuotaModels()})()
        result = answer_question("x?", retrieval=EVIDENCE, qa_config=QAConfig("key"),
                                 gemini_client=client)
        self.assertEqual("quota_exhausted", result["status"])
        self.assertEqual(2, len(result["citations"]))
        self.assertNotIn("secret-payload", result["warning"])
        self.assertIn("42", result["warning"])

    def test_transient_server_error_is_retried(self):
        models = TransientModels()
        client = type("Client", (), {"models": models})()
        result = answer_question("x?", retrieval=EVIDENCE, qa_config=QAConfig("key"),
                                 gemini_client=client)
        self.assertEqual("answered", result["status"])
        self.assertEqual(3, result["generation_call_count"])


if __name__ == "__main__": unittest.main()
