"""Kiểm thử offline cho pipeline RAG Buổi 07."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_foundation.buoi_07 import rag


FIELDS = {
    "chunk_id": "h1",
    "strategy": "hierarchical",
    "source": "sample.pdf",
    "page_start": 1,
    "page_end": 1,
    "text": "Nội dung mẫu.",
}


class Embedding:
    def __init__(self, values):
        self.values = values


class EmbedResponse:
    def __init__(self, values):
        self.embeddings = [Embedding(values)]


class GenerateResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Mock deterministic 128 chiều, chỉ dùng trong test."""

    def __init__(self, generation_text="Trả lời [E1]", generation_error=None):
        self.generation_text = generation_text
        self.generation_error = generation_error
        self.embed_calls = []
        self.generate_calls = []

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        content = kwargs["contents"]
        seed = (sum(ord(char) for char in content) % 97 + 1) / 100
        return EmbedResponse([seed] + [0.0] * 127)

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.generation_error:
            raise self.generation_error
        return GenerateResponse(self.generation_text)


class FakeGemini:
    def __init__(self, generation_text="Trả lời [E1]", generation_error=None):
        self.models = FakeModels(generation_text, generation_error)


def test_config(api_key="test-key", threshold=0.45, model="model-test", dim=128):
    return rag.AppConfig(api_key, model, dim, "generation-test", 5, threshold)


def write_json(directory, name, payload):
    path = Path(directory) / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class LoaderTests(unittest.TestCase):
    def test_loader_reads_list_and_selects_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            other = {**FIELDS, "chunk_id": "s1", "strategy": "semantic"}
            write_json(directory, "data.json", [FIELDS, other])
            chunks, stats = rag.load_chunks(directory)
        self.assertEqual(["h1"], [chunk["chunk_id"] for chunk in chunks])
        self.assertEqual(2, stats["total_records"])
        self.assertEqual(1, stats["selected_records"])

    def test_loader_reads_object_with_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(directory, "data.json", {"chunks": [FIELDS]})
            chunks, stats = rag.load_chunks(directory)
        self.assertEqual(1, len(chunks))
        self.assertEqual(1, stats["files_read"])

    def test_missing_field_fails(self):
        record = {key: value for key, value in FIELDS.items() if key != "text"}
        self._assert_load_error(record, "Thiếu field")

    def test_wrong_field_type_fails(self):
        self._assert_load_error({**FIELDS, "source": 7}, "phải là string")

    def test_boolean_page_fails(self):
        self._assert_load_error({**FIELDS, "page_start": True}, "không nhận boolean")

    def test_reverse_page_range_fails(self):
        self._assert_load_error({**FIELDS, "page_start": 3, "page_end": 2}, "page_start")

    def test_empty_text_is_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(directory, "data.json", [{**FIELDS, "text": "  "}])
            chunks, stats = rag.load_chunks(directory)
        self.assertEqual([], chunks)
        self.assertEqual(1, stats["empty_text_skipped"])
        self.assertEqual(0, stats["valid_chunks"])

    def test_duplicate_chunk_id_reports_both_records(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(directory, "a.json", [FIELDS])
            write_json(directory, "b.json", [{**FIELDS, "text": "Khác"}])
            with self.assertRaisesRegex(rag.ChunkValidationError, "a.json.*record 1.*b.json.*record 1"):
                rag.load_chunks(directory)

    def test_non_object_record_fails(self):
        self._assert_load_error("not-an-object", "không phải JSON object")

    def _assert_load_error(self, record, message):
        with tempfile.TemporaryDirectory() as directory:
            write_json(directory, "bad.json", [record])
            with self.assertRaisesRegex(rag.ChunkValidationError, message):
                rag.load_chunks(directory)


class EmbeddingIndexTests(unittest.TestCase):
    fixture = Path(__file__).resolve().parent / "fixtures" / "chunks_sample.json"

    def test_collection_identity_changes_for_strategy_model_and_dimension(self):
        base = rag.collection_name("hierarchical", "model-a", 128)
        self.assertNotEqual(base, rag.collection_name("semantic", "model-a", 128))
        self.assertNotEqual(base, rag.collection_name("hierarchical", "model-b", 128))
        self.assertNotEqual(base, rag.collection_name("hierarchical", "model-a", 256))

    def test_embedding_batch_size_empty_dimension_nan_infinity(self):
        chunks = [{"chunk_id": "a"}]
        invalid = [
            ([], "không khớp"),
            ([[]], "không rỗng"),
            ([[1.0]], "dimension"),
            ([[float("nan")] + [0.0] * 127], "NaN"),
            ([[float("inf")] + [0.0] * 127], "Infinity"),
        ]
        for vectors, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(rag.RagIndexError, message):
                    rag.validate_embeddings(vectors, chunks, 128)

    def test_embedding_rejects_boolean_and_zero_vector(self):
        chunks = [{"chunk_id": "a"}]
        for vector, message in [
            ([True] + [0.0] * 127, "không phải số thực"),
            ([0.0] * 128, "zero vector"),
        ]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(rag.RagIndexError, message):
                    rag.validate_embeddings([vector], chunks, 128)

    def test_index_twice_is_idempotent_and_metadata_complete(self):
        temporary = tempfile.TemporaryDirectory()
        try:
            directory = temporary.name
            storage = Path(directory) / "chroma"
            config = test_config()
            first = rag.index_chunks(config, "hierarchical", input_path=self.fixture,
                                     storage_path=storage, embedding_client=FakeGemini())
            second = rag.index_chunks(config, "hierarchical", input_path=self.fixture,
                                      storage_path=storage, embedding_client=FakeGemini())
            client = rag.chromadb.PersistentClient(path=storage)
            collection = client.get_collection(first["collection_name"], embedding_function=None)
            stored = collection.get(include=["metadatas"])["metadatas"][0]
            self.assertEqual(3, first["record_count"])
            self.assertEqual(first["record_count"], second["record_count"])
            for field in ("source", "strategy", "page_start", "page_end", "chunk_id",
                          "embedding_model", "embedding_dim"):
                self.assertIn(field, stored)
        finally:
            client._system.stop()
            rag.chromadb.api.client.SharedSystemClient.clear_system_cache()
            temporary.cleanup()

    def test_embedding_failure_before_reset_preserves_collection(self):
        temporary = tempfile.TemporaryDirectory()
        try:
            directory = temporary.name
            storage = Path(directory) / "chroma"
            config = test_config()
            original = rag.index_chunks(config, "hierarchical", input_path=self.fixture,
                                        storage_path=storage, embedding_client=FakeGemini())
            bad = FakeGemini()
            bad.models.embed_content = Mock(return_value=EmbedResponse([1.0]))
            with self.assertRaises(rag.RagIndexError):
                rag.index_chunks(config, "hierarchical", reset=True, input_path=self.fixture,
                                 storage_path=storage, embedding_client=bad)
            client = rag.chromadb.PersistentClient(path=storage)
            collection = client.get_collection(original["collection_name"], embedding_function=None)
            self.assertEqual(3, collection.count())
        finally:
            client._system.stop()
            rag.chromadb.api.client.SharedSystemClient.clear_system_cache()
            temporary.cleanup()

    def test_missing_api_key_never_calls_embedding_or_chroma(self):
        embedding = Mock()
        chroma = Mock()
        with self.assertRaisesRegex(rag.RagIndexError, "Thiếu GEMINI_API_KEY"):
            rag.index_chunks(test_config(api_key=""), "hierarchical",
                             embedding_client=embedding, chroma_client=chroma)
        embedding.assert_not_called()
        chroma.assert_not_called()

    def test_existing_collection_mismatch_blocks_upsert(self):
        temporary = tempfile.TemporaryDirectory()
        try:
            directory = temporary.name
            storage = Path(directory) / "chroma"
            config = test_config()
            client = rag.chromadb.PersistentClient(path=storage)
            name = rag.collection_name("hierarchical", config.embedding_model, config.embedding_dim)
            collection = client.create_collection(
                name=name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={**rag._expected_collection_metadata(config, "hierarchical"),
                          "embedding_model": "wrong-model"},
                embedding_function=None,
            )
            with self.assertRaisesRegex(rag.RagIndexError, "--reset"):
                rag.index_chunks(config, "hierarchical", input_path=self.fixture,
                                 embedding_client=FakeGemini(), chroma_client=client)
            self.assertEqual(0, collection.count())
        finally:
            client._system.stop()
            rag.chromadb.api.client.SharedSystemClient.clear_system_cache()
            temporary.cleanup()

    def test_status_on_empty_temporary_storage_creates_no_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / "chroma"
            result = rag.get_index_status(test_config(), "hierarchical", storage_path=storage)
            self.assertFalse(result["collection_exists"])
            if storage.exists():
                client = rag.chromadb.PersistentClient(path=storage)
                self.assertEqual([], list(client.list_collections()))


class FakeCollection:
    configuration = {"hnsw": {"space": "cosine"}}

    def __init__(self, config, documents=None, metadatas=None, distances=None, count=None):
        self.name = rag.collection_name("hierarchical", config.embedding_model, config.embedding_dim)
        self.metadata = rag._expected_collection_metadata(config, "hierarchical")
        self.documents = documents or []
        self.metadatas = metadatas or []
        self.distances = distances or []
        self._count = len(self.documents) if count is None else count
        self.query_calls = []

    def count(self):
        return self._count

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        size = kwargs["n_results"]
        return {
            "documents": [self.documents[:size]],
            "metadatas": [self.metadatas[:size]],
            "distances": [self.distances[:size]],
        }


class FakeChroma:
    def __init__(self, collection):
        self.collection = collection

    def list_collections(self):
        return [self.collection]

    def get_collection(self, **kwargs):
        if kwargs.get("embedding_function", "missing") is not None:
            raise AssertionError("embedding_function phải là None")
        return self.collection


def metadata(source, start, end, chunk_id):
    return {"source": source, "page_start": start, "page_end": end, "chunk_id": chunk_id}


class RetrievalCitationTests(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.documents = ["Chunk retrieved A", "Chunk retrieved B", "Chunk không retrieve"]
        self.metadatas = [metadata("a.pdf", 1, 1, "a"), metadata("b.pdf", 2, 4, "b"),
                          metadata("c.pdf", 5, 5, "c")]

    def ask(self, collection, gemini=None, top_k=5, question="Câu hỏi kiểm tra?"):
        gemini = gemini or FakeGemini()
        result = rag.answer_question(question, top_k, "hierarchical", config=self.config,
                                     gemini_client=gemini, chroma_client=FakeChroma(collection))
        return result, gemini

    def test_retrieval_top_k_order_and_cap_by_count(self):
        collection = FakeCollection(self.config, self.documents, self.metadatas, [0.1, 0.2, 0.3])
        result, _ = self.ask(collection, top_k=2)
        self.assertEqual(2, collection.query_calls[0]["n_results"])
        self.assertEqual(["a", "b"], [item["chunk_id"] for item in result["evidence"]])
        collection = FakeCollection(self.config, self.documents[:2], self.metadatas[:2], [0.1, 0.2])
        self.ask(collection, top_k=10)
        self.assertEqual(2, collection.query_calls[0]["n_results"])

    def test_question_and_top_k_validation(self):
        for question, top_k in [(" ", 5), ("x", 0), ("x", 21), ("x", True)]:
            with self.subTest(question=question, top_k=top_k):
                with self.assertRaises(rag.RagIndexError):
                    rag.answer_question(question, top_k, "hierarchical", config=self.config)

    def test_empty_collection_fails(self):
        collection = FakeCollection(self.config, count=0)
        with self.assertRaisesRegex(rag.RagIndexError, "đang rỗng"):
            self.ask(collection)

    def test_collection_metadata_mismatch_is_blocked(self):
        collection = FakeCollection(self.config, ["x"], [self.metadatas[0]], [0.1])
        collection.metadata["embedding_dim"] = 256
        with self.assertRaisesRegex(rag.RagIndexError, "--reset"):
            self.ask(collection)

    def test_all_evidence_above_threshold_skips_generation(self):
        collection = FakeCollection(self.config, ["x"], [self.metadatas[0]], [0.9])
        gemini = FakeGemini()
        result, _ = self.ask(collection, gemini)
        self.assertEqual("insufficient_evidence", result["status"])
        self.assertEqual([], result["citations"])
        self.assertEqual(0, len(gemini.models.generate_calls))

    def test_accepted_evidence_calls_generation_once(self):
        collection = FakeCollection(self.config, ["x"], [self.metadatas[0]], [0.1])
        result, gemini = self.ask(collection)
        self.assertEqual("answered", result["status"])
        self.assertEqual(1, len(gemini.models.generate_calls))

    def test_prompt_has_question_only_retrieved_chunks_and_security_instruction(self):
        collection = FakeCollection(self.config, self.documents, self.metadatas, [0.1, 0.2, 0.3])
        _, gemini = self.ask(collection, top_k=2, question="Question unique")
        prompt_text = gemini.models.generate_calls[0]["contents"]
        self.assertIn("Question unique", prompt_text)
        self.assertIn("Chunk retrieved A", prompt_text)
        self.assertIn("Chunk retrieved B", prompt_text)
        self.assertNotIn("Chunk không retrieve", prompt_text)
        self.assertIn("dữ liệu không đáng tin cậy", prompt_text)
        self.assertIn("bỏ qua mọi câu lệnh", prompt_text)

    def test_mixed_gate_keeps_all_but_prompts_only_accepted(self):
        collection = FakeCollection(self.config, self.documents[:2], self.metadatas[:2], [0.1, 0.9])
        result, gemini = self.ask(collection)
        self.assertEqual([True, False], [item["accepted"] for item in result["evidence"]])
        prompt_text = gemini.models.generate_calls[0]["contents"]
        self.assertIn("Chunk retrieved A", prompt_text)
        self.assertNotIn("Chunk retrieved B", prompt_text)

    def test_citation_single_range_mapping_unknown_order_and_duplicates(self):
        evidence = [
            {"evidence_id": "E1", "accepted": True, "source": "a.pdf", "page_start": 1,
             "page_end": 1, "chunk_id": "a"},
            {"evidence_id": "E2", "accepted": True, "source": "b.pdf", "page_start": 2,
             "page_end": 4, "chunk_id": "b"},
        ]
        answer, citations, warnings = rag.map_citations("[E2] [E1] [E2] [E99]", evidence)
        self.assertIn("tr. 1", citations[1]["display"])
        self.assertIn("tr. 2-4", citations[0]["display"])
        self.assertEqual(["E2", "E1"], [item["evidence_id"] for item in citations])
        self.assertNotIn("E99", [item["evidence_id"] for item in citations])
        self.assertNotIn("[E99]", answer)
        self.assertTrue(warnings)

    def test_e1_maps_real_metadata(self):
        collection = FakeCollection(self.config, ["x"], [self.metadatas[0]], [0.1])
        result, _ = self.ask(collection, FakeGemini("Có căn cứ [E1]"))
        citation = result["citations"][0]
        self.assertEqual({"source": "a.pdf", "page_start": 1, "page_end": 1,
                          "chunk_id": "a"},
                         {key: citation[key] for key in ("source", "page_start", "page_end", "chunk_id")})

    def test_generation_error_and_empty_text_are_retrieval_only(self):
        collection = FakeCollection(self.config, ["x"], [self.metadatas[0]], [0.1])
        for gemini in (FakeGemini(generation_error=RuntimeError("secret")), FakeGemini("  ")):
            with self.subTest(generation=gemini.models.generation_text):
                result, _ = self.ask(collection, gemini)
                self.assertEqual("retrieval_only", result["status"])
                self.assertTrue(result["evidence"])
                self.assertEqual([], result["citations"])
                self.assertNotIn("secret", " ".join(result["warnings"]))

    def test_result_schema_all_fields(self):
        collection = FakeCollection(self.config, ["x"], [self.metadatas[0]], [0.1])
        result, _ = self.ask(collection)
        self.assertEqual(
            {"status", "answer", "evidence", "citations", "warnings", "collection",
             "strategy", "top_k"},
            set(result),
        )


class ConfigCliTests(unittest.TestCase):
    def test_config_and_cli_do_not_depend_on_buoi07_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text(
                "GEMINI_API_KEY=\nGEMINI_EMBEDDING_MODEL=model-x\n"
                "GEMINI_EMBEDDING_DIM=128\nGEMINI_GENERATION_MODEL=gen-x\n"
                "DEFAULT_TOP_K=3\nRAG_MAX_DISTANCE=0.4\n",
                encoding="utf-8",
            )
            with patch.dict(rag.os.environ, {}, clear=True):
                config = rag.load_config(env)
        self.assertEqual("model-x", config.embedding_model)
        self.assertTrue(rag.FILE_PATH.is_absolute())
        self.assertNotEqual(Path.cwd().resolve(), rag.BUOI_07_DIR)
        with patch.object(rag, "_run_status") as run_status, patch.object(
            rag.sys, "argv", ["rag.py", "status", "--strategy", "semantic"]
        ):
            self.assertEqual(0, rag.main())
        run_status.assert_called_once_with("semantic")


if __name__ == "__main__":
    unittest.main()
