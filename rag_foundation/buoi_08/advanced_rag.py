"""Cấu hình Advanced RAG cho Buổi 08.

Module này chỉ nạp và validate cấu hình. Logic BM25, RRF và reranker sẽ được bổ
sung ở các bước sau; import module không tải model hoặc gọi API.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

try:
    from . import rag as semantic_baseline
except ImportError:  # Chạy trực tiếp: python advanced_rag.py ...
    import rag as semantic_baseline

load_chunks = semantic_baseline.load_chunks


FILE_PATH = Path(__file__).resolve()
BUOI_08_DIR = FILE_PATH.parent
ENV_PATH = BUOI_08_DIR / ".env"
RERANKER_CACHE_PATH = BUOI_08_DIR / "storage" / "huggingface"
_RERANKER_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
ANSWER_MODES = ("bm25", "semantic", "hybrid", "hybrid_rerank")
INSUFFICIENT_ANSWER = "Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp."
RETRIEVAL_ONLY_ANSWER = "Đã truy xuất được nguồn nhưng chưa thể tạo câu trả lời tổng hợp."


class AdvancedConfigError(ValueError):
    """Lỗi cấu hình an toàn để hiển thị, không chứa secret."""


@dataclass(frozen=True)
class AdvancedConfig:
    api_key: str
    embedding_model: str
    embedding_dim: int
    generation_model: str
    max_distance: float
    bm25_candidates: int
    semantic_candidates: int
    rrf_k: int
    rrf_bm25_weight: float
    rrf_semantic_weight: float
    rerank_candidates: int
    final_top_k: int
    reranker_model: str
    reranker_max_length: int
    rerank_batch_size: int
    rerank_min_score: float
    rerank_device: str

    def effective_rerank_candidates(self, union_count: int) -> int:
        """Giới hạn candidate theo union thực tế; union nhỏ không phải config lỗi."""
        if isinstance(union_count, bool) or not isinstance(union_count, int) or union_count < 0:
            raise AdvancedConfigError("union_count phải là integer không âm")
        return min(self.rerank_candidates, union_count)


@dataclass(frozen=True)
class BM25Corpus:
    """BM25 index in-memory cùng bản sao chunk và tokenized corpus."""

    index: BM25Okapi
    chunks: tuple[dict, ...]
    tokenized_corpus: tuple[tuple[str, ...], ...]


def tokenize_vi_legal(text: str) -> list[str]:
    """Tokenize Unicode đơn giản, giữ chữ tiếng Việt và số Điều/Khoản."""
    if not isinstance(text, str):
        raise TypeError("text phải là string")
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def build_bm25_corpus(chunks: list[dict]) -> BM25Corpus:
    """Tạo BM25Okapi in-memory từ chunk đã được baseline loader validate."""
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks phải là list không rỗng đã được validate")
    copied_chunks = tuple(deepcopy(chunk) for chunk in chunks)
    tokenized = tuple(tuple(tokenize_vi_legal(chunk["text"])) for chunk in copied_chunks)
    if any(not tokens for tokens in tokenized):
        raise ValueError("Mọi chunk trong BM25 corpus phải có ít nhất một token")
    return BM25Corpus(
        index=BM25Okapi([list(tokens) for tokens in tokenized]),
        chunks=copied_chunks,
        tokenized_corpus=tokenized,
    )


def bm25_search(question: str, chunks: list[dict], candidate_k: int) -> list[dict]:
    """Trả top BM25 candidate; score cao hơn tốt hơn và không phải xác suất."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question phải là string không rỗng")
    query_tokens = tokenize_vi_legal(question)
    if not query_tokens:
        raise ValueError("question không tạo được token hợp lệ")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1:
        raise ValueError("candidate_k phải là integer dương")

    corpus = build_bm25_corpus(chunks)
    scores = corpus.index.get_scores(query_tokens)
    ranked = sorted(
        zip(corpus.chunks, scores),
        key=lambda item: (-float(item[1]), item[0]["chunk_id"]),
    )[: min(candidate_k, len(corpus.chunks))]
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "bm25_rank": rank,
            "bm25_score": float(score),
        }
        for rank, (chunk, score) in enumerate(ranked, start=1)
    ]


def _baseline_config(config: AdvancedConfig) -> Any:
    """Chuyển config Advanced sang contract semantic baseline mà không tạo pipeline mới."""
    return semantic_baseline.AppConfig(
        api_key=config.api_key,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        generation_model=config.generation_model,
        default_top_k=config.final_top_k,
        max_distance=config.max_distance,
    )


def advanced_status(
    strategy: str,
    config: AdvancedConfig,
    input_path: Path = semantic_baseline.DEFAULT_INPUT_DIR,
    storage_path: Path = semantic_baseline.CHROMA_PATH,
    chroma_client: Any | None = None,
) -> dict[str, Any]:
    """Trả status BM25/semantic/reranker cache mà không gọi API hoặc tạo storage."""
    chunks, _ = load_chunks(input_path, strategy=strategy)
    build_bm25_corpus(chunks)
    semantic = semantic_baseline.get_index_status(
        _baseline_config(config),
        strategy,
        storage_path=storage_path,
        client=chroma_client,
    )
    return {
        "strategy": strategy,
        "corpus_size": len(chunks),
        "semantic_collection_name": semantic["collection_name"],
        "semantic_collection_exists": semantic["collection_exists"],
        "semantic_collection_count": semantic["record_count"],
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "bm25_ready": True,
        "reranker_model": config.reranker_model,
        "reranker_cache_exists": (BUOI_08_DIR / "storage" / "huggingface").is_dir(),
    }


def prepare_semantic(
    strategy: str,
    config: AdvancedConfig,
    input_path: Path = semantic_baseline.DEFAULT_INPUT_DIR,
    storage_path: Path = semantic_baseline.CHROMA_PATH,
    embedding_client: Any | None = None,
    chroma_client: Any | None = None,
) -> dict[str, Any]:
    """Chủ động index semantic bằng Gemini thật; idempotent theo baseline contract."""
    return semantic_baseline.index_chunks(
        _baseline_config(config),
        strategy,
        input_path=input_path,
        storage_path=storage_path,
        embedding_client=embedding_client,
        chroma_client=chroma_client,
    )


def semantic_candidates(
    question: str,
    candidate_k: int,
    strategy: str,
    config: AdvancedConfig,
    storage_path: Path = semantic_baseline.CHROMA_PATH,
    embedding_client: Any | None = None,
    chroma_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Trả semantic candidate theo đúng thứ tự cosine distance của Chroma."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question phải là string không rỗng")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1:
        raise ValueError("candidate_k phải là integer dương")
    baseline_config = _baseline_config(config)
    collection, _, count = semantic_baseline._query_collection(
        baseline_config, strategy, storage_path, chroma_client
    )
    vector = semantic_baseline.create_query_embedding(
        question.strip(), baseline_config, client=embedding_client
    )
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(candidate_k, count),
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    if not (len(documents) == len(metadatas) == len(distances)):
        raise ValueError("Kết quả semantic thiếu hoặc lệch document/metadata/distance")

    candidates: list[dict[str, Any]] = []
    for rank, (text, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        if not isinstance(text, str) or not isinstance(metadata, dict):
            raise ValueError(f"Semantic candidate rank {rank} không hợp lệ")
        required = ("chunk_id", "source", "page_start", "page_end")
        if any(key not in metadata for key in required):
            raise ValueError(f"Semantic candidate rank {rank} thiếu metadata")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError(f"Semantic distance rank {rank} không hợp lệ")
        numeric_distance = float(distance)
        if not math.isfinite(numeric_distance):
            raise ValueError(f"Semantic distance rank {rank} phải hữu hạn")
        candidates.append(
            {
                "chunk_id": metadata["chunk_id"],
                "text": text,
                "source": metadata["source"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "semantic_rank": rank,
                "semantic_distance": numeric_distance,
            }
        )
    return candidates


_FUSION_METADATA_FIELDS = ("text", "source", "page_start", "page_end")


def reciprocal_rank_fusion(
    bm25_candidates: list[dict[str, Any]],
    semantic_candidates_list: list[dict[str, Any]],
    rrf_k: int,
    bm25_weight: float,
    semantic_weight: float,
) -> list[dict[str, Any]]:
    """Hợp nhất rankings bằng RRF; raw score/distance không tham gia công thức."""
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k phải là integer dương")
    for name, weight in (("bm25_weight", bm25_weight), ("semantic_weight", semantic_weight)):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{name} phải là số")
        if not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{name} phải hữu hạn và không âm")
    if bm25_weight == 0 and semantic_weight == 0:
        raise ValueError("RRF weights không được đồng thời bằng 0")

    union: dict[str, dict[str, Any]] = {}

    def add_branch(candidates: list[dict[str, Any]], branch: str) -> None:
        rank_key = f"{branch}_rank"
        score_key = "bm25_score" if branch == "bm25" else "semantic_distance"
        seen: set[str] = set()
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Candidate {branch} thiếu chunk_id hợp lệ")
            if chunk_id in seen:
                raise ValueError(f"Candidate {branch} trùng chunk_id '{chunk_id}'")
            seen.add(chunk_id)
            rank = candidate.get(rank_key)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError(f"{rank_key} của chunk '{chunk_id}' không hợp lệ")
            if chunk_id not in union:
                union[chunk_id] = {
                    "chunk_id": chunk_id,
                    **{field: candidate.get(field) for field in _FUSION_METADATA_FIELDS},
                    "bm25_rank": None,
                    "bm25_score": None,
                    "semantic_rank": None,
                    "semantic_distance": None,
                    "rrf_score": 0.0,
                    "fused_rank": 0,
                    "matched_by": [],
                }
            fused = union[chunk_id]
            mismatches = [
                field for field in _FUSION_METADATA_FIELDS
                if fused[field] != candidate.get(field)
            ]
            if mismatches:
                raise ValueError(
                    f"Metadata mismatch cho chunk '{chunk_id}': {', '.join(mismatches)}"
                )
            fused[rank_key] = rank
            fused[score_key] = candidate.get(score_key)
            fused["matched_by"].append(branch)

    add_branch(bm25_candidates, "bm25")
    add_branch(semantic_candidates_list, "semantic")

    for candidate in union.values():
        score = 0.0
        if candidate["bm25_rank"] is not None:
            score += float(bm25_weight) / (rrf_k + candidate["bm25_rank"])
        if candidate["semantic_rank"] is not None:
            score += float(semantic_weight) / (rrf_k + candidate["semantic_rank"])
        candidate["rrf_score"] = score

    infinity = float("inf")
    ranked = sorted(
        union.values(),
        key=lambda candidate: (
            -candidate["rrf_score"],
            min(
                candidate["bm25_rank"] if candidate["bm25_rank"] is not None else infinity,
                candidate["semantic_rank"] if candidate["semantic_rank"] is not None else infinity,
            ),
            candidate["semantic_rank"] if candidate["semantic_rank"] is not None else infinity,
            candidate["bm25_rank"] if candidate["bm25_rank"] is not None else infinity,
            candidate["chunk_id"],
        ),
    )
    for fused_rank, candidate in enumerate(ranked, start=1):
        candidate["fused_rank"] = fused_rank
    return ranked


def hybrid_retrieve(
    question: str,
    strategy: str,
    chunks: list[dict],
    config: AdvancedConfig,
    bm25_retriever: Any = bm25_search,
    semantic_retriever: Any = semantic_candidates,
    semantic_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gọi mỗi retriever một lần, fusion bằng RRF và trả pipeline trace."""
    bm25_started = time.perf_counter()
    bm25_results = bm25_retriever(question, chunks, config.bm25_candidates)
    bm25_ms = (time.perf_counter() - bm25_started) * 1000

    semantic_started = time.perf_counter()
    semantic_results = semantic_retriever(
        question,
        config.semantic_candidates,
        strategy,
        config,
        **(semantic_options or {}),
    )
    semantic_ms = (time.perf_counter() - semantic_started) * 1000

    fusion_started = time.perf_counter()
    fused = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        config.rrf_k,
        config.rrf_bm25_weight,
        config.rrf_semantic_weight,
    )
    fusion_ms = (time.perf_counter() - fusion_started) * 1000
    bm25_ids = {item["chunk_id"] for item in bm25_results}
    semantic_ids = {item["chunk_id"] for item in semantic_results}
    return {
        "candidates": fused,
        "trace": {
            "bm25_candidate_count": len(bm25_results),
            "semantic_candidate_count": len(semantic_results),
            "union_count": len(bm25_ids | semantic_ids),
            "overlap_count": len(bm25_ids & semantic_ids),
            "fused_count": len(fused),
            "rrf_k": config.rrf_k,
            "rrf_bm25_weight": config.rrf_bm25_weight,
            "rrf_semantic_weight": config.rrf_semantic_weight,
            "latency_ms": {
                "tokenize_bm25": bm25_ms,
                "semantic": semantic_ms,
                "fusion": fusion_ms,
            },
        },
    }


def _resolve_reranker_device(requested_device: str) -> str:
    """Resolve device lazily; importing this module never imports torch."""
    import torch

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng cho reranker")
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_cross_encoder(model_name: str, device: str) -> tuple[Any, Any]:
    """Lazy-load và cache một cross-encoder trong process, không dùng remote code."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_key = (model_name, device)
    if cache_key not in _RERANKER_RUNTIME_CACHE:
        RERANKER_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=RERANKER_CACHE_PATH,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=RERANKER_CACHE_PATH,
            trust_remote_code=False,
        )
        model.to(device)
        model.eval()
        _RERANKER_RUNTIME_CACHE[cache_key] = (tokenizer, model)
    return _RERANKER_RUNTIME_CACHE[cache_key]


def _runtime_cross_encoder_logits(
    pairs: list[tuple[str, str]],
    config: AdvancedConfig,
) -> list[float]:
    """Score query-document pairs theo batch trong torch.no_grad()."""
    import torch

    device = _resolve_reranker_device(config.rerank_device)
    tokenizer, model = _load_cross_encoder(config.reranker_model, device)
    logits: list[float] = []
    for offset in range(0, len(pairs), config.rerank_batch_size):
        batch = pairs[offset : offset + config.rerank_batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.reranker_max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded).logits
        values = output.reshape(-1).detach().cpu().tolist()
        logits.extend(float(value) for value in values)
    return logits


def rerank_fused_candidates(
    question: str,
    fused_candidates: list[dict[str, Any]],
    config: AdvancedConfig,
    score_pairs: Any | None = None,
) -> dict[str, Any]:
    """Rerank top fused candidates; injected scorer chỉ dành cho test."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question phải là string không rỗng")
    limit = config.effective_rerank_candidates(len(fused_candidates))
    selected = sorted(
        (deepcopy(item) for item in fused_candidates),
        key=lambda item: (item["fused_rank"], item["chunk_id"]),
    )[:limit]
    pairs = [(question.strip(), item["text"]) for item in selected]
    started = time.perf_counter()
    try:
        raw_scores = (score_pairs or _runtime_cross_encoder_logits)(pairs, config)
        if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) != len(pairs):
            raise ValueError(
                f"Reranker phải trả đúng {len(pairs)} logit, nhận {len(raw_scores) if isinstance(raw_scores, (list, tuple)) else 'sai kiểu'}"
            )
        for candidate, raw_score in zip(selected, raw_scores):
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("Reranker logit phải là số và không chấp nhận boolean")
            raw = float(raw_score)
            if not math.isfinite(raw):
                raise ValueError("Reranker logit phải là số hữu hạn")
            candidate["rerank_raw_score"] = raw
            candidate["rerank_score"] = 1.0 / (1.0 + math.exp(-raw))
            candidate["reranker_model"] = config.reranker_model
        ranked = sorted(
            selected,
            key=lambda item: (-item["rerank_score"], item["fused_rank"], item["chunk_id"]),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rerank_rank"] = rank
            candidate["rank_change"] = candidate["fused_rank"] - rank
            candidate["rerank_latency_ms"] = latency_ms
        return {
            "status": "reranked",
            "candidates": ranked[: config.final_top_k],
            "reranked_count": len(ranked),
            "rerank_latency_ms": latency_ms,
            "warnings": [],
        }
    except Exception as exc:
        return {
            "status": "reranker_unavailable",
            "candidates": [],
            "reranked_count": 0,
            "rerank_latency_ms": (time.perf_counter() - started) * 1000,
            "warnings": [f"Reranker không khả dụng: {type(exc).__name__}: {exc}"],
        }


def hybrid_rerank(
    question: str,
    strategy: str,
    chunks: list[dict],
    config: AdvancedConfig,
    score_pairs: Any | None = None,
    hybrid_retriever: Any = hybrid_retrieve,
    hybrid_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chạy hybrid trước, rồi rerank candidate nhỏ; chưa generation."""
    hybrid = hybrid_retriever(
        question, strategy, chunks, config, **(hybrid_options or {})
    )
    result = rerank_fused_candidates(
        question, hybrid["candidates"], config, score_pairs=score_pairs
    )
    result["trace"] = hybrid["trace"]
    return result


def _complete_evidence(candidate: dict[str, Any], index: int, accepted: bool) -> dict[str, Any]:
    """Chuẩn hóa evidence để field không áp dụng là null, không bịa score/rank."""
    fields = (
        "bm25_rank", "bm25_score", "semantic_rank", "semantic_distance",
        "rrf_score", "fused_rank", "rerank_raw_score", "rerank_score",
        "rerank_rank", "rank_change",
    )
    return {
        "evidence_id": f"E{index}",
        "source": candidate["source"],
        "page_start": candidate["page_start"],
        "page_end": candidate["page_end"],
        "chunk_id": candidate["chunk_id"],
        "text": candidate["text"],
        **{field: candidate.get(field) for field in fields},
        "accepted": accepted,
    }


def _empty_trace() -> dict[str, Any]:
    return {
        "bm25_candidates": 0, "semantic_candidates": 0, "overlap": 0,
        "union": 0, "reranked": 0, "accepted": 0,
        "generation_called": False,
        "latency_ms": {
            "bm25": 0.0, "semantic": 0.0, "fusion": 0.0,
            "rerank": 0.0, "generation": 0.0, "total": 0.0,
        },
    }


def retrieve_mode(
    question: str,
    mode: str,
    strategy: str,
    chunks: list[dict],
    config: AdvancedConfig,
    bm25_retriever: Any = bm25_search,
    semantic_retriever: Any = semantic_candidates,
    reranker: Any = rerank_fused_candidates,
    semantic_options: dict[str, Any] | None = None,
    score_pairs: Any | None = None,
) -> dict[str, Any]:
    """Chạy đúng retrieval mode, chưa generation; hybrid_rerank là pipeline thật."""
    if mode not in ANSWER_MODES:
        raise ValueError(f"mode không hợp lệ: '{mode}'")
    started = time.perf_counter()
    trace = _empty_trace()
    bm25_results: list[dict[str, Any]] = []
    semantic_results: list[dict[str, Any]] = []

    if mode in {"bm25", "hybrid", "hybrid_rerank"}:
        stage = time.perf_counter()
        bm25_results = bm25_retriever(question, chunks, config.bm25_candidates)
        trace["latency_ms"]["bm25"] = (time.perf_counter() - stage) * 1000
        trace["bm25_candidates"] = len(bm25_results)
    # BM25 diagnostic answer still needs semantic corroboration for its gate.
    if mode in ANSWER_MODES:
        stage = time.perf_counter()
        semantic_results = semantic_retriever(
            question, config.semantic_candidates, strategy, config,
            **(semantic_options or {}),
        )
        trace["latency_ms"]["semantic"] = (time.perf_counter() - stage) * 1000
        trace["semantic_candidates"] = len(semantic_results)

    semantic_by_id = {item["chunk_id"]: item for item in semantic_results}
    if mode == "semantic":
        candidates = semantic_results
        trace["union"] = len(candidates)
    elif mode == "bm25":
        candidates = []
        for item in bm25_results:
            merged = deepcopy(item)
            semantic_item = semantic_by_id.get(item["chunk_id"])
            merged["semantic_rank"] = semantic_item.get("semantic_rank") if semantic_item else None
            merged["semantic_distance"] = semantic_item.get("semantic_distance") if semantic_item else None
            candidates.append(merged)
        trace["overlap"] = sum(item["chunk_id"] in semantic_by_id for item in bm25_results)
        trace["union"] = len({item["chunk_id"] for item in bm25_results} | set(semantic_by_id))
    else:
        stage = time.perf_counter()
        candidates = reciprocal_rank_fusion(
            bm25_results, semantic_results, config.rrf_k,
            config.rrf_bm25_weight, config.rrf_semantic_weight,
        )
        trace["latency_ms"]["fusion"] = (time.perf_counter() - stage) * 1000
        bm25_ids = {item["chunk_id"] for item in bm25_results}
        semantic_ids = set(semantic_by_id)
        trace["overlap"] = len(bm25_ids & semantic_ids)
        trace["union"] = len(bm25_ids | semantic_ids)
        if mode == "hybrid_rerank":
            reranked = reranker(question, candidates, config, score_pairs=score_pairs)
            trace["latency_ms"]["rerank"] = reranked["rerank_latency_ms"]
            trace["reranked"] = reranked["reranked_count"]
            if reranked["status"] == "reranker_unavailable":
                trace["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
                return {"status": "reranker_unavailable", "candidates": [],
                        "warnings": reranked["warnings"], "trace": trace}
            candidates = reranked["candidates"]
    trace["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
    return {"status": "retrieved", "candidates": candidates, "warnings": [], "trace": trace}


def _runtime_generate(prompt: str, config: AdvancedConfig) -> str:
    client = semantic_baseline.genai.Client(api_key=config.api_key)
    response = client.models.generate_content(model=config.generation_model, contents=prompt)
    return getattr(response, "text", None)


def answer_advanced(
    question: str,
    mode: str,
    strategy: str,
    chunks: list[dict],
    config: AdvancedConfig,
    retrieval: Any = retrieve_mode,
    generate: Any = _runtime_generate,
    retrieval_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advanced answer pipeline: retrieve, gate, grounded generation và citation."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question phải là string không rỗng")
    question = question.strip()
    if len(question) > 2000:
        raise ValueError("question dài tối đa 2000 ký tự")
    if mode not in ANSWER_MODES:
        raise ValueError(f"mode không hợp lệ: '{mode}'")
    total_started = time.perf_counter()
    retrieved = retrieval(
        question, mode, strategy, chunks, config, **(retrieval_options or {})
    )
    trace = retrieved["trace"]
    base = {
        "status": "insufficient_evidence", "mode": mode, "question": question,
        "answer": INSUFFICIENT_ANSWER, "evidence": [], "citations": [],
        "warnings": list(retrieved.get("warnings", [])), "trace": trace,
    }
    if retrieved["status"] == "reranker_unavailable":
        base["status"] = "reranker_unavailable"
        trace["latency_ms"]["total"] = (time.perf_counter() - total_started) * 1000
        return base

    evidence = []
    for index, candidate in enumerate(retrieved["candidates"], start=1):
        if mode == "hybrid_rerank":
            accepted = candidate.get("rerank_score") is not None and candidate["rerank_score"] >= config.rerank_min_score
        else:
            distance = candidate.get("semantic_distance")
            accepted = distance is not None and distance <= config.max_distance
        evidence.append(_complete_evidence(candidate, index, accepted))
    base["evidence"] = evidence
    accepted_count = sum(item["accepted"] for item in evidence)
    trace["accepted"] = accepted_count
    if not accepted_count:
        trace["latency_ms"]["total"] = (time.perf_counter() - total_started) * 1000
        return base

    prompt = semantic_baseline.build_generation_prompt(question, evidence)
    generation_started = time.perf_counter()
    trace["generation_called"] = True
    try:
        generated = generate(prompt, config)
        if not isinstance(generated, str) or not generated.strip():
            raise ValueError("generation trả text rỗng")
    except Exception as exc:
        trace["latency_ms"]["generation"] = (time.perf_counter() - generation_started) * 1000
        trace["latency_ms"]["total"] = (time.perf_counter() - total_started) * 1000
        base["status"] = "retrieval_only"
        base["answer"] = RETRIEVAL_ONLY_ANSWER
        base["warnings"].append(f"Generation lỗi đã làm sạch ({type(exc).__name__})")
        return base
    trace["latency_ms"]["generation"] = (time.perf_counter() - generation_started) * 1000
    mapped, citations, warnings = semantic_baseline.map_citations(generated.strip(), evidence)
    if not mapped:
        base["status"] = "retrieval_only"
        base["answer"] = RETRIEVAL_ONLY_ANSWER
        base["warnings"].extend(warnings + ["Generation không còn nội dung hợp lệ"])
    else:
        base["status"] = "answered"
        base["answer"] = mapped
        base["citations"] = citations
        base["warnings"].extend(warnings)
    trace["latency_ms"]["total"] = (time.perf_counter() - total_started) * 1000
    return base


def compare_modes(
    question: str,
    strategy: str,
    chunks: list[dict],
    config: AdvancedConfig,
    retrieval: Any = retrieve_mode,
    retrieval_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """So sánh bốn retrieval mode; tuyệt đối không generation."""
    rows: dict[str, dict[str, Any]] = {}
    latencies: dict[str, float] = {}
    for mode in ANSWER_MODES:
        result = retrieval(question, mode, strategy, chunks, config, **(retrieval_options or {}))
        latencies[mode] = result["trace"]["latency_ms"]["total"]
        for rank, candidate in enumerate(result.get("candidates", []), start=1):
            row = rows.setdefault(candidate["chunk_id"], {
                "chunk_id": candidate["chunk_id"], "modes": [], "ranks": {},
                "rank_movement": {},
            })
            row["modes"].append(mode)
            rank_field = {
                "bm25": "bm25_rank",
                "semantic": "semantic_rank",
                "hybrid": "fused_rank",
                "hybrid_rerank": "rerank_rank",
            }[mode]
            final_rank = candidate.get(rank_field) or rank
            row["ranks"][mode] = final_rank
            if candidate.get("rank_change") is not None:
                row["rank_movement"][mode] = candidate["rank_change"]
    return {"question": question, "strategy": strategy, "rows": list(rows.values()), "latency_ms": latencies}


def _required_text(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise AdvancedConfigError(f"{name} phải là string không rỗng")
    return value


def _integer(name: str, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "")
    try:
        value = int(raw)
    except ValueError as exc:
        raise AdvancedConfigError(f"{name} phải là integer") from exc
    if not minimum <= value <= maximum:
        raise AdvancedConfigError(f"{name} phải trong khoảng {minimum} đến {maximum}")
    return value


def _float(name: str) -> float:
    raw = os.getenv(name, "")
    try:
        value = float(raw)
    except ValueError as exc:
        raise AdvancedConfigError(f"{name} phải là float") from exc
    if not math.isfinite(value):
        raise AdvancedConfigError(f"{name} phải là số hữu hạn")
    return value


def load_advanced_config(env_path: Path = ENV_PATH) -> AdvancedConfig:
    """Nạp `.env` bằng path tuyệt đối suy ra từ vị trí module và validate."""
    resolved_env = env_path.resolve()
    if not resolved_env.is_file():
        raise AdvancedConfigError(f"Không tìm thấy file cấu hình: '{resolved_env}'")
    load_dotenv(dotenv_path=resolved_env, override=True)

    embedding_model = _required_text("GEMINI_EMBEDDING_MODEL")
    generation_model = _required_text("GEMINI_GENERATION_MODEL")
    reranker_model = _required_text("RERANKER_MODEL")
    embedding_dim = _integer("GEMINI_EMBEDDING_DIM", 128, 3072)
    bm25_candidates = _integer("BM25_CANDIDATES", 1, 100)
    semantic_candidates = _integer("SEMANTIC_CANDIDATES", 1, 100)
    rerank_candidates = _integer("RERANK_CANDIDATES", 1, 100)
    final_top_k = _integer("FINAL_TOP_K", 1, 100)
    if final_top_k > rerank_candidates:
        raise AdvancedConfigError("FINAL_TOP_K phải <= RERANK_CANDIDATES")

    rrf_k = _integer("RRF_K", 1, 2_147_483_647)
    bm25_weight = _float("RRF_BM25_WEIGHT")
    semantic_weight = _float("RRF_SEMANTIC_WEIGHT")
    if bm25_weight < 0 or semantic_weight < 0:
        raise AdvancedConfigError("RRF weights phải không âm")
    if bm25_weight == 0 and semantic_weight == 0:
        raise AdvancedConfigError("RRF weights không được đồng thời bằng 0")

    max_distance = _float("RAG_MAX_DISTANCE")
    if max_distance < 0:
        raise AdvancedConfigError("RAG_MAX_DISTANCE phải không âm")
    rerank_min_score = _float("RERANK_MIN_SCORE")
    if not 0 <= rerank_min_score <= 1:
        raise AdvancedConfigError("RERANK_MIN_SCORE phải trong khoảng 0 đến 1")
    rerank_device = _required_text("RERANK_DEVICE").lower()
    if rerank_device not in {"auto", "cpu", "cuda"}:
        raise AdvancedConfigError("RERANK_DEVICE chỉ nhận auto, cpu hoặc cuda")

    return AdvancedConfig(
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        generation_model=generation_model,
        max_distance=max_distance,
        bm25_candidates=bm25_candidates,
        semantic_candidates=semantic_candidates,
        rrf_k=rrf_k,
        rrf_bm25_weight=bm25_weight,
        rrf_semantic_weight=semantic_weight,
        rerank_candidates=rerank_candidates,
        final_top_k=final_top_k,
        reranker_model=reranker_model,
        reranker_max_length=_integer("RERANKER_MAX_LENGTH", 64, 4096),
        rerank_batch_size=_integer("RERANK_BATCH_SIZE", 1, 64),
        rerank_min_score=rerank_min_score,
        rerank_device=rerank_device,
    )


def _page_label(candidate: dict) -> str:
    if candidate["page_start"] == candidate["page_end"]:
        return str(candidate["page_start"])
    return f"{candidate['page_start']}-{candidate['page_end']}"


def _run_bm25(strategy: str, question: str, candidate_k: int) -> None:
    chunks, stats = load_chunks(strategy=strategy)
    candidates = bm25_search(question, chunks, candidate_k)
    print(
        f"BM25 lexical retrieval | strategy={strategy} | "
        f"valid_chunks={stats['valid_chunks']} | returned={len(candidates)}"
    )
    print("BM25 score cao hơn thường phù hợp từ khóa hơn; score không phải xác suất.")
    for item in candidates:
        preview = " ".join(item["text"].split())[:160]
        print(
            f"#{item['bm25_rank']} | score={item['bm25_score']:.6f} | "
            f"source={item['source']} | page={_page_label(item)} | "
            f"chunk_id={item['chunk_id']} | preview={preview}"
        )


def _config_for_status() -> AdvancedConfig:
    """Cho status dùng example khi chưa có .env; không tạo file hoặc key."""
    return load_advanced_config(ENV_PATH if ENV_PATH.is_file() else BUOI_08_DIR / ".env.example")


def _run_status(strategy: str) -> None:
    import json

    print(json.dumps(advanced_status(strategy, _config_for_status()), ensure_ascii=False, indent=2))


def _run_prepare_semantic(strategy: str) -> None:
    import json

    config = load_advanced_config(ENV_PATH)
    print(json.dumps(prepare_semantic(strategy, config), ensure_ascii=False, indent=2))


def _run_hybrid(strategy: str, question: str) -> None:
    config = load_advanced_config(ENV_PATH)
    chunks, _ = load_chunks(strategy=strategy)
    result = hybrid_retrieve(question, strategy, chunks, config)
    print("Hybrid RRF | raw BM25 score và cosine distance không được cộng trực tiếp.")
    for item in result["candidates"]:
        print(
            f"#{item['fused_rank']} | rrf={item['rrf_score']:.8f} | "
            f"bm25_rank={item['bm25_rank']} | bm25_score={item['bm25_score']} | "
            f"semantic_rank={item['semantic_rank']} | "
            f"distance={item['semantic_distance']} | chunk_id={item['chunk_id']} | "
            f"matched_by={','.join(item['matched_by'])}"
        )
    print("Trace:")
    import json

    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))


def _run_rerank(strategy: str, question: str) -> None:
    config = load_advanced_config(ENV_PATH)
    chunks, _ = load_chunks(strategy=strategy)
    print(
        f"Chuẩn bị cross-encoder {config.reranker_model}. Lần tải đầu có thể cần "
        "Internet, nhiều dung lượng đĩa và RAM."
    )
    result = hybrid_rerank(question, strategy, chunks, config)
    print(f"status={result['status']} | reranked_count={result['reranked_count']}")
    for item in result["candidates"]:
        print(
            f"#{item['rerank_rank']} | score={item['rerank_score']:.6f} "
            f"(không phải xác suất đúng) | fused_rank={item['fused_rank']} | "
            f"rank_change={item['rank_change']:+d} | chunk_id={item['chunk_id']}"
        )
    for warning in result["warnings"]:
        print(f"CẢNH BÁO: {warning}")


def _run_query(mode: str, strategy: str, question: str) -> None:
    import json

    config = load_advanced_config(ENV_PATH)
    chunks, _ = load_chunks(strategy=strategy)
    result = answer_advanced(question, mode, strategy, chunks, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_compare(strategy: str, question: str) -> None:
    config = load_advanced_config(ENV_PATH)
    chunks, _ = load_chunks(strategy=strategy)
    result = compare_modes(question, strategy, chunks, config)
    print("chunk_id | modes | final ranks | rank movement")
    for row in result["rows"]:
        print(
            f"{row['chunk_id']} | {','.join(row['modes'])} | "
            f"{row['ranks']} | {row['rank_movement']}"
        )
    print(f"latency_ms={result['latency_ms']}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Advanced RAG diagnostics Buổi 08")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bm25_parser = subparsers.add_parser("bm25", help="Chạy BM25 lexical retrieval")
    bm25_parser.add_argument(
        "--strategy",
        choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    bm25_parser.add_argument("--question", required=True)
    bm25_parser.add_argument("--candidate-k", type=int, default=20)
    status_parser = subparsers.add_parser("status", help="Xem Advanced RAG status read-only")
    status_parser.add_argument(
        "--strategy",
        choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    prepare_parser = subparsers.add_parser(
        "prepare-semantic", help="Chủ động tạo semantic index Buổi 08"
    )
    prepare_parser.add_argument(
        "--strategy",
        choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    hybrid_parser = subparsers.add_parser("hybrid", help="Chạy BM25 + semantic + RRF")
    hybrid_parser.add_argument(
        "--strategy",
        choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    hybrid_parser.add_argument("--question", required=True)
    rerank_parser = subparsers.add_parser("rerank", help="Chạy hybrid + cross-encoder rerank")
    rerank_parser.add_argument(
        "--strategy",
        choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    rerank_parser.add_argument("--question", required=True)
    query_parser = subparsers.add_parser("query", help="Advanced RAG answer pipeline")
    query_parser.add_argument("--mode", choices=ANSWER_MODES, default="hybrid_rerank")
    query_parser.add_argument(
        "--strategy", choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    query_parser.add_argument("--question", required=True)
    compare_parser = subparsers.add_parser("compare", help="So sánh retrieval, không generation")
    compare_parser.add_argument(
        "--strategy", choices=["hierarchical", "semantic", "fixed-size"],
        default="hierarchical",
    )
    compare_parser.add_argument("--question", required=True)
    args = parser.parse_args()
    try:
        if args.command == "bm25":
            _run_bm25(args.strategy, args.question, args.candidate_k)
        elif args.command == "status":
            _run_status(args.strategy)
        elif args.command == "prepare-semantic":
            _run_prepare_semantic(args.strategy)
        elif args.command == "hybrid":
            _run_hybrid(args.strategy, args.question)
        elif args.command == "rerank":
            _run_rerank(args.strategy, args.question)
        elif args.command == "query":
            _run_query(args.mode, args.strategy, args.question)
        elif args.command == "compare":
            _run_compare(args.strategy, args.question)
        return 0
    except (TypeError, ValueError, semantic_baseline.ChunkValidationError) as exc:
        print(f"LỖI: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
