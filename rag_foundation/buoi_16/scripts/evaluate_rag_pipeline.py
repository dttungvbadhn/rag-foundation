from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from src.secure_retriever import SecureRetriever  # noqa: E402

SOURCE_CSV = ROOT / "data" / "processed" / "chunks_secure.csv"
NORMALIZED_CSV = ROOT / "data" / "processed" / "chunks_normalized.csv"
QA_CSV = ROOT / "data" / "eval" / "qa_dataset.csv"
RESULTS_CSV = ROOT / "data" / "eval" / "evaluation_results.csv"
ANSWERS_CSV = ROOT / "data" / "eval" / "answered_checkpoint.csv"
REPORT_MD = ROOT / "outputs" / "ragas_evaluation_report.md"
GENERATOR_MODEL = "Qwen/Qwen3.5-9B:deepinfra"
JUDGE_MODEL = "openai/gpt-oss-20b:deepinfra"
HF_BASE_URL = "https://router.huggingface.co/v1"
FULL_ROLES = ["Admin", "HR", "Risk_Manager", "Staff"]
METRIC_COLUMNS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def configure() -> str:
    for env_path in (ROOT / ".env", WORKSPACE_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Thiếu HF_TOKEN trong buoi_16/.env hoặc .env ở workspace root")
    QA_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    return token


def classify_usecase(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ("title", "text", "document_type")).lower()
    if any(word in text for word in ("nhân sự", "lao động", "cán bộ", "nhân viên", "tuyển dụng")):
        return "HR"
    if any(word in text for word in ("rủi ro", "kiểm soát", "an toàn", "gian lận", "tín dụng", "bảo mật")):
        return "Risk"
    return "Common"


def ensure_secure_corpus() -> None:
    if SOURCE_CSV.exists():
        return
    if not NORMALIZED_CSV.exists():
        raise FileNotFoundError(f"Thiếu cả {SOURCE_CSV} và {NORMALIZED_CSV}")
    with NORMALIZED_CSV.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        fields = list(rows[0]) + ["allowed_roles"]
    for row in rows:
        usecase = classify_usecase(row)
        roles = FULL_ROLES if usecase == "Common" else ["Admin", "HR"] if usecase == "HR" else ["Admin", "Risk_Manager"]
        row["allowed_roles"] = json.dumps(roles, ensure_ascii=False)
    with SOURCE_CSV.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã tạo secure corpus: {SOURCE_CSV} ({len(rows)} chunks)")


def extract_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise ValueError("Generator không trả về JSON array hợp lệ")
        return json.loads(match.group(0))


def select_representative_chunks(rows: list[dict[str, str]], count: int = 15) -> list[dict[str, str]]:
    rng = random.Random(16)
    groups = {name: [] for name in ("HR", "Risk", "Common")}
    for row in rows:
        if len(row.get("text", "").strip()) >= 80:
            groups[classify_usecase(row)].append(row)
    selected = []
    for name in groups:
        candidates = groups[name]
        rng.shuffle(candidates)
        selected.extend(candidates[: max(1, count // 3)])
    if len(selected) < count:
        remaining = [row for row in rows if row not in selected and len(row.get("text", "")) >= 80]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def generate_golden_dataset(client: OpenAI) -> pd.DataFrame:
    rows = pd.read_csv(SOURCE_CSV).fillna("").to_dict("records")
    chunks = select_representative_chunks(rows)
    sources = [{"chunk_id": x["chunk_id"], "usecase": classify_usecase(x), "text": x["text"]} for x in chunks]
    items = []
    # Four smaller requests are substantially more reliable than one 7K-token response.
    for batch_index in range(4):
        batch_sources = sources[batch_index * 3 : batch_index * 3 + 6]
        if len(batch_sources) < 6:
            batch_sources += sources[: 6 - len(batch_sources)]
        prompt = f"""Bạn là chuyên gia tạo benchmark RAG tiếng Việt. Dựa CHỈ trên các chunks dưới đây, tạo đúng 5 mẫu hỏi đáp khác nhau.
Phân bố hợp lý các difficulty easy, medium, hard; usecase phải là HR, Risk hoặc Common. Ground truth phải đầy đủ và có căn cứ trực tiếp.
Trả về DUY NHẤT JSON array; mỗi object có: question, ground_truth, usecase, difficulty, source_chunk_ids (array).
Không markdown, không giải thích. Chunks:\n{json.dumps(batch_sources, ensure_ascii=False)}"""
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        batch_items = extract_json(response.choices[0].message.content or "")
        if len(batch_items) != 5:
            raise ValueError(f"Batch {batch_index + 1} phải có 5 mẫu, nhận được {len(batch_items)}")
        items.extend(batch_items)
        print(f"Đã sinh Golden Dataset batch {batch_index + 1}/4", flush=True)
    known_ids = {str(x["chunk_id"]) for x in chunks}
    records = []
    for item in items:
        ids = [str(x) for x in item.get("source_chunk_ids", []) if str(x) in known_ids]
        if not ids or item.get("usecase") not in {"HR", "Risk", "Common"} or item.get("difficulty") not in {"easy", "medium", "hard"}:
            raise ValueError(f"Mẫu golden không hợp lệ: {item}")
        records.append({**item, "source_chunk_ids": json.dumps(ids, ensure_ascii=False)})
    frame = pd.DataFrame(records)
    frame.to_csv(QA_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã lưu Golden Dataset: {QA_CSV}")
    return frame


def answer_questions(client: OpenAI, qa: pd.DataFrame, reuse_answers: bool = True) -> pd.DataFrame:
    retriever = SecureRetriever()
    records = []
    if reuse_answers and ANSWERS_CSV.exists():
        checkpoint = pd.read_csv(ANSWERS_CSV).fillna("")
        records = checkpoint.to_dict("records")[: len(qa)]
        print(f"Tiếp tục từ checkpoint: {len(records)}/{len(qa)} câu", flush=True)
    for index, row in qa.iterrows():
        if index < len(records):
            continue
        hits = retriever.retrieve(row["question"], FULL_ROLES, method="hybrid", top_k=5, candidate_k=20)
        contexts = [str(hit.get("text", "")) for hit in hits]
        prompt = """Trả lời ngắn gọn, chính xác bằng tiếng Việt và CHỈ dựa trên ngữ cảnh. Nếu ngữ cảnh không đủ, nói rõ không đủ thông tin. Không trình bày quá trình suy luận.

Câu hỏi: {question}

Ngữ cảnh:
{contexts}""".format(question=row["question"], contexts="\n\n".join(f"[{i}] {x}" for i, x in enumerate(contexts, 1)))
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        records.append({**row.to_dict(), "answer": response.choices[0].message.content or "", "contexts": json.dumps(contexts, ensure_ascii=False)})
        pd.DataFrame(records).to_csv(ANSWERS_CSV, index=False, encoding="utf-8-sig")
        print(f"Đã sinh và lưu câu trả lời {index + 1}/{len(qa)}", flush=True)
    return pd.DataFrame(records)


def parse_contexts(value) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(value)
    return [str(item) for item in parsed]


def score_with_ragas(frame: pd.DataFrame, token: str) -> pd.DataFrame:
    judge = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=HF_BASE_URL,
        api_key=token,
        temperature=0,
        max_retries=3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    dataset = Dataset.from_dict({
        "user_input": frame["question"].tolist(),
        "response": frame["answer"].tolist(),
        "retrieved_contexts": frame["contexts"].map(parse_contexts).tolist(),
        "reference": frame["ground_truth"].tolist(),
    })
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        raise_exceptions=False,
    ).to_pandas()
    output = frame.reset_index(drop=True).copy()
    for metric in METRIC_COLUMNS:
        output[metric] = pd.to_numeric(result[metric], errors="coerce")
    output.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã lưu kết quả Ragas: {RESULTS_CSV}")
    return output


def recommendation(metric: str) -> str:
    return {
        "context_precision": "Điều chỉnh RRF và bổ sung cross-encoder reranker để đẩy chunks liên quan lên đầu.",
        "context_recall": "Tăng top_k có kiểm soát, mở rộng truy vấn và bổ sung liên kết đồ thị lân cận.",
        "faithfulness": "Siết prompt chỉ dùng context, giảm nhiễu và rút gọn chunks trước khi sinh.",
        "answer_relevancy": "Yêu cầu câu trả lời trực tiếp, ngắn gọn và thêm few-shot theo usecase.",
    }[metric]


def write_report(frame: pd.DataFrame) -> str:
    means = frame[METRIC_COLUMNS].mean()
    lines = ["# Báo cáo đánh giá hệ thống RAG bằng Ragas", "", f"- Số mẫu: {len(frame)}", f"- Generator: `{GENERATOR_MODEL}`", f"- Evaluator: `{JUDGE_MODEL}`", "", "## Điểm trung bình", "", "| Metric | Điểm |", "|---|---:|"]
    lines.extend(f"| {metric} | {means[metric]:.4f} |" for metric in METRIC_COLUMNS)
    lines.extend(["", "## Các mẫu có điểm thấp (< 0.7)", ""])
    low_count = 0
    for index, row in frame.iterrows():
        low = [metric for metric in METRIC_COLUMNS if pd.notna(row[metric]) and row[metric] < 0.7]
        if not low:
            continue
        low_count += 1
        lines.extend([f"### Mẫu {index + 1}: {row['question']}", "", f"- Usecase/độ khó: {row['usecase']} / {row['difficulty']}", f"- Chỉ số thấp: {', '.join(f'{m}={row[m]:.3f}' for m in low)}", f"- Nhận định: {recommendation(min(low, key=lambda m: row[m]))}", ""])
    if not low_count:
        lines.append("Không có mẫu nào dưới ngưỡng 0.7.")
    lines.extend(["", "## Đề xuất tối ưu", ""])
    for metric in METRIC_COLUMNS:
        if pd.isna(means[metric]):
            lines.append(f"- **{metric}**: kiểm tra log/API vì metric không trả được điểm.")
        elif means[metric] < (0.8 if metric in {"faithfulness", "answer_relevancy"} else 0.7):
            lines.append(f"- **{metric}**: {recommendation(metric)}")
    lines.extend(["", "## Câu hỏi thảo luận", "", "1. Tách Generator và Evaluator giúp giảm self-preference bias; vẫn cần hiệu chuẩn judge bằng một tập nhỏ có nhãn người.", "2. Gửi dữ liệu nội bộ tới API công cộng có thể vi phạm chính sách phân loại dữ liệu. Nên ẩn danh hoặc dùng evaluator self-hosted trong hạ tầng được phê duyệt.", "3. Tăng `top_k` cải thiện recall nhưng cũng thêm nhiễu, khiến generator trộn thông tin và giảm faithfulness; cần rerank/compression và chọn top_k theo thực nghiệm.", "4. Dùng judge khác họ model, rubric cố định, đảo thứ tự/định dạng, chấm lặp và đối chiếu định kỳ với chuyên gia."])
    report = "\n".join(lines) + "\n"
    REPORT_MD.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command RAG evaluation pipeline")
    parser.add_argument("--reuse-qa", action="store_true", help="Dùng lại qa_dataset.csv")
    parser.add_argument("--sample-limit", type=int, default=None, help="Chỉ đánh giá N mẫu đầu tiên")
    parser.add_argument("--no-reuse-answers", action="store_true", help="Bỏ checkpoint câu trả lời hiện có")
    args = parser.parse_args()
    token = configure()
    ensure_secure_corpus()
    client = OpenAI(base_url=HF_BASE_URL, api_key=token, max_retries=3, timeout=120.0)
    qa = pd.read_csv(QA_CSV) if args.reuse_qa and QA_CSV.exists() else generate_golden_dataset(client)
    if args.sample_limit is not None:
        if args.sample_limit < 1:
            parser.error("--sample-limit phải lớn hơn 0")
        qa = qa.head(args.sample_limit).reset_index(drop=True)
    answered = answer_questions(client, qa, reuse_answers=not args.no_reuse_answers)
    results = score_with_ragas(answered, token)
    report = write_report(results)
    print("\nĐiểm trung bình:")
    print(results[METRIC_COLUMNS].mean().round(4).to_string())
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
