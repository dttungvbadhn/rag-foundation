"""RAG workshop: index JSON chunks, retrieve with ChromaDB, answer with Gemini."""

import json
import os
import sqlite3
from pathlib import Path

import chromadb
import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = BASE_DIR.parent / "buoi_05" / "output" / "chunks"
STORAGE_DIR = BASE_DIR / "storage"
SQLITE_PATH = STORAGE_DIR / "texts.db"
CHROMA_PATH = STORAGE_DIR / "chroma"
COLLECTION_NAME = "workshop_chunks"
EMBEDDING_MODEL = "gemini-embedding-2"
ANSWER_MODEL = "gemini-flash-lite-latest"
CHROMA_MODE = "Chưa kiểm tra"

load_dotenv(BASE_DIR / ".env")


def _postgres_connection(database=None):
    """Return a PostgreSQL connection, or None when PostgreSQL is unavailable."""
    try:
        return psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=database or os.getenv("POSTGRES_DB", "rag_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
    except psycopg.Error:
        return None


def _sqlite_connection():
    STORAGE_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(SQLITE_PATH)


def _chroma_collection():
    """Use a running Chroma server when possible, otherwise use local storage."""
    global CHROMA_MODE
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
        CHROMA_MODE = "Server"
    except Exception:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        CHROMA_MODE = "Embedded Local"
    return client.get_or_create_collection(COLLECTION_NAME)


def _gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    return genai.Client(api_key=api_key) if api_key else None


def _embed(texts):
    """Embed text with Gemini at the same 384 dimensions as Chroma's fallback."""
    client = _gemini_client()
    if client is None:
        return None

    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=384),
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings


def _read_chunks():
    chunks = []
    for json_file in sorted(CHUNKS_DIR.glob("*.json")):
        with json_file.open(encoding="utf-8") as file:
            items = json.load(file)

        for item in items:
            text = item.get("text", "").strip()
            if not text:
                continue

            source = item.get("source", json_file.stem)
            chunks.append(
                {
                    "id": f"{json_file.stem}:{item['chunk_id']}",
                    "document": source,
                    "text": text,
                    "metadata": {
                        "source": str(source),
                        "strategy": str(item.get("strategy", "")),
                        "page_start": int(item.get("page_start", 0)),
                        "page_end": int(item.get("page_end", 0)),
                    },
                }
            )
    return chunks


def _save_texts(chunks):
    """Store text in PostgreSQL, or in a local .db file when PostgreSQL is down."""
    connection = _postgres_connection()
    if connection is not None:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_name TEXT NOT NULL,
                        text_content TEXT NOT NULL,
                        metadata JSONB NOT NULL
                    )
                    """
                )
                cursor.execute("DELETE FROM rag_chunks")
                for chunk in chunks:
                    cursor.execute(
                        """
                        INSERT INTO rag_chunks (chunk_id, document_name, text_content, metadata)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            chunk["id"],
                            chunk["document"],
                            chunk["text"],
                            json.dumps(chunk["metadata"]),
                        ),
                    )
        connection.close()
        return "postgresql"

    connection = _sqlite_connection()
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                text_content TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM rag_chunks")
        connection.executemany(
            "INSERT INTO rag_chunks VALUES (?, ?, ?, ?)",
            [
                (chunk["id"], chunk["document"], chunk["text"], json.dumps(chunk["metadata"]))
                for chunk in chunks
            ],
        )
    connection.close()
    return "sqlite"


def _get_texts(chunk_ids):
    if not chunk_ids:
        return []

    connection = _postgres_connection()
    if connection is not None:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT chunk_id, text_content FROM rag_chunks WHERE chunk_id = ANY(%s)",
                (chunk_ids,),
            )
            rows = cursor.fetchall()
        connection.close()
    else:
        connection = _sqlite_connection()
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = connection.execute(
            f"SELECT chunk_id, text_content FROM rag_chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        connection.close()

    texts = {chunk_id: text for chunk_id, text in rows}
    return [texts[chunk_id] for chunk_id in chunk_ids if chunk_id in texts]


def index():
    """Read all chunk JSON files and rebuild the text and vector indexes."""
    chunks = _read_chunks()
    if not chunks:
        return {"documents": 0, "chunks": 0, "storage": "not indexed"}

    storage = _save_texts(chunks)
    collection = _chroma_collection()
    if collection.count():
        collection.delete(ids=collection.get()["ids"])

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = _embed(texts)

    if embeddings is None:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    else:
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    return {
        "documents": len({chunk["document"] for chunk in chunks}),
        "chunks": len(chunks),
        "storage": storage,
    }


def ask(question, k=3, include_context=False):
    """Retrieve top-k chunks and use Gemini to answer when an API key is available."""
    collection = _chroma_collection()
    if collection.count() == 0:
        result = {"chunks": [], "answer": "Chưa có dữ liệu. Hãy chạy index() trước."}
        return result if include_context else result["answer"]

    k = max(1, min(int(k), collection.count()))
    embeddings = _embed([question])
    if embeddings is None:
        result = collection.query(query_texts=[question], n_results=k)
    else:
        result = collection.query(query_embeddings=embeddings, n_results=k)

    texts = _get_texts(result["ids"][0])
    context = "\n\n---\n\n".join(texts)
    client = _gemini_client()
    if client is None:
        result = {"chunks": texts, "answer": None}
        return result if include_context else context

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=(
            "Trả lời câu hỏi chỉ dựa trên ngữ cảnh dưới đây. "
            "Nếu không đủ thông tin, hãy nói rõ.\n\n"
            f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"
        ),
    )
    result = {"chunks": texts, "answer": response.text}
    return result if include_context else result["answer"]


def status():
    """Return the number of source documents and chunks currently indexed."""
    collection = _chroma_collection()
    count = collection.count()
    if count == 0:
        return {"documents": 0, "chunks": 0}

    sources = collection.get(include=["metadatas"])["metadatas"]
    return {
        "documents": len({metadata["source"] for metadata in sources}),
        "chunks": count,
    }
