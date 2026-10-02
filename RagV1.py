"""
Phase 1: Naive RAG from scratch (no vector DB, no framework).

Setup:
    pip install pypdf numpy google-genai
    set GEMINI_API_KEY=your_key        (Windows CMD)
    export GEMINI_API_KEY=your_key     (Mac/Linux)

Usage:
    python rag_v1.py build Finalreportexcel.pdf
    python rag_v1.py ask "How do I use Goal Seek in Excel?"

Model names change over time. If you get a "model not found" error,
check the current names in the Gemini API docs and edit the two constants below.
"""
import time
import json
import os
import sys

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.8-flash"

CHUNK_SIZE = 800   # characters per chunk
OVERLAP = 150      # characters shared between neighbouring chunks
TOP_K = 4          # how many chunks to retrieve

CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ---------- 1. INGEST: PDF -> chunks ----------
def load_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def make_chunks(pdf_path):
    out = []
    for page_no, text in load_pages(pdf_path):
        for piece in chunk_text(text):
            if len(piece.strip()) > 40:  # skip near-empty chunks
                out.append({"page": page_no, "text": piece})
    return out


# ---------- 2. EMBED: text -> vectors ----------
def embed(texts, task_type):
    vectors = []
    for i in range(0, len(texts), 50):  # batch to stay under API limits
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in resp.embeddings)
    m = np.array(vectors, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)  # normalise: dot product == cosine similarity


def build(pdf_path):
    chunks = make_chunks(pdf_path)
    print(f"{len(chunks)} chunks created")
    vectors = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    np.save(VECTORS_FILE, vectors)
    print(f"Saved {CHUNKS_FILE} and {VECTORS_FILE} (shape {vectors.shape})")


# ---------- 3. RETRIEVE: question -> top-k chunks ----------
def retrieve(question, k=TOP_K):
    chunks = json.load(open(CHUNKS_FILE, encoding="utf-8"))
    vectors = np.load(VECTORS_FILE)
    q = embed([question], "RETRIEVAL_QUERY")[0]
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top]


# ---------- 4. GENERATE: chunks + question -> grounded answer ----------
PROMPT = """You are a helpful assistant answering questions about an Excel lab manual.
Use ONLY the context below. If the answer is not in the context, say
"I couldn't find that in the document." Cite page numbers like (p. 12).

Context:
{context}

Question: {question}
Answer:"""
def generate_with_retry(prompt, retries=6):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=GEN_MODEL, contents=prompt)
        except Exception as e:
            temporary = "503" in str(e) or "429" in str(e)
            if attempt == retries - 1 or not temporary:
                raise
            wait = min(5 * 2 ** attempt, 60)
            print(f"Server busy, retrying in {wait}s...")
            time.sleep(wait)

def ask(question):
    hits = retrieve(question)
    print("\n--- Retrieved chunks ---")
    for score, c in hits:
        preview = c["text"][:90].replace("\n", " ")
        print(f"[score {score:.3f}] page {c['page']}: {preview}...")

    context = "\n\n".join(f"[page {c['page']}]\n{c['text']}" for _, c in hits)
    resp = generate_with_retry(PROMPT.format(context=context, question=question))
    print("\n--- Answer ---")
    print(resp.text)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        print(__doc__)