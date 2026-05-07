from pathlib import Path
import json
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

# -------------------------
# PATHS
# -------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
KNOWLEDGE_FILE = DATA_DIR / "medical_knowledge.json"

_embeddings = None
_vectorstore: FAISS | None = None


# -------------------------
# EMBEDDINGS
# -------------------------
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FakeEmbeddings(size=384)
    return _embeddings


# -------------------------
# LOAD KNOWLEDGE BASE
# -------------------------
def load_documents():
    if not KNOWLEDGE_FILE.exists():
        return []

    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            docs.append(
                Document(page_content=item.get("text", ""))
            )

        return docs

    except Exception as e:
        print("⚠️ Failed to load knowledge file:", e)
        return []


# -------------------------
# BUILD VECTOR STORE
# -------------------------
def build_vectorstore():
    docs = load_documents()

    if not docs:
        return None

    embeddings = get_embeddings()

    db = FAISS.from_documents(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))

    return db


# -------------------------
# SAFE LOADER (FIXES YOUR ERROR)
# -------------------------
def get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    try:
        embeddings = get_embeddings()

        if INDEX_DIR.exists():
            _vectorstore = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            _vectorstore = build_vectorstore()

        return _vectorstore

    except Exception as e:
        print("❌ FAISS load failed, rebuilding index:", e)

        # 🔥 AUTO RECOVERY
        _vectorstore = build_vectorstore()
        return _vectorstore


# -------------------------
# MAIN SEARCH FUNCTION
# -------------------------
def similarity_search(query: str, k: int = 5):
    try:
        vs = get_vectorstore()

        if vs is None:
            return []

        return vs.similarity_search(query, k=k)

    except Exception as e:
        print("❌ similarity_search failed:", e)
        return []