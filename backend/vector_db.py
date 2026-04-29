import json
from pathlib import Path
from typing import Any

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
KNOWLEDGE_FILE = DATA_DIR / "medical_knowledge.json"

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FakeEmbeddings(size=384)
    return _embeddings


def _default_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "Type 2 diabetes management includes lifestyle modification, metformin as first-line, "
                "HbA1c monitoring every 3 months, foot and eye screening, and cardiology referral when indicated."
            ),
            metadata={"source": "guideline", "topic": "diabetes", "type": "medical"},
        ),
        Document(
            page_content=(
                "Hypertension: initial evaluation with BP monitoring, basic labs, assess ASCVD risk; "
                "first-line agents often include thiazide, ACE inhibitor, or ARB depending on comorbidities."
            ),
            metadata={"source": "guideline", "topic": "hypertension", "type": "medical"},
        ),
        Document(
            page_content=(
                "Asthma: spirometry for diagnosis, inhaled corticosteroids for persistent disease, "
                "written asthma action plan, and specialist referral for severe or uncontrolled symptoms."
            ),
            metadata={"source": "guideline", "topic": "asthma", "type": "medical"},
        ),
        Document(
            page_content=(
                "Doctor credentialing: verify medical license number, board certification in specialty, "
                "and hospital privileges before granting clinical system access."
            ),
            metadata={"source": "policy", "topic": "credentialing", "type": "doctor"},
        ),
    ]


def load_documents_from_json() -> list[Document]:
    if not KNOWLEDGE_FILE.is_file():
        return _default_documents()
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        payload = json.load(f)
    docs: list[Document] = []
    for item in payload.get("documents", []):
        docs.append(
            Document(
                page_content=item["content"],
                metadata=item.get("metadata", {}),
            )
        )
    return docs or _default_documents()


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    emb = _get_embeddings()
    if INDEX_DIR.is_dir() and any(INDEX_DIR.iterdir()):
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            emb,
            allow_dangerous_deserialization=True,
        )
        return _vectorstore
    docs = load_documents_from_json()
    _vectorstore = FAISS.from_documents(docs, emb)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore.save_local(str(INDEX_DIR))
    return _vectorstore


def add_documents(documents: list[Document]) -> None:
    global _vectorstore
    vs = get_vectorstore()
    vs.add_documents(documents)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))


def similarity_search(query: str, k: int = 4, filter_meta: dict[str, Any] | None = None) -> list[Document]:
    vs = get_vectorstore()
    fetch_k = k * 3 if filter_meta else k
    results = vs.similarity_search(query, k=fetch_k)
    if not filter_meta:
        return results[:k]
    out: list[Document] = []
    for d in results:
        if all(d.metadata.get(fk) == fv for fk, fv in filter_meta.items()):
            out.append(d)
        if len(out) >= k:
            break
    return out


def similarity_search_with_score(query: str, k: int = 4) -> list[tuple[Document, float]]:
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k)
