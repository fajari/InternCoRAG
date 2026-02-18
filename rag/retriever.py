from typing import List
import re

from langchain.schema import Document
from rag.vectorstore import get_vectorstore


# =====================================================
# CONFIG
# =====================================================

TOP_K = 8
TITLE_BOOST_WEIGHT = 4
KEYWORD_DENSITY_WEIGHT = 2


# =====================================================
# UTIL
# =====================================================

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def keyword_density_score(text: str, keywords: List[str]) -> float:
    words = tokenize(text)
    if not words:
        return 0.0

    count = sum(words.count(k) for k in keywords)
    return count / len(words)


def title_boost_score(title: str, keywords: List[str]) -> int:
    title = title.lower()
    score = 0

    for k in keywords:
        if k in title:
            score += TITLE_BOOST_WEIGHT

    return score


# =====================================================
# MAIN RETRIEVE FUNCTION
# =====================================================

def retrieve_documents(workspace_id: str, question: str) -> List[Document]:

    vectorstore = get_vectorstore()

    # 1️⃣ similarity search with metadata filter
    docs = vectorstore.similarity_search(
        question,
        k=TOP_K,
        filter={
            "workspace_id": workspace_id
        }
    )

    if not docs:
        return []

    keywords = tokenize(question)

    scored_docs = []

        # STRICT TITLE MATCH OVERRIDE
    for doc in docs:
        title = doc.metadata.get("section", "").lower()
        if all(k in title for k in keywords):
            return [doc]   # 🚀 langsung return satu section saja

    # 2️⃣ rerank using title + keyword density
    for rank, doc in enumerate(docs):

        title = doc.metadata.get("section", "")
        content = doc.page_content


        # Base score (inverse rank bias)
        base_score = (TOP_K - rank)

        boost = title_boost_score(title, keywords)
        density = keyword_density_score(content, keywords) * KEYWORD_DENSITY_WEIGHT

        final_score = base_score + boost + density

        scored_docs.append((final_score, doc))

    # 3️⃣ sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 4️⃣ return only documents
    return [doc for _, doc in scored_docs]
