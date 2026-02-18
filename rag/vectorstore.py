from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import QDRANT_URL, QDRANT_COLLECTION

# =====================================================
# QDRANT CLIENT
# =====================================================
client = QdrantClient(url=QDRANT_URL)

# =====================================================
# LOCAL EMBEDDING MODEL
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# ENSURE COLLECTION EXISTS (FOR INSERT)
# =====================================================
def ensure_collection():
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

# =====================================================
# VECTORSTORE FACTORY
# =====================================================
def get_vectorstore():
    ensure_collection()

    return Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,  # ✅ correct param
    )

# =====================================================
# CHECK IF WORKSPACE HAS DOCUMENTS (DEFENSIVE)
# =====================================================

def has_documents(workspace_id: str) -> bool:
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        scroll = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.workspace_id",
                        match=models.MatchValue(value=str(workspace_id))
                    )
                ]
            ),
            limit=1
        )

        points, _ = scroll
        return len(points) > 0

    except UnexpectedResponse:
        # collection belum ada / belum pernah diisi
        return False

    except Exception as e:
        # safety net
        print("has_documents error:", e)
        return False