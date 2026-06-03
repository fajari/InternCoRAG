import os
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return False

load_dotenv()

# =============================
# ENVIRONMENT
# =============================
ENV = os.getenv("ENV", "development")

# =============================
# QDRANT CONFIG
# =============================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_knowledge")

# =============================
# LLM CONFIG
# =============================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | openai

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "12"))

# =============================
# GRAPH RAG CONFIG
# =============================
ENABLE_GRAPH_RAG = os.getenv("ENABLE_GRAPH_RAG", "true").lower() == "true"
ENABLE_LLAMAINDEX_KG = os.getenv("ENABLE_LLAMAINDEX_KG", "false").lower() == "true"
ENABLE_NEO4J_SYNC = os.getenv("ENABLE_NEO4J_SYNC", "false").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GRAPH_RAG_PROJECT_ROOT = os.getenv("GRAPH_RAG_PROJECT_ROOT", "")

# =============================
# APP CONFIG
# =============================
APP_NAME = "Internal Company Knowledge Assistant"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 4

LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"

# =============================
# NARRATIVE-AWARE RAG CONFIG
# =============================
NARRATIVE_RAG_ENABLED = os.getenv("NARRATIVE_RAG_ENABLED", "true").lower() == "true"
NARRATIVE_LLM_CLASSIFIER_THRESHOLD = float(os.getenv("NARRATIVE_LLM_CLASSIFIER_THRESHOLD", "0.72"))
NARRATIVE_MAX_EVENTS = int(os.getenv("NARRATIVE_MAX_EVENTS", "80"))
