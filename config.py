import os
from dotenv import load_dotenv

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

# =============================
# APP CONFIG
# =============================
APP_NAME = "Internal Company Knowledge Assistant"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 4

LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"
