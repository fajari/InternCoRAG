from typing import List

def build_sources(docs):
    sources = []

    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "file_path": doc.metadata.get("file_path"),
        })

    return sources
