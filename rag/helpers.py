from typing import List

def build_sources(docs):
    sources = []
    seen = set()

    for doc in docs:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else doc.metadata
        item = {
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "file_path": metadata.get("file_path"),
            "section": metadata.get("section"),
        }
        key = (item["source"], item["file_path"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(item)

    return sources
