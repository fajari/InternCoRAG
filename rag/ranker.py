from rag.retriever import normalize, extract_query_words

def score_sections(docs: list, question: str) -> list[dict]:
    query_words = extract_query_words(question)
    results = []

    for doc in docs:
        title = doc.metadata.get("section", "")
        title_norm = normalize(title)
        body_norm = normalize(doc.page_content)
        toc_entries = doc.metadata.get("toc_entries", [])

        score = 0.0
        title_hit = False

        # 1️⃣ TITLE MATCH (PALING WAJIB)
        for w in query_words:
            if w in title_norm:
                score += 4.0
                title_hit = True

        # 2️⃣ TOC MATCH (STRUKTURAL)
        for entry in toc_entries:
            if any(w in normalize(entry) for w in query_words):
                score += 3.0
                title_hit = True

        # 3️⃣ BODY MATCH (HANYA JIKA TITLE HIT)
        if title_hit:
            for w in query_words:
                if w in body_norm:
                    score += 0.5

        # ❌ BODY ONLY → BUANG
        if not title_hit:
            continue

        results.append({
            "doc": doc,
            "title": title,
            "body": doc.page_content,
            "page": doc.metadata.get("page", 0),
            "score": round(score, 3)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
