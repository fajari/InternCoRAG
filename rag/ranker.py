import re

QUERY_STOPWORDS = {
    "how", "what", "when", "where", "which", "who", "why",
    "to", "do", "does", "is", "are", "the", "a", "an",
    "of", "for", "on", "in", "after",
}
SYNONYM_MAP = {
    "add": {"add", "create", "insert", "register", "new"},
    "create": {"create", "add", "insert", "register", "new"},
    "insert": {"insert", "add", "create"},
    "new": {"new", "add", "create"},
    "edit": {"edit", "update", "modify", "change", "revise"},
    "update": {"update", "edit", "modify", "change", "revise"},
    "modify": {"modify", "edit", "update", "change"},
    "change": {"change", "edit", "update", "modify"},
    "delete": {"delete", "remove", "erase", "drop"},
    "remove": {"remove", "delete", "erase", "drop"},
    "erase": {"erase", "delete", "remove"},
    "record": {"record", "records", "rr"},
    "records": {"records", "record", "rr"},
    "rr": {"rr", "record", "records", "resource"},
    "resource": {"resource", "resources", "rr"},
    "resources": {"resources", "resource", "rr"},
    "flush": {"flush", "clear", "purge", "refresh"},
    "clear": {"clear", "flush", "purge"},
    "resolve": {"resolve", "test", "lookup", "check", "verify"},
    "test": {"test", "verify", "resolve", "check"},
    "verify": {"verify", "test", "check", "resolve"},
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def extract_query_words(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", (question or "").lower())
    filtered = [word for word in words if word not in QUERY_STOPWORDS]
    return filtered or words


def canonicalize_token(token: str) -> str:
    token = (token or "").lower().strip()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def canonicalize_words(words: list[str]) -> list[str]:
    return [canonicalize_token(word) for word in words if word]


def expand_query_words(words: list[str]) -> list[str]:
    expanded: list[str] = []
    seen = set()

    for word in words:
        candidates = [word, *sorted(SYNONYM_MAP.get(word, set()))]
        for candidate in candidates:
            candidate = canonicalize_token(candidate)
            if candidate in QUERY_STOPWORDS or candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)

    return expanded


def section_depth(section_title: str) -> int:
    match = re.match(r"^\s*((?:\d+\.)*\d+)", section_title or "")
    if not match:
        return 0
    return match.group(1).count(".")


def extract_primary_action(words: list[str]) -> str:
    primary_actions = ("add", "edit", "delete", "create", "update", "remove")
    word_set = set(words)

    for action in primary_actions:
        if action in word_set:
            return action

    return ""


def contains_exact_action_phrase(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False

    text_tokens = canonicalize_words(extract_query_words(text))
    text_joined = " ".join(text_tokens)

    for size in range(min(3, len(keywords)), 1, -1):
        for start in range(0, len(keywords) - size + 1):
            phrase = " ".join(keywords[start:start + size])
            if phrase and phrase in text_joined:
                return True

    return False


def intent_penalty_score(title: str, raw_question_words: list[str]) -> float:
    title_words = set(extract_query_words(title))
    query_words = set(raw_question_words)
    penalty = 0.0

    if "after" in title_words and "after" not in query_words:
        penalty += 8.0

    primary_actions = {"add", "edit", "delete", "create", "update", "remove"}
    matched_actions = title_words & primary_actions
    query_actions = query_words & primary_actions

    if len(matched_actions) >= 2 and len(query_actions) <= 1:
        penalty += 3.0

    query_action = extract_primary_action(list(query_words))
    title_action = extract_primary_action(list(title_words))

    if query_action and title_action and query_action != title_action:
        allowed_pairs = {
            ("add", "create"),
            ("create", "add"),
            ("edit", "update"),
            ("update", "edit"),
            ("delete", "remove"),
            ("remove", "delete"),
        }
        if (query_action, title_action) not in allowed_pairs:
            penalty += 12.0

    return penalty


def score_sections(docs: list, question: str) -> list[dict]:
    raw_query_words = extract_query_words(question)
    query_words = canonicalize_words(raw_query_words)
    expanded_query_words = expand_query_words(query_words)
    query_word_set = set(query_words)
    expanded_query_word_set = set(expanded_query_words)
    results = []

    for doc in docs:
        title = doc.metadata.get("section", "")
        title_norm = normalize(title)
        body_norm = normalize(doc.page_content)
        toc_entries = doc.metadata.get("toc_entries", [])

        score = 0.0
        title_hit = False

        # 1️⃣ TITLE MATCH (PALING WAJIB)
        title_tokens = set(canonicalize_words(extract_query_words(title_norm)))
        strict_title_overlap = len(query_word_set & title_tokens)
        synonym_title_overlap = len(expanded_query_word_set & title_tokens)
        if strict_title_overlap or synonym_title_overlap:
            score += strict_title_overlap * 4.0
            score += max(synonym_title_overlap - strict_title_overlap, 0) * 2.0
            title_hit = True

            if strict_title_overlap == len(query_word_set) and strict_title_overlap > 0:
                score += 6.0 + (section_depth(title) * 2.0)

            if contains_exact_action_phrase(title, query_words):
                score += 12.0

        # 2️⃣ TOC MATCH (STRUKTURAL)
        best_toc_score = 0.0
        for entry in toc_entries:
            entry_norm = normalize(entry)
            entry_tokens = set(canonicalize_words(extract_query_words(entry_norm)))
            strict_overlap = len(query_word_set & entry_tokens)
            synonym_overlap = len(expanded_query_word_set & entry_tokens)
            entry_score = 0.0

            if strict_overlap == len(query_word_set) and strict_overlap > 0:
                entry_score += 9.0
            elif strict_overlap >= 2:
                entry_score += 6.0
            elif strict_overlap == 1:
                entry_score += 3.0
            elif synonym_overlap >= 2:
                entry_score += 4.0
            elif synonym_overlap == 1:
                entry_score += 2.0

            # Give extra weight when the TOC entry directly matches the section.
            if entry_score > 0 and entry_norm and entry_norm == title_norm:
                entry_score += 6.0

            best_toc_score = max(best_toc_score, entry_score)

        if best_toc_score > 0:
            score += best_toc_score
            title_hit = True

        # 3️⃣ BODY MATCH (HANYA JIKA TITLE / TOC HIT)
        if title_hit:
            for w in expanded_query_words:
                if w in body_norm:
                    score += 0.5

        score -= intent_penalty_score(title, raw_query_words)

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
