import fitz  # PyMuPDF
import os
import hashlib
import re

HIGHLIGHT_COLORS = {
    "yellow": (1.0, 1.0, 0.0),
    "green": (0.65, 0.95, 0.55),
    "blue": (0.55, 0.8, 1.0),
    "pink": (1.0, 0.72, 0.82),
    "orange": (1.0, 0.82, 0.45),
}


# =========================================================
# NORMALIZE TEXT
# =========================================================
def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_compare(text: str) -> str:
    normalized = _normalize(text).lower()
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    return normalized


def _unique_rects(rects):
    unique = []
    seen = set()

    for rect in rects:
        key = tuple(round(value, 2) for value in (rect.x0, rect.y0, rect.x1, rect.y1))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rect)

    return unique


def _rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.width) * max(0.0, rect.height)


def _merge_touching_rects(rects):
    merged = []

    for rect in sorted(rects, key=lambda item: (round(item.y0, 1), round(item.x0, 1))):
        current = fitz.Rect(rect)
        merged_any = False

        for index, existing in enumerate(merged):
            same_line = abs(existing.y0 - current.y0) <= 6 or abs(existing.y1 - current.y1) <= 6
            close_horizontally = current.x0 <= existing.x1 + 12 and current.x1 >= existing.x0 - 12
            overlap = existing.intersects(current)
            if overlap or (same_line and close_horizontally):
                merged[index] = existing | current
                merged_any = True
                break

        if not merged_any:
            merged.append(current)

    return _unique_rects(merged)


def _build_search_candidates(text: str) -> list[str]:
    normalized = _normalize(text)
    if len(normalized) < 8:
        return []

    candidates: list[str] = [normalized]
    parts = re.split(r"(?<=[.!?;:])\s+|,\s+", normalized)
    words = normalized.split()

    for part in parts:
        part = part.strip(" .:-")
        if len(part) >= 20:
            candidates.append(part)

    window_sizes = (12, 10, 8, 6)
    for size in window_sizes:
        if len(words) < size:
            continue
        step = max(2, size // 2)
        for start in range(0, len(words) - size + 1, step):
            snippet = " ".join(words[start:start + size]).strip()
            if len(snippet) >= 20:
                candidates.append(snippet)

    ordered = []
    seen = set()
    for candidate in sorted(candidates, key=len, reverse=True):
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)

    return ordered


def _search_by_line_overlap(page: fitz.Page, text: str):
    candidate_tokens = [token for token in _normalize_for_compare(text).split() if token]
    if len(candidate_tokens) < 2:
        return []

    best_rects = []
    best_score = 0.0

    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = _normalize(" ".join(span.get("text", "") for span in spans))
            line_tokens = [token for token in _normalize_for_compare(line_text).split() if token]
            if len(line_tokens) < 2:
                continue

            overlap = len(set(candidate_tokens) & set(line_tokens))
            coverage = overlap / max(1, min(len(candidate_tokens), len(line_tokens)))
            if overlap < 2 or coverage < 0.45:
                continue

            rects = []
            for span in spans:
                bbox = span.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                rects.append(fitz.Rect(*bbox))

            merged_rects = _merge_touching_rects(rects)
            if merged_rects and coverage > best_score:
                best_rects = merged_rects
                best_score = coverage

    return best_rects


def _words_for_compare(page: fitz.Page):
    words = []
    for item in page.get_text("words"):
        if len(item) < 8:
            continue
        x0, y0, x1, y1, token, block_no, line_no, word_no = item[:8]
        normalized = _normalize_for_compare(token)
        if not normalized:
            continue
        words.append({
            "rect": fitz.Rect(x0, y0, x1, y1),
            "text": token,
            "normalized": normalized,
            "block_no": block_no,
            "line_no": line_no,
            "word_no": word_no,
        })
    return words


def _search_by_word_windows(page: fitz.Page, text: str):
    words = _words_for_compare(page)
    if not words:
        return []

    candidates = _build_search_candidates(text)
    if not candidates:
        return []

    for candidate in candidates:
        candidate_norm = _normalize_for_compare(candidate)
        candidate_tokens = [token for token in candidate_norm.split() if token]
        if len(candidate_tokens) < 3:
            continue

        window_size = len(candidate_tokens)
        for start in range(0, len(words) - window_size + 1):
            window = words[start:start + window_size]
            blocks = {item["block_no"] for item in window}
            lines = {(item["block_no"], item["line_no"]) for item in window}
            if len(blocks) > 1 or len(lines) > 3:
                continue

            window_tokens = [item["normalized"] for item in window]
            joined_window = " ".join(window_tokens)
            if candidate_norm != joined_window:
                continue

            rects = [item["rect"] for item in window]
            merged_rects = _merge_touching_rects(rects)
            if merged_rects:
                return merged_rects

    return []


# =========================================================
# EXACT SEARCH WITH WORD BOUNDARY
# =========================================================
def _search_exact(page: fitz.Page, text: str):
    """
    Exact search:
    - Case insensitive
    - Word boundary safe
    - Avoid partial numeric collision (6.2.5 vs 6.3)
    """

    matches = []
    for candidate in _build_search_candidates(text):
        try:
            candidate_matches = page.search_for(candidate, quads=False)
        except Exception:
            candidate_matches = []

        if candidate_matches:
            matches.extend(candidate_matches)
            if len(candidate) >= max(28, len(_normalize(text)) // 2):
                break

    return _unique_rects(matches)


# =========================================================
# FUZZY FALLBACK (CONTROLLED)
# =========================================================
def _search_fallback(page: fitz.Page, text: str):
    """
    Controlled fallback:
    - Only first 150 chars
    - Avoid tiny prefix collision
    """

    normalized_page_text = _normalize_for_compare(page.get_text("text"))
    if not normalized_page_text:
        return []

    for candidate in _build_search_candidates(text):
        candidate_norm = _normalize_for_compare(candidate)
        if len(candidate_norm) < 18:
            continue

        if candidate_norm not in normalized_page_text:
            continue

        try:
            matches = page.search_for(candidate, quads=False)
        except Exception:
            matches = []

        if matches:
            return _unique_rects(matches)

    return []


def _search_title_fallback(page: fitz.Page, text: str):
    normalized = _normalize(text)
    if len(normalized) < 6:
        return []

    try:
        matches = page.search_for(normalized, quads=False)
    except Exception:
        matches = []

    if matches:
        return _unique_rects(matches)

    return _search_by_line_overlap(page, normalized)


# =========================================================
# MAIN FUNCTION
# =========================================================
def highlight_pdf(
    pdf_path: str,
    highlights: list[str],
    pages: list[int],
    section_title: str = None,
    color: str = "yellow",
) -> str:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if section_title:
        highlights = [section_title] + highlights
    stroke_color = HIGHLIGHT_COLORS.get((color or "yellow").lower(), HIGHLIGHT_COLORS["yellow"])

    try:
        for page_index in pages:

            if page_index < 0 or page_index >= len(doc):
                continue

            page = doc[page_index]
            highlighted_rect_keys = set()

            for text in highlights:

                if not text or len(text.strip()) < 6:
                    continue

                # --------------------------------------------
                # 1️⃣ TRY EXACT MATCH FIRST
                # --------------------------------------------
                rects = _search_exact(page, text)

                # --------------------------------------------
                # 2️⃣ FALLBACK ONLY IF NOTHING FOUND
                # --------------------------------------------
                if not rects:
                    rects = _search_fallback(page, text)

                if not rects:
                    rects = _search_by_word_windows(page, text)

                if not rects:
                    rects = _search_by_line_overlap(page, text)

                if not rects:
                    rects = _search_title_fallback(page, text)

                for rect in rects:
                    if _rect_area(rect) <= 0:
                        continue
                    rect_key = tuple(round(value, 2) for value in (rect.x0, rect.y0, rect.x1, rect.y1))
                    if rect_key in highlighted_rect_keys:
                        continue
                    highlighted_rect_keys.add(rect_key)
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=stroke_color)
                    annot.set_opacity(0.35)
                    annot.update()

        # -----------------------------------------------------
        # SAVE OUTPUT
        # -----------------------------------------------------
        digest = hashlib.md5(
            (pdf_path + "".join(highlights) + "".join(str(page) for page in pages) + str(color)).encode()
        ).hexdigest()[:8]

        base, ext = os.path.splitext(pdf_path)
        out_path = f"{base}_highlighted_{digest}{ext}"

        doc.save(out_path, garbage=4, deflate=True)

    finally:
        doc.close()

    return out_path
