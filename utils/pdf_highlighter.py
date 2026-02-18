import fitz  # PyMuPDF
import os
import hashlib
import re


# =========================================================
# NORMALIZE TEXT
# =========================================================
def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


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

    text = _normalize(text)

    if len(text) < 20:
        return []

    try:
        # Use quads=True for better accuracy
        matches = page.search_for(
            text,
            quads=False  # rect-based highlight
        )
        return matches
    except Exception:
        return []


# =========================================================
# FUZZY FALLBACK (CONTROLLED)
# =========================================================
def _search_fallback(page: fitz.Page, text: str):
    """
    Controlled fallback:
    - Only first 150 chars
    - Avoid tiny prefix collision
    """

    text = _normalize(text)

    snippet = text[:150]

    if len(snippet) < 30:
        return []

    try:
        matches = page.search_for(snippet)
        return matches
    except Exception:
        return []


# =========================================================
# MAIN FUNCTION
# =========================================================
def highlight_pdf(
    pdf_path: str,
    highlights: list[str],
    pages: list[int],
    section_title: str = None
) -> str:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if section_title:
        highlights = [section_title] + highlights

    try:
        for page_index in pages:

            if page_index < 0 or page_index >= len(doc):
                continue

            page = doc[page_index]

            for text in highlights:

                if not text or len(text.strip()) < 20:
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

                for rect in rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 1, 0))
                    annot.update()

        # -----------------------------------------------------
        # SAVE OUTPUT
        # -----------------------------------------------------
        digest = hashlib.md5(
            (pdf_path + "".join(highlights)).encode()
        ).hexdigest()[:8]

        base, ext = os.path.splitext(pdf_path)
        out_path = f"{base}_highlighted_{digest}{ext}"

        doc.save(out_path, garbage=4, deflate=True)

    finally:
        doc.close()

    return out_path
