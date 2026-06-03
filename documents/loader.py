import os
import shutil
import subprocess
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

try:
    import fitz
except Exception:
    fitz = None


MIN_READABLE_PAGE_CHARS = 80


def load_pdf_with_pypdf(file_path: str) -> List[Document]:
    return PyPDFLoader(file_path).load()


def remove_repeated_page_lines(pages: List[str]) -> List[str]:
    line_counts: dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        candidates = lines[:3] + lines[-3:]
        for line in candidates:
            if len(line) <= 4:
                continue
            line_counts[line] = line_counts.get(line, 0) + 1

    repeated = {
        line for line, count in line_counts.items()
        if count >= max(2, len(pages) // 2)
    }
    cleaned_pages = []
    for page in pages:
        cleaned_lines = [
            line for line in page.splitlines()
            if line.strip() not in repeated
        ]
        cleaned_pages.append("\n".join(cleaned_lines).strip())
    return cleaned_pages


def load_pdf_layout_aware(file_path: str) -> List[Document]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not available")

    pdf = fitz.open(file_path)
    raw_pages: List[str] = []
    page_metadata: List[dict] = []

    try:
        for page_index, page in enumerate(pdf):
            blocks = page.get_text("blocks", sort=True)
            lines = []
            table_like_lines = 0
            for block in blocks:
                if len(block) < 5:
                    continue
                text = str(block[4] or "").strip()
                if not text:
                    continue
                if "\t" in text or text.count("  ") >= 3:
                    table_like_lines += 1
                lines.append(text)
            raw_pages.append("\n\n".join(lines).strip())
            page_metadata.append(
                {
                    "page": page_index,
                    "parser": "pymupdf_layout",
                    "layout_blocks": len(blocks),
                    "table_like_blocks": table_like_lines,
                    "ocr_applied": False,
                }
            )
    finally:
        pdf.close()

    cleaned_pages = remove_repeated_page_lines(raw_pages)
    documents = [
        Document(page_content=text, metadata=page_metadata[index])
        for index, text in enumerate(cleaned_pages)
        if text.strip()
    ]
    if not documents:
        raise RuntimeError("PyMuPDF returned no readable page content")
    return documents


def ocr_pdf_with_tesseract(file_path: str) -> List[Document]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not available for OCR rendering")
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract is not available on this system")

    documents: List[Document] = []
    pdf = fitz.open(file_path)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for page_index, page in enumerate(pdf):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_path = os.path.join(tmpdir, f"page-{page_index + 1}.png")
                output_base = os.path.join(tmpdir, f"page-{page_index + 1}")
                pixmap.save(image_path)
                result = subprocess.run(
                    ["tesseract", image_path, output_base, "-l", "ind+eng", "--psm", "6"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    result = subprocess.run(
                        ["tesseract", image_path, output_base, "-l", "eng", "--psm", "6"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                text_path = f"{output_base}.txt"
                if not os.path.exists(text_path):
                    continue
                with open(text_path, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read().strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page": page_index,
                                "parser": "tesseract_ocr",
                                "ocr_applied": True,
                            },
                        )
                    )
    finally:
        pdf.close()

    if not documents:
        raise RuntimeError("OCR returned no readable page content")
    return documents


def has_enough_readable_text(documents: List[Document]) -> bool:
    readable_pages = [
        doc for doc in documents
        if len((doc.page_content or "").strip()) >= MIN_READABLE_PAGE_CHARS
    ]
    return len(readable_pages) >= max(1, len(documents) // 3)


def load_pdf_with_pdftotext(file_path: str) -> List[Document]:
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext is not available on this system")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_output:
        output_path = tmp_output.name

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", file_path, output_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "pdftotext failed to extract the PDF")

        with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
            raw_text = handle.read()

        pages = raw_text.split("\f")
        documents: List[Document] = []
        for page_index, page_text in enumerate(pages):
            cleaned = page_text.strip()
            if not cleaned:
                continue
            documents.append(
                Document(
                    page_content=cleaned,
                    metadata={"page": page_index},
                )
            )

        if not documents:
            raise RuntimeError("pdftotext returned no readable page content")

        return documents
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def load_document(file_path: str, original_filename: str):
    ext = os.path.splitext(original_filename)[1].lower()

    if ext == ".pdf":
        try:
            docs = load_pdf_layout_aware(file_path)
            if has_enough_readable_text(docs):
                return docs
        except Exception:
            pass

        try:
            docs = ocr_pdf_with_tesseract(file_path)
            if docs:
                return docs
        except Exception:
            pass

        try:
            return load_pdf_with_pdftotext(file_path)
        except Exception:
            return load_pdf_with_pypdf(file_path)

    if ext in [".txt", ".md"]:
        return TextLoader(
            file_path,
            encoding="utf-8",
            autodetect_encoding=True
        ).load()

    raise ValueError(f"Unsupported file type: {ext}")

def is_table_of_contents(text: str) -> bool:
    t = text.lower()
    return (
        "table of contents" in t
        or "contents" in t
        or t.count("....") > 3
    )
