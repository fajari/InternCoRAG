import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(file_path: str, original_filename: str):
    ext = os.path.splitext(original_filename)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()

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

