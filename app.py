import streamlit as st
import tempfile
import os
import base64
import streamlit.components.v1 as components

from documents.loader import load_document
from documents.chunker import chunk_documents
from rag.vectorstore import get_vectorstore, has_documents, clear_workspace_documents
from rag.chain import get_chain
from utils.pdf_highlighter import highlight_pdf


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Internal Company Knowledge Assistant",
    layout="wide",
)

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("workspace_id", "dev-workspace")
st.session_state.setdefault("open_pdf_index", None)
st.session_state.setdefault("last_result", None)


def render_answer(text: str):
    if not text:
        st.write("")
        return

    st.markdown(text)


def clear_workspace_state():
    st.session_state.chat_history = []
    st.session_state.open_pdf_index = None
    st.session_state.last_result = None


def delete_workspace_uploads(upload_dir: str, workspace_id: str):
    prefix = f"{workspace_id}_"

    for filename in os.listdir(upload_dir):
        if not filename.startswith(prefix):
            continue

        file_path = os.path.join(upload_dir, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)


# =====================================================
# SAFE PDF VIEWER
# =====================================================
def render_pdf(path: str, page_index: int):
    try:
        with open(path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        components.html(
            f"""
            <iframe
                src="data:application/pdf;base64,{base64_pdf}#page={page_index + 1}"
                width="100%"
                height="720"
                style="border:none;">
            </iframe>
            """,
            height=740,
        )
    except Exception as e:
        st.error(f"PDF render error: {e}")


# =====================================================
# HEADER
# =====================================================
st.title("🏢 Internal Company Knowledge Assistant")


# =====================================================
# SIDEBAR UPLOAD
# =====================================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Indexing document..."):

            file_path = os.path.join(
                UPLOAD_DIR,
                f"{st.session_state.workspace_id}_{uploaded_file.name}"
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(uploaded_file.getbuffer())
            tmp.close()

            docs = load_document(tmp.name, uploaded_file.name)
            chunks = chunk_documents(docs, uploaded_file.name)

            for d in chunks:
                d.metadata.update({
                    "workspace_id": st.session_state.workspace_id,
                    "file_path": file_path,
                    "source": uploaded_file.name,
                })

            clear_workspace_documents(st.session_state.workspace_id)
            get_vectorstore().add_documents(chunks)
            os.unlink(tmp.name)

            st.success("Document indexed")
            st.rerun()

    if st.button("Clear Index"):
        clear_workspace_documents(st.session_state.workspace_id)
        delete_workspace_uploads(UPLOAD_DIR, st.session_state.workspace_id)
        clear_workspace_state()
        st.success("Index cleared")
        st.rerun()


# =====================================================
# MAIN AREA
# =====================================================
st.divider()

if not has_documents(st.session_state.workspace_id):
    st.info("Please upload a document first.")
    st.stop()


# =====================================================
# CHAT HISTORY
# =====================================================
history_items = st.session_state.chat_history
hide_last_history_answer = (
    st.session_state.open_pdf_index is not None
    and st.session_state.last_result is not None
    and bool(history_items)
)

for index, h in enumerate(history_items):
    with st.chat_message("user"):
        st.write(h["question"])
    if hide_last_history_answer and index == len(history_items) - 1:
        continue
    with st.chat_message("assistant"):
        render_answer(h["answer"])


# =====================================================
# CHAT INPUT
# =====================================================
question = st.chat_input("Ask a question...")

if question:

    # Reset PDF state when new question
    st.session_state.open_pdf_index = None

    with st.chat_message("user"):
        st.write(question)

    chain = get_chain(st.session_state.workspace_id)
    result = chain(question)

    # Save last result for rerun stability
    st.session_state.last_result = result

    st.session_state.chat_history.append({
        "question": question,
        "answer": result.get("answer", "")
    })

    st.session_state.chat_history = st.session_state.chat_history[-5:]


# =====================================================
# DISPLAY LAST RESULT (RERUN SAFE)
# =====================================================
if st.session_state.last_result:

    result = st.session_state.last_result
    show_answer_block = st.session_state.open_pdf_index is None

    with st.chat_message("assistant"):

        if show_answer_block:
            render_answer(result.get("answer", ""))

            for h in result.get("highlight", []):
                st.markdown(f"> {h}")

        # ================================
        # READ MORE SECTION
        # ================================
        for i, s in enumerate(result.get("sources", [])):

            file_path = s.get("file_path")

            if not file_path:
                continue

            if st.button("📄 Read More (Show PDF)", key=f"read_more_{i}"):
                st.session_state.open_pdf_index = i

            if st.session_state.open_pdf_index == i:

                try:
                    pdf = highlight_pdf(
                        file_path,
                        result.get("highlight", []),
                        result.get("highlight_pages", []),
                        section_title=result.get("selected_section_title") or s.get("section")
                    )

                    render_pdf(
                        pdf,
                        result.get("highlight_pages", [0])[0]
                    )

                    with open(pdf, "rb") as f:
                        st.download_button(
                            "⬇️ Download highlighted PDF",
                            f,
                            file_name=os.path.basename(pdf),
                            mime="application/pdf",
                            key=f"download_{i}"
                        )

                except Exception as e:
                    st.error(f"PDF highlight error: {e}")

def build_overview_from_keypoints(keypoints: list[str]) -> str:
    if not keypoints:
        return ""

    first = keypoints[0]

    # potong maksimal 25 kata
    words = first.split()
    return " ".join(words[:25]) + "..."
