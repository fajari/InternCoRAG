import warnings

try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
except Exception:
    UnsupportedFieldAttributeWarning = UserWarning

warnings.filterwarnings(
    "ignore",
    message=r".*validate_default.*Field\(\).*",
    category=UnsupportedFieldAttributeWarning,
)

import streamlit as st
import tempfile
import os
import re
import html
import uuid

from documents.loader import load_document
from documents.chunker import chunk_documents
from rag.graph_pipeline import build_graph_bundle
from rag.chain import build_actor_profiles, build_social_relationships, collect_case_reading, collect_semantic_chunks, pack_semantic_summary_chunks
from rag.vectorstore import get_vectorstore, has_documents, clear_workspace_documents
from rag.chain import get_chain


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Analisa Keputusan Mahkamah Agung",
    layout="wide",
)

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("workspace_id", "dev-workspace")
st.session_state.setdefault("last_result", None)

CASE_ANALYSIS_DISABLED = True


def render_answer(text: str):
    if not text:
        st.write("")
        return

    pattern = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    last_end = 0

    for match in pattern.finditer(text):
        before = text[last_end:match.start()].strip()
        if before:
            st.markdown(before)

        render_mermaid(match.group(1).strip())
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        st.markdown(remaining)


def render_mermaid(diagram: str):
    if not diagram:
        return

    container_id = f"mermaid-{uuid.uuid4().hex}"
    toolbar_id = f"{container_id}-toolbar"
    download_jpg_id = f"{container_id}-jpg"
    download_svg_id = f"{container_id}-svg"
    zoom_in_id = f"{container_id}-zoom-in"
    zoom_out_id = f"{container_id}-zoom-out"
    reset_zoom_id = f"{container_id}-zoom-reset"
    fullscreen_id = f"{container_id}-fullscreen"
    zoom_label_id = f"{container_id}-zoom-label"
    hint_id = f"{container_id}-hint"
    safe_diagram = html.escape(diagram)
    st.iframe(
        f"""
        <div id="{toolbar_id}">
          <div class="diagram-meta">
            <div class="diagram-title-wrap">
              <div class="diagram-title">Diagram Investigasi</div>
              <div class="diagram-subtitle">Zoom, fullscreen, atau ekspor diagram saat dibutuhkan.</div>
            </div>
            <div id="{zoom_label_id}" class="zoom-chip">100%</div>
          </div>
          <div class="diagram-actions">
            <button id="{zoom_out_id}" type="button">-</button>
            <button id="{zoom_in_id}" type="button">+</button>
            <button id="{reset_zoom_id}" type="button">Reset</button>
            <button id="{fullscreen_id}" type="button">Fullscreen</button>
            <button id="{download_jpg_id}" type="button">JPG</button>
            <button id="{download_svg_id}" type="button">SVG</button>
          </div>
        </div>
        <div id="{hint_id}" class="diagram-hint">Geser panel untuk melihat area yang terpotong saat zoom diperbesar.</div>
        <div id="{container_id}-frame">
          <div id="{container_id}" class="mermaid">
            {safe_diagram}
          </div>
        </div>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{
            startOnLoad: true,
            securityLevel: "loose",
            theme: "default",
            flowchart: {{ useMaxWidth: true, htmlLabels: true }}
          }});
          const el = document.getElementById("{container_id}");
          const frame = document.getElementById("{container_id}-frame");
          const jpgButton = document.getElementById("{download_jpg_id}");
          const svgButton = document.getElementById("{download_svg_id}");
          const zoomInButton = document.getElementById("{zoom_in_id}");
          const zoomOutButton = document.getElementById("{zoom_out_id}");
          const resetZoomButton = document.getElementById("{reset_zoom_id}");
          const fullscreenButton = document.getElementById("{fullscreen_id}");
          const zoomLabel = document.getElementById("{zoom_label_id}");
          let zoomLevel = 1;

          const applyZoom = () => {{
            if (!el) return;
            el.style.transform = `scale(${{zoomLevel}})`;
            el.style.transformOrigin = "top left";
            if (zoomLabel) {{
              zoomLabel.textContent = `${{Math.round(zoomLevel * 100)}}%`;
            }}
          }};

          const downloadDataUrl = (dataUrl, filename) => {{
            const link = document.createElement("a");
            link.href = dataUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
          }};

          const buildSvgData = () => {{
            const svg = el?.querySelector("svg");
            if (!svg) return null;
            const serializer = new XMLSerializer();
            let source = serializer.serializeToString(svg);
            if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {{
              source = source.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
            }}
            if (!source.includes('xmlns:xlink="http://www.w3.org/1999/xlink"')) {{
              source = source.replace("<svg", '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
            }}
            return {{
              source,
              dataUrl: "data:image/svg+xml;charset=utf-8," + encodeURIComponent(source),
              svg,
            }};
          }};

          svgButton?.addEventListener("click", () => {{
            const svgData = buildSvgData();
            if (!svgData) return;
            downloadDataUrl(svgData.dataUrl, "diagram-investigasi.svg");
          }});

          zoomInButton?.addEventListener("click", () => {{
            zoomLevel = Math.min(zoomLevel + 0.2, 3);
            applyZoom();
          }});

          zoomOutButton?.addEventListener("click", () => {{
            zoomLevel = Math.max(zoomLevel - 0.2, 0.5);
            applyZoom();
          }});

          resetZoomButton?.addEventListener("click", () => {{
            zoomLevel = 1;
            applyZoom();
            frame?.scrollTo({{ top: 0, left: 0, behavior: "smooth" }});
          }});

          fullscreenButton?.addEventListener("click", async () => {{
            if (!frame) return;
            if (document.fullscreenElement === frame) {{
              await document.exitFullscreen();
              return;
            }}
            await frame.requestFullscreen();
          }});

          document.addEventListener("fullscreenchange", () => {{
            if (!fullscreenButton || !frame) return;
            fullscreenButton.textContent = document.fullscreenElement === frame ? "Exit Fullscreen" : "Fullscreen";
          }});

          jpgButton?.addEventListener("click", async () => {{
            const svgData = buildSvgData();
            if (!svgData) return;
            const svgRect = svgData.svg.getBoundingClientRect();
            const width = Math.max(1600, Math.ceil(svgRect.width) || 1600);
            const height = Math.max(900, Math.ceil(svgRect.height) || 900);
            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, width, height);
            const img = new Image();
            img.onload = () => {{
              const scale = Math.min(width / img.width, height / img.height);
              const drawWidth = img.width * scale;
              const drawHeight = img.height * scale;
              const x = (width - drawWidth) / 2;
              const y = (height - drawHeight) / 2;
              ctx.drawImage(img, x, y, drawWidth, drawHeight);
              downloadDataUrl(canvas.toDataURL("image/jpeg", 0.95), "diagram-investigasi.jpg");
            }};
            img.src = svgData.dataUrl;
          }});

          if (el) {{
            try {{
              await mermaid.run({{ nodes: [el] }});
              applyZoom();
            }} catch (err) {{
              const detail = err?.message || err?.str || JSON.stringify(err, null, 2) || String(err);
              el.innerHTML = `<div style="border:1px solid #fecaca;background:#fef2f2;color:#991b1b;border-radius:12px;padding:1rem;font:500 13px/1.5 sans-serif;"><div style="font-weight:700;margin-bottom:0.4rem;">Diagram Mermaid gagal dirender</div><div style="white-space:pre-wrap;">${{detail}}</div></div>`;
            }}
          }}
        </script>
        <style>
          #{toolbar_id} {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.9rem;
            margin-bottom: 0.55rem;
            flex-wrap: wrap;
          }}
          #{toolbar_id} .diagram-meta {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
          }}
          #{toolbar_id} .diagram-title-wrap {{
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
          }}
          #{toolbar_id} .diagram-title {{
            font: 700 14px/1.2 sans-serif;
            color: #111827;
          }}
          #{toolbar_id} .diagram-subtitle {{
            font: 500 12px/1.35 sans-serif;
            color: #6b7280;
          }}
          #{toolbar_id} .diagram-actions {{
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
          }}
          #{zoom_label_id} {{
            border: 1px solid #dbe4f0;
            background: #f8fafc;
            color: #0f172a;
            border-radius: 999px;
            padding: 0.38rem 0.72rem;
            font: 700 12px/1 sans-serif;
          }}
          #{toolbar_id} button {{
            appearance: none;
            border: 1px solid #d1d5db;
            background: #f9fafb;
            color: #111827;
            border-radius: 999px;
            min-width: 44px;
            padding: 0.5rem 0.85rem;
            font: 600 13px/1 sans-serif;
            cursor: pointer;
            transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
          }}
          #{toolbar_id} button:hover {{
            background: #f3f4f6;
            border-color: #9ca3af;
          }}
          #{toolbar_id} button:active {{
            transform: translateY(1px);
          }}
          #{hint_id} {{
            font: 500 12px/1.4 sans-serif;
            color: #6b7280;
            margin-bottom: 0.6rem;
          }}
          #{container_id}-frame {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem 1rem 1.25rem;
            overflow-x: auto;
            overflow-y: auto;
            min-height: 820px;
            max-height: 80vh;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
          }}
          #{container_id}-frame:fullscreen {{
            max-height: none;
            height: 100vh;
            padding: 1.25rem;
            background: #ffffff;
          }}
          #{container_id} {{
            display: inline-block;
            min-width: 100%;
          }}
          #{container_id} svg {{
            width: max-content;
            min-width: 100%;
          }}
          @media (max-width: 700px) {{
            #{toolbar_id} {{
              align-items: stretch;
            }}
            #{toolbar_id} .diagram-actions {{
              width: 100%;
            }}
            #{toolbar_id} button {{
              flex: 1 1 auto;
            }}
          }}
        </style>
        """,
        height=920,
    )


def clear_workspace_state():
    st.session_state.chat_history = []
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
# HEADER
# =====================================================
st.title("Analisa Kasus Keputusan Mahkamah Agung")
st.caption("Unggah PDF kasus, pilih mode analisis yang sesuai, lalu lanjutkan dengan pertanyaan bebas bila perlu.")
if CASE_ANALYSIS_DISABLED:
    st.warning("Sementara mode Analisis Isi PDF dinonaktifkan.")


# =====================================================
# SIDEBAR UPLOAD
# =====================================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

with st.sidebar:
    st.markdown("### Alur Penggunaan")
    st.markdown(
        """
        1. **Upload PDF**
        2. **Process Document**
        3. **Pilih mode analisis di halaman utama**
        4. **Lanjutkan dengan chat bila perlu**
        """
    )
    st.caption("Sidebar dipakai untuk menyiapkan dokumen. Analisis dan diagram dijalankan dari area utama.")
    st.divider()

    st.markdown("### Step 1 · Upload")
    uploaded_file = st.file_uploader(
        "Pilih file PDF kasus",
        type=["pdf"]
    )

    if uploaded_file:
        st.success(f"File siap diproses: `{uploaded_file.name}`")
    else:
        st.info("Belum ada file yang dipilih.")

    st.markdown("### Step 2 · Proses")
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

            combined_text = "\n".join(
                chunk.page_content.strip()
                for chunk in chunks
                if getattr(chunk, "page_content", "").strip()
            ).strip()
            case_reading = collect_case_reading(chunks, combined_text)
            narrative_text = case_reading.get("narrative_text") or combined_text
            semantic_chunks = case_reading.get("narrative_chunks") or collect_semantic_chunks(chunks, narrative_text) or pack_semantic_summary_chunks(narrative_text)
            actors, relationships = build_social_relationships(narrative_text)
            actor_profiles = build_actor_profiles(actors, narrative_text)
            build_graph_bundle(
                workspace_id=st.session_state.workspace_id,
                question="Buat graph kasus dari dokumen yang diunggah.",
                semantic_chunks=semantic_chunks,
                actors=actors,
                relationships=relationships,
                actor_profiles=actor_profiles,
            )

            os.unlink(tmp.name)

            st.success("Document indexed")
            st.rerun()
    elif uploaded_file:
        st.caption("Klik `Process Document` untuk menyiapkan isi PDF ke mode analisis.")

    st.markdown("### Step 3 · Reset")
    if st.button("Clear Index"):
        clear_workspace_documents(st.session_state.workspace_id)
        delete_workspace_uploads(UPLOAD_DIR, st.session_state.workspace_id)
        clear_workspace_state()
        st.success("Index cleared")
        st.rerun()

    st.divider()
    st.markdown("### Setelah Itu")
    st.markdown(
        """
        - **Buat Ringkasan Kasus** aktif untuk menampilkan executive summary
        - **Analisis Isi PDF** sementara dinonaktifkan
        - Gunakan **Diagram Jaringan Sosial** untuk memetakan aktor dan relasi
        - Gunakan **chat lanjutan** untuk pertanyaan spesifik
        """
    )


# =====================================================
# MAIN AREA
# =====================================================
st.divider()

if not has_documents(st.session_state.workspace_id):
    st.info("Silakan unggah PDF terlebih dahulu dari sidebar untuk memulai analisis dokumen.")
    st.stop()


st.markdown(
    """
    <style>
    .mode-card-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.9rem;
      margin: 0.35rem 0 1rem 0;
    }
    .mode-card {
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      padding: 1rem 1rem 0.95rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .mode-card h4 {
      margin: 0 0 0.35rem 0;
      font: 700 15px/1.2 sans-serif;
      color: #111827;
    }
    .mode-card p {
      margin: 0;
      font: 500 12.5px/1.5 sans-serif;
      color: #4b5563;
    }
    .section-label {
      font: 800 13px/1 sans-serif;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 0.35rem;
    }
    .section-copy {
      margin: 0 0 0.95rem 0;
      font: 500 13px/1.55 sans-serif;
      color: #475569;
    }
    @media (max-width: 900px) {
      .mode-card-grid {
        grid-template-columns: 1fr;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Mode Analisis</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Gunakan aksi cepat jika ingin hasil siap pakai. Setiap mode punya fokus yang berbeda, jadi Anda tidak perlu menebak prompt dari awal.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="mode-card-grid">
      <div class="mode-card">
        <h4>Ringkasan & Analisis PDF</h4>
        <p>Sementara dinonaktifkan. Gunakan chat lanjutan untuk pertanyaan spesifik atau diagram jaringan sosial.</p>
      </div>
      <div class="mode-card">
        <h4>Diagram Jaringan Sosial</h4>
        <p>Fokus pada pihak yang terlibat, peran forensik, relasi antar pihak, dan diagram investigasi Mermaid.</p>
      </div>
      <div class="mode-card">
        <h4>Pertanyaan Chat Bebas</h4>
        <p>Dipakai setelah analisis awal jika Anda ingin bertanya detail, menguji dugaan, atau menelusuri bagian tertentu.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# CHAT HISTORY
# =====================================================
history_items = st.session_state.chat_history
hide_last_history_answer = (
    st.session_state.last_result is not None
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
# RUNNER
# =====================================================
def run_question(question_text: str):
    with st.chat_message("user"):
        st.write(question_text)

    chain = get_chain(st.session_state.workspace_id)
    result = chain(question_text)

    st.session_state.last_result = result
    st.session_state.chat_history.append({
        "question": question_text,
        "answer": result.get("answer", "")
    })
    st.session_state.chat_history = st.session_state.chat_history[-8:]


# =====================================================
# QUICK ACTIONS
# =====================================================
st.markdown('<div class="section-label">Aksi Cepat</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Pilih salah satu untuk langsung menghasilkan output sesuai kebutuhan tanpa mengetik pertanyaan manual.</p>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(
        "Buat Ringkasan Kasus",
        use_container_width=True,
    ):
        run_question("Buat ringkasan atau rangkuman dari kasus pada PDF ini.")

with col2:
    if st.button(
        "Analisis Isi PDF",
        use_container_width=True,
        disabled=CASE_ANALYSIS_DISABLED,
    ):
        run_question("Lakukan analisis kasus dari PDF ini.")

with col3:
    if st.button("Diagram Jaringan Sosial", use_container_width=True):
        run_question("Buat diagram jaringan sosial pihak yang terlibat pada kasus di PDF ini.")


# =====================================================
# CHAT INPUT
# =====================================================
st.markdown('<div class="section-label">Chat Lanjutan</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Gunakan kolom chat untuk pertanyaan susulan, misalnya menguji satu nama, satu transfer, satu kronologi, atau meminta penjelasan lebih rinci dari hasil analisis sebelumnya.</p>',
    unsafe_allow_html=True,
)
question = st.chat_input("Contoh: siapa aktor yang paling sering muncul, apa indikasi aliran dana, atau jelaskan kronologi pada bulan tertentu...")

if question:
    run_question(question)


# =====================================================
# DISPLAY LAST RESULT (RERUN SAFE)
# =====================================================
if st.session_state.last_result:

    result = st.session_state.last_result

    with st.chat_message("assistant"):
        render_answer(result.get("answer", ""))

def build_overview_from_keypoints(keypoints: list[str]) -> str:
    if not keypoints:
        return ""

    first = keypoints[0]

    # potong maksimal 25 kata
    words = first.split()
    return " ".join(words[:25]) + "..."
