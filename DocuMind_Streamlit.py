"""DocuMind AI backend: PDF processing, hybrid retrieval, Qwen QA and summarization."""

import hashlib
import numpy as np
import torch
import fitz
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

BGE_NAME = "BAAI/bge-base-en-v1.5"
QWEN_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_TOKENIZER_NAME = "bert-base-uncased"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CANDIDATES = 10
TOP_K = 6
RRF_K = 60


def load_models():
    """Load tokenizers and models once; call this from a cached UI function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_tokenizer = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER_NAME)
    embedding_model = SentenceTransformer(BGE_NAME, device=device)
    generation_tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME)
    dtype = torch.float16 if device == "cuda" else torch.float32
    generation_model = AutoModelForCausalLM.from_pretrained(
        QWEN_NAME, torch_dtype=dtype
    ).to(device)
    generation_model.eval()
    return {
        "device": device,
        "chunk_tokenizer": chunk_tokenizer,
        "embedding_model": embedding_model,
        "generation_tokenizer": generation_tokenizer,
        "generation_model": generation_model,
    }


def extract_pdf_pages(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    doc.close()
    return pages


def create_page_chunks(page_data, tokenizer):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for page in page_data:
        token_ids = tokenizer(
            page["text"], add_special_tokens=False
        )["input_ids"]
        start = 0
        while start < len(token_ids):
            chunk_text = tokenizer.decode(
                token_ids[start:start + CHUNK_SIZE],
                skip_special_tokens=True,
            ).strip()
            if chunk_text:
                chunks.append({"page": page["page"], "text": chunk_text})
            start += step
    return chunks


def build_document_indexes(pdf_bytes, models):
    pages = extract_pdf_pages(pdf_bytes)
    if not pages:
        raise ValueError(
            "No selectable text found. This may be a scanned PDF that needs OCR."
        )

    chunks = create_page_chunks(pages, models["chunk_tokenizer"])
    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    texts = [chunk["text"] for chunk in chunks]
    vectors = models["embedding_model"].encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(vectors)

    vector_index = faiss.IndexFlatIP(vectors.shape[1])
    vector_index.add(vectors)
    bm25_index = BM25Okapi([text.lower().split() for text in texts])

    return {
        "page_data": pages,
        "page_chunks": chunks,
        "vector_index": vector_index,
        "bm25_index": bm25_index,
        "document_hash": hashlib.sha256(pdf_bytes).hexdigest(),
    }


def hybrid_retrieval(question, document_data, models, top_k=TOP_K):
    chunks = document_data["page_chunks"]
    count = min(CANDIDATES, len(chunks))

    query_vector = models["embedding_model"].encode(
        [question], convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(query_vector)
    _, bge_indices = document_data["vector_index"].search(query_vector, k=count)

    bm25_scores = document_data["bm25_index"].get_scores(question.lower().split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:count]

    fused = {}
    for rank, idx in enumerate(bge_indices[0], start=1):
        if idx >= 0:
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(bm25_indices, start=1):
        fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(fused, key=fused.get, reverse=True)[:min(top_k, len(fused))]
    return [
        {
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "rrf_score": float(fused[idx]),
            "chunk_index": idx,
        }
        for idx in ranked
    ]


def generate_with_qwen(messages, models, max_new_tokens=350):
    tokenizer = models["generation_tokenizer"]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=7000
    ).to(models["device"])

    with torch.inference_mode():
        output_ids = models["generation_model"].generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_question(question, document_data, models):
    results = hybrid_retrieval(question, document_data, models, top_k=TOP_K)
    context = "\n\n".join(
        f"[Page {result['page']}]\n{result['text']}" for result in results
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are DocuMind AI, an assistant for understanding research papers. "
                "Answer using only the supplied PDF context. Be accurate and clear. "
                "Do not invent facts. If the context does not contain the answer, "
                "say that the supplied context is insufficient. Do not guess or "
                "invent page numbers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PDF context:\n{context}\n\nQuestion: {question}\n\n"
                "Give a complete, concise answer with important technical details."
            ),
        },
    ]
    answer = generate_with_qwen(messages, models, max_new_tokens=350)
    pages = sorted({result["page"] for result in results})
    return answer, results, pages


def summarize_document(document_data, models, progress_callback=None):
    summaries = []
    chunks = document_data["page_chunks"]

    for i, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "Faithfully summarize this research paper section. Preserve methods, "
                    "findings, equations, and technical details. Do not invent facts. "
                    "Ignore acknowledgements and references unless essential."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this section from page {chunk['page']}:\n\n"
                    f"{chunk['text']}"
                ),
            },
        ]
        summary = generate_with_qwen(messages, models, max_new_tokens=140)
        summaries.append(f"Page {chunk['page']}: {summary}")
        if progress_callback:
            progress_callback(i, len(chunks))

    combined = "\n\n".join(summaries)
    final_messages = [
        {
            "role": "system",
            "content": (
                "Create a faithful research paper summary using only the notes. "
                "Use headings where information is available: Research problem, "
                "Proposed method, Methodology, Experiments and results, Conclusion. "
                "Do not add unsupported claims. Omit missing details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Combine these section summaries into one coherent summary:\n\n"
                + combined
            ),
        },
    ]
    return generate_with_qwen(final_messages, models, max_new_tokens=700)


"""DocuMind AI: single-file Streamlit application with integrated RAG backend and UI."""

import streamlit as st

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")

st.title("📚 DocuMind AI")
st.caption(
    "Your AI research paper assistant — upload one PDF, ask questions, "
    "and explore its contents."
)

with st.sidebar:
    st.header("About DocuMind AI")
    st.write(
        "A single-document research assistant using BGE embeddings, FAISS, "
        "BM25, Reciprocal Rank Fusion, and Qwen."
    )
    st.divider()
    st.write("**Supported file:** PDF")
    st.write("**Retrieval:** BGE + BM25 + RRF")
    st.write("**Generation:** Qwen2.5-1.5B-Instruct")
    st.caption("Scanned PDFs need OCR before upload.")


@st.cache_resource(show_spinner="Loading DocuMind AI models...")
def get_models():
    return load_models()


if "document_data" not in st.session_state:
    st.session_state.document_data = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "document_hash" not in st.session_state:
    st.session_state.document_hash = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

uploaded_file = st.file_uploader(
    "Upload your research paper", type=["pdf"],
    help="Upload a text-based PDF. Scanned PDFs require OCR first."
)

if uploaded_file is not None:
    st.info(f"Selected PDF: {uploaded_file.name}")
    if st.button("Process PDF", type="primary", use_container_width=True):
        try:
            models = get_models()
            pdf_bytes = uploaded_file.getvalue()
            with st.spinner("Extracting text, creating chunks, and building indexes..."):
                document_data = build_document_indexes(pdf_bytes, models)
            st.session_state.document_data = document_data
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_hash = document_data["document_hash"]
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.success(
                f"Processed {len(document_data['page_data'])} pages and "
                f"indexed {len(document_data['page_chunks'])} chunks."
            )
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

if st.session_state.document_data is not None:
    data = st.session_state.document_data
    st.success(f"Ready: {st.session_state.document_name}")

    metric1, metric2 = st.columns(2)
    metric1.metric("Pages with extracted text", len(data["page_data"]))
    metric2.metric("Indexed chunks", len(data["page_chunks"]))

    qa_tab, summary_tab = st.tabs(["Ask your PDF", "Document summary"])

    with qa_tab:
        st.subheader("Ask a question")
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("sources"):
                    with st.expander("Retrieved passages"):
                        for source in item["sources"]:
                            st.markdown(f"**Page {source['page']}**")
                            st.write(source["text"])

        question = st.chat_input("Ask something about your research paper...")
        if question:
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    models = get_models()
                    with st.spinner("Searching the paper and generating an answer..."):
                        answer, results, pages = answer_question(
                            question, data, models
                        )
                    st.markdown(answer)
                    st.caption(
                        "Retrieved source pages: "
                        + (", ".join(map(str, pages)) if pages else "None")
                    )
                    with st.expander("Retrieved passages"):
                        for result in results:
                            st.markdown(f"**Page {result['page']}**")
                            st.write(result["text"])
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": results,
                        }
                    )
                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")

    with summary_tab:
        st.subheader("Document summary")
        if st.session_state.summary is None:
            if st.button("Generate summary", type="primary"):
                try:
                    models = get_models()
                    progress_bar = st.progress(0, text="Preparing summary...")

                    def update_progress(done, total):
                        progress_bar.progress(
                            done / total,
                            text=f"Summarizing chunk {done}/{total}...",
                        )

                    with st.spinner("Generating document summary. This may take a while..."):
                        st.session_state.summary = summarize_document(
                            data, models, progress_callback=update_progress
                        )
                    progress_bar.empty()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not generate the summary: {exc}")
        else:
            st.markdown(st.session_state.summary)
            st.download_button(
                "Download summary",
                data=st.session_state.summary,
                file_name="DocuMind_AI_Summary.txt",
                mime="text/plain",
            )
else:
    st.info("Upload a research paper PDF and select **Process PDF** to get started.")

st.caption(
    "Educational research assistant. Verify important answers against the original PDF. "
    "Displayed page references identify retrieved passages; they are not claim-by-claim citation verification."
)
