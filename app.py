import os
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

APP_TITLE = "Chat With Your Documents"
DB_PATH = Path(".chroma_db")
COLLECTION_NAME = "uploaded_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title("📚 Chat With Your Documents")
st.caption("A Retrieval-Augmented Generation (RAG) app that answers questions using your uploaded files.")

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            reader = PdfReader(tmp_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        finally:
            os.unlink(tmp_path)
    if name.endswith(".txt") or name.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            doc = Document(tmp_path)
            return "\n".join(p.text for p in doc.paragraphs)
        finally:
            os.unlink(tmp_path)
    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, or Markdown.")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def index_files(uploaded_files):
    collection = get_collection()
    embedder = get_embedder()
    indexed_chunks = 0
    for file in uploaded_files:
        text = extract_text(file)
        chunks = chunk_text(text)
        if not chunks:
            st.warning(f"No extractable text found in {file.name}. Scanned PDFs may need OCR.")
            continue
        # Stable IDs prevent collisions between files with identical chunk indexes.
        ids = [f"{file.name}-{i}-{abs(hash(chunks[i]))}" for i in range(len(chunks))]
        embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": file.name, "chunk": i} for i in range(len(chunks))]
        )
        indexed_chunks += len(chunks)
    return indexed_chunks

def retrieve(question, top_k):
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    query_embedding = get_embedder().encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )
    matches = []
    for doc, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        matches.append({
            "text": doc,
            "source": meta.get("source", "Unknown"),
            "chunk": meta.get("chunk", "?"),
            "distance": distance
        })
    return matches

def generate_answer(question, matches, api_key, model):
    client = OpenAI(api_key=api_key)
    context = "\n\n".join(
        f"[Source: {m['source']} | chunk {m['chunk']}]\n{m['text']}"
        for m in matches
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the supplied document context. "
                    "If the context does not contain the answer, say you could not find "
                    "enough information in the uploaded documents. Do not invent facts. "
                    "Cite supporting sources by filename in square brackets."
                )
            },
            {
                "role": "user",
                "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"
            }
        ]
    )
    return response.choices[0].message.content

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Chat model", value="gpt-4.1-mini")
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4)
    st.caption("Your API key is used only for the request from this app. Do not commit it to GitHub.")
    if st.button("Clear indexed documents", type="secondary"):
        client = chromadb.PersistentClient(path=str(DB_PATH))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        st.cache_resource.clear()
        st.success("Local document index cleared. Refresh the app if needed.")

st.subheader("1. Upload documents")
uploads = st.file_uploader(
    "Choose PDF, DOCX, TXT, or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)
if st.button("Process documents", type="primary", disabled=not uploads):
    with st.spinner("Extracting text, creating chunks, and building embeddings..."):
        try:
            total = index_files(uploads)
            st.success(f"Processed {len(uploads)} file(s). Indexed {total} text chunk(s).")
        except Exception as exc:
            st.error(f"Could not process the files: {exc}")

st.subheader("2. Ask a question")
question = st.text_input("Ask something that your documents can answer")
if st.button("Get answer", disabled=not question.strip()):
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            try:
                matches = retrieve(question, top_k)
                if not matches:
                    st.warning("No indexed documents found. Upload and process files first.")
                else:
                    answer = generate_answer(question, matches, api_key, model)
                    st.markdown("### Answer")
                    st.write(answer)
                    with st.expander("Retrieved context (for transparency)"):
                        for i, match in enumerate(matches, 1):
                            st.markdown(f"**{i}. {match['source']} — chunk {match['chunk']}**")
                            st.write(match["text"])
                            st.caption(f"Cosine distance: {match['distance']:.4f}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

st.divider()
st.caption("Pipeline: Document → Text extraction → Chunking → Embeddings → Chroma vector search → LLM answer.")
