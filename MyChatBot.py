"""
NoteBot — Streamlit + LangChain (single file)
-------------------------------------------------
Quick start:
1) Install deps (one-time):
   pip install -U streamlit PyPDF2 langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu

2) Set your OpenAI API key (any one of these):
   - In shell:  export OPENAI_API_KEY="sk-..."   (Windows PowerShell:  $env:OPENAI_API_KEY="sk-...")
   - In .streamlit/secrets.toml:  OPENAI_API_KEY = "sk-..."
   - Or type it into the sidebar field when the app runs.

3) Run the app:
   streamlit run app.py
"""

import io
import os
import hashlib
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="NoteBot", page_icon="📓", layout="wide")
st.title("📓 NoteBot")

# -------------------------------
# Helpers
# -------------------------------
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if needed

def get_api_key() -> str | None:
    # Priority: Streamlit secrets -> ENV -> user input
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        with st.sidebar:
            st.info("Add your OpenAI API key to continue.")
            key = st.text_input("OpenAI API key", type="password")
    return key

@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_bytes: bytes, api_key: str) -> FAISS:
    """Reads PDF bytes, extracts text by page, splits into chunks, and returns a FAISS store."""
    reader = PdfReader(io.BytesIO(pdf_bytes))

    # Extract per-page text and preserve page numbers in metadata
    page_docs: list[Document] = []
    for idx, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        if raw.strip():
            page_docs.append(Document(page_content=raw, metadata={"page": idx + 1}))

    if not page_docs:
        raise ValueError("No extractable text found in the PDF. Try another file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    chunked_docs = splitter.split_documents(page_docs)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    store = FAISS.from_documents(chunked_docs, embeddings)
    return store

# -------------------------------
# Sidebar: file upload
# -------------------------------
with st.sidebar:
    st.header("My Notes")
    uploaded = st.file_uploader("Upload a PDF and start asking questions", type=["pdf"]) 

api_key = get_api_key()
if not api_key:
    st.stop()

# -------------------------------
# Load / cache vector store for the uploaded file
# -------------------------------
if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    # Build a stable cache key for this exact file content
    cache_key = hashlib.md5(pdf_bytes).hexdigest()

    try:
        with st.spinner("Indexing your notes (embeddings + FAISS)..."):
            vector_store = build_vector_store(pdf_bytes, api_key)
        st.success("Notes are ready. Ask a question below! ✅")
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        st.stop()

    # -------------------------------
    # Query UI
    # -------------------------------
    user_query = st.text_input("Type your question about the PDF", placeholder="e.g., Summarize section 2, or What does theorem 3 state?")

    if user_query:
        # Retrieve similar chunks
        with st.spinner("Searching relevant chunks..."):
            matching_chunks = vector_store.similarity_search(user_query, k=4)

        # Define LLM and prompt
        llm = ChatOpenAI(
            api_key=api_key,
            model=MODEL_NAME,
            temperature=0,
            max_tokens=300,
        )
        prompt = ChatPromptTemplate.from_template(
            (
                "You are a helpful tutor. Answer the user's question using ONLY the provided context.\n"
                "If the answer cannot be found in the context, reply exactly with: I don't know Jenny\n\n"
                "Context:\n{context}\n\n"
                "Question: {input}"
            )
        )
        chain = create_stuff_documents_chain(llm, prompt)

        with st.spinner("Thinking..."):
            result = chain.invoke({"input": user_query, "input_documents": matching_chunks})

        # The chain commonly returns a string, but handle dict just in case
        if isinstance(result, dict):
            answer_text = result.get("output", result.get("answer", str(result)))
        else:
            answer_text = str(result)

        st.subheader("Answer")
        st.markdown(answer_text)

        # Show sources (pages) used
        with st.expander("Show context excerpts (sources)"):
            for i, d in enumerate(matching_chunks, start=1):
                p = d.metadata.get("page", "?")
                snippet = d.page_content.strip().replace("\n", " ")
                st.markdown(f"**Chunk {i} — Page {p}:**\n\n{snippet[:600]}{'…' if len(snippet) > 600 else ''}")
else:
    st.info("Upload a PDF from the sidebar to begin.")
