# app.py
import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- API key (Streamlit Cloud) ---
# Add to .streamlit/secrets.toml: OPENAI_API_KEY = "sk-..."
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()  # fail fast with a clear message on UI
# ----------------------------------

st.set_page_config(page_title="NoteBot", page_icon="📝")
st.header("📝 NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is None:
    st.info("Upload a PDF on the left to begin.")
    st.stop()

# --- Extract text from PDF ---
try:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        # PyPDF2 can return None; guard it
        page_text = page.extract_text() or ""
        text += page_text
except Exception as e:
    st.error(f"Failed to read PDF: {e}")
    st.stop()

if not text.strip():
    st.warning("I couldn't extract text from this PDF. Try another file or a text-based PDF.")
    st.stop()

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)
chunks = splitter.split_text(text)

# --- Embeddings + Vector store ---
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # uses text-embedding-3-large by default
vector_store = FAISS.from_texts(chunks, embeddings)

# --- UI for query ---
user_query = st.text_input("Type your query here")

if not user_query:
    st.stop()

# --- Retrieve top chunks ---
matching_chunks = vector_store.similarity_search(user_query, k=4)

# --- LLM (use a current model name) ---
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",   # modern, inexpensive, good for RAG
    temperature=0,
    max_tokens=300,
)

# --- Prompt & chain ---
prompt = ChatPromptTemplate.from_template(
    """You are my assistant tutor. Answer the question based on the following context.
If the answer is not in the context, reply exactly: "I don't know Jenny".

<context>
{context}
</context>

Question: {input}
"""
)

chain = create_stuff_documents_chain(llm, prompt=prompt)

# The similarity search returns a List[Document]; pass directly as {context}
result = chain.invoke({
    "context": matching_chunks,
    "input": user_query
})

# `chain.invoke` often returns a dict or string depending on LC version.
# Handle both cleanly:
if isinstance(result, dict) and "answer" in result:
    st.write(result["answer"])
else:
    st.write(result)
