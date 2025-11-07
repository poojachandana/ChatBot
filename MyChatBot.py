import io
import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="NoteBot", page_icon="📝")
st.header("📝 NoteBot")

# -----------------------------
# Secrets / API key
# -----------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.stop()  # Fail early so we don't proceed without a key

# -----------------------------
# Sidebar: Upload & Build Index
# -----------------------------
with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")
    build_btn = st.button("Build / Rebuild Index", disabled=(file is None))
    st.caption("Tip: Rebuild the index if you upload a new file.")

# -----------------------------
# Helpers
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a Streamlit uploaded PDF file."""
    # Streamlit gives a BytesIO-like object; ensure PyPDF2 gets bytes
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_chunks = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_chunks.append(page_text)
    return "\n".join(text_chunks)

def build_vectorstore_from_text(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks were created from the PDF. Check the PDF content.")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.from_texts(chunks, embeddings)

def get_llm():
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=300,
    )

# -----------------------------
# Build vector store when asked
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

if build_btn and file is not None:
    with st.spinner("Reading PDF and building vector index..."):
        try:
            text = extract_pdf_text(file)
            vector_store = build_vectorstore_from_text(text)
            st.session_state.vector_store = vector_store
            st.session_state.file_name = file.name
            st.success(f"Index built from: {file.name}")
        except Exception as e:
            st.error(f"Failed to build index: {e}")

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Ask a question about your notes...")
if user_query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            st.warning("Please upload a PDF and click **Build / Rebuild Index** first.")
    else:
        # Retrieve relevant chunks
        try:
            matching_chunks = st.session_state.vector_store.similarity_search(user_query, k=4)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Similarity search failed: {e}")
            st.stop()

        # Define LLM and prompt
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context.
If you do not find the answer in the context, reply exactly with: "I don't know Jenny".

<context>
{context}
</context>

Question: {input}"""
        )

        # Create the chain that stuffs the retrieved docs into {context}
        chain = create_stuff_documents_chain(llm, prompt=prompt)

        # Run the chain
        try:
            output = chain.invoke({
                "context": matching_chunks,  # list[Document]
                "input": user_query          # user question
            })
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"LLM chain failed: {e}")
            st.stop()

        # Display assistant answer
        with st.chat_message("assistant"):
            # chain.invoke can return a string or an AIMessage depending on LC version
            response_text = getattr(output, "content", output)
            st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

# -----------------------------
# Footer info
# -----------------------------
st.caption(
    "Notes are embedded locally in-memory using FAISS. "
    "Re-run the index if you upload a new PDF."
)
