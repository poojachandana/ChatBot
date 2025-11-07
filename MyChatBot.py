import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load your OpenAI API key from Streamlit secrets
OpenAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

# extracting the text from pdf file
if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()

    # break it into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(text)

    # creating Object of OpenAIEmbeddings class that let us connect with OpenAI's Embedding Models
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    # Creating VectorDB & Storing embeddings into it
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user query
    user_query = st.text_input("Type your query here")

    # semantic search from vector store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # define our LLM
        llm = ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo",
        )

        # Approach: Generate Response using a customized prompt (no chains import needed)
        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context and
if you did not get the context simply say "I don't know Jenny":
{context}
Question: {input}
"""
        )

        context_text = "\n\n".join(doc.page_content for doc in matching_chunks)
        messages = customized_prompt.format_messages(context=context_text, input=user_query)
        output = llm.invoke(messages)

        st.write(output.content)
