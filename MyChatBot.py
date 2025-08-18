import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load API key from .env file
OpenAI_API_KEY = st.secrets["OPENAI_API_KEY"]


st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is not None:
    # Extract text from PDF
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user query
    user_query = st.text_input("Type your query here")

    if user_query:
        # Semantic search
        matching_chunks = vector_store.similarity_search(user_query)

        # Define LLM
        llm = ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        # Define customized prompt with {context} variable
        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context,
            and if you do not get the answer, simply say "I don't know Jenny":

            {context}

            Question: {input}
            """
        )

        # Create chain
        chain = create_stuff_documents_chain(llm, prompt=customized_prompt)

        # Invoke chain with correct keys
        output = chain.invoke({
            "context": matching_chunks,  # matches {context} in prompt
            "input": user_query          # matches {input} in prompt
        })

        st.write(output)
