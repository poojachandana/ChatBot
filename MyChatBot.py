import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

OpenAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        llm = ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        customized_prompt = ChatPromptTemplate.from_template(
            """ You are my assistant tutor. Answer the question based on the following context and
            if you did not get the answer from context simply say "I don't know Jenny":
            {context}
            Question: {input}
            """
        )

        chain = create_stuff_documents_chain(llm, customized_prompt)
        output = chain.invoke({"input": user_query, "input_documents": matching_chunks})
        st.write(output)
