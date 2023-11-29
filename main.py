import os
import streamlit as st
import time
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Berry - The Bot")
st.sidebar.title("Blog URL")

urls = []

url = st.sidebar.text_input(f"URL")
urls.append(url)

process_url_clicked = st.sidebar.button("Sumit")


main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Please Wait while we load the data :)")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    # main_placeholder.text()
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    vectorstore_openai.save_local("faiss_index")
    main_placeholder.text("Thanks for your Patience")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

query = st.chat_input("Whats up?")
if query:
    with st.chat_message('user'):
        st.markdown(query)

    st.session_state.messages.append({'role': 'user', 'content': query})
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    ans = chain({"question": query}, return_only_outputs=True)

    print(ans)
    response = ans['answer']
    # print(result)
    with st.chat_message('assistant'):
        st.markdown(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})
