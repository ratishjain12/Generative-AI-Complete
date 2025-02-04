import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

def extract_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = splitter.split_text(raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vector_store.as_retriever()
    )
    return chain


def handle_userinput(user_query):
    response = st.session_state.conversation({'question': user_query})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.title("Chat with multiple PDFs :books:")
    
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_query = st.text_input("Enter your query here:")
    if(user_query):
        handle_userinput(user_query)
    

    with st.sidebar:
        st.subheader("Documents:")
        pdf_docs = st.file_uploader("Upload PDFs here:", type="pdf",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Process the PDFs
                raw_text = extract_text(pdf_docs)

                #get text chunks
                text_chunks = get_text_chunks(raw_text)

                #store in vector db
                vector_store = get_vector_store(text_chunks)

                #get conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()