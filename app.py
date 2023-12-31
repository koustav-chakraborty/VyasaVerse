import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,  HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size = 100000000000,
        chunk_overlap = 200000,
        # length_func = len

    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding= embeddings)
    return vectorstore

def get_conversation_chain(vectorestore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorestore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question' : user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']


    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:    
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title='VYASA VERSE', page_icon="https://e7.pngegg.com/pngimages/486/216/png-clipart-telegram-bot-api-chatbot-internet-bot-application-programming-interface-others-miscellaneous-blue-thumbnail.png")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("GET HELP WITH QUERIES RELATED TO YOUR DOCUMENT")
    user_question = st.text_input("Ask Your Query : ")

    if user_question:
        # handle_user_input(user_question)
        try:
            handle_user_input(user_question)
        except:
            st.error("PLEASE UPLOAD YOUR PDF")
            

    # st.write(user_template.replace("{{MSG}}", "HELLO ROBOT"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "HELLO HUMAN"), unsafe_allow_html=True)

    with st.sidebar:
        
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload Your PDF", accept_multiple_files=True)
        if st.button("PROCESS"):
            if pdf_docs:

                    with st.progress(0):
                        progress_bar = st.progress(0, "PROCESSING ...")
                        for percent_complete in range(100):
                            time.sleep(0.03)
                            progress_bar.progress(percent_complete + 1, text="PROCESSING ...")
                            
                        raw_text = get_pdf_text(pdf_docs)
                        # get pdf text
                        # st.write(raw_text)

                        # get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        # st.write(text_chunks)
                        st.success("PROCESSED SUCCESFULLY")

                        # vectorestore
                        vectorestore = get_vectorestore(text_chunks)

                        # conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorestore)
            else:
                st.error("PLEASE UPLOAD YOUR PDF")
            
if __name__ == '__main__':
    
    main()
    