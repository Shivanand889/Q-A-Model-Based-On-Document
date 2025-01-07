import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="Chatbot",
    initial_sidebar_state="auto"
)

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

groq_model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='Llama3-8b-8192')

chat_prompt_template = ChatPromptTemplate.from_template("""
Please provide answer to the questions based on given information. 
If you are not able to find the answer, reply: 
"Sorry, I didn’t understand your question. Do you want to connect with a live agent?".

<context>
{context}
<context>
Questions:{input}
""")

if 'document_vectors' not in st.session_state:
    st.session_state.update({
        'embeddings_model': HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        ),
        'text_splitter': RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200),
        'documents': [],
        'split_documents': [],
        'chat_history': []
    })


def reset_session_state():
    st.session_state.clear()


def generate_vector_embeddings(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    document_list = loader.load()
    split_docs = st.session_state.text_splitter.split_documents(document_list)

    st.session_state.documents.extend(document_list)
    st.session_state.split_documents.extend(split_docs)
    st.session_state.document_vectors = FAISS.from_documents(st.session_state.split_documents, st.session_state.embeddings_model)



st.title("Chatbot")
st.sidebar.title("Upload PDF")
st.sidebar.write("Please upload a document to ask questions.")

uploaded_file = st.sidebar.file_uploader("", accept_multiple_files=False, type=["pdf"])
if uploaded_file:
    generate_vector_embeddings(uploaded_file)


user_input = st.chat_input("Write your query")
if user_input:
    st.session_state.chat_history.append({"user": user_input})
    if 'document_vectors' in st.session_state:
        try:
            document_chain = create_stuff_documents_chain(groq_model, chat_prompt_template)
            retriever = st.session_state.document_vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': user_input})
            st.session_state.chat_history.append({"bot": response["answer"]})
        except Exception as error:
            st.session_state.chat_history.append({"bot": f"An error occurred: {error}"})
    else:
        st.session_state.chat_history.append({"bot": "Please upload a document first."})

for message in st.session_state.chat_history:
    if "user" in message:
        st.chat_message("User").write(message["user"])
    if "bot" in message:
        st.chat_message("Assistant").write(message["bot"])
