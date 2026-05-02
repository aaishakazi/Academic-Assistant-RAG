import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain # New Chain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.set_page_config(page_title="Academic Assistant", page_icon="📚", layout="wide")

# --- 🎨 Custom CSS for Chat Alignment ---
st.markdown("""
    <style>
        /* Identify the message container and check if it's from the user */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            text-align: right;
            margin-left: auto;
            width: 80%;

        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Academic Assistant RAG")
st.markdown("---")

# --- 1. Memory & History Setup ---
msgs = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    chat_memory=msgs,
    return_messages=True,
    k=5,
    output_key="answer"
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- 2. Sidebar Setup ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload your Notes (PDF)", type="pdf")
    if st.button("Clear Chat History"):
        msgs.clear()

# --- 3. Main Section: Analyzing Logic ---
if uploaded_file and st.session_state.vector_db is None:
    # This spinner now appears in the main window
    with st.spinner("Analyzing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_db = Chroma.from_documents(chunks, embeddings)
        st.success("Analysis Complete!")

# --- 4. Display Chat Messages ---
for msg in msgs.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# --- 5. Chat Input ---
if prompt := st.chat_input("Ask a follow-up question..."):
    
    st.chat_message("human").write(prompt)

    if st.session_state.vector_db is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vector_db.as_retriever(),
                    memory=memory,
                    return_source_documents=True
                )
                
                response = qa_chain.invoke({"question": prompt})
                st.markdown(response["answer"])
                
                with st.expander("📚 View Sources"):
                    for doc in response["source_documents"]:
                        st.write(f"**Page {doc.metadata.get('page', 0) + 1}:** {doc.page_content[:300]}...")
    else:
        st.info("Please upload a PDF to begin.")