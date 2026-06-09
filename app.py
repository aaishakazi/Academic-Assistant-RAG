print("Testing import: click")
from click import prompt
print("Testing import: streamlit")
import streamlit as st
print("Testing import: os, base64, uuid, PIL, BytesIO")
import os
import base64
import uuid
import PIL.Image
from io import BytesIO
print("Testing import: dotenv, datetime")
from functions import purge_expired_guest_files, get_hybrid_retriever, get_suggested_prompts
from eval import grade_answer
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
print("Testing import: mongoclient")
from pymongo import MongoClient
print("Testing import: supabase")
from supabase import create_client
print("Testing import: streamlit_local_storage")
from streamlit_local_storage import LocalStorage
print("testing import: sentence_transformers")
# from sentence_transformers import CrossEncoder
# from flashrank import Ranker, RerankRequest

# LangChain Imports
print("Testing import: langchain modules")
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
print("Testing import: messages,langchain_huggingface")
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
print("Testing import: langchain_classic.chain module")
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
print("Testing import: langchain_classic.memory")
from langchain.memory import ConversationBufferWindowMemory
print("Testing import: langchain retrievers")
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
print("mongodb, document loaders, text splitters")
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("✅ All imports successful. Starting app...")

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Academic AI", page_icon="🎓", layout="wide")

# --- 🎨 Custom CSS ---
st.markdown("""
    <style>
            
        .fixed-header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 999;
            background-color: #0E1117;
            padding-top: 45px;
            border-bottom: 1px solid #31333F;
        }
            
            
        /* Container for the bottom profile/logout card */
        div[data-testid="stVerticalBlock"] > div:has(.sidebar-footer) {
            position: absolute;
            width: 100%;
            }

        /* Ensure the sidebar has enough height for the absolute positioning */
        [data-testid="stSidebarUserContent"] {
            height: auto;
            padding-bottom: 0 !important;
        }
        
        [data-testid="stSidebarContent"] {
            # display: flex;
            # flex-direction: column;
            height: 100vh;
            padding-bottom: 5px !important;
        }  
        
        # /* Ensure the sidebar container uses the full height and flex layout */
        # [data-testid="stSidebarUserContent"] > div:first-child {
        #     display: flex;
        #     flex-direction: column;
        #     height: auto;
        # }

        # # /* This is the magic spring that adjusts its size */
        # # .sidebar-spacer {
        # #     flex-grow: 1;
        # #     # min-height: 100px; /* Prevents it from disappearing entirely */
        # # }

        # /* Targets the footer container specifically */
        # .sidebar-footer {
        #     margin-top: auto;
        #     padding-top: 10px;
        # }
            
        .custom-caption-link {
            color: var(--secondary-text-color, #6e7787) !important; 
            font-family: var(--font) !important;
            font-size: 15px !important;
            text-decoration: none !important; 
            transition: color 0.2s ease-in-out !important;
            display: inline-block;
            cursor: pointer;
            display: inline-block !important;
            max-width: 210px; 
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            vertical-align: middle;
        }
        .custom-caption-link:hover {
            color: #FE2F4A !important;
        }

        div[data-baseweb="tab-list"] {
            height: 50px;
            gap: 40px;   
            border-bottom: 2px solid #2D313E !important;
        }
            
        button[data-baseweb="tab"] {
            font-size: 17px !important;      
            font-weight: 600 !important;      /* Makes the font bold and crisp */
            height: 100% !important; 
            padding-top: 10px !important;
            padding-left: 20px !important;    
            padding-right: 20px !important;  
            letter-spacing: 0.5px;           
        }
        
        /* Fix inner text positioning inside the button wrapper */
        button[data-baseweb="tab"] div {
            font-size: inherit !important;  /* Ensure child div elements inherit the 17px rule */
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            margin-left: auto;
            margin-right: 0; 
            max-width: 80%;  
            border-radius: 15px;
        }
    
        .main .block-container {
            padding-top: 200px !important;
            padding-bottom: 100px !important; /* Space for the fixed input */
            max-width: 900px; /* Limits width for readability */ 
            # margin: 0 auto;
        }
            
        div[data-testid="stBottomBlockContainer"] {
            background-color: #0E1117 !important; /* Matches your theme background */
            z-index: 10000;
            padding-bottom: 20px;
        }
            
        [data-testid="stChatInput"] {
            position: fixed;
            bottom: 20px;
            left: 58%; /* Adjusted to account for sidebar presence */
            transform: translateX(-50%);
            width: 70%; /* Slightly narrower for a cleaner look */
            z-index: 1000;
            background-color: #0E1117;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="fixed-header"><h1 style="margin-left: 20px;">🎓 Academic AI</h1></div>', unsafe_allow_html=True)

# --- ⚙️ Environment & Setup ---
MONGO_CONN_STR = os.getenv("MONGO_CONNECTION_STRING")
DB_DIR = "./chroma_db"
DBG_DIR = "./chroma_dbg"

@st.cache_resource(show_spinner=False)
def init_resources():
    print("Initializing resources...")
    # embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GOOGLE_API_KEY"))
    print("HF model loaded")
    client = MongoClient(MONGO_CONN_STR, serverSelectionTimeoutMS=5000) 
    print("MongoDB client initialized")
    db = client["academic_assistant"]
    vector_db = Chroma(persist_directory=DBG_DIR, embedding_function=embeddings_model)
    print("Chroma vector store initialized")
    return embeddings_model, db, vector_db

@st.cache_resource(show_spinner=False)
def cached_hybrid_retriever(user_id, _vector_db, token):
    return get_hybrid_retriever(user_id, _vector_db, token)

@st.cache_resource
def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"))
    return llm

@st.cache_resource
def get_ocr_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

with st.spinner("Initializing.."): 
    try:
        embeddings, db, vector_db = init_resources()
    except Exception as e:
        st.error(f"Failed to initialize resources: {e}")
        st.stop()
users_collection = db["users"]
history_collection = db["chat_history"]
localS = LocalStorage()

try:
    purge_expired_guest_files(supabase)  
except Exception as e:
    # Logs the issue to the console but lets the app continue loading seamlessly
    print(f"Gfiles cleanup skipped: {e}")

saved_user = localS.getItem("saved_user")
saved_pass = localS.getItem("saved_pass")

# If local storage is empty, check if we just set them in session_state
if not saved_user:
    saved_user = st.session_state.get("just_logged_in_user")
    saved_pass = st.session_state.get("just_logged_in_pass")

final_user_id = None
final_passcode = None
auth_status = "unauthorized"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


with st.sidebar:
    st.markdown('<div class="sidebar-top">', unsafe_allow_html=True)
    st.title("📚")
    st.markdown("### Your Library & Profile")
    top_container = st.container()
    st.markdown('</div>', unsafe_allow_html=True)

    sidebar_spacer_placeholder = st.empty()

    # --- 🔐 AUTHENTICATION LOGIC ---
    if "guest_id" in st.session_state:
        final_user_id = st.session_state.guest_id
        auth_status = "guest"
    
    elif saved_user and saved_pass:
        # Using session state to avoid querying Mongo every single rerun
        if st.session_state.get("is_verified") or users_collection.find_one({"username": saved_user}):
            final_user_id = saved_user
            final_passcode = saved_pass
            auth_status = "logged_in"
            st.session_state["is_verified"] = True

    # --- FOOTER CONTAINER ---
    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    bottom_container = st.container(border=True)
    button_placeholder = st.empty() 
    
    with bottom_container:
        if auth_status == "guest":
            col1, col2 = st.columns([1, 4])
            with col1: st.write("🌐")
            with col2: st.markdown("**Guest Mode**")
            if button_placeholder.button(" Sign in to Account ", type="primary", use_container_width=True):
                    st.session_state.pop("guest_id", None)
                    st.session_state.pop("msgs", None)
                    st.session_state.chat_loaded = False
                    st.session_state.chat_ui = []
                    if "processed_files" in st.session_state:
                        try:
                            file_paths_to_remove = [f"{final_user_id}/{f['name']}" for f in st.session_state.processed_files]
                            if file_paths_to_remove:
                                supabase.storage.from_("PDFs").remove(file_paths_to_remove)
                        except Exception as e:
                            print(f"Failed guest file purge: {e}")
                        st.session_state.pop("processed_files", None)
                    st.session_state.pop("is_verified", None)
                    st.session_state.pop("chat_user", None)
                    st.toast("Switching to User Account...", icon="👤")
                    st.rerun()

        elif auth_status == "logged_in":
            col1, col2 = st.columns([1, 4])
            with col1: st.write("👤")
            with col2: st.markdown(f"**{saved_user}**")
            with button_placeholder:
                with st.popover(f"Account Options", type="primary", use_container_width=True):
                    if st.button("🚪 Logout", use_container_width=True):
                        if localS:
                            try:
                                localS.deleteItem("saved_user", key="s_u_l")
                                localS.deleteItem("saved_pass", key="s_p_l")
                                localS.deleteAll(key="logout_clear")
                            except Exception:
                                pass
                        keys_to_flush = [
                            "just_logged_in_user", "just_logged_in_pass", "is_verified", 
                            "chat_user", "chat_ui", "chat_loaded", "msgs", "processed_files"
                        ]
                        for key in keys_to_flush:
                            st.session_state.pop(key, None)
                        
                        st.session_state['is_verified'] = False
                        st.session_state["uploader_key"] = 0
                        st.rerun()
                    if st.button("Continue as Guest", use_container_width=True):
                        keys_to_clear = ["just_logged_in_user", "just_logged_in_pass", "is_verified", "msgs", "chat_ui", "processed_files"]
                        for key in keys_to_clear:
                            st.session_state.pop(key, None)
                            
                        st.session_state['is_verified'] = False
                        st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:8]}"
                        st.session_state.chat_loaded = False
                        st.rerun()

        elif auth_status == "unauthorized":
            tab1, tab2 = st.tabs(["👤 Login", "🌐 Guest"])
            
            with tab1:
                st.markdown("#### User Access")
                u_input = st.text_input("Username:", key="login_u").strip()
                p_input = st.text_input("Passcode:", type="password", key="login_p")
                
                if st.button("Login", use_container_width=True):
                    if u_input and p_input:
                        existing = users_collection.find_one({"username": u_input})
                        if existing:
                            stored_p = existing.get("passcode")
                            if stored_p == p_input:
                                if existing and existing.get("passcode") == p_input:
                                    if localS:
                                        try:
                                            localS.setItem("saved_user", u_input, key="s_u_l")
                                            localS.setItem("saved_pass", p_input, key="s_p_l")
                                        except Exception:
                                            pass
                                st.session_state["just_logged_in_user"] = u_input
                                st.session_state["just_logged_in_pass"] = p_input
                                st.rerun()
                            else:
                                st.sidebar.error("🚫 Incorrect passcode for this username.")
                        else:
                            st.sidebar.warning("❓ Username not found. Please register first.")
                    else:
                        st.error("Please enter both username and passcode.")


                if st.button("Register", use_container_width=True):
                    if u_input and p_input:
                        # Checking if username is already taken
                        existing = users_collection.find_one({"username": u_input})
                        if existing:
                            st.error("⚠️ Username already taken. Please choose another or login.")
                        else:
                            # Creating the profile in MongoDB
                            users_collection.insert_one({
                                "username": u_input,
                                "passcode": p_input,
                                "created_at": datetime.now().isoformat(),
                                "files": []
                            })

                            # Log them in automatically after registration
                            localS.setItem("saved_user", u_input, key="s_u_r")
                            localS.setItem("saved_pass", p_input, key="s_p_r")
                            st.session_state["just_logged_in_user"] = u_input
                            st.session_state["just_logged_in_pass"] = p_input
                                # st.success("🎉 New Account")
                            st.spinner("Creating account...")
                            st.rerun()
                    else:
                        st.error("Please provide a username and passcode to register.")

            with tab2:
                st.markdown("#### Temporary Access")
                st.warning("History will clear on refresh.")
                if st.button("Continue as Guest", use_container_width=True):
                    st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:8]}"
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    with sidebar_spacer_placeholder:
        h = 50 if auth_status == "unauthorized" else 200
        st.container(height=h, border=False)

    if auth_status == "unauthorized":
        st.stop()

if final_user_id:
    # --- 🗄️ Database & History Setup ---
    if st.session_state.get("chat_user") != final_user_id:
        st.session_state.pop("msgs", None)
        st.session_state.chat_user = final_user_id
        st.session_state.chat_ui = []
        st.session_state.chat_loaded = False
        st.session_state['uploader_key'] += 1

    if "msgs" not in st.session_state:
        st.session_state.msgs = MongoDBChatMessageHistory(
            connection_string=MONGO_CONN_STR,
            session_id=final_user_id,
            database_name="academic_assistant",
            collection_name="chat_history",
        )
    msgs = st.session_state.msgs

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        chat_memory=msgs,
        return_messages=True,
        k=3,
        output_key="answer"
    )

    # --- 💬 Fast UI Chat Cache ---
    # Load history only once per user
    if "chat_ui" not in st.session_state:
        st.session_state.chat_ui = []
    if "chat_loaded" not in st.session_state:
        st.session_state.chat_loaded = False
    if not st.session_state.chat_loaded:
        history = msgs.messages

        st.session_state.chat_ui = [
            {
                "role": m.type,
                "content": m.content,
                "sources": m.additional_kwargs.get("metadata", {}).get("sources", [])
            }
            for m in history
        ]
        st.session_state.chat_loaded = True

    if "processed_files" not in st.session_state:
        # Guest mode
        if auth_status == "guest":
            st.session_state.processed_files = []

        # Logged-in users
        else:
            user_data = users_collection.find_one(
                {"username": final_user_id},
                {"files": 1}
            )
            if user_data and "files" in user_data:
                st.session_state.processed_files = user_data["files"]
            else:
                st.session_state.processed_files = []

    if "file_change_token" not in st.session_state:
        st.session_state.file_change_token = 0

    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"

    retriever = cached_hybrid_retriever(final_user_id, vector_db, st.session_state.file_change_token)

    # --- 🤖 LLM Setup ---
    llm = get_llm()
    ocr_llm = get_ocr_llm()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # QA prompt: answers based on retrieved documents
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant for academic notes. Answer the user's question based on the following retrieved context. If the answer is not in the context, say 'I don't have information about that in your notes.'\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # retriever = vector_db.as_retriever(
    #     search_kwargs={
    #         "k": 3,
    #         "filter": {"user_id": final_user_id}
    #     }
    # )

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

   
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=advanced_retriever, 
    #     return_source_documents=True
    # )

    @st.fragment
    def upload_controls():

        # Upload New Files
        uploaded_files = st.file_uploader("Upload Your Notes", type="pdf", accept_multiple_files=True, key=f"pdf_uploader_{st.session_state.uploader_key}")
        
        if uploaded_files:
            existing_names = [f["name"] for f in st.session_state.processed_files]
            new_files = [f for f in uploaded_files if f.name not in existing_names]
            
            if new_files:
                with st.spinner(f"Processing..."):
                    for uploaded_file in new_files:
                        file_bytes = uploaded_file.getvalue()

                        # Temporary local save for PyPDFLoader
                        tempf_path = f"./temp_{final_user_id}_{uploaded_file.name}"
                        with open(tempf_path, "wb") as f:
                            f.write(file_bytes)

                        # Uploading to supabase storage
                        supabase_path = f"{final_user_id}/{uploaded_file.name}"
                        supabase.storage.from_("PDFs").upload(
                            path=supabase_path,
                            file=file_bytes,
                            file_options={
                                "content-type": "application/pdf",
                                "x-upsert": "true"
                            }
                        )

                        file_path = supabase.storage.from_("PDFs").get_public_url(supabase_path)

                        try:
                            loader = PyPDFLoader(tempf_path)
                            raw_docs = loader.load()
                                
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                            chunks = text_splitter.split_documents(raw_docs)
                                
                                # Tag metadata for Multi-User Isolation
                            for chunk in chunks:
                                chunk.metadata["user_id"] = final_user_id
                                chunk.metadata["source"] = uploaded_file.name 
                                
                            vector_db.add_documents(chunks)

                        finally:
                            if os.path.exists(tempf_path):
                                os.remove(tempf_path) 
                            
                        new_file_entry = {"name": uploaded_file.name, "url": file_path}
                        if auth_status != "guest":
                            users_collection.update_one(
                                {"username": final_user_id},
                                {"$addToSet": {"files": new_file_entry}}
                            )
                        st.session_state.processed_files.append(new_file_entry)

                    # After successful upload/delete:
                    st.session_state.file_change_token += 1
                    st.toast(f"✅ {len(new_files)} files added!", icon='📚')
                    with sidebar_spacer_placeholder:
                        st.container(height=50, border=False)
                    st.rerun()

        if not st.session_state.processed_files and not uploaded_files:
            # st.stop()
            st.info("Upload PDFs to begin.")
            return

        if st.button("🗑️ Clear Chat History"):
            msgs.clear()
            st.session_state.chat_ui = []
            st.rerun()

    def get_image_text_efficiently(uploaded_file):
        img = PIL.Image.open(uploaded_file)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{img_str}" # Converted to data url so it can be read by the LLM directly without needing external hosting
        
        response = ocr_llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Look at this image. Extract ONLY what seems like a question. And If there is a table that might seem like a part of the question, represent it strictly in Markdown format. If there is a diagram, describe its structure briefly within the question text. Ignore everything else."},
                {"type": "image_url", "image_url": image_data_url}
            ])
        ])
        return response.content

    # --- Sidebar ---
    with top_container:
        upload_controls()
        st.subheader("📚 Documents")
        # Displaying Current Files
        # if st.session_state.processed_files:
        #     for f_name in st.session_state.processed_files:
        #         st.caption(f"📄 {f_name}")
        # else:
        #     st.caption("No documents uploaded yet.")

        if st.session_state.processed_files:
            for file_data in st.session_state.processed_files:
                name = file_data["name"]
                url = file_data["url"]

                col1, col2 = st.columns([4, 1], vertical_alignment="center")
                with col1:
                    custom_link =f"""<a href="{url}" target="_blank" class="custom-caption-link" title="{name}">📄{name}</a>"""
                    st.markdown(custom_link, unsafe_allow_html=True)

                with col2:
                    if st.button("🗑️", key=f"del_{name}"):

                        # delete from supabase
                        supabase.storage.from_("PDFs").remove(
                            [f"{final_user_id}/{name}"]
                        )

                        # delete from chroma
                        results = vector_db.get(
                            where={
                                "$and": [
                                    {"user_id": final_user_id},
                                    {"source": name}
                                ]})

                        ids_to_delete = results["ids"]

                        if ids_to_delete:
                            vector_db.delete(ids=ids_to_delete)

                        # remove from session
                        st.session_state.processed_files = [
                            f for f in st.session_state.processed_files
                            if f["name"] != name
                        ]

                        # remove from mongo
                        if auth_status != "guest":
                            users_collection.update_one(
                                {"username": final_user_id},
                                {"$pull": {"files": {"name": name}}}
                            )
                        st.session_state.uploader_key += 1
                        st.session_state.file_change_token += 1 # Triggering retriever cache refresh
                        st.toast(f"Deleted {name}")
                        st.rerun()

        else:
            st.caption("No documents uploaded yet.")

        st.markdown("---")
        uploaded_image = st.file_uploader("📸 Scan a Question", type=["jpg", "jpeg", "png"], key=f"img_uploader_{st.session_state.uploader_key}")
        if uploaded_image:
            st.image(uploaded_image, use_container_width=True)
            if st.button("🔍 Analyze Image"):
                with st.spinner("Detecting text..."):
                    question = get_image_text_efficiently(uploaded_image)
                    st.success("✅ Question Detected!")
                    st.session_state.chat_ui.append({
                        "role": "human",
                        "content": question,
                        "sources": []
                    })


                with st.spinner("Searching your notes..."):
                    # hist = memory.load_memory_variables({})["chat_history"]
                    # response = qa_chain.invoke({"question": question,
                    #     "chat_history": hist})
                    chat_history = msgs.messages 
                    response = rag_chain.invoke({
                        "input": question,
                        "chat_history": chat_history
                    })

                    # Saving sources in metadata 
                    source_meta = []
                    no_info_phrases = ["i don't know", "not mentioned in the document", "not found in the notes", "does not contain"]
                    answer_lower = response["answer"].lower()

                    # Check if the answer seems to be based on the docs
                    if not any(phrase in answer_lower for phrase in no_info_phrases):
                        for d in response.get("context", []):
                            file_name = d.metadata.get('source', 'Unknown')
                            file_data = next(
                                (f for f in st.session_state.processed_files if f["name"] == file_name
                                ), None)

                            file_url = file_data["url"] if file_data else ""

                            source_meta.append({
                                "file": file_name,
                                "url": file_url,
                                "page": d.metadata.get('page', 0) + 1,
                                "text": d.page_content[:200],
                                "chunk": d.page_content
                            })
                    #---------- SAVE TO UI CACHE ----------
                    st.session_state.chat_ui.append({
                            "role": "ai",
                            "content": response["answer"],
                            "sources": source_meta
                        })

                    # ---------- SAVE TO MONGO ----------
                    meta = {"sources": source_meta}
                    msgs.add_user_message(question)
                    msgs.add_message(
                        AIMessage(
                            content=response["answer"],
                            additional_kwargs={"metadata": meta}
                        )
                    )
                    st.rerun()

    # --- 💬 CHAT FRAGMENT ---
    @st.fragment
    def render_chat():

        master_container = st.container()
        
        with master_container:
            #Scrollable container
            chat_container = st.container(height=505, border=False)

            # ---------- RENDER CHAT HISTORY ----------
            with chat_container:
                # SHOW STARTER CARDS IF CHAT LOG IS EMPTY
                if not st.session_state.chat_ui:
                    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
                    st.markdown("<h4 style='text-align: center; color: #8E93A6;'>💡 Tap a starter card to begin studying:</h4>", unsafe_allow_html=True)
                    
                    # Fetch prompts dynamically
                    prompts = get_suggested_prompts(llm, vector_db, final_user_id)
                    
                    # Render cards using columns side by side
                    card_cols = st.columns(3)
                    selected_prompt = None
                    
                    for idx, prompt_text in enumerate(prompts):
                        with card_cols[idx]:
                            # Style each card option beautifully inside a bordered area
                            with st.container(border=True):
                                st.markdown(f"<p style='font-size:14px; min-height:55px; margin-bottom:10px;'><b>{prompt_text}</b></p>", unsafe_allow_html=True)
                                if st.button("🚀 Ask AI", key=f"starter_card_{idx}", use_container_width=True, type="secondary"):
                                    selected_prompt = prompt_text
                    
                    # If a card was triggered, programmatically set it as a query submit execution
                    if selected_prompt:
                        st.session_state["card_submit_query"] = selected_prompt
                        st.rerun()

                # Otherwise, render standard chat text histories
                else:
                    for msg in st.session_state.chat_ui:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])
                            if msg["role"] == "ai" and msg.get("sources"):
                                with st.expander("📚 View Sources"):
                                    for s in msg["sources"]:
                                        file_path = f"{s['url']}#page={s['page']}"
                                        st.markdown(f'📄 <a href="{file_path}" target="_blank">**{s["file"]}** (Page {s["page"]})</a>', unsafe_allow_html=True)
                                        st.caption(f"{s['text']}...")

                if st.session_state.chat_ui:
                    st.markdown(
                        '<div id="end-of-chat" style="height:0px; margin:0; padding:0;"></div>', 
                        unsafe_allow_html=True
                    )
                    # Simple text input hack that auto-focuses an invisible target at the bottom
                    st.components.v1.html(
                        """
                        <script>
                            var chatWindow = window.parent.document.querySelector('.stDocstring').closest('[data-testid="stVComponentBlock"]');
                            if (chatWindow) {
                                chatWindow.scrollTop = chatWindow.scrollHeight;
                            }
                        </script>
                        """,
                        height=0
                    )

            # ---------- CHAT INPUT ----------
            prompt = st.chat_input("Ask a question from your notes...")
            
            # Check if a starter card clicked triggered a queued submit execution
            if "card_submit_query" in st.session_state:
                prompt = st.session_state.pop("card_submit_query")

            if prompt:
                with chat_container:
                    with st.chat_message("human"):
                        st.write(prompt)

                st.session_state.chat_ui.append({"role": "human", "content": prompt, "sources": []})

                # Force assistant message to render inside the scrollable view
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Searching the docs..."):
                            # hist = memory.load_memory_variables({})["chat_history"]
                            # response = qa_chain.invoke({
                            #     "question": prompt,
                            #     "chat_history": hist
                            # })

                            chat_history = msgs.messages  
                            response = rag_chain.invoke({
                                "input": prompt,
                                "chat_history": chat_history
                            })
                            answer = response["answer"]
                            st.write(answer)

                            # ---------- PROCESS SOURCES ----------
                            source_meta = []
                            no_info_phrases = ["i don't know", "not mentioned in the document", "not found in the notes", "does not contain"]
                            answer_lower = response["answer"].lower()

                            if not any(phrase in answer_lower for phrase in no_info_phrases):
                                for d in response.get("context", []):
                                    file_name = d.metadata.get('source', 'Unknown')
                                    file_data = next((f for f in st.session_state.processed_files if f["name"] == file_name), None)
                                    file_url = file_data["url"] if file_data else ""

                                    source_meta.append({
                                        "file": file_name,
                                        "url": file_url,
                                        "page": d.metadata.get('page', 0) + 1,
                                        "text": d.page_content[:200],
                                        "chunk": d.page_content
                                    })
                            if source_meta:
                                with st.expander("📚 View Sources"):
                                    for s in source_meta:
                                        file_path = f"{s['url']}#page={s['page']}"
                                        st.markdown(f'📄 <a href="{file_path}" target="_blank">**{s["file"]}** (Page {s["page"]})</a>', unsafe_allow_html=True)
                                        st.caption(f"{s['text']}...")

                            # ---------- SAVE TO UI CACHE ----------
                            st.session_state.chat_ui.append({
                                "role": "ai",
                                "content": answer,
                                "sources": source_meta
                            })

                            # ---------- SAVE TO MONGO ----------
                            meta = {"sources": source_meta}
                            msgs.add_user_message(prompt)
                            msgs.add_message(AIMessage(content=answer, additional_kwargs={"metadata": meta}))
    # Run fragment
    render_chat()

    # --- 📊 EVALUATION DASHBOARD FRAGMENT ---
    # @st.fragment
    # def render_evaluation_dashboard():
    #     st.subheader("📊 RAG System Accuracy Audit")
    #     st.write("This diagnostic test runs your RAG pipeline through benchmark exam questions to evaluate generation accuracy.")
        
    #     # Define evaluation data matching your uploaded documents
    #     EVAL_DATASET = [
    #         {
    #             "query": "Explain how an organization can ensure its systems are up and running even during peak traffic hours.",
    #             "ground_truth": "Organizations utilize backup systems, firewalls, and anti-DDoS protections to maintain availability, ensuring websites like online shopping portals remain online to serve customers during peak sales periods."
    #         },
    #         {
    #             "query": "What security principle ensures that user permissions are validated at every single resource request?",
    #             "ground_truth": "Complete mediation is the security principle stating that every time a user requests a resource, the system must verify authorization and not trust previous permissions by default."
    #         },
    #         {
    #             "query": "What are stack canaries and ASLR used for?",
    #             "ground_truth": "Stack canaries and address space layout randomization (ASLR) are memory protection mechanisms used by developers to protect software applications against buffer overflow attacks."
    #         },
    #         {
    #             "query": "How do IPSec and SSL/TLS differ in terms of the network layer they operate on and their primary use cases?",
    #             "ground_truth": "IPSec operates at the network layer to provide secure communication across IP networks (commonly used in VPNs), whereas SSL/TLS operates at the transport layer to encrypt communication between a web browser and a web server via HTTPS."
    #         }
    #     ]

    #     if st.button("🚀 Start Diagnostic Suite", type="primary"):
    #         progress_bar = st.progress(0.0)
    #         status_text = st.empty()
            
    #         scores = []
    #         detailed_results = []
    #         total_tests = len(EVAL_DATASET)
            
    #         for index, test in enumerate(EVAL_DATASET):
    #             query = test["query"]
    #             ground_truth = test["ground_truth"]
                
    #             status_text.markdown(f"⏳ **Processing ({index+1}/{total_tests}):** *\"{query[:40]}...\"*")
                
    #             # Run through your live RAG chain
    #             response = rag_chain.invoke({
    #                 "input": query,
    #                 "chat_history": [] # Blank context for evaluation purity
    #             })
    #             generated_answer = response["answer"]
                
    #             # Grade the response via LLM-as-a-judge
    #             score, reasoning = grade_answer(llm, query, ground_truth, generated_answer)
    #             scores.append(score)
                
    #             detailed_results.append({
    #                 "query": query,
    #                 "generated": generated_answer,
    #                 "ground_truth": ground_truth,
    #                 "score": score,
    #                 "reasoning": reasoning
    #             })
                
    #             progress_bar.progress((index + 1) / total_tests)
                
    #         status_text.success("✅ Evaluation Diagnostics Complete!")
            
    #         # Compute Metrics
    #         avg_score = sum(scores) / len(scores)
    #         accuracy_percentage = (avg_score / 5.0) * 100
            
    #         st.markdown("---")
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.metric(
    #                 label="Overall System Accuracy", 
    #                 value=f"{accuracy_percentage:.1f}%", 
    #                 delta="Optimal" if accuracy_percentage >= 80 else "Needs Tuning"
    #             )
    #         with col2:
    #             st.metric(
    #                 label="Average GPA Rating", 
    #                 value=f"{avg_score:.2f} / 5.0"
    #             )
                
    #         # Question breakdown accordion layout
    #         st.subheader("📝 Granular Question Breakdown")
    #         for res in detailed_results:
    #             emoji = "🟢" if res["score"] >= 4 else "🟡" if res["score"] == 3 else "🔴"
    #             with st.expander(f"{emoji} Q: {res['query']}"):
    #                 st.markdown(f"**🤖 System Output:**\n*{res['generated']}*")
    #                 st.markdown(f"**🎯 Expected Ground Truth:**\n*{res['ground_truth']}*")
    #                 st.markdown(f"**🏅 Score:** `{res['score']}/5` — *{res['reasoning']}*")
    

    # tab_chat, tab_metrics = st.tabs(["💬 Virtual Assistant Chat", "📊 RAG Performance Audit"])
    
    # with tab_chat:
    #     render_chat()
        
    # with tab_metrics:
    #     render_evaluation_dashboard()
