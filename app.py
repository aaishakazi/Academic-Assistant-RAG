from click import prompt
import streamlit as st
import os
import base64
import uuid
import PIL.Image
from io import BytesIO
from functions import purge_expired_guest_files
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from supabase import create_client
from streamlit_local_storage import LocalStorage
from flashrank import Ranker, RerankRequest

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_mongodb import MongoDBChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank import FlashRankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

@st.cache_resource(show_spinner=False)
def init_resources():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = MongoClient(MONGO_CONN_STR)
    db = client["academic_assistant"]
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)
    return embeddings_model, db, vector_db

@st.cache_resource
def load_reranker():
    return Ranker()

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
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

with st.spinner("Initializing.."): 
    embeddings, db, vector_db = init_resources()
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
                        localS.deleteItem("saved_user", key="s_u_l")
                        localS.deleteItem("saved_pass", key="s_p_l")
                        localS.deleteAll(key="logout_clear")
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
                                localS.setItem("saved_user", u_input, key="s_u_l")
                                localS.setItem("saved_pass", p_input, key="s_p_l")
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
        h = 50 if auth_status == "unauthorized" else 300
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


    # --- 🤖 LLM Setup ---
    llm = get_llm()
    ocr_llm = get_ocr_llm()

    # retriever = vector_db.as_retriever(
    #     search_kwargs={
    #         "k": 3,
    #         "filter": {"user_id": final_user_id}
    #     }
    # )

    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     return_source_documents=True
    # )
    base_retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 10,  
            "filter": {"user_id": final_user_id} })

    compressor = FlashRankRerank(top_n=3)

    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=advanced_retriever, 
        return_source_documents=True
    )

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
                    # st.info(f"**Extracted:** {question}")
                    st.session_state.chat_ui.append({
                        "role": "human",
                        "content": question,
                        "sources": []
                    })


                with st.spinner("Searching your notes..."):
                    hist = memory.load_memory_variables({})["chat_history"]
                    response = qa_chain.invoke({"question": question,
                        "chat_history": hist})

                    # Saving sources in metadata 
                    source_meta = []
                    no_info_phrases = ["i don't know", "not mentioned in the document", "not found in the notes", "does not contain"]
                    answer_lower = response["answer"].lower()

                    # Check if the answer seems to be based on the docs
                    if not any(phrase in answer_lower for phrase in no_info_phrases):
                        for d in response.get("source_documents", []):
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
            if prompt := st.chat_input("Ask a question from your notes..."):
                
                with chat_container:
                    with st.chat_message("human"):
                        st.write(prompt)

                # Append to UI cache right away
                st.session_state.chat_ui.append({
                    "role": "human",
                    "content": prompt,
                    "sources": []
                })

                # Force assistant message to render inside the scrollable view
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Searching the docs..."):
                            hist = memory.load_memory_variables({})["chat_history"]
                            response = qa_chain.invoke({
                                "question": prompt,
                                "chat_history": hist
                            })
                            answer = response["answer"]
                            st.write(answer)

                            # ---------- PROCESS SOURCES ----------
                            source_meta = []
                            no_info_phrases = ["i don't know", "not mentioned in the document", "not found in the notes", "does not contain"]
                            answer_lower = response["answer"].lower()

                            if not any(phrase in answer_lower for phrase in no_info_phrases):
                                for d in response.get("source_documents", []):
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
