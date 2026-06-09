import streamlit as st
from datetime import datetime, timedelta, timezone

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

import json

@st.cache_resource
def load_reranker_model():
    return HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

def purge_expired_guest_files(supabase_client):
    """Finds and deletes any file in guest folders that is older than 24 hours."""
    try:
        # 1. List the root folders/items inside your bucket
        bucket_items = supabase_client.storage.from_("PDFs").list()
        file_deleted = 0
        
        for item in bucket_items:
            folder_name = item.get("name", "")
            
            # Target only folders prefixed with 'guest_'
            if folder_name.startswith("guest_"):
                
                # 2. Inspect the files inside this specific guest folder
                guest_files = supabase_client.storage.from_("PDFs").list(path=folder_name)
                
                for file in guest_files:
                    created_at_str = file.get("created_at")  # ISO timestamp from Supabase
                    
                    if created_at_str:
                        # Convert Supabase ISO timestamp string to a timezone-aware Python datetime object
                        # (Replacing 'Z' with UTC offset syntax for seamless parsing)
                        file_time = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        current_time = datetime.now(timezone.utc)
                        
                        if current_time - file_time > timedelta(hours=48):
                            full_storage_path = f"{folder_name}/{file['name']}"
                            
                            # Vaporize it from Supabase
                            supabase_client.storage.from_("PDFs").remove([full_storage_path])
                            print(f"Deleted expired file: {full_storage_path}")
                            file_deleted +=1
        if file_deleted == 0:
            print("No files to clean up.")
                            
    except Exception as e:
        # Fails silently in production so it never crashes your user interface
        print(f"Background cleanup exception: {e}")

# ________

def get_hybrid_retriever(user_id, vector_store, token=0):
    """
    Builds an ensemble retriever (BM25 + dense) with cross-encoder reranking.
    token is used to invalidate cache when files change.
    """
    try:
        # Get all user documents from Chroma
        results = vector_store.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        if not results or not results["documents"]:
            # Fallback to dense only
            return vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"user_id": user_id}})
        
        # Reconstruct LangChain Documents
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(results["documents"], results["metadatas"])
        ]
        
        # BM25 retriever
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = 10
        
        # Dense vector retriever
        dense = vector_store.as_retriever(
            search_kwargs={"k": 10, "filter": {"user_id": user_id}}
        )
        
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[0.5, 0.5]
        )
        
        # Reranker
        model = load_reranker_model()
        reranker = CrossEncoderReranker(model=model, top_n=3)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble
        )
        print("Hybrid retriever created.")
        return final_retriever
        
    except Exception as e:
        print(f"⚠️ Hybrid retriever creation failed: {e}")
        # Fallback
        return vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"user_id": user_id}})
    
#________

def get_suggested_prompts(llm, vector_db, user_id):
    """
    Looks into the user's vector store records, pulls sample textual data, 
    and lets the LLM formulate 3 specific contextual study questions.
    """
    fallback_prompts = [
        "📚 What are the core themes covered inside my notes?",
        "🔑 Extract the most critical key concepts and definitions.",
        "📝 Generate a short mock quiz based on my documents."
    ]
    
    try:
        # Pull a few document segments uploaded by this unique user id
        db_content = vector_db.get(where={"user_id": user_id}, limit=3)
        if not db_content or not db_content.get("documents"):
            return fallback_prompts
            
        sample_text = "\n".join(db_content["documents"][:2])
        
        prompt = f"""You are an academic study planner. Based on this small snippet of a student's study guide material, formulate exactly 3 highly specific, separate study exam questions that a student might want to ask a tutor. 
        
        Rules:
        - Keep each question under 15 words.
        - Make them highly relevant to the text content.
        - Return them STRICTLY as a valid JSON list of strings, with no extra text or markdown formatting.
        
        Text Snippet:
        {sample_text}
        """
        
        response = llm.invoke(prompt)
        # Handle cleaning out markdown json codeblocks if added by the model
        clean_res = response.content.strip().lstrip("```json").rstrip("```").strip()
        prompts_list = json.loads(clean_res)
        
        if isinstance(prompts_list, list) and len(prompts_list) >= 3:
            return prompts_list[:3]
    except Exception as e:
        print(f"Failed to generate dynamic cards: {e}")
        
    return fallback_prompts