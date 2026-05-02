from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the existing database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Define your question
query = "What are the main topics discussed in this document?" 

# 3. Search the database
# 'k=3' means find the 3 most relevant chunks of text
docs = vector_db.similarity_search(query, k=3)

print(f"🔍 Searching for: {query}\n")
print("--- Relevant Chunks Found ---")

for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:")
    print(doc.page_content[:500] + "...") # Print first 500 characters