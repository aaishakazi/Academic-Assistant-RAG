from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the PDF
# Replace 'notes.pdf' with your actual file name
loader = PyPDFLoader("notes.pdf")
data = loader.load()

# 2. Split the text into manageable chunks
# We use 'Recursive' because it tries to keep paragraphs and sentences together.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
chunks = text_splitter.split_documents(data)

print(f" Loaded {len(data)} pages.\n")
print(f" Split into {len(chunks)} chunks.")

# Let's look at the first chunk to see what it looks like
print("\n--- Sample Chunk ---")
print(chunks[0].page_content)

# 3. Choose an Embedding Model (This runs locally on your CPU/GPU)
# This model is small but very powerful for student projects.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the Vector Database
# This will create a folder named 'db' and save your "smart" data there.
print("⏳ Turning text into math (embeddings)... This might take a minute.")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("🚀 Success! Your Vector Database is ready in the 'chroma_db' folder.")