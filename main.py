import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv()

# 2. Load the database (The Memory)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Initialize the LLM (The Brain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# 4. Create the RetrievalQA Chain
# This 'chain' automatically handles: Question -> Search DB -> Send to LLM -> Final Answer
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "Stuffing" the chunks into the prompt
    retriever=vector_db.as_retriever(),
    return_source_documents=True 
)

# 5. Ask a question!
question = "What are the types of experiment in probability?"
response = qa_chain.invoke(question)

print(f"❓ Question: {question}")
print(f"🤖 Answer: {response['result']}")

print("\n📚 SOURCED FROM:")
for doc in response["source_documents"]:
    page_num = doc.metadata.get("page", "Unknown")
    # We add 1 because lists start at 0, but book pages start at 1
    print(f"-> Page: {page_num + 1}")