# Academic-Assistant
A Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware answers for my college notes. Built with LangChain, ChromaDB, and Hugging Face Embeddings to eliminate AI hallucinations and ensure source-backed responses.

# 🎓 Academic AI — Production-Grade Academic RAG Assistant

Academic AI is a sophisticated Retrieval-Augmented Generation (RAG) platform designed to transform dense academic documents (PDFs, textbook chapters, lecture notes) into interactive, context-aware chat companions.

Instead of simply forwarding prompts to an LLM, this platform orchestrates a complete multi-stage RAG pipeline involving semantic chunking, vector embedding generation, contextual retrieval, prompt orchestration, and low-latency inference to deliver grounded, hallucination-resistant academic answers.

---

<p align="center">
  <img src="./assets/app-preview.png" width="100%" alt="AcaDocMine AI Screenshot"/>
</p>

<p align="center">
  <a href="https://academic-assistant-rag-ouddaajucgs7vvr3ax44ww.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</p>

---

## 🚀 Live Demo

🔗 **Try the App Here:**  
(https://academic-assistant-rag-ouddaajucgs7vvr3ax44ww.streamlit.app/)

---

## 📸 Application Preview

> Replace the image path below with your actual screenshot path.

```md
![AcaDocMine Preview](./assets/app-preview.png)

---

# 🏗️ Architecture & Data Flow

```text
[User Document]
        │
        ▼
[PyTorch / Transformers Processing Pipeline]
        │
        ▼
[Semantic Chunking Engine]
        │
        ▼
[Vector Embeddings Generation]
        │
        ▼
[Supabase Vector Database (PGVector)]
        ▲
        │
[User Query]
        │
        ▼
[Semantic Similarity Search]
        │
        ▼
[Retrieved Context Chunks]
        │
        ▼
[Prompt Orchestration Layer]
(Query + Context Injection)
        │
        ▼
[Groq / Gemini Inference]
        │
        ▼
[AI Response]
        │
        ▼
[MongoDB Chat Logging]
```

---

# ⚙️ Core System Workflow

## 1. Intelligent Document Ingestion

Uploaded PDFs and academic notes are processed using custom extraction pipelines powered by:

- `PyTorch`
- `torchvision`
- Hugging Face `transformers`

The system extracts raw textual structure while preserving semantic coherence for downstream retrieval.

---

## 2. Semantic Chunking & Vectorization

Extracted content is segmented into semantically meaningful chunks using recursive text splitting strategies.

Each chunk is transformed into dense vector embeddings and stored inside:

- **Supabase PGVector**

This enables high-speed cosine similarity search across user-specific document collections.

---

## 3. Retrieval-Augmented Generation (RAG)

When a student asks a question:

1. The query is vectorized
2. Top-k semantically similar chunks are retrieved
3. Retrieved context is dynamically injected into the prompt
4. The final grounded prompt is sent to the LLM

This dramatically reduces hallucinations while increasing answer precision.

---

## 4. Multi-LLM Inference Layer

The platform supports multiple inference backends:

### Groq API
- Ultra-low latency inference
- Llama-based architectures

### Google Gemini API
- OCR & multimodal reasoning
- Image-based academic question extraction

---

## 5. Persistent Conversational Memory

Conversation history and metadata are stored using:

- **MongoDB Atlas**

This enables:
- Persistent chat memory
- Multi-user isolation
- Session continuity
- Context-aware follow-up questioning

---

# 🛠️ Deep Tech Stack

## Frontend
- Streamlit
- Custom responsive UI
- Stateful session handling
- Fragment-based rerender optimization

---

## AI / ML Infrastructure
- PyTorch
- torchvision
- Hugging Face transformers
- Sentence Transformers
- LangChain

---

## Large Language Models
- Groq Inference Engine
- Google Gemini API
- Llama-based architectures

---

## Vector Database
- Supabase
- PostgreSQL + PGVector

---

## NoSQL Database
- MongoDB Atlas

Used for:
- Chat history persistence
- User metadata
- Session tracking
- Conversational memory

---

# 💡 Engineering Challenges & Key Learnings

Building this project pushed me beyond simple API-wrapper applications into full-scale AI systems engineering.

---

## 1. Solving Streamlit Memory Leaks & Infinite Reloads

### The Problem
Streamlit reruns the entire script after every interaction.

This caused:
- repeated model initialization
- database reconnections
- excessive memory consumption
- deployment instability

Especially problematic with:
- `torch`
- `transformers`
- external API clients

---

### The Solution

I implemented a layered caching strategy using:

```python
@st.cache_resource
@st.cache_data
```

This optimized:
- model loading
- database connections
- embedding initialization
- vector store access

Result:
- significantly lower latency
- smoother rerenders
- stable cloud deployment

---

## 2. Hybrid Database Architecture (SQL + NoSQL)

### The Problem

Vector embeddings and conversational histories have fundamentally different storage requirements.

Using a single database for both introduced:
- scaling inefficiencies
- retrieval overhead
- poor schema flexibility

---

### The Solution

I designed a dual-database architecture.

### Supabase (PostgreSQL + PGVector)
Handles:
- semantic vectors
- cosine similarity search
- retrieval indexing

### MongoDB
Handles:
- flexible chat trees
- session persistence
- metadata storage
- conversational memory

This separation improved both:
- scalability
- maintainability

---

## 3. Multi-User Vector Isolation

### The Problem

Ensuring uploaded academic documents remain isolated per user inside a shared vector database.

---

### The Solution

Each chunk is tagged with:

```python
chunk.metadata["user_id"]
```

Retriever-level filtering ensures:
- zero cross-user leakage
- secure semantic retrieval
- isolated RAG pipelines

---

# 🚀 Local Installation & Verification

## 1. Clone Repository

```bash
git clone https://github.com/aaishakazi/academic-assistant-rag.git
cd academic-assistant-rag
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Important Note

CPU-optimized builds of:
- `torch`
- `torchvision`

were intentionally targeted to:
- reduce deployment image size
- prevent out-of-memory crashes
- improve cloud compatibility

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
MONGO_CONNECTION_STRING=your_mongodb_connection_string
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_public_key
```

---

## 5. Launch the Application

```bash
streamlit run app.py
```

---

# 📂 Key Features

- 📚 PDF Upload & Semantic Retrieval
- 🧠 Multi-LLM Routing
- 💬 Persistent Conversational Memory
- 🔍 Context-Aware RAG Pipeline
- 🖼️ OCR-Based Question Scanning
- 👥 Multi-User Isolation
- ☁️ Cloud-Native Deployment
- ⚡ Low-Latency Inference
- 📄 Source Citation Tracking
- 🔗 Direct PDF Source Navigation

---

# 📈 Future Roadmap

- [ ] Hybrid Search (BM25 + Dense Retrieval)
- [ ] Retrieval Visualization Graphs
- [ ] Cross-Document Reasoning
- [ ] Citation-Aware Responses
- [ ] Streaming Token Generation
- [ ] Fine-Tuned Academic Embedding Models
- [ ] Adaptive Chunking Strategies
- [ ] PDF Page-Level Deep Linking

---

# 🌐 Deployment

## Streamlit Cloud

Deploy seamlessly using:
- GitHub integration
- Streamlit Secrets Manager
- Supabase cloud infrastructure
- MongoDB Atlas


---

# ⭐ Final Note

This project represents a transition from building simple AI wrappers to engineering scalable, production-grade AI systems involving:

- Retrieval-Augmented Generation
- Vector Databases
- Multi-Model Inference
- Cloud Infrastructure
- Stateful Session Management
- Distributed Data Architectures

The goal was not just to build a chatbot —
but to engineer a reliable academic intelligence system capable of grounded, context-aware reasoning over user-provided knowledge bases.