# 📄 RAG PDF App — Fast Local Retrieval-Augmented Generation with Groq API

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that lets you **upload PDFs**, extract context, and **query them using LLMs**. It uses **Qdrant** for vector storage, **SentenceTransformer** for embeddings, and **Groq’s open-source GPT-OSS-20B model** for ultra-fast responses.

---

## 🚀 Features

* PDF text extraction and chunking
* Embedding generation using huggingFace `all-MiniLM-L6-v2`
* Vector search with Qdrant
* Query answering using `openai/gpt-oss-20b` via Groq API
* Modular architecture with FastAPI + Streamlit UI + Inngest event functions

---

## 🧩 Project Structure

```
RAG_PDF_App/
├── main.py                # FastAPI backend with RAG pipeline
├── streamlit_app.py       # Frontend interface for users
├── vector_db.py           # Qdrant vector database logic
├── data_loader.py         # PDF loader, chunking, and embedding
├── custom_types.py        # Pydantic models for data flow
├── .env                   # Environment variables
└── requirements.txt       # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/RAG_PDF_App.git
cd RAG_PDF_App
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # For Windows
```

### 3️⃣ Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4️⃣ Add environment variables

Create a file named `.env` in your project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
QDRANT_URL=http://localhost:6333
```

---

## 🧠 Running the App

### 🖥️ 1. Run the backend (FastAPI)

```bash
uv run uvicorn main:app
```

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 🧱 Run Qdrant Locally via Docker

If you don’t have Qdrant running yet, use Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
``` 
Expected output:

```
✅ Qdrant will now be available at: http://localhost:6333
```

### 🧩 2. Run the Inngest worker (for async jobs)

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

Expected output:

```
HTTP Request: POST http://127.0.0.1:8288/fn/register "HTTP/1.1 200 OK"
```

### 🌐 3. Run the Streamlit UI

```bash
uv run streamlit run .\streamlit_app.py
```

Then open your browser:

```
Local URL: http://localhost:8501
Network URL: http://10.x.x.x:8501
```

---

## 🧩 How It Works

1. **PDF Upload:** The user uploads a document via Streamlit.
2. **Chunking:** The text is split into chunks using `data_loader.py`.
3. **Embedding:** Each chunk is encoded using the `intfloat/e5-base-v2` model.
4. **Storage:** Embeddings are stored in Qdrant.
5. **Query:** The user asks a question → The system searches for the most relevant chunks.
6. **Generation:** Context + Question → Sent to `openai/gpt-oss-20b` model (Groq API).
7. **Response:** The model returns an accurate answer grounded in the document.

---

## ⚡ Groq Integration Details

We use Groq’s **OpenAI-compatible API** endpoint:

```python
API_URL = "https://api.groq.com/openai/v1/chat/completions"
model = "openai/gpt-oss-20b"
```

Groq’s API key format: `gsk_...`

This gives lightning-fast inference (≈ 100 tokens/sec) with no external dependencies.

---

## 🧪 Test the Model Directly

Run this in PowerShell to verify your Groq API key:

```powershell
curl -X POST "https://api.groq.com/openai/v1/chat/completions" `
     -H "Authorization: Bearer gsk_your_groq_api_key_here" `
     -H "Content-Type: application/json" `
     -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello from Groq!"}]}'
```

Expected output:

```json
{"choices":[{"message":{"role":"assistant","content":"Hello! How can I help you?"}}]}
```

---

## 🧰 Tech Stack

| Component       | Technology                       |
| --------------- | -------------------------------- |
| **Frontend**    | Streamlit                        |
| **Backend**     | FastAPI                          |
| **Vector DB**   | Qdrant                           |
| **Embeddings**  | SentenceTransformer (all-MiniLM-L6) |
| **LLM**         | openai/gpt-oss-20b (Groq API)    |
| **Async Jobs**  | Inngest                          |
| **Environment** | Python 3.12                      |

---

## 🧾 Example Workflow

1. Upload a PDF
2. Ingest it → vectorized in Qdrant
3. Ask: “What are the main findings of section 3?”
4. The system finds matching chunks
5. Sends context + question to Groq model
6. Displays the final summarized answer

---

## 🧑‍💻 Author

Built by **Chethan N.** — Innovating AI-powered retrieval syst
