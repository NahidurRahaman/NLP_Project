# Smart RAG API + Streamlit UI 🚀

A full-stack Retrieval-Augmented Generation (RAG) system using:
- **FastAPI** (Backend API)
- **LangChain + FAISS** (Vector Search)
- **Google Gemini (Generative AI)**
- **PostgreSQL** (Chat history & users)
- **Streamlit** (Frontend UI)
- **OCR support** (Images & scanned PDFs)

---

## ✨ Features
- Upload documents: **PDF, DOCX, TXT, CSV, Images, SQLite DB**
- OCR support for images & scanned PDFs
- Semantic search using FAISS
- Chat history per user
- Image-based question answering
- Streamlit-based chat UI

---

## 📦 Tech Stack
- Python 3.10+
- FastAPI
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Google Gemini API
- PostgreSQL
- Tesseract OCR

---

## ⚙️ Environment Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/smart-rag.git
cd smart-rag

## 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

## 3️⃣ Install Dependencies
pip install -r requirements.txt

# 📄 Sample .env

DATABASE_URL=postgresql://postgres:password@localhost:5432/smart_rag
GOOGLE_API_KEY=your_google_gemini_api_key

# 🗄️ Database Setup

Create tables in PostgreSQL:

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL
);

CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    prompt TEXT,
    answer TEXT
);

## 🚀 Run the Application
Backend (FastAPI)
uvicorn main:app --reload


## API runs at:

http://localhost:8000

Frontend (Streamlit)
streamlit run app.py


## UI runs at:

http://localhost:8501

