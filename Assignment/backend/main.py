import os
import uuid
import base64
import io
import sqlite3
from typing import Optional, List

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from pathlib import Path
import pandas as pd
import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)

# ================= ENV =================
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

UPLOAD_DIR = os.path.join(os.getcwd(), "../uploads/")
os.makedirs(UPLOAD_DIR, exist_ok=True)
FAISS_DIR = Path("../faiss_index/")


# ================= APP =================
app = FastAPI(title="Smart RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= POSTGRES =================
def get_db_conn():
    return psycopg2.connect(DATABASE_URL)

# ================= OCR =================
def ocr_from_path(path: str) -> str:
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("OCR path error:", e)
        return ""


def ocr_from_base64(b64: str) -> str:
    try:
        if not b64:
            return ""

        # Remove data:image/...;base64, prefix if exists
        if "," in b64:
            b64 = b64.split(",")[1]

        # Fix incorrect padding
        missing_padding = len(b64) % 4
        if missing_padding:
            b64 += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))

        return pytesseract.image_to_string(img)

    except Exception as e:
        print("OCR base64 error:", e)
        return ""

# ================= SQLITE EXTRACT =================
def extract_sqlite(db_path: str) -> List[Document]:
    docs = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for (table,) in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        docs.append(
            Document(
                page_content=df.to_string(),
                metadata={"table": table}
            )
        )

    conn.close()
    return docs

# ================= DOCUMENT PROCESSOR =================
class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process(self, path: str, ext: str) -> List[Document]:
        docs: List[Document] = []

        if ext == "pdf":
            docs = PyPDFLoader(path).load()
            if not any(d.page_content.strip() for d in docs):
                text = ocr_from_path(path)
                if text.strip():
                    docs = [Document(page_content=text)]

        elif ext in ["docx", "doc"]:
            docs = Docx2txtLoader(path).load()

        elif ext == "txt":
            docs = TextLoader(path, encoding="utf-8").load()

        elif ext in ["jpg", "jpeg", "png"]:
            text = ocr_from_path(path)
            if text.strip():
                docs = [Document(page_content=text)]

        elif ext == "csv":
            try:
                docs = CSVLoader(path).load()
            except Exception:
                df = pd.read_csv(path, encoding="utf-8", errors="ignore")
                docs = [Document(page_content=df.to_string())]

        elif ext == "db":
            docs = extract_sqlite(path)

        else:
            raise HTTPException(400, "Unsupported file type")

        # 🔐 Split + REMOVE empty chunks
        chunks = self.splitter.split_documents(docs)
        chunks = [d for d in chunks if d.page_content.strip()]

        if not chunks:
            raise ValueError("No readable content found in file")

        for i, d in enumerate(chunks):
            d.metadata = d.metadata or {}
            d.metadata["chunk_index"] = i

        return chunks

processor = DocumentProcessor()

# ================= VECTOR STORE =================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Ensure FAISS_DIR exists
FAISS_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = None

# Try to load existing FAISS index
if FAISS_DIR.exists():
    try:
        vectorstore = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully.")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        vectorstore = None

# If no FAISS exists, create empty vectorstore
if vectorstore is None:
    print("Creating a new empty FAISS vectorstore...")
    empty_doc = [Document(page_content="", metadata={})]
    vectorstore = FAISS.from_documents(empty_doc, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print("New FAISS index created.")

def save_vectorstore():
    if vectorstore:
        vectorstore.save_local(FAISS_DIR)

# ================= LLM + PROMPT =================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

SYSTEM_PROMPT = '''
You are a helpful assistant for students.
Use the context to answer the question in max three sentences.
For general greetings or conversation, respond naturally.
Keep responses friendly and helpful.but If you don't know question answer, just say don't know.
Context: {context}
Chat History: {chat_history}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

# ================= MODELS =================
class UploadResponse(BaseModel):
    file_id: str
    chunks: int
    message: str

class QueryRequest(BaseModel):
    user_id: int
    text: Optional[str] = None
    image_base64: Optional[str] = None

class HistoryRequest(BaseModel):
    user_id: int

class UserRequest(BaseModel):
    username: str
    
# ================= ENDPOINTS =================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

    # ---- Save file ----
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(400, f"Upload failed: File write error → {str(e)}")

    # ---- Process document ----
    try:
        docs = processor.process(file_path, ext)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Upload failed: Error loading {file.filename} → {str(e)}"
        )

    # ---- Metadata ----
    for i, d in enumerate(docs):
        d.metadata.update({
            "filename": file.filename,
            "chunk_index": i
        })

    # ---- FAISS SAFE ADD ----
    global vectorstore
    try:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(docs, embeddings)
        else:
            vectorstore.add_documents(docs)

        vectorstore.save_local(FAISS_DIR)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vectorstore error → {str(e)}"
        )

    return {
        "file_id": file_id,
        "chunks": len(docs),
        "message": f"✅ {file.filename} uploaded successfully ({len(docs)} chunks)"
    }



@app.post("/get_history")
def get_history(req: HistoryRequest):
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT prompt, answer FROM chat_history WHERE user_id=%s ORDER BY id ASC",
        (req.user_id,)
    )
    rows = cur.fetchall()

    cur.close()
    conn.close()

    history = []
    for q, a in rows:
        history.append({"role": "human", "content": q})
        history.append({"role": "ai", "content": a})

    return {"history": history}


@app.post("/query")
def query_rag(req: QueryRequest):
    if vectorstore is None:
        raise HTTPException(400, "No documents uploaded yet")

    question = req.text or ""

    if req.image_base64:
        ocr_text = ocr_from_base64(req.image_base64)
        if ocr_text.strip():
            question += "\n" + ocr_text

    # ===== LOAD CHAT HISTORY =====
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT prompt, answer FROM chat_history WHERE user_id=%s ORDER BY id ASC",
        (req.user_id,)
    )
    rows = cur.fetchall()

    chat_history_messages = []
    for q, a in rows:
        chat_history_messages.append(HumanMessage(content=q))
        chat_history_messages.append(AIMessage(content=a))

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history_messages
    })

    answer = response.get("answer", "I don't know")

    cur.execute(
        "INSERT INTO chat_history (user_id, prompt, answer) VALUES (%s,%s,%s)",
        (req.user_id, question, answer)
    )
    conn.commit()
    cur.close()
    conn.close()

    sources = []
    context_preview = []
    seen = set()

    for doc in response.get("context", []):
        key = (doc.metadata.get("filename"), doc.metadata.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": doc.metadata.get("filename"),
                "chunk": doc.metadata.get("chunk_index")
            })
            context_preview.append(doc.page_content[:300])

    return {
        "answer": answer,
        "context": context_preview,
        "sources": sources
    }
    


@app.post("/get_or_create_user")
def get_or_create_user(req: UserRequest):
    conn = get_db_conn()
    cur = conn.cursor()
    
    # 1. Try to find the user
    cur.execute("SELECT id FROM users WHERE username = %s", (req.username,))
    user_row = cur.fetchone() #(1,)

    
    if user_row:
        user_id = user_row[0] #1
    else:
        # 2. If not found, create them
        cur.execute("INSERT INTO users (username) VALUES (%s) RETURNING id", (req.username,))
        conn.commit()
        user_id = cur.fetchone()[0] #2
        
    cur.close()
    conn.close()
    return {"user_id": user_id, "username": req.username}


@app.get("/")
def root():
    return {"status": "Smart RAG API Running 🚀"}
