import streamlit as st
import requests
import base64
from datetime import datetime

API_URL = "http://localhost:8000"  # FastAPI URL

# ================= SESSION STATE =================
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= UTILS =================
def login_user(username):
    res = requests.post(f"{API_URL}/get_or_create_user", json={"username": username})
    if res.status_code == 200:
        data = res.json()
        st.session_state.user_id = data["user_id"]
        st.session_state.username = data["username"]
        st.session_state.chat_history = get_chat_history()
        st.success(f"Logged in as {st.session_state.username}")
    else:
        st.error("Login failed")

def get_chat_history():
    res = requests.post(f"{API_URL}/get_history", json={"user_id": st.session_state.user_id})
    if res.status_code == 200:
        return res.json().get("history", [])
    return []

def render_message(message):
    role = message["role"]
    content = message["content"]
    if role == "human":
        st.markdown(f"<div style='text-align:right;background:#DCF8C6;padding:8px;border-radius:8px;margin:5px'>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left;background:#F1F0F0;padding:8px;border-radius:8px;margin:5px'>{content}</div>", unsafe_allow_html=True)

def render_message_with_sources(message):
    """AI message with sources and colored text"""
    role = message["role"]
    content = message["content"]

    if role == "human":
        html = f"""
        <div style='
            text-align:right;
            background:#DCF8C6;
            padding:8px;
            border-radius:8px;
            margin:5px;
            color:#0B3D91;  /* Dark blue text */
            font-weight:500;
        '>{content}</div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        html = f"""
        <div style='
            text-align:left;
            background:#F1F0F0;
            padding:8px;
            border-radius:8px;
            margin:5px;
            color:#8B0000;  /* Dark red text */
            font-weight:500;
        '>{content}
        """
        sources = message.get("sources")
        if sources:
            html += "<div style='margin-top:5px;font-size:0.85rem;color:#FF9800;font-weight:600'>📚 Sources:</div>"
            for s in sources:
                html += f"<div style='font-size:0.8rem;color:#000000'>• {s['filename']} (Chunk {s['chunk']})</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ================= LOGIN =================
if st.session_state.user_id is None:
    st.title("Login / Signup")
    username = st.text_input("Enter username")
    if st.button("Login"):
        if username.strip():
            login_user(username)
else:
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.chat_history = []
        st.rerun()

# ================= FILE UPLOAD =================
st.title("Smart RAG Chat UI")
uploaded_file = st.file_uploader("Upload file (PDF, DOCX, TXT, CSV, Images, SQLite)", type=["pdf","docx","txt","csv","jpg","jpeg","png","db"])
if uploaded_file:
    
    # Call your FastAPI upload endpoint
    try:
        files = {"file": (uploaded_file.name, open(f"temp_{uploaded_file.name}", "rb"))}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            st.success(f"✅ Upload successful: {uploaded_file.name} ({response.json()['chunks']} chunks created)")
        else:
            st.error(f"❌ Upload failed: {response.text}")
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
# ================= QUESTION / CHAT =================
st.subheader("Ask a question")
question = st.text_area("Your question here:")
image_file = st.file_uploader("Optional: Upload image for OCR", type=["jpg","jpeg","png"])
if st.button("Send"):
    if not question and not image_file:
        st.warning("Enter a question or upload an image!")
    else:
        payload = {"user_id": st.session_state.user_id, "text": question}
        if image_file:
            img_bytes = image_file.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            payload["image_base64"] = img_b64
        res = requests.post(f"{API_URL}/query", json=payload)
        if res.status_code == 200:
            data = res.json()
            ai_msg = {"role":"ai","content":data["answer"], "sources":data.get("sources",[])}
            human_msg = {"role":"human","content":question}
            st.session_state.chat_history.append(human_msg)
            st.session_state.chat_history.append(ai_msg)
        else:
            st.error(res.json().get("detail","Error fetching answer"))

# ================= RENDER CHAT =================
st.subheader("Chat History")
for msg in st.session_state.chat_history:
    render_message_with_sources(msg)
