import io
import gradio as gr
import requests
import PyPDF2
import docx

ROOT_API = "https://nahidur415-rsmsapi.hf.space"


# ------------------------------
# File Text Extraction
# ------------------------------
def extract_text_from_pdf(file):
    if isinstance(file, dict):  # NamedString from Gradio
        file_bytes = io.BytesIO(file['data'])
    else:
        file_bytes = file
    pdf_reader = PyPDF2.PdfReader(file_bytes)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    if isinstance(file, dict):
        file_bytes = io.BytesIO(file['data'])
    else:
        file_bytes = file
    doc = docx.Document(file_bytes)
    text = ''
    for p in doc.paragraphs:
        text += p.text + '\n'
    return text


def extract_text_from_txt(file):
    # If file is a NamedString from Gradio
    if hasattr(file, "value"):
        text = file.value  # .value contains the text content
    else:
        try:
            text = file.read().decode('utf-8')
        except UnicodeDecodeError:
            text = file.read().decode('latin-1')
    return text



def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower() if hasattr(uploaded_file, 'name') else \
    uploaded_file['name'].split('.')[-1].lower()

    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type! Upload PDF, DOCX or TXT.")


# ------------------------------
# API Call Function
# ------------------------------
def call_predict_api(text: str) -> dict:
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(ROOT_API + "/predict", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return {"status": "success", "sentiment": result.get("sentiment")}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}


# ------------------------------
# Single Prediction
# ------------------------------
def gradio_predict(file):
    try:
        content = handle_file_upload(file)
        category = call_predict_api(content)
        if category["status"] == "success":
            return f"📄 Predicted Category: {category['sentiment']}"
        else:
            return f"❌ Error: {category['error']}"
    except Exception as e:
        return f"❌ Error: {e}"


# ------------------------------
# Dark Mode CSS + Bigger Fonts
# ------------------------------
dark_css = """
body {
    background-color: #0d0d0d; 
    color: #FFFFFF; 
    font-family: 'Arial', sans-serif;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 48px;
    color: #FFFFFF;
    text-align: center;
    margin-bottom: 30px;
    text-shadow: 1px 1px 5px #ffcc00;
}

#component-0, #component-1, #component-2, #component-3, input, textarea {
    background-color: #1a1a1a !important; 
    color: #FFFFFF !important; 
    border: 2px solid #FFFFFF !important;
    border-radius: 12px !important;
    font-size: 22px !important;
    padding: 15px !important;
    box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
}

.gr-button {
    background-color: #ffcc00 !important;
    color: #0d0d0d !important;
    font-size: 22px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 25px !important;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}

.gr-button:hover {
    background-color: #e6b800 !important;
    transform: scale(1.05);
}

.gr-box {
    border: 2px solid #FFFFFF !important;
    border-radius: 15px !important;
    padding: 20px !important;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
    margin-top: 20px !important;
}

"""

# ------------------------------
# Gradio Interface
# ------------------------------
with gr.Blocks(css=dark_css, title="Resume Category Prediction") as demo:
    gr.HTML("<h1 style='text-align:center;'>📄 Resume Category Prediction (APP)</h1>")

    uploaded_file = gr.File(label="Upload Resume (PDF/DOCX/TXT)", file_types=[".pdf", ".docx", ".txt"])
    output_text = gr.Textbox(label="Predicted Category", interactive=False, lines=2)

    uploaded_file.change(fn=gradio_predict, inputs=uploaded_file, outputs=output_text)

if __name__ == "__main__":
    demo.launch(debug=True)
