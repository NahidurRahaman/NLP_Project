import requests
import json
import gradio as gr
import os
import pandas as pd
import plotly.express as px
ROOT_API = "https://nahidur415-myabsa.hf.space"


def call_greets_json():
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(ROOT_API + "/greet", json={}, headers=headers)
        response.raise_for_status()
        result = response.json()
        return {
            "status": "success",
            "response": result,
            "message": "Endpoint is working correctly"
        }

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "error": f"HTTP error occurred: {str(http_err)}",
            "status_code": response.status_code
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": "error",
            "error": f"Request error occurred: {str(req_err)}"
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Invalid JSON response from API"
        }




def call_predict_api(text: str, aspect: str) -> dict:
    payload = {
        "text": text,
        "aspect": aspect
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(ROOT_API + "/predict", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return {
            "status": "success",
            "sentiment": result.get("sentiment"),
            "probabilities": result.get("probabilities"),
            "raw_response": result
        }
    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "error": f"HTTP error occurred: {str(http_err)}",
            "status_code": response.status_code
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": "error",
            "error": f"Request error occurred: {str(req_err)}"
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Invalid JSON response from API",
            "raw_response": response.text
        }


# -------------------------------------------------
#  SINGLE PREDICTION
# -------------------------------------------------
def gradio_predict(text: str, aspect: str):
    result = call_predict_api(text, aspect)

    if result["status"] != "success":
        return f"<span style='color:red; font-size:22px;'>❌ Error: {result['error']}</span>"

    sentiment = result["sentiment"]
    probs = result["probabilities"]

    # If probs is a list, convert to dict
    if isinstance(probs, list):
        # Assume order: [positive, neutral, negative]
        probs = {
            "positive": probs[2],
            "neutral": probs[1],
            "negative": probs[0]
        }

    output = f"""
    <div style="font-size:22px;">
        <b>Sentiment:</b> {sentiment.capitalize()} <br><br>
        <b>Probabilities:</b><br>
        • Positive: {probs['positive']:.2f} <br>
        • Neutral: {probs['neutral']:.2f} <br>
        • Negative: {probs['negative']:.2f}
    </div>
    """

    return output


# -------------------------------------------------
#  CSV BATCH PROCESSING
# -------------------------------------------------
def process_csv(file, text_col, aspect_col):
    if file is None:
        return "❌ Please upload a CSV file.", None, None

    df = pd.read_csv(file.name)

    if text_col not in df.columns or aspect_col not in df.columns:
        return "❌ Column name not found in CSV!", None, None

    sentiments = []
    for _, row in df.iterrows():
        result = call_predict_api(row[text_col], row[aspect_col])
        sentiments.append(result["sentiment"])

    df["predicted_sentiment"] = sentiments

    # Sentiment Distribution
    fig1 = px.histogram(
        df,
        x="predicted_sentiment",
        title="Sentiment Distribution",
        color="predicted_sentiment"
    )

    # Aspect vs Sentiment
    fig2 = px.histogram(
        df,
        x=aspect_col,
        color="predicted_sentiment",
        title="Sentiment per Aspect",
        barmode="group"
    )

    return df, fig1, fig2


# -------------------------------------------------
#  CUSTOM CSS FOR FONT SIZE
# -------------------------------------------------
custom_css = """
#component-0, #component-1, #component-2, #component-3 {
    font-size: 20px !important;
}
h1, h2, h3, p {
    font-size: 24px !important;
}
"""


# -------------------------------------------------
#  BUILD INTERFACE
# -------------------------------------------------
with gr.Blocks(css=custom_css, title="Aspect-Based Sentiment Analysis Dashboard") as demo:

    gr.Markdown(
        """
        <h1 style='font-size:36px;'> Aspect-Based Sentiment Analysis</h1>
        <p style='font-size:22px;'>Analyze single text inputs or run batch predictions on a CSV file with visualization.</p>
        """,
    )

    with gr.Tab("🔍 Single Prediction"):
        text_input = gr.Textbox(lines=3, label="Text", elem_id="input_text")
        aspect_input = gr.Textbox(lines=1, label="Aspect", elem_id="input_aspect")

        submit_btn = gr.Button("Predict Sentiment", variant="primary")
        output_box = gr.HTML()

        submit_btn.click(
            gradio_predict,
            inputs=[text_input, aspect_input],
            outputs=output_box
        )

    with gr.Tab(" CSV Batch Processing"):
        csv_file = gr.File(label="Upload CSV")
        text_column = gr.Textbox(label="Text Column Name (e.g., review)")
        aspect_column = gr.Textbox(label="Aspect Column Name (e.g., service)")

        btn_process = gr.Button("Run Batch Prediction", variant="primary")

        df_output = gr.DataFrame(label="Predicted Results")
        graph1 = gr.Plot(label="Sentiment Distribution")
        graph2 = gr.Plot(label="Sentiment per Aspect")

        btn_process.click(
            process_csv,
            inputs=[csv_file, text_column, aspect_column],
            outputs=[df_output, graph1, graph2]
        )


if __name__ == "__main__":
    demo.launch(debug=True)
