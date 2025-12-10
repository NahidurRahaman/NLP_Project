import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn as nn
import pickle
import re

token_2_id = None
# Load the dictionary later
with open(r"vocab.pkl", "rb") as f:
    token_2_id = pickle.load(f)
print(token_2_id)

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text


def tokenize(text):
    tokens = text.split()
    return tokens


def convert_tokens_2_ids(tokens):
    input_ids = [
        token_2_id.get(token, token_2_id['<UNK>']) for token in tokens
    ]
    return input_ids


def process_text(text, aspect):
    text_aspect_pair = text + ' ' + aspect
    normalized_text = normalize(text_aspect_pair)
    tokens = tokenize(normalized_text)
    input_ids = convert_tokens_2_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    return input_ids


class ABSA(nn.Module):
    def __init__(self, vocab_size, num_labels=3):
        super(ABSA, self).__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=256
        )
        # Sequence to sequence learning
        self.lstm_layer = nn.LSTM(
            input_size=256,
            hidden_size=512,
            batch_first=True, # as batch is first dimension in the input shape
        )

        self.fc_layer = nn.Linear(
            in_features=512,
            out_features=self.num_labels
        )
    def forward(self, x):
        # input: (B, L)
        embeddings = self.embedding_layer(x) # (B, L, 256)
        lstm_out, _ = self.lstm_layer(embeddings) # (B, L, 512)
        logits = self.fc_layer(lstm_out[:, -1, :])
        return logits

model = ABSA(vocab_size=len(token_2_id.keys()), num_labels=3)
model.load_state_dict(torch.load(r'model_weights.pth'))
model.eval()
print("Model loaded successfully")

app = FastAPI()

# space url = "https://masumbhuiyan-myabsaservice.hf.space/"
# greet api = "https://masumbhuiyan-myabsaservice.hf.space/greet"
# post api = "https://masumbhuiyan-myabsaservice.hf.space/predict"
@app.get("/greet")
def greet_json():
    return {"message": "Hello World"}


class TextAspectInput(BaseModel):
    text: str
    aspect: str


SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


@app.post("/predict")
async def predict_sentiment(input_data: TextAspectInput):
    try:
        text = input_data.text
        aspect = input_data.aspect
        input_ids = process_text(text, aspect)

        try:
            with torch.no_grad():
                logits = model(input_ids)
                probs = torch.softmax(logits, dim=-1)
                prediction = probs.argmax(dim=-1).item()
                sentiment = SENTIMENT_LABELS[prediction]
        except Exception as e:
            raise Exception(e)

        return {"sentiment": sentiment, "probabilities": probs.squeeze().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))