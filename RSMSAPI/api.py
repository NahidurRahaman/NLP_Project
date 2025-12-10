import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
import pickle
import re

token_2_id = None
# Load the dictionary later
with open(r"vocab.pkl", "rb") as f:
    token_2_id = pickle.load(f)
print(token_2_id)

def normalize(text):
    #convert the text into lowercase
    text = text.lower()
    # remove punctuations, special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # remove extra whitespaces
    text = ' '.join(text.split())

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)

    return text


def tokenize(text):
    tokens = text.split()
    return tokens


def convert_tokens_2_ids(tokens):
    input_ids = [
        token_2_id.get(token, token_2_id['<UNK>']) for token in tokens
    ]
    return input_ids


def process_text(text):
    normalized_text = normalize(text)
    tokens = tokenize(normalized_text)
    input_ids = convert_tokens_2_ids(tokens)
    input_ids = torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
    return input_ids


class RSMModel(nn.Module):
    def __init__(self, vocab_size, num_labels=25, embed_dim=512, hidden_size=512, num_layers=2):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_labels = num_labels

        # Embedding: Convert each token id is represented  by a vector
        # For example: input_ids = [124, 14, 35]
        # input ids shape = (B, 3,)
        # After that, shape = (B, 3, 256)
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        # MULTI-LAYER BI-LSTM
        self.lstm_layer = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,        # MULTI-LAYER এখানে!
            batch_first=True,
            dropout=0.3,                  # Layer এর মাঝে dropout
            bidirectional=True            # BiLSTM
        )

        # Output dimension = hidden_size * 2 (because bidirectional)
        self.fc_layer = nn.Linear(hidden_size * 2, num_labels)


    def forward(self, x):
        embeddings = self.embedding_layer(x)
        lstm_out, _ = self.lstm_layer(embeddings)
        # 👉 Mean pooling (last timestep use করলে accuracy কমে)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc_layer(pooled)
        return logits

model = RSMModel(
    vocab_size=len(token_2_id),
    num_labels=25,
    embed_dim=512,
    hidden_size=512,
    num_layers=2
)
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


SENTIMENT_LABELS = {
    0: "ETL Developer",
    1: "Data Science",
    2: "Civil Engineer",
    3: "Mechanical Engineer",
    4: "Python Developer",
    5: "Arts",
    6: "Blockchain",
    7: "Testing",
    8: "Automation Testing",
    9: "Electrical Engineering",
    10: "SAP Developer",
    11: "Network Security Engineer",
    12: "HR",
    13: "DevOps Engineer",
    14: "Database",
    15: "Java Developer",
    16: "Business Analyst",
    17: "Operations Manager",
    18: "Advocate",
    19: "DotNet Developer",
    20: "Hadoop",
    21: "PMO",
    22: "Health and fitness",
    23: "Sales",
    24: "Web Designing"
}


@app.post("/predict")
async def predict_sentiment(input_data: TextAspectInput):
    try:
        text = input_data.text
        input_id = process_text(text)

        try:
            with torch.no_grad():
                logits = model(input_id)
                probs = torch.softmax(logits, dim=-1)
                prediction = probs.argmax(dim=-1).item()
                sentiment = SENTIMENT_LABELS[prediction]
        except Exception as e:
            raise Exception(e)

        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))