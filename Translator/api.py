import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mt_pretrained_model_name = "shhossain/opus-mt-en-to-bn"

class MTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # load pretrained model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(mt_pretrained_model_name)
        # load pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(mt_pretrained_model_name)

    def preprocess(self, text: str):
        """Preprocess input text"""
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

    def postprocess(self, generated_ids):
        """Convert generated token IDs back to text"""
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def translate(self, english_text: str) -> str:
        """Translate English text to Bangla"""
        input_ids, attention_mask = self.preprocess(english_text)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7
            )

        return self.postprocess(generated_ids)[0]

    def translate_batch(self, english_texts: List[str]) -> List[str]:
        """Translate multiple English texts to Bangla"""
        translations = []
        for text in english_texts:
            translations.append(self.translate(text))
        return translations

    def forward(self,
                src_input_ids,
                src_attention_mask,
                tgt_input_ids,
                tgt_attention_mask
        ):
        outputs = self.model(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=tgt_input_ids[:, :-1],
            decoder_attention_mask=tgt_attention_mask[:, :-1]
        )
        return outputs

# Load model
try:
    model = MTModel()
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# FastAPI app
app = FastAPI(title="English to Bangla Translator")


class TranslationRequest(BaseModel):
    text: str


class BatchTranslationRequest(BaseModel):
    texts: List[str]


@app.get("/")
def root():
    return {"message": "Translation API - Use /translate endpoint"}

@app.get("/health")
async def health_check():
    """Check if the API is healthy"""
    try:
        # Test translation
        test_result = model.translate("Hello")
        return {
            "status": "healthy",
            "model": mt_pretrained_model_name,
            "device": str(device),
            "test_translation": test_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/translate")
def translate_get(text: str):
    if not text.strip():
        raise HTTPException(500, "Text cannot be empty")
    return {"translation": model.translate(text)}


@app.post("/translate")
def translate_post(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(500, "Text cannot be empty")
    return {"translation": model.translate(request.text)}


@app.post("/translate/batch")
def translate_batch(request: BatchTranslationRequest):
    if not request.texts:
        raise HTTPException(500, "Texts list cannot be empty")

    valid_texts = [text for text in request.texts if text.strip()]
    if not valid_texts:
        raise HTTPException(500, "No valid texts to translate")

    return {"translations": model.translate_batch(valid_texts)}



