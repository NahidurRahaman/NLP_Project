import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------------------------
# Character Set (AS YOU PROVIDED)
# -------------------------------------------------
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = [
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z'
]

ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# -------------------------------------------------
# OCR Model (nn.Module)
# -------------------------------------------------
class OCR(nn.Module):
    def __init__(self, num_chars=ALL_CHAR_SET_LEN, max_length=MAX_CAPTCHA):
        super(OCR, self).__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.char_set = ALL_CHAR_SET

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first conv layer for 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_chars * max_length)
        )

        self.transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor()
        ])

    def forward(self, x):
        x = self.resnet(x)
        return x.view(-1, self.max_length, self.num_chars)

    def predict_image(self, image_path, device):
        img = Image.open(image_path).convert("L")
        img = self.transform(img).unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            output = self(img)               # (1, 4, 36)
            pred = output.softmax(-1).argmax(-1)

        text = "".join(self.char_set[i] for i in pred[0])
        return text




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OCR(num_chars=ALL_CHAR_SET_LEN,max_length=MAX_CAPTCHA)

# Load trained weights
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")
# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="OCR FastAPI (numbers + lowercase)")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files are allowed"}
        )

    image_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        prediction = model.predict_image(image_path, device)
        return {
            "filename": file.filename,
            "prediction": prediction
        }

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
