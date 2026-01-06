import os
import json
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image
import timm
import io
import numpy as np
from torchvision import transforms

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
MODEL_PATH = "model.pth"
CLASSES_FILE = "classes.json"
MODEL_NAME = "efficientnet_b0"
IMG_SIZE = 224

# Initialize FastAPI
app = FastAPI(title="Plant Disease Detector API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------------------------------------
# LOAD CLASS NAMES
# ------------------------------------------------
def load_classes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing")
    
    with open(path, "r") as f:
        classes = json.load(f)
    
    cleaned = []
    for c in classes:
        cleaned.append(c.replace("___", " ").replace("__", " ").replace("_", " "))
    return cleaned

try:
    class_names = load_classes(CLASSES_FILE)
    num_classes = len(class_names)
except:
    class_names = ["Error"]
    num_classes = 1

# ------------------------------------------------
# DISEASE INFORMATION
# ------------------------------------------------
disease_info = {
    "Apple Scab": "Fungal disease causing olive-green to black spots on leaves and fruits.",
    "Apple Black Rot": "Fungus causes purple spots that enlarge to brown lesions with concentric rings.",
    "Cedar Apple Rust": "Orange-yellow spots on leaves, caused by fungus alternating between apple and cedar trees.",
    "Apple Healthy": "No disease symptoms detected. Leaves are green and healthy.",
    "Blueberry Healthy": "Plant shows no signs of disease. Leaves are vibrant and healthy.",
    "Cherry Powdery Mildew": "White powdery fungal growth on leaves and shoots.",
    "Cherry Healthy": "Healthy cherry plant with no disease symptoms.",
    "Corn Cercospora Leaf Spot": "Gray leaf spots with tan centers and reddish-brown borders.",
    "Corn Common Rust": "Reddish-brown pustules on both leaf surfaces.",
    "Corn Northern Leaf Blight": "Large cigar-shaped grayish lesions on leaves.",
    "Corn Healthy": "Healthy corn plant with no disease symptoms.",
    "Grape Black Rot": "Brown circular lesions with black fruiting bodies on leaves and fruits.",
    "Grape Esca": "Also called Black Measles, causes tiger-stripe patterns on leaves.",
    "Grape Leaf Blight": "Dark spots with yellow halos caused by Isariopsis fungus.",
    "Grape Healthy": "Healthy grape vine with no disease symptoms.",
    "Orange Citrus Greening": "Yellowing of leaves, asymmetric blotchy mottling.",
    "Peach Bacterial Spot": "Water-soaked lesions on leaves that turn brown and fall out.",
    "Peach Healthy": "Healthy peach tree with no disease symptoms.",
    "Bell Pepper Bacterial Spot": "Small water-soaked spots that become brown with yellow halos.",
    "Bell Pepper Healthy": "Healthy pepper plant with no disease symptoms.",
    "Potato Early Blight": "Dark concentric rings on leaves, resembles target patterns.",
    "Potato Late Blight": "Water-soaked lesions that rapidly enlarge and turn black.",
    "Potato Healthy": "Healthy potato plant with no disease symptoms.",
    "Raspberry Healthy": "Healthy raspberry plant with no disease symptoms.",
    "Soybean Healthy": "Healthy soybean plant with no disease symptoms.",
    "Squash Powdery Mildew": "White powdery coating on leaves, stems, and fruits.",
    "Strawberry Leaf Scorch": "Purple to brown spots on leaves that may cause defoliation.",
    "Strawberry Healthy": "Healthy strawberry plant with no disease symptoms.",
    "Tomato Bacterial Spot": "Small dark lesions on leaves surrounded by yellow halos.",
    "Tomato Early Blight": "Bull's-eye pattern lesions with concentric rings.",
    "Tomato Late Blight": "Water-soaked greasy lesions that spread rapidly in humid conditions.",
    "Tomato Leaf Mold": "Yellow spots on upper leaf surface with grayish-purple mold underneath.",
    "Tomato Septoria Leaf Spot": "Small circular spots with dark borders and light centers.",
    "Tomato Spider Mites": "Yellow stippling on leaves caused by tiny spider mites feeding.",
    "Tomato Target Spot": "Brown spots with concentric rings and yellow halos.",
    "Tomato Yellow Leaf Curl Virus": "Upward curling of leaves, yellowing, and stunted growth.",
    "Tomato Mosaic Virus": "Mottled light and dark green pattern on leaves.",
    "Tomato Healthy": "Healthy tomato plant with vigorous growth and no disease symptoms."
}

def get_description(disease):
    for key in disease_info:
        if key.lower() in disease.lower():
            return disease_info[key]
    return "No description available."

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    state = torch.load(MODEL_PATH, map_location="cpu")
    
    clean_state = {k.replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model

try:
    model = load_model()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# ------------------------------------------------
# TRANSFORM
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ------------------------------------------------
# RISK COLOR LOGIC
# ------------------------------------------------
def get_risk_color(prob):
    if prob >= 0.70:
        return {"text": "High Risk", "color": "danger", "icon": "🔴"}
    elif prob >= 0.30:
        return {"text": "Moderate Risk", "color": "warning", "icon": "🟡"}
    else:
        return {"text": "Low Risk", "color": "success", "icon": "🟢"}

# ------------------------------------------------
# PREDICT FUNCTION
# ------------------------------------------------
async def predict_image(image_bytes):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            out = model(image_tensor)
            probs = torch.softmax(out, dim=1)[0]
        
        # Get all predictions
        predictions = []
        for i, class_name in enumerate(class_names):
            prob = float(probs[i])
            risk_info = get_risk_color(prob)
            predictions.append({
                "disease": class_name,
                "confidence": round(prob * 100, 2),
                "description": get_description(class_name),
                "risk_level": risk_info["text"],
                "risk_color": risk_info["color"],
                "risk_icon": risk_info["icon"]
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "top_disease": predictions[0]["disease"],
            "top_confidence": predictions[0]["confidence"],
            "all_predictions": predictions[:10]  # Top 10
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ------------------------------------------------
# API ROUTES
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint for predictions"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    image_bytes = await file.read()
    
    # Predict
    result = await predict_image(image_bytes)
    
    return result

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": num_classes
    }

# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000