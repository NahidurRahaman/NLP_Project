// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const cameraBtn = document.getElementById('cameraBtn');
const cameraPreview = document.getElementById('cameraPreview');
const cameraStream = document.getElementById('cameraStream');
const captureBtn = document.getElementById('captureBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const predictionsList = document.getElementById('predictionsList');
const topDisease = document.getElementById('topDisease');
const confidenceBar = document.getElementById('confidenceBar');
const topDescription = document.getElementById('topDescription');

let stream = null;
let capturedImage = null;

// Event Listeners
imageInput.addEventListener('change', handleImageUpload);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'rgba(40, 167, 69, 0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = 'rgba(40, 167, 69, 0.05)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'rgba(40, 167, 69, 0.05)';
    
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

cameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', captureImage);
stopCameraBtn.addEventListener('click', stopCamera);
analyzeBtn.addEventListener('click', analyzeImage);

// Functions
function handleImageUpload(e) {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        alert('File size should be less than 5MB');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('d-none');
        capturedImage = file;
        analyzeBtn.disabled = false;
        stopCamera(); // Stop camera if running
    };
    reader.readAsDataURL(file);
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        cameraStream.srcObject = stream;
        cameraPreview.classList.remove('d-none');
        cameraBtn.classList.add('d-none');
    } catch (err) {
        alert('Camera access denied or not available');
        console.error('Camera error:', err);
    }
}

function captureImage() {
    const canvas = document.createElement('canvas');
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraStream, 0, 0);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        handleFile(file);
        stopCamera();
    }, 'image/jpeg', 0.9);
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        cameraPreview.classList.add('d-none');
        cameraBtn.classList.remove('d-none');
        stream = null;
    }
}

async function analyzeImage() {
    if (!capturedImage) {
        alert('Please upload an image first');
        return;
    }

    // Show loading
    loadingSpinner.classList.remove('d-none');
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', capturedImage);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error('Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed. Please try again.');
    } finally {
        loadingSpinner.classList.add('d-none');
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    // Update top prediction
    topDisease.textContent = data.top_disease;
    topDescription.textContent = data.all_predictions[0].description;
    
    // Animate progress bar
    confidenceBar.style.width = `${data.top_confidence}%`;
    confidenceBar.textContent = `${data.top_confidence}%`;
    
    // Update progress bar color based on confidence
    if (data.top_confidence >= 70) {
        confidenceBar.className = 'progress-bar bg-danger';
    } else if (data.top_confidence >= 30) {
        confidenceBar.className = 'progress-bar bg-warning';
    } else {
        confidenceBar.className = 'progress-bar bg-success';
    }
    
    // Display all predictions
    predictionsList.innerHTML = '';
    data.all_predictions.forEach(pred => {
        const card = document.createElement('div');
        card.className = 'card prediction-card';
        
        card.innerHTML = `
            <div class="card-body">
                <div class="row align-items-center">
                    <div class="col-md-3">
                        <h5 class="card-title mb-0">${pred.disease}</h5>
                    </div>
                    <div class="col-md-3">
                        <div class="progress">
                            <div class="progress-bar bg-${pred.risk_color}" 
                                 role="progressbar" 
                                 style="width: ${pred.confidence}%">
                                ${pred.confidence}%
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <span class="risk-badge risk-${pred.risk_color}">
                            ${pred.risk_icon} ${pred.risk_level}
                        </span>
                    </div>
                    <div class="col-md-3">
                        <p class="text-muted mb-0 small">${pred.description}</p>
                    </div>
                </div>
            </div>
        `;
        
        predictionsList.appendChild(card);
    });
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check API health
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            console.log('API Health:', data);
            if (!data.model_loaded) {
                alert('Warning: AI model is not loaded properly');
            }
        })
        .catch(err => console.error('Health check failed:', err));
});