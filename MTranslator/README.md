---
title: 🌐 AI Multi-Language Translator
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🌐 AI Multi-Language Translator

Translate text between 50+ languages using Facebook's mBART-50 model. This application provides professional-grade translation with support for multiple language pairs.

## 🚀 Features

- **50+ Languages**: Translate between English, Bangla, Hindi, Arabic, French, German, Chinese, Korean and more
- **Real-time Translation**: Instant translation with AI-powered accuracy
- **User-Friendly Interface**: Clean and intuitive design for easy use
- **Professional Quality**: Uses state-of-the-art mBART-50 model
- **Open Source**: Completely free to use and modify

## 📋 Supported Languages

| Language Code | Language Name | Country Flag |
|--------------|---------------|--------------|
| `en_XX` | English | 🇺🇸 |
| `bn_IN` | Bangla | 🇧🇩 |
| `hi_IN` | Hindi | 🇮🇳 |
| `ar_AR` | Arabic | 🇸🇦 |
| `fr_XX` | French | 🇫🇷 |
| `de_DE` | German | 🇩🇪 |
| `zh_CN` | Chinese | 🇨🇳 |
| `ko_KR` | Korean | 🇰🇷 |
| `ja_XX` | Japanese | 🇯🇵 |
| `es_XX` | Spanish | 🇪🇸 |
| `ru_RU` | Russian | 🇷🇺 |
| `pt_XX` | Portuguese | 🇵🇹 |
| ...and 40+ more | | |

## 🎯 How to Use

1. **Enter Text**: Type or paste your text in the input box
2. **Select Languages**: Choose source and target languages from dropdown menus
3. **Click Translate**: Press the translate button to get instant results
4. **View Translation**: See the translated text displayed below

### 📝 Example Translations

- **English → Bangla**: "Hello, how are you?" → "হ্যালো, আপনি কেমন আছেন?"
- **Bangla → English**: "আপনার নাম কি?" → "What is your name?"
- **English → Hindi**: "Thank you very much" → "बहुत बहुत धन्यवाद"
- **English → Arabic**: "Good morning" → "صباح الخير"

## 🏗️ Technical Details

### Model Information
- **Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Architecture**: mBART-50 (Multilingual BART)
- **Parameters**: 610 million
- **Training Data**: 50 languages, 25K sentences per language
- **Size**: ~1.7 GB

### Technology Stack
- **Frontend**: Streamlit
- **Backend**: PyTorch, Transformers
- **Model**: Hugging Face Transformers
- **Deployment**: Hugging Face Spaces

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- 8GB+ RAM (recommended)
- 2GB+ free disk space

### Installation
```bash
# Clone the repository
git clone https://huggingface.co/spaces/your-username/translator-app

# Navigate to directory
cd translator-app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
