from transformers import MBartForConditionalGeneration,MBart50TokenizerFast
import torch
import streamlit as st

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def translate_text(text, source_lang, target_lang):
    # set the language codes
    tokenizer.src_lang = source_lang
    encoded_text = tokenizer(text, return_tensors="pt")

    # generated the translation
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])

    # Decode the output
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

# Page configuration FIRST

st.title("Language Translation...")
st.markdown("Translation text in multiple languages using AI-powered machine learning Model from huggingface...")
# Input text box
text = st.text_area("Enter text to translate", "")

# Language selection
# Language selection - UNIQUE KEYS ADDED
source_language = st.selectbox(
    "Select source language",
    ["en_XX", "de_DE", "fr_FR", "fr_XX", "id_ID", "ar_AR", "bn_IN", "hi_IN", "ko_KR", "zh_CN"],
    key="source_lang_select"  # ✅ UNIQUE KEY
)

target_language = st.selectbox(
    "Select target language",  # ✅ Fixed label
    ["en_XX", "de_DE", "fr_FR", "fr_XX", "id_ID", "ar_AR", "bn_IN", "hi_IN", "ko_KR", "zh_CN"],
    key="target_lang_select"  # ✅ UNIQUE KEY
)

if st.button("Translate", key="translate_button"):
    if text.strip() == "":
        st.warning("Please enter a text to translate")
    else:
        with st.spinner("Translating..."):
            translation = translate_text(text, source_language, target_language)

            st.success("Translation complete!")
            st.write("## Translated Text:")
            st.write(translation)

# Example section
st.sidebar.header("Examples")
st.sidebar.markdown("**Try Translating!**")
st.sidebar.markdown("- 'Hello, how are you?' from English")
st.sidebar.markdown("- 'content ca va?' from French")

# Footer
st.markdown("___")
st.markdown("Built with using (Hugging Face)(https://huggingface.co/) and (Streamlit)(https://streamlit.io/)")