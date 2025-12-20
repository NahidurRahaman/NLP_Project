import streamlit as st
import requests

# Page config
st.set_page_config(page_title="English to Bangla", page_icon="🌐")

# Title
st.title("🌐 English to Bangla Translator")

# API URL input
api_url = st.text_input(
    "Enter API URL:",
    value="https://nahidur415-translator.hf.space",
    help="Enter your translation API endpoint"
)

# Remove trailing slash
if api_url:
    api_url = api_url.rstrip('/')

# Test connection
if st.button("Test Connection"):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ Cannot connect")

st.divider()

# Translation section
st.subheader("Translate Text")

# Text input
text = st.text_area("Enter English text:", height=100)

# Translate button
if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter text")
    elif not api_url:
        st.warning("Please enter API URL")
    else:
        try:
            # Call API
            response = requests.post(
                f"{api_url}/translate",
                json={"text": text},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Translation Complete")

                # Show results
                st.markdown("**English:**")
                st.write(text)

                st.markdown("**Bangla:**")
                st.write(result.get("translation", ""))
            else:
                st.error(f"Error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection failed: {str(e)}")

st.divider()

# Batch translation
st.subheader("Batch Translation")

batch_text = st.text_area(
    "Enter English texts (one per line):",
    height=150,
    placeholder="Hello\nGood morning\nThank you"
)

if st.button("Translate Batch"):
    if not batch_text.strip():
        st.warning("Please enter texts")
    elif not api_url:
        st.warning("Please enter API URL")
    else:
        texts = [line.strip() for line in batch_text.split('\n') if line.strip()]

        if texts:
            try:
                response = requests.post(
                    f"{api_url}/translate/batch",
                    json={"texts": texts},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Translated {len(texts)} texts")

                    # Show results in table
                    import pandas as pd

                    df = pd.DataFrame({
                        "English": texts,
                        "Bangla": result.get("translations", [])
                    })
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"Error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection failed: {str(e)}")