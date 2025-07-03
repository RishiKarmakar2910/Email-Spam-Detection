import streamlit as st
import joblib
import re
import pandas as pd

# --- Load model and vectorizer ---
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Page setup ---
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# --- Sidebar ---
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app classifies SMS or Email messages as **Spam** or **Not Spam** using a Naive Bayes model trained on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
    """
)
st.sidebar.markdown("### 💻 Model Info")
st.sidebar.markdown("- Algorithm: Naive Bayes  \n- Vectorizer: TF-IDF  \n- Accuracy: 96.6%")

# --- Header ---
st.title("📧 SMS / Email Spam Detector")
st.markdown("Paste your message below to check if it's **Spam** or **Not Spam**.")

# --- Session state to manage sample input ---
if "input" not in st.session_state:
    st.session_state["input"] = ""

# --- Sample buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🔍 Try Sample Spam"):
        st.session_state["input"] = "Congratulations! You've won a free iPhone. Click here to claim now!"

with col2:
    if st.button("✅ Try Sample Ham"):
        st.session_state["input"] = "Hi John, are we still meeting at 3pm today?"

# --- Text input area ---
input_text = st.text_area("📨 Enter message here:", value=st.session_state["input"], height=150)

# --- Clean text function ---
def clean_text(text):
    return re.sub(r"\W", " ", text.lower())

# --- Prediction logic ---
if input_text.strip():
    cleaned_text = clean_text(input_text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]
    spam_prob = probs[1]
    ham_prob = probs[0]

    st.markdown("### 🧹 Cleaned Text")
    st.code(cleaned_text, language='text')

    st.markdown("### 📊 Prediction")
    if prediction == 1:
        st.error(f"🚫 This message is **Spam**")
    else:
        st.success(f"✅ This message is **Not Spam**")

    st.markdown("### 📈 Confidence Score")
    st.progress(spam_prob)
    st.write(f"**Spam Probability:** {spam_prob*100:.2f}%")
    st.write(f"**Not Spam Probability:** {ham_prob*100:.2f}%")
else:
    st.info("Enter a message above to get a prediction.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Rishi Karmakar 2024")
