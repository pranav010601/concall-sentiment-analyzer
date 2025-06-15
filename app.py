import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import re

st.set_page_config(page_title="Concall Sentiment Analyzer", layout="wide")
st.title("📊 Earnings Call Sentiment Analyzer (FinBERT)")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    labels = ['positive', 'negative', 'neutral']
    sentiment = labels[torch.argmax(probs)]
    scores = dict(zip(labels, [round(p.item(), 3) for p in probs[0]]))
    return sentiment, scores

uploaded_file = st.file_uploader("Upload a Concall Transcript (.txt)", type=["txt"])

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    clean_text = preprocess_text(raw_text)
    
    st.subheader("Analyzing Sentiment...")
    sentiment, scores = analyze_sentiment(clean_text)

    st.success(f"📌 Overall Sentiment: **{sentiment.upper()}**")
    st.write("Confidence Scores:")
    st.json(scores)

    with st.expander("📄 View Full Transcript"):
        st.write(raw_text)
