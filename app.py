import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import re

# Load FinBERT model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

# Financial negative keywords (custom override)
negative_keywords = [
    "decline", "decrease", "drop", "fall", "weak", "loss", "pressure", "delay",
    "headwind", "slowdown", "inflation", "higher cost", "margin pressure",
    "underperformed", "reduced", "cut", "shortfall", "challenging", "sluggish", "impact"
]

# Clean the text
def preprocess_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Hybrid chunk-based analyzer
def analyze_transcript_hybrid(text, chunk_size=512):
    sentences = text.split(".")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())

    summary = {"positive": 0, "neutral": 0, "negative": 0}

    for chunk in chunks:
        lower_chunk = chunk.lower()

        # Rule-based check
        if any(neg_word in lower_chunk for neg_word in negative_keywords):
            summary["negative"] += 1
            continue

        # FinBERT prediction
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        labels = ['positive', 'negative', 'neutral']
        sentiment = labels[torch.argmax(probs)]
        summary[sentiment] += 1

    return summary

# Streamlit UI
st.set_page_config(page_title="Concall Sentiment Analyzer", layout="centered")
st.title("📊 Earnings Concall Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload Concall Transcript (.txt format)", type="txt")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    clean_text = preprocess_text(raw_text)

    st.subheader("Sentiment Analysis In Progress...")
    with st.spinner("Analyzing sentiment by segment..."):
        sentiment_counts = analyze_transcript_hybrid(clean_text)
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    st.success(f"📌 Overall Sentiment: **{dominant_sentiment.upper()}**")

    st.write("🔍 Sentiment Breakdown:")
    st.json(sentiment_counts)

    with st.expander("📄 View Full Transcript"):
        st.write(raw_text)
