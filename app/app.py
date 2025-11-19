import streamlit as st
import pickle
from newspaper import Article
from youtube_transcript_api import YouTubeTranscriptApi
import instaloader

# Load model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ------------------------------
# Extract text from Website URL
# ------------------------------
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

# ------------------------------
# Extract text from YouTube link
# ------------------------------
def extract_youtube_text(url):
    try:
        if "watch?v=" in url:
            video_id = url.split("v=")[-1]
        else:
            video_id = url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])
        return full_text
    except:
        return None

# ------------------------------
# Extract text from Instagram link
# ------------------------------
def extract_instagram_caption(url):
    try:
        L = instaloader.Instaloader()
        shortcode = url.split("/")[-2]
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return post.caption
    except:
        return None

# ------------------------------
# Predict Fake or Real
# ------------------------------
def predict_news(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return pred, prob

# ------------------------------
# Streamlit Web App UI
# ------------------------------
st.title("Real-Time Fake News Detector")
st.write("Paste any *news link* (Website, YouTube, Instagram) or enter text manually.")

# URL Input
url = st.text_input("Paste a link here")

if st.button("Check Link"):
    extracted_text = None

    if "youtube.com" in url or "youtu.be" in url:
        extracted_text = extract_youtube_text(url)
    elif "instagram.com" in url:
        extracted_text = extract_instagram_caption(url)
    else:
        extracted_text = extract_text_from_url(url)

    if extracted_text:
        pred, prob = predict_news(extracted_text)
        st.subheader("Prediction Result")

        if pred == 1:
            st.success(f"REAL NEWS | Confidence: {prob:.2f}")
        else:
            st.error(f"FAKE NEWS | Confidence: {prob:.2f}")
    else:
        st.error("Unable to extract text from this link.")

st.write("---")

# Manual Text Input
st.subheader("Or Enter News Text Manually")
text_input = st.text_area("Paste news text here")

if st.button("Check Text"):
    if text_input.strip():
        pred, prob = predict_news(text_input)
        st.subheader("Prediction Result")

        if pred == 1:
            st.success(f"REAL NEWS | Confidence: {prob:.2f}")
        else:
            st.error(f"FAKE NEWS | Confidence: {prob:.2f}")
    else:
        st.error("Please enter some text.")
