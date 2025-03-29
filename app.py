import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load and Clean Data
def load_data():
    df = pd.read_csv("Amazon Reviews.csv")
    df = df[['product_name', 'rating', 'review_text', 'is_verified', 'review_country', 'brand']]
    df.dropna(subset=['review_text'], inplace=True)
    df['brand'].fillna("Unknown", inplace=True)
    df['review_country'].fillna("Unknown", inplace=True)
    return df

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else ("Negative" if analysis.sentiment.polarity < 0 else "Neutral")

# Fake Review Detection (Simple TF-IDF & RandomForest Model)
def detect_fake_reviews(df):
    vectorizer = TfidfVectorizer(max_features=500)
    model = RandomForestClassifier()
    
    # Generating dummy labels for training (for demonstration)
    np.random.seed(42)
    df['is_fake'] = np.random.choice([0, 1], size=len(df))
    
    X = vectorizer.fit_transform(df['review_text'])
    y = df['is_fake']
    
    model.fit(X, y)
    df['Fake Review Prediction'] = model.predict(X)
    df['Fake Review Prediction'] = df['Fake Review Prediction'].apply(lambda x: "Fake" if x == 1 else "Genuine")
    return df

# Main App
def main():
    st.title("📊 Amazon Reviews Sentiment & Fake Review Analysis Dashboard")
    df = load_data()
    df['Sentiment'] = df['review_text'].apply(get_sentiment)
    df = detect_fake_reviews(df)
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    sentiment_filter = st.sidebar.multiselect("Select Sentiments", df['Sentiment'].unique(), default=df['Sentiment'].unique())
    verified_filter = st.sidebar.radio("Verified Purchase", ["All", "Verified", "Not Verified"], index=0)
    country_filter = st.sidebar.multiselect("Select Countries", df['review_country'].unique(), default=df['review_country'].unique())
    fake_filter = st.sidebar.radio("Filter by Fake Reviews", ["All", "Fake", "Genuine"], index=0)
    
    # Apply Filters
    filtered_df = df[df['Sentiment'].isin(sentiment_filter) & df['review_country'].isin(country_filter)]
    if verified_filter == "Verified":
        filtered_df = filtered_df[filtered_df['is_verified']]
    elif verified_filter == "Not Verified":
        filtered_df = filtered_df[~filtered_df['is_verified']]
    if fake_filter == "Fake":
        filtered_df = filtered_df[filtered_df['Fake Review Prediction'] == "Fake"]
    elif fake_filter == "Genuine":
        filtered_df = filtered_df[filtered_df['Fake Review Prediction'] == "Genuine"]
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    # Word Cloud
    st.subheader("Word Cloud of Reviews")
    text = " ".join(filtered_df['review_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
    
    # Display Reviews
    st.subheader("Sample Reviews")
    st.dataframe(filtered_df[['product_name', 'brand', 'rating', 'review_text', 'Sentiment', 'is_verified', 'review_country', 'Fake Review Prediction']].head(10))
    
if __name__ == "__main__":
    main()
