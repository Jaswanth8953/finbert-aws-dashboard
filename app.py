import streamlit as st
import pandas as pd
import plotly.express as px
import feedparser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime
import time

# Load model once and cache it
@st.cache_resource
def get_model():
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

# Get news from RSS feed
def get_news_from_feed(url):
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            articles.append({
                'title': entry.title,
                'link': entry.link,
                'source': url
            })
        return articles
    except:
        return []

# Analyze sentiment
def check_sentiment(articles, model, tokenizer):
    if not articles:
        return articles

    titles = [a['title'] for a in articles]
    inputs = tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**inputs)

    sentiments = torch.argmax(output.logits, dim=1)
    labels = ['positive', 'negative', 'neutral']

    for i, article in enumerate(articles):
        article['sentiment'] = labels[sentiments[i]]

    return articles


st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Financial News Sentiment Analysis")

# Sidebar setup
st.sidebar.header("Settings")

# RSS feeds list
feeds = [
    'http://feeds.marketwatch.com/marketwatch/topstories/',
    'https://feeds.cnbc.com/cnbc/snw/',
    'https://feeds.reuters.com/news/finance',
]

st.sidebar.write(f"Analyzing {len(feeds)} RSS feeds")

# Main app
if st.sidebar.button("Analyze Now"):
    st.write("Loading model...")
    model, tokenizer = get_model() #finbert

    all_articles = []

    # Get articles from all feeds
    st.write("Fetching articles...")
    for feed_url in feeds:
        articles = get_news_from_feed(feed_url)
        all_articles.extend(articles)

    # Check sentiment
    st.write(f"Analyzing {len(all_articles)} articles...")
    articles_with_sentiment = check_sentiment(all_articles, model, tokenizer)

    # Make dataframe
    df = pd.DataFrame(articles_with_sentiment)

    # Show stats
    st.subheader("Results")
    col1, col2, col3, col4 = st.columns(4)

    pos = len(df[df['sentiment'] == 'positive'])
    neg = len(df[df['sentiment'] == 'negative'])
    neu = len(df[df['sentiment'] == 'neutral'])

    col1.metric("Total", len(df))
    col2.metric("Positive ", pos)
    col3.metric("Negative ", neg)
    col4.metric("Neutral ", neu)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Charts", "Articles", "Download"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            sentiment_count = df['sentiment'].value_counts()
            # st.write("dataframe shape:", df.shape)
            # st.write(df.head(10))
            fig = px.pie(names=sentiment_count.index, values=sentiment_count.values,
                        title="Sentiment Distribution",
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bar chart
            # st.write("dataframe shape:", df.shape)
            # st.write(df.head(10))
            fig = px.bar(x=sentiment_count.index, y=sentiment_count.values,
                        title="Count",
                        color=sentiment_count.index,
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### Articles")

        # Filter by sentiment
        sentiment_filter = st.multiselect("Filter by sentiment",
                                         options=['positive', 'negative', 'neutral'],
                                         default=['positive', 'negative', 'neutral'])

        filtered = df[df['sentiment'].isin(sentiment_filter)]

        for idx, row in filtered.iterrows():
            emoji = {'positive': '+++', 'negative': '---', 'neutral': '____'}[row['sentiment']]
            st.write(f"{emoji} **{row['title'][:100]}**")
            st.write(f"Sentiment: {row['sentiment']} | [Read →]({row['link']})")
            st.write("")

    with tab3:
        # Download
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="sentiment.csv", mime="text/csv")

        # Show table
        st.write("### Data")
        st.dataframe(df, use_container_width=True)
