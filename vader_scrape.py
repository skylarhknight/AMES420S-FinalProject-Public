# vader_scrape.py

import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Setup
sia = SentimentIntensityAnalyzer()
keywords = [...]   # (Paste your full keywords list here)
pop_culture_terms = [...]  # (Paste your pop culture list here)

def load_links(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset='link')
    return df

def extract_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except:
        return ''

def analyze_articles(df_links):
    records = []
    for idx, row in df_links.iterrows():
        text = extract_text(row['link'])
        if len(text) < 50:
            continue
        sentiment = sia.polarity_scores(text)['compound']
        word_counts = Counter(re.findall(r'\w+', text.lower()))
        keyword_counts = {kw: word_counts.get(kw, 0) for kw in keywords}
        records.append({**keyword_counts, 'sentiment': sentiment, 'source': row['source'], 'url': row['link'], 'text': text})
    return pd.DataFrame(records)

def main():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    df_links = load_links('article_links.csv')
    df = analyze_articles(df_links)
    df.to_csv('outputs/vader_analysis.csv', index=False)
    print('Scraping and analysis complete. Saved to outputs/vader_analysis.csv')

if __name__ == "__main__":
    main()
