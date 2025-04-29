# absa_analysis_clean.py

# Import packages
import os
import re
import pandas as pd
import torch
import torch.nn.functional as F  # <-- NEW (for softmax correction)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup

# --- SETUP ---

# Model for Aspect-Based Sentiment Analysis
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

# --- FUNCTIONS ---

def load_links(filepath):
    """Load links and source labels from a CSV."""
    df = pd.read_csv(filepath, names=['link', 'source'])
    df = df.drop_duplicates(subset='link')
    return df

def extract_text(url):
    """Scrape main article text from a URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ' '.join([p.get_text() for p in paragraphs])
        return article.strip()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ''

def predict_sentiment(text, aspect="surveillance"):
    """Predict sentiment focused on a specific aspect (corrected version)."""
    prompt = f"[CLS] {aspect} [SEP] {text} [SEP]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()

    # --- Apply Softmax to convert logits into probabilities ---
    probs = F.softmax(logits, dim=0).cpu().numpy()

    # Probabilities: [negative, neutral, positive]
    negative_prob = probs[0]
    neutral_prob = probs[1]
    positive_prob = probs[2]

    # Continuous sentiment score: weighted from -1 to 1
    score = (-1) * negative_prob + (0) * neutral_prob + (1) * positive_prob

    return score

def analyze_articles(df_links):
    """Analyze each article for surveillance-related sentiment."""
    records = []

    for idx, row in df_links.iterrows():
        url = row['link']
        source = row['source']

        print(f"Processing ({source}): {url}")

        text = extract_text(url)
        if not text or len(text) < 50:
            continue

        # Predict surveillance-specific sentiment
        sentiment_score = predict_sentiment(text, aspect="surveillance")

        records.append({
            "source": source,
            "url": url,
            "sentiment_surveillance": sentiment_score,
            "text": text
        })

    return pd.DataFrame(records)

def save_results(df):
    """Save results to CSV and export summary report."""
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    df.to_csv("outputs/surveillance_sentiment_analysis.csv", index=False)
    print("Data saved to 'outputs/surveillance_sentiment_analysis.csv'.")

    with open("outputs/summary_report.txt", "w", encoding="utf-8") as f:
        total_articles = len(df)
        avg_sentiment = df['sentiment_surveillance'].mean()
        f.write("SURVEILLANCE MEDIA ANALYSIS REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Total Articles Analyzed: {total_articles}\n")
        f.write(f"Average Surveillance Sentiment Score: {avg_sentiment:.4f}\n\n")
        
        f.write("Western Media Average: {:.4f}\n".format(
            df[df['source'] == 'Western']['sentiment_surveillance'].mean()))
        f.write("Chinese Media Average: {:.4f}\n".format(
            df[df['source'] == 'Chinese']['sentiment_surveillance'].mean()))
        f.write("\n")
        
        f.write("Top Positive Surveillance Article:\n")
        f.write(df.loc[df['sentiment_surveillance'].idxmax()]['url'] + "\n\n")

        f.write("Top Negative Surveillance Article:\n")
        f.write(df.loc[df['sentiment_surveillance'].idxmin()]['url'] + "\n\n")
    
    print("Summary report saved to 'outputs/summary_report.txt'.")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    df_links = load_links("article_links.csv")
    df_results = analyze_articles(df_links)

    if df_results.empty:
        print("No valid articles processed.")
    else:
        save_results(df_results)
        print("Analysis complete!")
