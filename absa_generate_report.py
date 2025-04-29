# absa_generate_report.py

import os
import pandas as pd
from datetime import datetime

# --- SETTINGS ---

def load_data():
    """Load surveillance sentiment analysis results."""
    df = pd.read_csv('outputs/surveillance_sentiment_analysis.csv')
    return df

def save_report(text):
    """Save the report text into a file."""
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    filename = f'reports/surveillance_sentiment_report.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Report saved successfully to {filename}")

# --- REPORT GENERATOR ---

def generate_report(df):
    """Generate a clean statistical report."""
    lines = []

    def add(line=""):
        print(line)  # Also print to console
        lines.append(line)

    # Header
    add("="*80)
    add("SURVEILLANCE MEDIA SENTIMENT ANALYSIS REPORT")
    add(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("="*80)
    add()

    # Basic stats
    total = len(df)
    western = len(df[df['source'] == 'Western'])
    chinese = len(df[df['source'] == 'Chinese'])

    add(f"Total Articles Analyzed: {total}")
    add(f"  - Western Articles: {western}")
    add(f"  - Chinese Articles: {chinese}")
    add()

    # Average sentiment
    avg_total = df['sentiment_surveillance'].mean()
    avg_west = df[df['source'] == 'Western']['sentiment_surveillance'].mean()
    avg_china = df[df['source'] == 'Chinese']['sentiment_surveillance'].mean()

    add(f"Overall Average Sentiment (Surveillance-focused): {avg_total:.4f}")
    add(f"  - Western Media: {avg_west:.4f}")
    add(f"  - Chinese Media: {avg_china:.4f}")
    add()

    # Sentiment distributions
    add("Descriptive Statistics (Overall Sentiment):")
    desc = df['sentiment_surveillance'].describe().round(4)
    add(str(desc))
    add()

    # Top 5 Positive Articles
    add("Top 5 Most Positive Articles:")
    top_pos = df.sort_values('sentiment_surveillance', ascending=False).head(5)
    for idx, row in top_pos.iterrows():
        add(f"  {row['sentiment_surveillance']:.3f}  - {row['source']}  - {row['url']}")
    add()

    # Top 5 Most Negative Articles
    add("Top 5 Most Negative Articles:")
    top_neg = df.sort_values('sentiment_surveillance', ascending=True).head(5)
    for idx, row in top_neg.iterrows():
        add(f"  {row['sentiment_surveillance']:.3f}  - {row['source']}  - {row['url']}")
    add()

    # Notes
    add("="*80)
    add("Notes:")
    add("- Sentiment scores range from roughly -1 (very negative) to +1 (very positive).")
    add("- Positive scores suggest surveillance framed positively (e.g., safety, security).")
    add("- Negative scores suggest surveillance framed critically (e.g., control, repression).")
    add("- Neutral scores suggest either balanced or vague coverage.")
    add("="*80)

    report_text = "\n".join(lines)
    save_report(report_text)

# --- MAIN ---

if __name__ == "__main__":
    df = load_data()

    if df.empty:
        print("No data found. Please make sure you've run the ABSA analysis first.")
    else:
        generate_report(df)
