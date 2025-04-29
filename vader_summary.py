# vader_report.py

import pandas as pd
from datetime import datetime
import os

def load_data():
    return pd.read_csv('outputs/vader_analysis.csv')

def save_report(text):
    if not os.path.exists('reports'):
        os.makedirs('reports')
    with open('reports/vader_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(text)

def generate_report(df):
    lines = []
    lines.append('='*80)
    lines.append('SURVEILLANCE MEDIA VADER ANALYSIS REPORT')
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('='*80)

    lines.append(f"Total Articles: {len(df)}")
    lines.append(f"Western Articles: {len(df[df['source'] == 'Western'])}")
    lines.append(f"Chinese Articles: {len(df[df['source'] == 'Chinese'])}")
    lines.append("")

    lines.append("Sentiment Statistics:")
    lines.append(df['sentiment'].describe().to_string())
    lines.append("")

    lines.append("Western Sentiment:")
    lines.append(df[df['source'] == 'Western']['sentiment'].describe().to_string())
    lines.append("")

    lines.append("Chinese Sentiment:")
    lines.append(df[df['source'] == 'Chinese']['sentiment'].describe().to_string())
    lines.append("")

    lines.append('='*80)
    return "\n".join(lines)

def main():
    df = load_data()
    report = generate_report(df)
    save_report(report)
    print('Summary report created in reports/vader_summary_report.txt')

if __name__ == "__main__":
    main()
