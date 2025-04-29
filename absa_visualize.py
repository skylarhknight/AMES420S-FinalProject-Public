# absa_visualize.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

# --- LOAD DATA ---

def load_data():
    df = pd.read_csv('outputs/surveillance_sentiment_analysis.csv')
    return df

# --- SAVE FIGURE UTILITY ---

def save_figure(name):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/{name}.png', bbox_inches='tight', dpi=300)

# --- VISUALIZATIONS ---

def plot_average_sentiment(df):
    avg_sentiment = df.groupby('source')['sentiment_surveillance'].mean().reset_index()

    plt.figure(figsize=(8,6))
    sns.barplot(x='source', y='sentiment_surveillance', data=avg_sentiment, palette='Set2', edgecolor='black')
    
    plt.title('Average Surveillance Sentiment by Source', fontsize=20, fontweight='bold')
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xlabel('')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure('average_sentiment_by_source')
    plt.show()

def plot_sentiment_violin(df):
    plt.figure(figsize=(10,6))
    sns.violinplot(
        y='source', x='sentiment_surveillance',
        data=df,
        palette='Set2',
        inner='box'
    )
    plt.title('Sentiment Distribution (Violin Plot)', fontsize=20, fontweight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=14)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    save_figure('sentiment_violin_plot')
    plt.show()

def plot_sentiment_boxplot(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(
        y='source', x='sentiment_surveillance',
        data=df,
        palette='Set2'
    )
    plt.title('Sentiment Distribution (Box and Whisker)', fontsize=20, fontweight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=14)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    save_figure('sentiment_box_plot')
    plt.show()

def plot_individual_articles(df):
    # Sort to make it prettier
    df_sorted = df.sort_values('sentiment_surveillance')

    # Color mapping
    color_map = {'Western': '#4C72B0', 'Chinese': '#DD8452'}
    bar_colors = df_sorted['source'].map(color_map)

    plt.figure(figsize=(10,12))
    plt.barh(
        y=range(len(df_sorted)),
        width=df_sorted['sentiment_surveillance'],
        color=bar_colors,
        edgecolor='black'
    )

    plt.title('Individual Article Sentiment on Surveillance', fontsize=20, fontweight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=14)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Remove y-ticks
    plt.yticks([], [])

    # Legend manually
    import matplotlib.patches as mpatches
    west_patch = mpatches.Patch(color='#4C72B0', label='Western Media')
    china_patch = mpatches.Patch(color='#DD8452', label='Chinese Media')
    plt.legend(handles=[west_patch, china_patch], loc='lower right', fontsize=12)

    plt.tight_layout()
    save_figure('individual_article_sentiment')
    plt.show()

def plot_sentiment_histogram(df):
    plt.figure(figsize=(10,6))
    sns.histplot(
        df['sentiment_surveillance'],
        bins=20,
        kde=True,
        color='#4C72B0'
    )
    plt.title('Overall Sentiment Histogram', fontsize=20, fontweight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    save_figure('sentiment_histogram')
    plt.show()

# --- MAIN ---

if __name__ == "__main__":
    df = load_data()

    if df.empty:
        print("No data found. Please run the ABSA analysis first.")
    else:
        plot_average_sentiment(df)
        plot_sentiment_violin(df)
        plot_sentiment_boxplot(df)
        plot_individual_articles(df)
        plot_sentiment_histogram(df)

        print("Visualizations complete! All figures saved in 'figures/' folder.")
