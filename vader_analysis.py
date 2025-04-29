# Import Packages
import os
import re
import nltk
import requests
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from datetime import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import matplotlib.patches as mpatches
from nltk.sentiment import SentimentIntensityAnalyzer

import matplotlib.colors as mcolors
import matplotlib.cm as cm
scientific_cmap = cm.cividis

# --- SETTINGS ---

# Download VADER lexicon - only needed once, after that comment out
#nltk.download('vader_lexicon')

# Initialize VADER analyzer
sia = SentimentIntensityAnalyzer()

# Global Visualization Settings
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

# SETUP ====================================================

# Keywords
keywords = [
    'surveillance', 'monitoring', 'privacy', 'security', 'freedom', 'control',
    'oversight', 'censorship', 'safety', 'regulation', 'governance', 'threat',
    'authoritarian', 'intrusive', 'transparency', 'repression',
    'human rights', 'stability', 'efficiency', 'AI', 'facial recognition',
    'big data', 'social credit', 'cybersecurity', 'biometrics',
    'enforcement', 'dissent', 'propaganda', 'public safety',
    'national security', 'digital rights', 'internet freedom',
    'online tracking', 'mass surveillance', 'civil liberties',
    'tyranny', 'data privacy', 'geolocation', 'biometric data',
    'data collection', 'state control', 'social stability', 'digital authoritarianism'
]

# Pop culture references
pop_culture_terms = [
    'big brother', 'minority report', 'orwellian'
]


# --- FUNCTIONS ---

def load_links_from_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset='link')
    western_links = df[df['source'] == 'Western']['link'].dropna().tolist()
    chinese_links = df[df['source'] == 'Chinese']['link'].dropna().tolist()
    return western_links, chinese_links

def extract_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ' '.join([p.get_text() for p in paragraphs])
        return article
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ''

def get_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound']

def analyze_articles(article_links, label):
    data = []
    for link in article_links:
        text = extract_text(link)
        if len(text) > 50:
            word_counts = Counter(re.findall(r'\w+', text.lower()))
            keyword_counts = {kw: word_counts.get(kw, 0) for kw in keywords}
            sentiment = get_sentiment(text)
            data.append({**keyword_counts, 'sentiment': sentiment, 'source': label, 'url': link, 'text': text})
    return data

def gather_all():
    western_articles, chinese_articles = load_links_from_csv('article_links.csv')
    western_data = analyze_articles(western_articles, 'Western')
    chinese_data = analyze_articles(chinese_articles, 'Chinese')
    return pd.DataFrame(western_data + chinese_data)

def save_to_csv(df):
    df.to_csv('articles_analysis.csv', index=False)
    print('Data saved to articles_analysis.csv')

def save_figure(name):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/{name}.png', bbox_inches='tight', dpi=300)
    
    
# --- VISUALIZATIONS ---

def visualize_keyword_comparison(df):
    selected_keywords = ['security', 'safety', 'privacy', 
                         'control', 'freedom', 'threat', 
                         'control', 'authoritarian', 'tyranny']
    
    summary = df.groupby('source')[selected_keywords].mean()
    
    colors = sns.color_palette("rocket", n_colors=len(selected_keywords))

    summary.plot(kind='bar', figsize=(12,8), color=colors)
    plt.title('Keyword Frequency Comparison', fontsize=18, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Average Count', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', color='#cccccc', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    save_figure('keyword_comparison')
    plt.show()

def visualize_keyword_comparison_by_keyword(df):
    selected_keywords = [
        'security', 'safety', 'privacy', 
        'control', 'freedom', 'threat', 
        'authoritarian', 'tyranny', 'monitoring',
        'censorship', 'stability', 'transparency'
    ]

    # Prepare data for plotting
    df_melted = df.melt(
        id_vars='source',
        value_vars=selected_keywords,
        var_name='keyword',
        value_name='count'
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_melted,
        x='keyword',
        y='count',
        hue='source',
        palette='Set2',
        edgecolor=None
    )

    plt.title('Keyword Frequency Comparison by Keyword', fontsize=20, fontweight='bold')
    plt.xlabel('Keyword', fontsize=14)
    plt.ylabel('Average Mention Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Source', fontsize=12, title_fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_figure('keyword_comparison_by_keyword')
    plt.show()
    

def visualize_pop_culture_references(df):
    all_text = ''
    for url in df['url']:
        text = extract_text(url)
        all_text += text.lower() + ' '

    pop_counts = {term: all_text.count(term) for term in pop_culture_terms}

    plt.figure(figsize=(10,6))
    colors = sns.color_palette("muted", n_colors=len(pop_counts))

    plt.bar(pop_counts.keys(), pop_counts.values(), color=colors)
    plt.title('Pop Culture References Related to Surveillance Using VADER Sentiment Analysis', fontsize=18, fontweight='bold')
    plt.xlabel('Reference', fontsize=14)
    plt.ylabel('Mentions', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', color='#cccccc', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    save_figure('pop_culture_references')
    plt.show()



def visualize_sentiment(df):
    sns.set_theme(style="whitegrid")  # <-- Set the theme
    df_grouped = df.groupby('source')['sentiment'].mean()
    colors = sns.color_palette("Set2") 
    
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(10,6))
    bars = plt.bar(df_grouped.index, df_grouped.values, color=colors[:len(df_grouped.index)])
    plt.title('Average Sentiment Comparison Using VADER Sentiment Analysis', fontsize=18, fontweight='bold')
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='-', color='#cccccc', linewidth=0.7, alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.01, f'{yval:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    save_figure('average_sentiment')
    plt.show()

def visualize_individual_sentiments(df):
    plt.figure(figsize=(12,8))
    
    # Professional muted colors
    western_color = '#4C72B0'   # muted blue
    chinese_color = '#DD8452'   # muted orange
    
    colors = df['source'].map({'Western': western_color, 'Chinese': chinese_color})
    
    plt.barh(range(len(df)), df['sentiment'], color=colors, edgecolor='black', linewidth=0.5)

    plt.axvline(x=0, color='black', linewidth=1)
    
    plt.title('Sentiment of Individual Articles Using VADER Sentiment Analysis', fontsize=20, fontweight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=14)
    plt.ylabel('')  # No label for y-axis
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks([], [])  # Remove y-axis ticks

    # Legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color=western_color, label='Western Media')
    orange_patch = mpatches.Patch(color=chinese_color, label='Chinese Media')
    plt.legend(handles=[blue_patch, orange_patch], loc='upper right', fontsize=12, title='Source')

    plt.tight_layout()
    save_figure('individual_sentiments')
    plt.show()


def visualize_avg_sentiment_per_source(df):
    avg_sentiment = df.groupby(['source'])['sentiment'].mean().reset_index()
    source_colors = {'Western': sns.color_palette('rocket')[0], 'Chinese': sns.color_palette('rocket')[2]}

    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(8,6))
    plt.bar(avg_sentiment['source'], avg_sentiment['sentiment'], color=[source_colors[src] for src in avg_sentiment['source']])
    plt.title('Average Sentiment per News Source Using VADER Sentiment Analysis', fontsize=18, fontweight='bold')
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, axis='y', which='major', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_figure('avg_sentiment_per_source')
    plt.show()
    
    
def visualize_sentiment_boxplot(df):
    plt.figure(figsize=(10,6))
    
    plt.rcParams['font.family'] = 'Times New Roman'
    
    
    data = [
        df[df['source'] == 'Western']['sentiment'],
        df[df['source'] == 'Chinese']['sentiment']
    ]
    
    colors = sns.color_palette('Spectral')

    # Make a horizontal boxplot
    box = plt.boxplot(data, vert=False, patch_artist=True, labels=['Western', 'Chinese'],
                      boxprops=dict(facecolor=colors[0], color=colors[0]),
                      capprops=dict(color=colors[0]),
                      whiskerprops=dict(color=colors[0]),
                      flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none'),
                      medianprops=dict(color='black'))

    plt.title('Sentiment Distribution in Western vs Chinese Media Using VADER Sentiment Analysis', fontsize=18, fontweight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=14)
    plt.ylabel('')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', color='#cccccc', linewidth=0.7, alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    save_figure('sentiment_boxplot')
    plt.show()

def visualize_sentiment_violinplot(df):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'

    # Prepare data
    data = [
        df[df['source'] == 'Western']['sentiment'],
        df[df['source'] == 'Chinese']['sentiment']
    ]

    # Create violin plot
    parts = plt.violinplot(
        data,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        vert=False  # Make it horizontal
    )

    # Customize body colors
    for pc in parts['bodies']:
        pc.set_facecolor('#88c0d0')   # Soft blue
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Customize center lines (no loop needed)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)

    if 'cmedians' in parts:
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)

    # Layout adjustments
    plt.yticks([1, 2], ['Western Media', 'Chinese Media'], fontsize=12)
    plt.xlabel('Sentiment Polarity (Negative → Positive)', fontsize=14)
    plt.title('Sentiment Distribution by Media Source Using VADER', fontsize=20, fontweight='bold')

    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)  # Center line

    plt.tight_layout()
    save_figure('sentiment_violinplot')
    plt.show()

    

def co_occurrence_network(df, label):
    # Gather text
    all_text = ''
    for url in df[df['source'] == label]['url']:
        article_text = extract_text(url)
        all_text += article_text.lower() + ' '

    # Tokenize words
    words = re.findall(r'\w+', all_text)
    co_occur = Counter()
    window_size = 5

    for i in range(len(words) - window_size):
        window = words[i:i+window_size]
        for w1 in window:
            for w2 in window:
                if w1 != w2:
                    pair = tuple(sorted([w1, w2]))
                    co_occur[pair] += 1

    # Build graph
    G = nx.Graph()
    for (w1, w2), count in co_occur.items():
        if w1 in keywords and w2 in keywords and count >= 3:
            G.add_edge(w1, w2, weight=count)

    if len(G.nodes) == 0:
        print(f"No strong co-occurrences found for {label}.")
        return

    # Node properties
    keyword_counts = Counter(words)
    node_sizes = [400 + keyword_counts.get(node, 1) * 40 for node in G.nodes]
    node_colors = [keyword_counts.get(node, 1) for node in G.nodes]

    # Layout and figure
    fig, ax = plt.subplots(figsize=(14,12), facecolor='#f7f7f7')
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Rescaled Spectral colormap
    spectral = plt.cm.Spectral
    spectral_rescaled = mcolors.LinearSegmentedColormap.from_list(
        'spectral_rescaled', spectral(np.linspace(0.2, 0.85, 256))
    )

    # Draw edges
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for (u, v, d) in edges]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_weights,
        edge_cmap=plt.cm.coolwarm,
        edge_vmin=min(edge_weights),
        edge_vmax=max(edge_weights),
        width=[0.2 + w*0.3 for w in edge_weights],
        alpha=0.4
    )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=node_colors,
        cmap=spectral_rescaled,
        edgecolors='white',   # Soft white border
        linewidths=1.5,
        alpha=0.95,
        ax=ax
    )

    # Draw black labels with smart scaling
    for node, (x, y) in pos.items():
        freq = keyword_counts.get(node, 1)

        font_size = 5 + 2.2 * np.sqrt(freq)
        font_size = min(font_size, 13)

        ax.text(
            x, y, node,
            fontsize=font_size,
            fontweight='semibold',
            fontfamily='DejaVu Sans',
            color='black',
            ha='center', va='center'
        )

    # Node colorbar
    sm_nodes = plt.cm.ScalarMappable(cmap=spectral_rescaled, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.7, location='right', pad=0.01)
    cbar_nodes.set_label('Keyword Frequency', fontsize=12)

    # Edge colorbar
    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=ax, shrink=0.5, location='bottom', pad=0.05, orientation='horizontal')
    cbar_edges.set_label('Connection Strength (Co-occurrence)', fontsize=12)

    # Title and note
    ax.set_title(f'Keyword Co-occurrence Network ({label})', fontsize=24, fontweight='bold', pad=20)
    ax.text(0.5, -0.08, '(Edge width/color = strength of co-occurrence)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.axis('off')
    plt.tight_layout()
    save_figure(f'co_occurrence_network_{label}')
    plt.show()



    # Colorbars
    sm_nodes = plt.cm.ScalarMappable(cmap=spectral_rescaled, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.7, location='right', pad=0.01)
    cbar_nodes.set_label('Keyword Frequency', fontsize=12)

    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=ax, shrink=0.5, location='bottom', pad=0.05, orientation='horizontal')
    cbar_edges.set_label('Connection Strength (Co-occurrence)', fontsize=12)

    # Titles and Notes
    ax.set_title(f'Keyword Co-occurrence Network ({label})', fontsize=24, fontweight='bold', pad=20)
    ax.text(0.5, -0.08, '(Edge width/color = strength of co-occurrence)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.axis('off')
    plt.tight_layout()
    save_figure(f'co_occurrence_network_{label}')
    plt.show()
    


# SUMMARY REPORT ==========================================
    """
    Prints statistics and metrics from the analysis.
    """


def report_metrics(df):
    report_lines = []

    def add_line(text=""):
        print(text)
        report_lines.append(text)

    if not os.path.exists('reports'):
        os.makedirs('reports')

    add_line("="*90)
    add_line("SURVEILLANCE MEDIA ANALYSIS REPORT")
    add_line(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_line("="*90)

    # --- Basic counts ---
    total_articles = len(df)
    western_articles = len(df[df['source'] == 'Western'])
    chinese_articles = len(df[df['source'] == 'Chinese'])

    add_line(f"Total Articles: {total_articles}")
    add_line(f"  - Western Sources: {western_articles}")
    add_line(f"  - Chinese Sources: {chinese_articles}")
    add_line()

    # --- Word counts ---
    total_words = sum(df['text'].dropna().apply(lambda x: len(x.split())))
    western_words = sum(df[df['source'] == 'Western']['text'].dropna().apply(lambda x: len(x.split())))
    chinese_words = sum(df[df['source'] == 'Chinese']['text'].dropna().apply(lambda x: len(x.split())))

    add_line(f"Total Words Analyzed: {total_words:,}")
    add_line(f"  - Western Words: {western_words:,}")
    add_line(f"  - Chinese Words: {chinese_words:,}")
    add_line()

    # --- Sentiment ---
    overall_sentiment = df['sentiment'].mean()
    western_sentiment = df[df['source'] == 'Western']['sentiment'].mean()
    chinese_sentiment = df[df['source'] == 'Chinese']['sentiment'].mean()

    add_line("Sentiment Averages:")
    add_line(f"  - Overall: {overall_sentiment:.4f}")
    add_line(f"  - Western Media: {western_sentiment:.4f}")
    add_line(f"  - Chinese Media: {chinese_sentiment:.4f}")
    add_line()

    # --- Keyword totals ---
    total_keywords = df[keywords].sum().sum()
    avg_keywords_per_article = total_keywords / total_articles

    western_keywords_total = df[df['source'] == 'Western'][keywords].sum().sum()
    chinese_keywords_total = df[df['source'] == 'Chinese'][keywords].sum().sum()

    western_keywords_avg = western_keywords_total / western_articles if western_articles else 0
    chinese_keywords_avg = chinese_keywords_total / chinese_articles if chinese_articles else 0

    add_line("Keyword Mentions:")
    add_line(f"  - Total Keyword Mentions: {total_keywords:,}")
    add_line(f"    - Western Total: {western_keywords_total:,}")
    add_line(f"    - Chinese Total: {chinese_keywords_total:,}")
    add_line(f"  - Average Keyword Mentions per Article:")
    add_line(f"    - Overall: {avg_keywords_per_article:.2f}")
    add_line(f"    - Western: {western_keywords_avg:.2f}")
    add_line(f"    - Chinese: {chinese_keywords_avg:.2f}")
    add_line()

    # --- Per Keyword Averages ---
    add_line("Average Mentions of Each Keyword per Article:")
    keyword_means = df.groupby('source')[keywords].mean().T.round(3)  # Transpose for easier view

    for keyword in keyword_means.index:
        western_avg = keyword_means.loc[keyword, 'Western'] if 'Western' in keyword_means.columns else 0
        chinese_avg = keyword_means.loc[keyword, 'Chinese'] if 'Chinese' in keyword_means.columns else 0
        overall_avg = df[keyword].mean()
        add_line(f"  - {keyword.title()}:")
        add_line(f"     * Western Avg: {western_avg:.3f}")
        add_line(f"     * Chinese Avg: {chinese_avg:.3f}")
        add_line(f"     * Overall Avg: {overall_avg:.3f}")
    add_line()

    # --- Top Differences ---
    diff_series = (keyword_means['Western'] - keyword_means['Chinese']).sort_values(key=lambda x: abs(x), ascending=False)
    add_line("Top Keywords by Difference Between Western and Chinese Media:")
    for keyword, diff in diff_series.items():
        direction = "Western" if diff > 0 else "Chinese"
        add_line(f"  - {keyword}: {diff:.3f} ({direction}-leaning)")
    add_line()

    add_line("="*90)

    # --- Save file ---
    with open('reports/media_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print("\nDetailed summary report saved to 'reports/media_analysis_report.txt'.")

# VISUALIZATON ==========================================

def visualize_pop_culture_sentiment(df):
    """
    Visualizes pop culture references with sentiment and media source.
    """
    # Prepare dataset
    records = []
    
    for idx, row in df.iterrows():
        text = row['text'].lower()
        sentiment = row['sentiment']
        source = row['source']
        
        for term in pop_culture_terms:
            if term in text:
                records.append({
                    'pop_culture': term.title(),
                    'sentiment_label': 'Positive' if sentiment >= 0 else 'Negative',
                    'source': source
                })
                
    if not records:
        print("No pop culture references found.")
        return
    
    pop_df = pd.DataFrame(records)
    
    # Plot
    plt.figure(figsize=(12,8))
    
    sns.countplot(
        data=pop_df,
        x='pop_culture',
        hue='sentiment_label',
        palette={'Positive': 'seagreen', 'Negative': 'indianred'},
        dodge=True
    )
    
    plt.title('Pop Culture References by Sentiment (VADER)', fontsize=20, fontweight='bold')
    plt.xlabel('Pop Culture Reference', fontsize=14)
    plt.ylabel('Number of Articles', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Sentiment', fontsize=12, title_fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_figure('pop_culture_sentiment')
    plt.show()



# --- MAIN ---

if __name__ == "__main__":
    df = gather_all()
    if df.empty:
        print("No articles found. Please check your URLs.")
    else:
        print(df.head())
        save_to_csv(df)
        
        # Draw Figures
        visualize_keyword_comparison(df)
        visualize_pop_culture_references(df)
        visualize_sentiment(df)
        visualize_individual_sentiments(df)
        visualize_avg_sentiment_per_source(df)
        visualize_sentiment_boxplot(df)
        visualize_sentiment_violinplot(df)
        co_occurrence_network(df, 'Western')
        co_occurrence_network(df, 'Chinese')
        visualize_pop_culture_sentiment(df)
        
        print("Figures complete.")
        
        # Generate summary report
        report_metrics(df)
        
        print("Analysis complete.")
