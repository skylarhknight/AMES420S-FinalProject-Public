import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import Counter
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
from googletrans import Translator

# GLOBAL VISTUALIZATON SETTINGS ============================

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['grid.alpha'] = 0.7


# SETUP ====================================================

# Keywords
keywords = [
    'surveillance', 'monitoring', 'privacy', 'security', 
    'freedom', 'control', 'oversight', 'social stability', 
    'censorship', 'public safety', 'regulation', 'governance',
    'fear', 'opression', 'dystopia', 'authoritarian', 'totalitarian',
    'safety'
]

# Pop culture references
pop_culture_terms = [
    'black mirror', 'big brother', 'minority report', '1984', 'orwellian'
]

translator = Translator()

# FUNCTIONS  ====================================================

# Load article links from CSV
def load_links_from_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset='link')  # Remove duplicate links
    western_links = df[df['source'] == 'Western']['link'].dropna().tolist()
    chinese_links = df[df['source'] == 'Chinese']['link'].dropna().tolist()
    return western_links, chinese_links

# Fetch and extract article text
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

# Analyze articles
def analyze_articles(article_links, label):
    data = []
    for link in article_links:
        text = extract_text(link)
        if len(text) > 50:
            try:
                detected_lang = translator.detect(text).lang
                if detected_lang != 'en' or label == 'Chinese':
                    translated = translator.translate(text, dest='en')
                    text = translated.text
            except:
                pass

            word_counts = Counter(re.findall(r'\w+', text.lower()))
            keyword_counts = {kw: word_counts.get(kw, 0) for kw in keywords}
            sentiment = TextBlob(text).sentiment.polarity
            data.append({**keyword_counts, 'sentiment': sentiment, 'source': label, 'url': link})
    return data

# Gather all articles
def gather_all():
    western_articles, chinese_articles = load_links_from_csv('article_links.csv')
    western_data = analyze_articles(western_articles, 'Western')
    chinese_data = analyze_articles(chinese_articles, 'Chinese')
    return pd.DataFrame(western_data + chinese_data)

# Save analysis to CSV
def save_to_csv(df):
    df.to_csv('articles_analysis.csv', index=False)
    print('Data saved to articles_analysis.csv')


# VISUALIZATION FUNCTIONS ====================================================

def visualize_keyword_comparison(df):
    summary = df.groupby('source')[keywords].mean()
    summary.plot(kind='bar', figsize=(12,8))
    plt.title('Keyword Frequency Comparison', fontsize=18)
    plt.ylabel('Average Count', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.xlabel('') # hide x-label
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_sentiment(df):
    df_grouped = df.groupby('source')['sentiment'].mean()
    colors = {'Western': '#1f77b4', 'Chinese': '#d62728'}

    plt.figure(figsize=(10,6))
    bars = plt.bar(df_grouped.index, df_grouped.values, color=[colors[src] for src in df_grouped.index])
    plt.title('Average Sentiment Comparison', fontsize=18)
    plt.xlabel('Source', fontsize=14)
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xlabel('') # hide x-label
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.01, f'{yval:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    
def visualize_individual_sentiments(df):
    colors = df['source'].map({'Western': 'blue', 'Chinese': 'red'})
    plt.figure(figsize=(12,8))
    
    plt.barh(range(len(df)), df['sentiment'], color=colors)  # <-- HORIZONTAL bars now
    plt.title('Sentiment of Individual Articles', fontsize=18, fontweight='bold')
    plt.ylabel('Article Index', fontsize=14)
    plt.xlabel('Sentiment Polarity (Negative → Positive)', fontsize=14)
    plt.yticks(range(len(df)), labels=range(len(df)), fontsize=12)  # Label each bar with its index
    plt.xticks(fontsize=12)
    plt.grid(axis='x', alpha=0.7)
    
    # Legend
    blue_patch = mpatches.Patch(color='blue', label='Western')
    red_patch = mpatches.Patch(color='red', label='Chinese')
    plt.legend(handles=[blue_patch, red_patch], loc='lower right', title='Source')

    plt.axvline(x=0, color='black', linewidth=1)  # <-- Center line at x=0
    plt.tight_layout()
    plt.show()

def visualize_avg_sentiment_per_source(df):
    avg_sentiment = df.groupby(['source'])['sentiment'].mean().reset_index()
    colors = avg_sentiment['source'].map({'Western': 'blue', 'Chinese': 'red'})
    plt.figure(figsize=(8,6))
    plt.bar(avg_sentiment['source'], avg_sentiment['sentiment'], color=colors)
    plt.title('Average Sentiment per News Source', fontsize=18)
    plt.xlabel('News Source', fontsize=14)
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.show()


def co_occurrence_network(df, label):
    # Extract article texts
    all_text = ''
    for url in df[df['source'] == label]['url']:
        article_text = extract_text(url)
        all_text += article_text.lower() + ' '
    
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

    G = nx.Graph()

    # Add edges for any keyword pairs that co-occur
    for (w1, w2), count in co_occur.items():
        if w1 in keywords and w2 in keywords and count >= 1:
            G.add_edge(w1, w2, weight=count)

    if len(G.nodes) == 0:
        print(f"No strong co-occurrences found for {label}.")
        return

    # Node sizes and colors based on keyword frequency
    keyword_counts = Counter(words)
    node_sizes = [500 + keyword_counts.get(node, 1) * 100 for node in G.nodes]
    node_colors = [keyword_counts.get(node, 1) for node in G.nodes]

    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G, k=0.5, seed=42)  # <-- k controls node repulsion/spacing

    # Draw edges thicker and semi-transparent
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap=plt.cm.plasma)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                               norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Keyword Frequency', fontsize=12)

    plt.title(f'Keyword Co-occurrence Network ({label})', fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_pop_culture_references(df):
    all_text = ''
    for url in df['url']:
        text = extract_text(url)
        all_text += text.lower() + ' '

    pop_counts = {term: all_text.count(term) for term in pop_culture_terms}

    plt.figure(figsize=(10,6))
    plt.bar(pop_counts.keys(), pop_counts.values(), color='purple')
    plt.title('Pop Culture References Related to Surveillance', fontsize=18)
    plt.xlabel('Reference', fontsize=14)
    plt.ylabel('Mentions', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_sentiment_by_pop_culture(df):
    pop_culture_flags = []
    for url in df['url']:
        text = extract_text(url).lower()
        if any(term in text for term in pop_culture_terms):
            pop_culture_flags.append('Mentions Pop Culture')
        else:
            pop_culture_flags.append('No Pop Culture')

    df['pop_culture_mentioned'] = pop_culture_flags

    grouped = df.groupby(['source', 'pop_culture_mentioned'])['sentiment'].mean().unstack()

    grouped.plot(kind='bar', figsize=(10,6))
    plt.title('Sentiment by Pop Culture Reference and Source', fontsize=18)
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xlabel('Source', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y',  alpha=0.7)
    plt.legend(title='Pop Culture Mentioned', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.show()



# --- MAIN ---

if __name__ == "__main__":
    df = gather_all()
    if df.empty:
        print("No articles found. Please check your URLs.")
    else:
        print(df.head())
        save_to_csv(df)
        visualize_keyword_comparison(df)
        visualize_sentiment(df)
        visualize_individual_sentiments(df)
        visualize_avg_sentiment_per_source(df)
        co_occurrence_network(df, 'Western')
        co_occurrence_network(df, 'Chinese')
        visualize_pop_culture_references(df)
        visualize_sentiment_by_pop_culture(df)
