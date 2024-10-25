# basic libs
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# for graphs
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from networkx.algorithms.community import girvan_newman

# for concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# for NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# configure tqdm with pandas
tqdm.pandas()

#------------------------------------------------------------------------------
# Helper Functions #

def extract_common_keywords(titles, stop_words):
    """
    Extracts the most common non-stopword keywords from a list of titles.
    Args:
        titles (List[str]): List of titles to extract keywords from.
    Returns:
        common_keywords (str): The most common keyword(s) or subject(s).
    """
    # Tokenize words and remove stopwords
    words = [word.lower() for title in titles for word in word_tokenize(title, language="english") if word.isalnum()]
    words = [word for word in words if word not in stop_words]

    # Get the most common words
    most_common_words = Counter(words).most_common(3)  # Top 3 common words
    return ', '.join([word[0] for word in most_common_words])  # Return top words as a string

def validate_data(df, id_col, title_col, text_col):
    """
    Validates the input data.
    """

    # filter out rows without texts
    df = df[df[text_col].apply(lambda x: isinstance(x, str) and len(x) > 0)]

    # get rid of non-letter chars
    df[text_col] = df[text_col].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)

    # drop duplicates by id
    df = df.dropna(subset=[id_col])

    # convert ids to string if needed
    df[id_col] = df[id_col].astype(str)

    return df

#------------------------------------------------------------------------------
# Embedding the Input Texts #

def batch_encode(texts, model, batch_size=32):
    """
    Encodes texts in batches using the provided model.

    Args:
        texts (list): List of text data to encode.
        model (SentenceTransformer): Model for encoding.
        batch_size (int): Batch size for encoding.

    Returns:
        List of embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    return embeddings

def generate_embeddings(df, text_col, model, batch_size=32, num_threads=4):
    """
    Gets a df and a column name and returns a df with an embedding column.

    Args:
        df(DataFrame): df with text column, should also have an id and title column.
        text_col(String): name of the text column.
        model(SentenceTransformer): SentenceTransformer model.
        batch_size(int): Number of texts to encode in one batch for efficiency. Default is 32.

    Returns:
        df(DataFrame): df with an embedding column.
    """

    # Prepare text data as a list to avoid pandas row overhead
    texts = df[text_col].values
    n = len(texts)

    # Split data for multithreading
    splits = np.array_split(texts, num_threads)

    # Use multithreading to process each batch
    embeddings = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(batch_encode, split, model, batch_size) for split in splits]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings in parallel"):
            embeddings.extend(future.result())

    # Convert list of embeddings to a NumPy array and add to dataframe
    df['embedding'] = np.array(embeddings).tolist()

    return df

#------------------------------------------------------------------------------
# Creating the Graphs #

def add_edges(G, data, i, similarity_matrix, threshold, id_col):
    """Helper function to add edges to the graph."""
    edges = []
    for j in range(i + 1, len(data)):
        score = similarity_matrix[i, j]
        if score > threshold:
            edges.append((data.iloc[i][id_col], data.iloc[j][id_col], {"score": float(score)}))
    return edges

def create_similarity_nx(data, id_col, model, threshold):
    """
    Creates a graph data structure with text similarity as edge score.

    Args:
        data(DataFrame): df with an embedding column.
        id_col(String): name of the id column.

    Returns:
        G(Graph): graph data structure with text similarity as edge score.
    """

    G = nx.Graph()

    # Convert embeddings to a NumPy array
    embeddings = np.array(data["embedding"].tolist())

    # Calculate pairwise similarities using a dot product and normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(embeddings, embeddings.T) / (norms * norms.T)

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(len(data)), desc="Finding edges"):
            futures.append(executor.submit(add_edges, G, data, i, similarity_matrix, threshold, id_col))

    # Collect results and add edges to the graph
    for future in tqdm(futures, desc="Adding edges to graph"):
        edges = future.result()
        G.add_edges_from(edges)

    return G

def G_to_net(G, id_col, title_col, text_col, data, stop_words):
    """
    Creates a Pyvis network from a graph data structure.

    Args:
        G(Graph): graph data structure with text similarity as edge score.
        id_col(String): name of the id column.
        title_col(String): name of the title column.
        data(DataFrame): df with an embedding column.

    Returns:
        net(Network): Pyvis network.
    """

    # find communities
    comp = girvan_newman(G)
    communities = next(comp)
    colors = plt.cm.get_cmap("tab10", len(communities))

    # Create a Pyvis network
    net = Network(notebook=True)

    # add nodes to pyvis
    for idx, community in tqdm(enumerate(communities), desc="Building pyvis graph by communities - nodes"):

        # Get titles for this community
        texts = [data.loc[data[id_col] == node, title_col].values[0] for node in community]

        # Create a community label (e.g., most common title)
        community_node_id = f'community_{idx}'
        community_title = community_node_id + " - " + extract_common_keywords(texts, stop_words)

        # Add a community node (outer circle)
        net.add_node(community_node_id, label=community_title, title=community_title,
                     color=f'rgba({colors(idx)[0] * 255}, {colors(idx)[1] * 255}, {colors(idx)[2] * 255}, 0.2)',
                     size=40)  # Larger size for visibility

        # Connect community node to its members
        for node in community:
            title_value = data.loc[data[id_col]==node, title_col].values[0]

            # deal with long node titles
            if len(title_value) > 35:
                title_value = title_value[:35] + "..."

            net.add_node(node, title=title_value, label=title_value) # get title from data df
            net.nodes[-1]['color'] = f'rgba({colors(idx)[0] * 255}, {colors(idx)[1] * 255}, {colors(idx)[2] * 255}, 0.7)'

            net.add_edge(community_node_id, node, color='rgba(0, 0, 0, 0)', value=0)  # Invisible edges for connection

    # add edges to pyvis
    for u, v, val in tqdm(G.edges(data=True), desc="Building pyvis graph by communities - edges"):
        net.add_edge(u, v, value=val['score'] * 10, label=val['score'])  # Scale scores for edge visibility

    # Set dark mode options
    net.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 16,
                    "color": "white",
                    "face": "Arial"
                },
                "borderWidth": 1,
                "borderWidthSelected": 2
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100
            },
            "physics": {
                "enabled": true
            },
            "layout": {
                "improvedLayout": true
            }
        }
    """)
    net.background_color = "#222222"  # Dark background

    # Show the network
    return net

#------------------------------------------------------------------------------
# Running the whole process #

def df2net(df, id_col, title_col, text_col, threshold, lang):
    """
    Creates a network visualization from a dataframe. Combines all above functions.

    Args:
        df(DataFrame): the user's data.
        id_col(String): column name of the unique identifier.
        title_col(String): column name of the title.
        text_col(String): column name of the text.
        threshold(Float): threshold for the edge score.
        lang(String): language of the text. Used for choosing the right model.

    Returns:
        net(Pyvis): a pyvis network graph. 
    """

    # dict for choosing a model for each optional language
    language_models = {"eng": "all-MiniLM-L6-v2", "heb": "imvladikon/sentence-transformers-alephbert"}

    # choose the right model for user input language
    model = SentenceTransformer(language_models[lang])

    # read the dataset and create embedding column
    data = validate_data(df, id_col, title_col, text_col)
    data = generate_embeddings(data, text_col, model)

    # create a graph data structure with text similarity as edge score
    G = create_similarity_nx(data, id_col, model, threshold)

    # convert the nx graph to a net graph
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    net = G_to_net(G, id_col, title_col, text_col, data, stop_words)

    return net
