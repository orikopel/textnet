# basic libs
import numpy as np
import pandas as pd
from tqdm import tqdm

# for graphs
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from networkx.algorithms.community import girvan_newman

# for concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# for NLP
from sentence_transformers import SentenceTransformer

# inner modules
from helper_funcs import extract_common_keywords, validate_data

# configure tqdm with pandas
tqdm.pandas()

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
            edges.append((data.iloc[i][id_col], data.iloc[j][id_col], float(score)))
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

def G_to_net(G, id_col, title_col, text_col, data):
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
        titles = [data.loc[data[id_col] == node, title_col].values[0] for node in community]
        texts = [data.loc[data[id_col] == node, title_col].values[0] for node in community]

        # Create a community label (e.g., most common title)
        community_node_id = f'community_{idx}'
        community_title = community_node_id + " - " + extract_common_keywords(texts)

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

    net.set_options("""
        var options = {
        "nodes": {
            "font": {
            "size": 14
            }
        },
        "edges": {
            "smooth": {
            "type": "continuous"
            }
        },
        "physics": {
            "enabled": true
        }
        }
        """)

    # Show the network
    return net

#------------------------------------------------------------------------------
# Displaying the Graph as HTML Page #

def create_with_header(save_path, net, html_header_path):
    """
    Creates an html page with a header and a network visualization.

    Args:
        save_path(String): path to save the html page.
        net(Network): Pyvis network.
        html_header(String): html header.
    """

    with open(html_header_path, "r", encoding="utf-8") as file:
        html_header = file.read()

    # Combine the header with the network output
    with open(save_path, "w") as f:
        f.write(html_header)
        f.write(net.generate_html())  # Include the network visualization
        f.write("</body></html>")  # Close the HTML tags

#------------------------------------------------------------------------------
# Running the whole process #

def df2net(data_path, save_path, id_col, title_col, text_col, html_header_path, threshold, lang):
    """
    Creates a network visualization from a dataframe. Combines all above functions.

    Args:
        data_path(String): path to the dataframe.
        save_path(String): path to save the html page.
        id_col(String): column name of the unique identifier.
        title_col(String): column name of the title.
        text_col(String): column name of the text.
        html_header(String): header of the html page.
        lang(String): language of the text. Used for choosing the right model.
        threshold(Float): threshold for the edge score.
    """

    # dict for choosing a model for each optional language
    language_models = {"eng": "all-MiniLM-L6-v2", "heb": "imvladikon/sentence-transformers-alephbert"}

    # choose the right model for user input language
    model = SentenceTransformer(language_models[lang])
    print(f"1 - Selected model: {language_models[lang]}")

    # read the dataset and create embedding column
    data = pd.read_csv(data_path, encoding='latin1').head(500)
    data = validate_data(data, id_col, title_col, text_col)
    data = generate_embeddings(data, text_col, model)
    print("2 - Created embeddings for the data")

    # create a graph data structure with text similarity as edge score
    G = create_similarity_nx(data, id_col, model, threshold)
    print("3 - Created graph data structure with text similarity as edge score")

    # convert the nx graph to a net graph
    net = G_to_net(G, id_col, title_col, text_col, data)
    print("4 - Created pyvis network")

    # save the graph as an html page
    create_with_header(save_path, net, html_header_path)
    print("5 - Saved the graph as an html page")

df2net("McDonald_s_Reviews.csv", "mcdonalds.html", "reviewer_id", "review", "review", "header.html", 0.7, "eng")
