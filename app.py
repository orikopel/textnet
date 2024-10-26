import tempfile
import pandas as pd
import streamlit as st
from create_textnet import *



st.title("Network Graph of Text Similarity")
st.markdown("A tool for visualizing similarity between texts as a network graph for you to analyze.")
st.markdown("The program uses language models in multiple languages for your choice and visualizes connections where the similarity is above the user defined threshold.")

# File upload option
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    
    # Choose between CSV or Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='latin1')
    else:
        df = pd.read_excel(uploaded_file, encoding='latin1')

    # Column selection for id, title, and text
    id_col = st.selectbox("Select the ID column", df.columns)
    title_col = st.selectbox("Select the Title column", df.columns)
    text_col = st.selectbox("Select the Text column", df.columns)

    # Additional parameters
    threshold = st.slider("Select Similarity Threshold", 0.0, 1.0, 0.75)
    lang = st.selectbox("Select Language Model", ["eng", "heb"])

    if st.button("Generate Graph"):
        
        # create the graph
        net = df2net(df, id_col, title_col, text_col, threshold, lang)

        # Save the network to a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.show(tmp_file.name)
            tmp_html_path = tmp_file.name

        # Display the interactive graph in Streamlit
        st.components.v1.html(open(tmp_html_path).read(), height=800, width=800)





