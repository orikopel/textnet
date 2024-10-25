# Text Similarity Network Graph App

A Streamlit application that calculates the similarity between texts in a given CSV or Excel file, visualizes the results as a network graph, and helps users identify relationships between similar texts.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)

## Overview

This app allows users to upload a CSV or Excel file containing a column of text data. The application computes the similarity between each pair of texts using a chosen similarity metric and generates a network graph representing the relationships among similar texts. This can be particularly useful for tasks such as clustering, text analysis, and data exploration.

![image](https://github.com/user-attachments/assets/9fe220bd-26e1-4fc8-8de2-7844df204526)


## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/text-similarity-network-app.git
   cd text-similarity-network-app
2. Install the required packages:
   pip install -r requirements.txt

## Usage

1. Run the Streamlit app:
   streamlit run app.py
2. Open your web browser and go to http://localhost:8501.
3. Upload your CSV or Excel file. Ensure that the file has a column with text data.
4. Provide inputs such as the name of the text column and the similarity threshold.
5. Optional - filter your data by a list of keywords.
7. Click the "Generate Graph" button to process the data and generate the network graph.
8. Explore the network graph to identify similar texts and their relationships.

## Features

* Supports both CSV and Excel file formats.
* Calculates text similarity using various language models.
* Currently supports English, Hebrew and Arabic.
* Interactive network graph visualization.
* User-friendly interface built with Streamlit.

## License

© 2024 Ori Kopolovich. All rights reserved.
License: Use for personal or internal business purposes only. No modifications or redistribution allowed.

## Contact

Email: orikopel@gmail.com
GitHub: orikopel



  
