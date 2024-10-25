import re
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def extract_common_keywords(titles):
    """
    Extracts the most common non-stopword keywords from a list of titles.
    Args:
        titles (List[str]): List of titles to extract keywords from.
    Returns:
        common_keywords (str): The most common keyword(s) or subject(s).
    """
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for title in titles for word in word_tokenize(title) if word.isalnum()]
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