import re
import pandas as pd
import numpy as np
# feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# to clean movie title 
# (remove special char cuz it's make difficault for searching)
def clean_title(title):
    # remove char that not in below reg
    return re.sub("[^a-zA-Z0-9]", " ", title)


