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


def recommend_movies(movie, rating, movie_id):
    # Find user similar to us
    ## user who like the same movie(movie_id) which rate the move more than 4
    user_who_like_mov = rating[rating['movieId'] == 1]
    user_who_like_mov_idx = user_who_like_mov[user_who_like_mov['rating'] > 4]['userId'].unique()
    ## movies that the user like (score > 4)
    rec_mov = rating[(rating['userId'].isin(user_who_like_mov_idx))&(rating['rating'] > 4)]
    
    # Adjusting over 10% of the user recommend that particular movie
    ## filter movie that more than 10% of similar people like the movie
    rec_mov = rec_mov['movieId'].value_counts()/len(user_who_like_mov_idx)
    rec_mov = rec_mov[rec_mov > 0.1]
    
    # finding how common the reccommendations were among all of the users
    ## all user user who ever rate the movie (that the movie rate more than 10%) and the rateing is more than 4
    all_user = rating[(rating['movieId'].isin(rec_mov.index)) & (rating["rating"] > 4)]
    all_user_rec = all_user["movieId"].value_counts()/len(all_user["userId"].unique())

    # creating the score
    # Compare the percent
    rec_percentages = pd.concat([rec_mov, all_user_rec], axis = 1)
    rec_percentages.columns = ['recommend score','common score']
    rec_percentages['Overall Score'] = rec_percentages['recommend score']/rec_percentages['common score']
    rec_percentages =rec_percentages.sort_values("Overall Score", ascending=False)
    # merge the top 10 score with movie table
    return rec_percentages.head(10).merge(movie, left_index=True, right_on="movieId")
