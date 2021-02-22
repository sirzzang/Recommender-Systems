# module import
import pandas as pd
import numpy as np
from utils import load_data
from dimensionality_reduction import svd
from recommendation import topn_recommendations

# load data
ratings = load_data('./data/movielens/ml-1m/ratings.dat')
movies = load_data('./data/movielens/ml-1m/movies.dat')

# rename columns
ratings = ratings.rename(columns={0: 'UserID', 1: 'MovieID', 2: 'Rating'})
movies = movies.rename(columns={0: 'MovieID', 1: 'Title', 2: 'Genre'})

# user-item matrix
user_item_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating')

# nan imputation # TODO
matrix = pd.DataFrame(user_item_matrix.fillna(0), columns=user_item_matrix.columns)

# dimensionality reduction and reconstruction
reconstructed_matrix = svd(matrix, ratio=90)

# plain top N recommendations
user_idx = np.random.randint(1, user_item_matrix.index.max())
recommendations = topn_recommendations(user_idx, movies, user_item_matrix, reconstructed_matrix, n=10)
print(recommendations)