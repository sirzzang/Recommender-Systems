# module import
import pandas as pd
import numpy as np
from utils import load_data
from dimensionality_reduction import svd
from recommendation import topn_recommendations

# load data
ratings = load_data('./data/movielens/ml-1m/ratings.dat')
movies = load_data('./data/movielens/ml-1m/movies.dat')
users = load_data('./data/movielens/ml-1m/users.dat')

# rename columns
ratings = ratings.rename(columns={0: 'UserID', 1: 'MovieID', 2: 'Rating', 3:'Timestamp'})
movies = movies.rename(columns={0: 'MovieID', 1: 'Title', 2: 'Genre'})
users = users.rename(columns={0: 'UserID', 1: 'Gender', 2: 'Age', 3: 'Occupation', 4: 'Zip-code'})

# user-item matrix
user_item_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating')

# nan imputation # TODO
matrix = pd.DataFrame(user_item_matrix.fillna(0), columns=user_item_matrix.columns)

# dimensionality reduction and reconstruction
reconstructed_matrix = svd(matrix, ratio=0.9)

# top N recommendations
user_idx = np.random.randint(1, user_item_matrix.index.max())
user_profile = users[users['UserID'] == user_idx]
print('user 정보\n', user_profile)
recommendations = topn_recommendations(user_idx, movies, user_item_matrix, reconstructed_matrix, n=10)
print(recommendations)