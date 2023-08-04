import pandas as pd  # Used to read .xls file into pandas DataFrame and convert it to numpy matrix. In order to do this, one might need to install "xlrd" with "pip install xlrd"
import numpy as np
'''
We use here the small MovieLens latest dataset [https://grouplens.org/datasets/movielens/latest/], of 100,000 ratings 
and 3,600 tag applications applied to 9,000 movies by 600 users. 
Each row corresponds to a user and each column to a movie. 
The ratings go from 0.5 to 5.
'''

ratings_data = pd.read_csv('Data/MovieLens.csv')  # Read the .csv file into a pandas DataFrame
matrix_pandas = ratings_data.pivot(index='userId', columns='movieId', values='rating')  # Reshape the data into a matrix of nbr_user x nbr_movies dimensions filled with the ratings
movielens = matrix_pandas.to_numpy()  # Convert the DataFrame to a numpy matrix
movielens_normalized = (movielens - 0.5) / 4.5  # Normalization to [0, 1]
matrix[matrix == 0] = 'nan'

# Save the modified matrix as a new Excel file
modified_movie_lens = pd.DataFrame(movielens_normalized)  # Convert into xls. file
writer = pd.ExcelWriter('Data/movielens_normalized.xlsx', engine='xlsxwriter')
modified_movie_lens.to_excel(writer, index=False)
writer.save()


