import math
import statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import euclidean as scipy_euclidean, cosine as scipy_cosine
from starter import *
import pandas as pd

#load and parse movielense.txt, train_{a,b,c}.txt, valid_{a,b,c}.txt, and test_{a,b,c}.txt
#each line in movielens.txt should contain a user, movie, and rating
#matrix build where rows represent users and columns represent movies
def data_preprocessing(file_name, train_files, valid_files, test_files):
    df = pd.read_csv(file_name, delimiter='\t')

    main_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')

    #filling in the matrix's NaN values with Scale's mean (notes from data science)
    #the movie rating scale is 1-to-5, so the mean of the scale is 3
    #center ratings around the mean of the scale
    main_matrix = main_matrix - 3

    #fill in NaN values with 0
    main_matrix = main_matrix.fillna(0)

    #process training, validation, and test sets to maintain data consistency across training, validation, and testing
    def process_file(file):
        df = pd.read_csv(file_name, delimiter='\t')
        matrix = df.pivot(index='user_id', columns='movie_id', values='rating')
        matrix = matrix - 3
        matrix = matrix.fillna(0)
        return matrix

    train_matrix = [process_file(f) for f in train_files]
    valid_matrix = [process_file(f) for f in valid_files]
    test_matrix = [process_file(f) for f in test_files]

    return main_matrix, train_matrix, valid_matrix, test_matrix




#using distance metrics to calculate similarities between users based on movie ratings
#look up each target user's top K most similar users and use ratings to suggest new movies
def user_similarity(user_ratings, metric):
    number_users = user_ratings.shape[0]

    #similarity matrix with zero initalization (each entry is similarity between user pairs)
    similarity_matrix = np.zeros((number_users, number_users))

    #loop through each user to calculate similarity with other users
    for i in range (number_users):
        for j in range(i + 1, number_users):
            if metric == 'cosim';
                similarity = 1 - cosim(user_ratings[i], user_ratings[j])
            if metric == 'euclidian':
                similarity = euclidean(user_ratings[i], user_ratings[j])
            if metric == 'pearson':
                similarity = pearson_correlation(user_ratings[i], user_ratings[j])
            if metric == 'hamming':
                similarity = hamming(user_ratings[i], user_ratings[j])

            #fill in calculated similarity for i and j, 
            similarity_matrix[i,j] = similarity
            similarity_matrix[j,i] = similarity

    return similarity_matrix



#recommend top M movies to each target user based on movies that similar users have highly rated
def recommend_movies():
    pass



#implement precision, recall, and F-1 score for measuring performance
def evaluation_metrics():
    pass





if __name__ == "__main__":
    #data preprocessing for a data
    data_preprocessing('movielens.txt', 'train_a.txt', 'valid_a.txt', 'test_a.txt')

    #data processing for b data
    #data_preprocessing('movielens.txt', 'train_b.txt', 'valid_b.txt', 'test_b.txt')

    #data processing for c data
    #data_preprocessing('movielens.txt', 'train_c.txt', 'valid_c.txt', 'test_c.txt')
