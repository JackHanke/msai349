import math
import statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.spatial.distance import euclidean as scipy_euclidean, cosine as scipy_cosine
from starter import *
import pandas as pd

#load and parse movielense.txt, train_{a,b,c}.txt, valid_{a,b,c}.txt, and test_{a,b,c}.txt
#each line in movielens.txt should contain a user, movie, and rating
#matrix build where rows represent users and columns represent movies
def data_preprocessing(file_name, columns = None):
    df = pd.read_csv(file_name, delimiter='\t')
    print(df.head())
    print(df.dtypes)
    #taking into account the user demographics and combining it to a one dimensional vector
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['gender', 'occupation', 'genre'])
    df_encoded.fillna(method='ffill', inplace=True)
    # Taking all values of one iser into one row using the mean 
    user_features = df_encoded.groupby('user_id').mean(numeric_only=True).reset_index()
    print(user_features)
    
    # Debugging: Print the columns of user_features
    print("Columns in user_features after one-hot encoding:", user_features.columns)
    relevant_features = user_features.drop(columns=['user_id'], errors='ignore')

    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(relevant_features)
    user_features_scaled = scaler.fit_transform(user_features)
    print(user_features_scaled)
    print("Shape of user_features_scaled:", user_features_scaled.shape)

    # # Check if the number of columns match before creating DataFrame
    # if user_features_scaled.shape[1] != len(scaled_columns):
    #     print("Mismatch in column counts:", user_features_scaled.shape[1], "vs", len(scaled_columns))
    # Create a DataFrame for the scaled features, including user_id
    # user_features_scaled_df = pd.DataFrame(user_features_scaled, columns=user_features.columns[1:])
    user_features_scaled_df = pd.DataFrame(user_features_scaled)
    user_features_scaled_df['user_id'] = user_features['user_id']

    # Merging the movie features and user features 
    movie_features = df[['user_id', 'movie_id', 'rating']]
    user_movie_features = pd.merge(user_features_scaled_df, movie_features, on='user_id')

    X = user_movie_features.drop(columns=['movie_id', 'rating'])
    print(X)  # Drop movie_id and rating
    y = user_movie_features['movie_id']  # Set movie_id as the target variable

    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    
    return X, y, X.columns
    # main_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')

    # #filling in the matrix's NaN values with Scale's mean (notes from data science)
    # #the movie rating scale is 1-to-5, so the mean of the scale is 3
    # #center ratings around the mean of the scale
    # main_matrix = main_matrix - 3
    # print(main_matrix)
    # #fill in NaN values with 0
    # main_matrix = main_matrix.fillna(0)
    # print(main_matrix)
    # #process training, validation, and test sets to maintain data consistency across training, validation, and testing
    # def process_file(file):
    #     df = pd.read_csv(file_name, delimiter='\t')
    #     matrix = df.pivot(index='user_id', columns='movie_id', values='rating')
    #     matrix = matrix - 3
    #     matrix = matrix.fillna(0)
    #     return matrix

    # train_matrix = [process_file(f) for f in train_files]
    # valid_matrix = [process_file(f) for f in valid_files]
    # test_matrix = [process_file(f) for f in test_files]

    # return main_matrix, train_matrix, valid_matrix, test_matrix
    # return user_features




#using distance metrics to calculate similarities between users based on movie ratings
#look up each target user's top K most similar users and use ratings to suggest new movies
# def user_similarity(user_ratings, metric):
#     number_users = user_ratings.shape[0]

#     #similarity matrix with zero initalization (each entry is similarity between user pairs)
#     similarity_matrix = np.zeros((number_users, number_users))

#     #loop through each user to calculate similarity with other users
#     for i in range (number_users):
#         for j in range(i + 1, number_users):
#             if metric == 'cosim':
#                 similarity = 1 - cosim(user_ratings[i], user_ratings[j])
#             if metric == 'euclidian':
#                 similarity = euclidean(user_ratings[i], user_ratings[j])
#             if metric == 'pearson':
#                 similarity = pearson_correlation(user_ratings[i], user_ratings[j])
#             if metric == 'hamming':
#                 similarity = hamming(user_ratings[i], user_ratings[j])

#             #fill in calculated similarity for i and j, 
#             similarity_matrix[i,j] = similarity
#             similarity_matrix[j,i] = similarity

#     return similarity_matrix


def recommend_movies(X_train, y_train, X_valid, y_valid):
    # Call the knn function to get predictions for the validation set
    predicted_labels = knn(train=np.column_stack((y_train, X_train)), 
                            query=X_valid, 
                            metric='euclidean')
    
    # Evaluate predictions
    accuracy = np.mean(predicted_labels == y_valid)
    print(f'Accuracy: {accuracy:.2f}')
    pass

def recommend_movies_sk(X_train, y_train, X_valid, y_valid, metric='euclidean', k=10):

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print("Recommendation Accuracy:", accuracy)
    
    # Print predictions for recommendations
    # recommendations = pd.DataFrame({'User': X_valid.index, 'Recommended_Movie': y_pred})
    # print("Recommendations for Validation Users:\n", recommendations.head())




#implement precision, recall, and F-1 score for measuring performance
def evaluation_metrics():
    pass





if __name__ == "__main__":
    #data preprocessing for a data
    training_data_X, training_data_y, train_columns=data_preprocessing('train_a.txt')
    validation_data_X, validation_data_y, _=data_preprocessing('valid_a.txt', columns=train_columns)
    
    recommend_movies_sk(training_data_X, training_data_y, validation_data_X, validation_data_y)
    

    #data processing for b data
    #data_preprocessing('movielens.txt', 'train_b.txt', 'valid_b.txt', 'test_b.txt')

    #data processing for c data
    #data_preprocessing('movielens.txt', 'train_c.txt', 'valid_c.txt', 'test_c.txt')
