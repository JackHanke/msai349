import pandas as pd
from starter import euclidean, cosim, find_k_neighbors
import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def get_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t')
        return df

    def pivot_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_pivoted = pd.pivot(df, index='user_id', columns='movie_id', values='rating')
        return df_pivoted

    def impute_nans_with_scale_mean(self, df: pd.DataFrame, scale_mean: int = 3) -> pd.DataFrame:
        return df.fillna(scale_mean)

    def __call__(self, path: str) -> pd.DataFrame:
        df = self.get_dataset(path)
        df = self.pivot_dataframe(df)
        return self.impute_nans_with_scale_mean(df)
    

def reduce_movies(movie_lens_df: pd.DataFrame, query_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie_lens_cols = set(movie_lens_df.columns.tolist())
    query_cols = set(query_df.columns.tolist())
    reduced_list = sorted(list(query_cols.intersection(movie_lens_cols)))
    query_df_reduced = query_df[reduced_list]
    movie_df_reduced = movie_lens_df[reduced_list]
    return movie_df_reduced, query_df_reduced


# good function
def good_to_mid(df: pd.DataFrame) -> list[int, np.ndarray]:
    return_arr = []
    for tup in zip(df.index, df.values.tolist()):
        return_arr.append(tup)
    return return_arr


# preprocess_and_model preprocesses the data at train_path and query_path and finds the k similar users to query and recommends M movies
def preprocess_and_model(train_path, query_path, k, M):
    # preprocess movie and query data
    preprocessor = Preprocessor()
    movie_lens_df = preprocessor(train_path)
    query_df = preprocessor(query_path)
    movie_lens_reduced, query_df_reduced = reduce_movies(
        movie_lens_df=movie_lens_df, 
        query_df=query_df
    )
    #  puts data in the "mid" format that knn expects
    movie_mid = good_to_mid(movie_lens_reduced)
    query_mid = good_to_mid(query_df_reduced)

    # hyperparameter k is defined in find_k_neighbors to ensure consistency with Part I of assignment
    similar_users = find_k_neighbors(
        train=movie_mid,
        query=query_mid,
        metric='euclidean',
        find_mode=False,
        k=k
    )
    movies_to_recommend = list(set(movie_lens_df.columns)- set(query_df.columns))
    filtered_movies = movie_lens_df.iloc[similar_users][movies_to_recommend]
    mean_ratings = filtered_movies.mean() # TODO retire from coding
    # we need M row labels with highest mean ratings
    mean_ratings_sorted = mean_ratings.sort_values(ascending=False)
    recommended_movies = mean_ratings_sorted.index.tolist()[:M]
    return recommended_movies


# evaluated query set (validation & testing) the recommended movies produced by the preprocess_and_model function
# returns precision, recall, and F1 score
def eval_reced_movies(movies_reced, query_path):
    # handle query data
    preprocessor = Preprocessor()
    query_df = preprocessor(query_path)
    query_movies = query_df.columns.tolist() # this is the movies the user has seen for val or test

    # calc precision, TODO what
    precision = len(set(movies_reced).intersection(set(query_movies)))/len(movies_reced)
    # calc recal, TODO what
    recall = len(set(movies_reced).intersection(set(query_movies)))/len(query_movies)
    # calc F1
    f_1 = 2*precision*recall / (precision+recall)

    return precision, recall, f_1

if __name__ == '__main__':
    for user_letter in ['a','b','c']:
        print(f'Training and testing on user {user_letter}...')
        train_path = f'train_{user_letter}.txt'
        valid_path = f'valid_{user_letter}.txt'
        test_path = f'test_{user_letter}.txt'

        # grid search hyperparams k and M
        for k in [5,10,50,100]:
            for M in [5,10,50]:
                movies_reced = preprocess_and_model(train_path='movielens.txt', query_path=train_path, k=k, M=M)
                print(f'We recommend the following movies: {movies_reced}') # TODO get movie names for final string output
                # validation
                val_prec, val_recall, val_f_1 = eval_reced_movies(movies_reced=movies_reced, query_path=valid_path)
                print(f'---Validation (k={k}, M={M})---\nPrecision: {val_prec}\nRecall: {val_recall}\nF1: {val_f_1}')
        print('testing...')
        # test
        test_prec, test_recall, test_f_1 = eval_reced_movies(movies_reced=movies_reced, query_path=test_path)
        print(f'---Test---\nPrecision: {val_prec}\nRecall: {val_recall}\nF1: {val_f_1}')
        



