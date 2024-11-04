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

    def __call__(self, path: str, extra_features) -> pd.DataFrame:
        df = self.get_dataset(path)
        # TODO muster up the will to handle all of these
        # user_df = df[['user_id', 'age', 'gender', 'occupation']].drop_duplicates()
        user_df = df[['user_id','gender', 'age']].drop_duplicates()
        user_df.set_index('user_id', drop=True, inplace=True)
        user_df['gender'] = user_df['gender'].map({'M':0, 'F':1})
        user_df.rename(columns={"gender":0, "age":-1}, inplace=True) # NOTE this is horrible
        piv_df = self.pivot_dataframe(df)
        imputed_df = self.impute_nans_with_scale_mean(piv_df)
        if extra_features: imputed_df = pd.concat([user_df, imputed_df], axis=1)
        return imputed_df
    

# filters movies and query dataframes to shared features (special features and each movie column)
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
def preprocess_and_model(train_path, query_path, k, M, extra_features):
    # preprocess movie and query data
    preprocessor = Preprocessor()
    movie_lens_df = preprocessor(path=train_path, extra_features=extra_features)
    query_df = preprocessor(path=query_path, extra_features=extra_features)
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
    movies_to_recommend = list(set(movie_lens_df.columns) - set(query_df.columns))
    temp_df = movie_lens_df.loc[similar_users] # I hate pandas
    filtered_movies = temp_df[movies_to_recommend]
    mean_ratings = filtered_movies.mean() # TODO retire from coding
    # we need M row labels with highest mean ratings
    mean_ratings_sorted = mean_ratings.sort_values(ascending=False)
    recommended_movies = mean_ratings_sorted.index.tolist()[:M]
    return recommended_movies


# evaluated query set (validation & testing) the recommended movies produced by the preprocess_and_model function
# returns precision, recall, and F1 score
def eval_reced_movies(movies_reced, query_path, extra_features):
    # handle query data
    preprocessor = Preprocessor()
    query_df = preprocessor(query_path, extra_features=extra_features)
    query_movies = query_df.columns.tolist() # this is the movies the user has seen for val or test

    # calc recall, the number of movies suggested in the query movies set over the total number of movies reced
    recall = len(set(movies_reced).intersection(set(query_movies)))/len(query_movies)
    # calc precision, the number of movies suggested in the query movies set over the total number of query movies
    precision = len(set(movies_reced).intersection(set(query_movies)))/len(movies_reced)
    # calc F1
    if (precision + recall) == 0: return precision, recall, 0
    f_1 = (2*precision*recall) / (precision + recall)
    return precision, recall, f_1


# validates and tests rec system for different hyperparameters (hardcoded)
def val_and_test(extra_features, verbose=False):
    if extra_features: print(f'Testing with special features...')
    for user_letter in ['a', 'b', 'c']:
        if verbose: print(f'Training and testing on user {user_letter}...')
        train_path = f'train_{user_letter}.txt'
        valid_path = f'valid_{user_letter}.txt'
        test_path = f'test_{user_letter}.txt'

        # grid search hyperparams k and M
        best_f_1 = 0
        best_k, best_M = -1, -1
        for k in [5, 50, 100]:
            for M in [5, 50, 100, 200]:
                if verbose: print(f'Running on k={k} and M={M}')
                movies_reced = preprocess_and_model(train_path='movielens.txt', query_path=train_path, k=k, M=M, extra_features=extra_features)
                if verbose: print(f'We recommend the following movies: {movies_reced}') # TODO get movie names for final string output
                # validation
                val_prec, val_recall, val_f_1 = eval_reced_movies(movies_reced=movies_reced, query_path=valid_path, extra_features=extra_features)
                if verbose: print(f'---Validation (k={k}, M={M})---\nPrecision: {val_prec}\nRecall: {val_recall}\nF1: {val_f_1}')
                if val_f_1 > best_f_1: best_f_1, best_k, best_M = val_f_1, k, M # comment on top of it

        if verbose: print('Testing...')
        movies_reced = preprocess_and_model(train_path='movielens.txt', query_path=train_path, k=best_k, M=best_M, extra_features=extra_features)
        test_prec, test_recall, test_f_1 = eval_reced_movies(movies_reced=movies_reced, query_path=test_path, extra_features=extra_features)
        print(f'---Test for User {user_letter} (with best k = {best_k} and best M = {best_M})---\nPrecision: {test_prec}\nRecall: {test_recall}\nF1: {test_f_1}')
    
    return test_prec, test_recall, test_f_1


if __name__ == '__main__':
    # validate and test recs with just movies
    val_and_test(extra_features=False, verbose=False)

    # validate and test recs with movies and extra features 
    val_and_test(extra_features=True, verbose=False)



