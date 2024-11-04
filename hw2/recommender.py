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


def good_to_mid(df: pd.DataFrame) -> list[int, np.ndarray]:
    return_arr = []
    for tup in zip(df.index, df.values.tolist()):
        return_arr.append(tup)
    return return_arr


if __name__ == '__main__':
    # preprocess movie and query data
    preprocessor = Preprocessor()
    movie_lens_df = preprocessor('movielens.txt')
    query_df = preprocessor('train_a.txt')
    movie_lens_reduced, query_df_reduced = reduce_movies(
        movie_lens_df=movie_lens_df, 
        query_df=query_df
    )
    # puts data in the "mid" format that knn expects
    movie_mid = good_to_mid(movie_lens_reduced)
    query_mid = good_to_mid(query_df_reduced)
    # hyperparameter k is defined in find_k_neighbors to ensure consistency with Part I of assignment
    similar_users = find_k_neighbors(
        train=movie_mid,
        query=query_mid,
        metric='euclidean',
        find_mode=False
    )
    
    movies_to_recommend = list(set(movie_lens_df.columns)- set(query_df.columns))
    filtered_movies = movie_lens_df.iloc[similar_users][movies_to_recommend]
    mean_ratings = filtered_movies.mean() # TODO retire from coding
    # hyperparameter M is the number of movies recommended
    M = 3
    # we need M row labels with highest mean ratings
    mean_ratings_sorted = mean_ratings.sort_values(ascending=False)
    recommended_movies = mean_ratings_sorted.index.tolist()[:M]
    print(f'we recommend {recommended_movies}')





    
    
