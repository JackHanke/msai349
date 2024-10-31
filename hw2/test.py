import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import pearsonr

def load_data(file_name):
    df = pd.read_csv(file_name, delimiter='\t')
    df = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return df

def calculate_similarity(train_data, target_user_id, metric='pearson'):
    target_user_ratings = train_data.loc[target_user_id]
    similarities = {}

    for user_id in train_data.index:
        if user_id != target_user_id:
            other_user_ratings = train_data.loc[user_id]

            # Only compute similarity if there are overlapping rated movies
            common_ratings = np.logical_and(target_user_ratings > 0, other_user_ratings > 0)
            if common_ratings.sum() > 0:
                # Calculate Pearson correlation on overlapping ratings
                similarity, _ = pearsonr(target_user_ratings[common_ratings], other_user_ratings[common_ratings])
                if not np.isnan(similarity):  # Ensure similarity is valid
                    similarities[user_id] = similarity
            else:
                print(f"No common ratings between user {target_user_id} and user {user_id}.")

    if not similarities:
        print(f"No similar users found for user {target_user_id}.")
    else:
        print(f"Similar users for user {target_user_id}: {similarities}")

    return similarities

def recommend_movies(train_data, target_user_id, K, M, metric='pearson'):
    similarities = calculate_similarity(train_data, target_user_id, metric)
    
    if not similarities:
        all_ratings = train_data.mean().sort_values(ascending=False)
        return all_ratings.index[:M].tolist()
    
    top_k_users = sorted(similarities, key=similarities.get, reverse=True)[:K]
    
    movie_scores = {}
    for user_id in top_k_users:
        similar_user_ratings = train_data.loc[user_id]
        for movie_id, rating in similar_user_ratings.items():
            if rating > 0 and train_data.loc[target_user_id, movie_id] == 0:
                movie_scores[movie_id] = movie_scores.get(movie_id, 0) + rating * similarities[user_id]
    
    if not movie_scores:
        print(f"No specific recommendations for user {target_user_id}. Recommending highest-rated unseen movies within training set.")
        global_movie_ratings = train_data.mean().sort_values(ascending=False)
        recommendations = [movie for movie in global_movie_ratings.index if train_data.loc[target_user_id, movie] == 0]
        return recommendations[:M]
    
    recommendations = sorted(movie_scores, key=movie_scores.get, reverse=True)[:M]
    print(f"Recommendations for user {target_user_id}: {recommendations}")
    
    return recommendations

def evaluate_recommendations(true_ratings, recommended_movies):
    y_true = [1 if movie in true_ratings else 0 for movie in recommended_movies]
    y_pred = [1] * len(recommended_movies)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return precision, recall, f1

def collaborative_filter_evaluation(train_file, valid_file, K, M, metric='pearson'):
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)
    
    for target_user_id in train_data.index:
        recommendations = recommend_movies(train_data, target_user_id, K, M, metric)
        
        valid_ratings = valid_data.loc[target_user_id][valid_data.loc[target_user_id] > 0].index.tolist()
        
        if valid_ratings:
            precision_val, recall_val, f1_val = evaluate_recommendations(valid_ratings, recommendations)
            print(f"Validation Set - User {target_user_id}: Precision={precision_val:.2f}, Recall={recall_val:.2f}, F1-score={f1_val:.2f}")
        else:
            print(f"No validation ratings available for user {target_user_id}.")
            precision_val, recall_val, f1_val = 0.0, 0.0, 0.0

# Hyperparameters
K = 5  # Number of users to consider
M = 10  # Number of movies to recommend
metric = 'pearson'  # Similarity metric set to Pearson correlation

for user_file in ['train_a.txt', 'train_b.txt', 'train_c.txt']:
    print(f"\nEvaluating recommendations for dataset: {user_file}")
    collaborative_filter_evaluation(user_file, f'valid_{user_file[-5]}.txt', K, M, metric)
