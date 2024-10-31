import math
import statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Literal, Union
from scipy.spatial.distance import euclidean as scipy_euclidean, cosine as scipy_cosine
import pandas as pd

# returns Euclidean distance between vectors a and b
def euclidean(a,b):
    if len(a)!=len(b): 
        raise("Dimensions of a and b not the same")
    a = [float(i) for i in a]  # Convert to float
    b = [float(i) for i in b] 

    #calculate Euclidean distance between vectors a and b
    dist = math.sqrt(sum([(a[i]-b[i])**2 for i in range(len(a))]))
    return dist 
        
# returns Cosine Similarity between vectors a and b
def cosim(a,b):
    #ensures vectors have same dimensions
    if len(a)!=len(b): 
        raise("Dimensions of a and b not the same")
    else:
        zip(a, b) #traversing the vectors for every dimension
        numerator = sum(x * y for x, y in zip(a,b)) #compute dot product
        mod_a = math.sqrt(sum(x ** 2 for x in a)) #compute magnitude of vector a
        mod_b = math.sqrt(sum(y ** 2 for y in b)) #compute magnitudeof vector b
        denom = mod_a*mod_b 
        if denom!=0:
            #calculate cosine similarity
            dist=numerator/denom
            return (1-dist)
        return 0

#returns the pearson correlation between vectors 'a' and 'b'
def pearson_correlation(a,b):
    if len(a)!=len(b):
        print("Dimensions of a and b not the same")
    else:
        zip(a, b) #traversing the vectors for every dimension
        mean_x = sum(a)/len(a)
        mean_y = sum(b)/len(b)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(a, b))
        mod_a = sum((x-mean_x) ** 2 for x in a)
        mod_b = sum((y-mean_y) ** 2 for y in b)
        denom = math.sqrt(mod_a*mod_b)
        while denom!=0:
            dist=numerator/denom
            return(dist)
        return(0)

#returns the hamming distance between vectors 'a' and 'b'
def hamming(a,b):
    if len(a)!=len(b):
        print("Dimensions of a and b not the same")
    else:
        dist=sum(x != y for x, y in zip(a, b))
        return dist
    return(0)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train: np.ndarray, query: np.ndarray, metric: Literal['euclidean', 'cosim']) -> np.ndarray:
    def distance(a,b):
        if metric == 'euclidean': 
            return euclidean(a, b) 
        elif metric == 'cosim': 
            return cosim(a, b)
        else: 
            raise ValueError
    k = 10
    predicted_labels = []
    for q in query:
        query_features = q[-1]
        distances=[(distance(item[1:], query_features), item[0]) for item in train]
        nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k] #select k nearest neighbors
        nearest_labels = [(label) for _distance, label in nearest_neighbors]
        max_labels = statistics.mode(nearest_labels) #assign most common label among neighbors
        predicted_labels.append(max_labels) 
    return np.array(predicted_labels, dtype=np.int64)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train: np.ndarray, query: np.ndarray, metric: Literal['euclidean', 'cosim'], verbose: bool = True) -> np.ndarray:
    def distance(a,b):
        if metric == 'euclidean': 
            return euclidean(a, b)
        elif metric == 'cosim': 
            return cosim(a, b)
        else: 
            raise ValueError

    # get closest mean to a given data point
    def get_nearest_centroid(data_point, means):
        distances = [distance(means[x], data_point) for x in range(k)]
        return np.argmin(distances) # Return minimum distance
    
    # Init centroids
    def init_centroids(train_data: np.ndarray):
        data_dim = train_data.shape[-1]
        return np.mean(train_data, axis=0) + (2 * np.random.rand(k, data_dim) - np.ones((k, data_dim)))
    
    # Function to cluster data
    def cluster_data(data: np.ndarray, means: np.ndarray) -> np.ndarray:
        return np.array([get_nearest_centroid(row, means) for row in data])

    # k number of means
    k = 10

    # Init means
    means = init_centroids(train_data=train)

    #iterative updating of cluster centers (TRAINING)
    e = 1e-4
    max_iters = 500
    for i in range(max_iters):
        movements = np.zeros(k) # This will store the distances of movements of centroids
        centroids = cluster_data(train, means=means)

        # update means
        for k_i in range(k):
            current_cluster = train[centroids == k_i, :] # Filter data to current cluster
            new_mean = np.mean(current_cluster, axis=0) # Compute new mean of the cluster
            cluster_movement_distance = distance(means[k_i], new_mean) # Calculate distance from old mean to new mean
            means[k_i] = new_mean # Update
            movements[k_i] = cluster_movement_distance # Append cluster movements

        # If the max distance moved is greater than or equal to epsilon, it has converged
        max_distance = movements.max()
        if verbose:
            print(f"Iteration {i + 1}, max centroid movement: {max_distance:.6f}")
        if max_distance <= e:
            print('Converged!')
            break

    # Predict clusters on validation set (VALIDATION)
    query_clusters = cluster_data(query, means=means)
    return query_clusters, means

'''
Paragraph for Part I #3.

For the k-means classifier

For the quantitative metric for measuring how well the clusters align with the labels, we want data with known labels to tend to be in the same cluster, regardless of the number of clusters k. We compute the entropy of the data labelled with a specifc label over all clusters k. 

we decided that it 
'''


def cluster_allignment(query, means, metric='euclidean'):
    def distance(a,b):
        if metric == 'euclidean': 
            return euclidean(a, b)
        elif metric == 'cosim': 
            return cosim(a, b)
        else: 
            raise ValueError
    # get closest mean to a given data point
    def get_nearest_centroid(data_point, means):
        distances = [distance(means[x], data_point) for x in range(k)]
        return np.argmin(distances) # Return minimum distance

    def entropy_calc(arr, base=2): 
        # Helper function to avoid math error for log(0)
        def entropy_term(freq, base):
            if freq == 0 or freq == 1: return 0
            return -freq * math.log(freq, base)
        return sum([entropy_term(freq, base=base) for freq in arr])
    # segment the test set into collections of datapoints by label ie 1 group, 2 group, etc
    num_labels, k = 10, 10
    query_grouping = [[] for _ in range(num_labels)]
    for label, feature in query:
        query_grouping[int(label)].append([label, feature])
    # within each of these segments (ie within the 1 group), how many datapoints are in each cluster
    final_entropy = 0
    for group in query_grouping:
        # this gives a distribution over the clusters, where we can calculate an entropy for a specific segment, base would be k
        # freq is the number of means assigned to the index-th mean
        freq = [0 for _ in range(num_labels)]
        for label, feature in group:
            index = get_nearest_centroid(data_point=feature, means=means)
            freq[index] += (1/len(group))

        group_entropy = entropy_calc(freq, base=k)
        # average the entropy over all the segments, return the entropy 
        final_entropy += ((group_entropy)/num_labels)
    return final_entropy


def generate_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, n_labels: int = 10) -> np.ndarray:
    assert y_pred.shape == y_true.shape
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64) # (true, preds)
    for pred, true in zip(y_pred, y_true):
        matrix[true, pred] += 1 # Iterate count
    return matrix

#reads data from a file and processes it into a usable dataset format
def read_data(file_name):
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
#visualizes data from the file, displaying either as pixels or numerical values
def show(file_name,mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

#applies PCA for dimensionality reduction on the given train and query datasets
def apply_pca(train_data, query_data, n_components=50, return_labels=True):
    
    # Extract features and labels from the training data
    train_features = [item[-1] for item in train_data]
    query_features = [item[-1] for item in query_data]
    
    # Fit PCA on the combined train and query data to ensure both are transformed in the same space
    pca = PCA(n_components=n_components)
    
    train_features = np.array(train_features)
    
    # Apply PCA
    reduced_train_features = pca.fit_transform(train_features)
    reduced_query_features = pca.transform(query_features)

    if return_labels:
        new_train_set = [(train_data[i][0], reduced_train_features[i]) for i in range(len(train_features))]
        new_query_set = [(query_data[i][0], reduced_query_features[i]) for i in range(len(query_features))]
        return new_train_set, new_query_set
    return reduced_train_features, reduced_query_features


def main(algorithm):
    from sklearn.metrics import accuracy_score
    # Load in data
    training_data = read_data('mnist_train.csv')
    validation_data = read_data('mnist_valid.csv')
    test_data = read_data('mnist_test.csv')

    if algorithm == 'knn':
        # Apply PCA for KNN
        print('Applying PCA for KNN model...')
        reduced_train_data, reduced_query_data = apply_pca(training_data, validation_data, n_components=50)

        # Fit and predict KNN on validation
        print('Fitting KNN on training data and getting preds on validation...')
        y_pred = knn(train=reduced_train_data, query=reduced_query_data, metric='euclidean') # validation preds
        y_true = np.array([item[0] for item in reduced_query_data], dtype=np.int64) # validation true
        print(y_true.shape, y_pred.shape)

        # Getting Accuracy
        print('Getting accuracy...')
        print(accuracy_score(y_true=y_true, y_pred=y_pred))
        
        # Generate confusion matrix
        print('Generating confusion matrix...')
        conf_mat = generate_confusion_matrix(y_pred=y_pred, y_true=y_true)
        print(conf_mat)

    elif algorithm == 'kmeans':
        # "train" generates means
        reduced_train_data, reduced_query_data = apply_pca(training_data, validation_data, n_components=50, return_labels=False)
        query_clusters, means = kmeans(train=reduced_train_data, query=reduced_query_data, metric='euclidean')

        # test alignment with cluster allignment 
        reduced_train_data, reduced_test_data = apply_pca(training_data, validation_data, n_components=50, return_labels=True)
        quant_metric = cluster_allignment(query=reduced_test_data, means=means)
        print(f'Cluster Allighment (entropy) is {quant_metric}')

if __name__ == "__main__":
    main(algorithm='kmeans')