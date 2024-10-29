import math
import statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import euclidean as scipy_euclidean, cosine as scipy_cosine

# returns Euclidean distance between vectors a and b
def euclidean(a,b):
    if len(a)!=len(b): 
        raise("Dimensions of a and b not the same")
    a = [float(i) for i in a]  # Convert to float
    b = [float(i) for i in b] 

    #calculate Euclidean distance bewteen vectors a and b
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
def knn(train, query, metric, k=10):
    def distance(a,b):
        if metric == 'euclidean': return euclidean(a, b) 
        elif metric == 'cosim': return cosim(a, b)
        else: return("error")
        
    predicted_labels = []
    for q in query:
        query_features = q[-1]
        distances=[(distance(item[-1], query_features), item[0]) for item in train]
        nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k] #select k nearest neighbors
        nearest_labels = [(label) for _distance, label in nearest_neighbors]
        max_labels = statistics.mode(nearest_labels) #assign most common label among neighbors
        predicted_labels.append(max_labels) 
    return predicted_labels

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    def distance(a,b):
        if metric == 'euclidean': 
            delta = np.sum(np.square(a-b))
            return np.sqrt(delta)
        elif metric == 'cosim': 
            return 1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        else: return("error")

    # get closest mean to a given data point
    def get_nearest_centroid(data_point, means):
        distances = [distance(means[x], data_point) for x in range(k)]
        return distances.index(min(distances)) # Return minimum distance
    
    # Init centroids
    def init_centroids():
        return np.mean(train_data, axis=0) + (2 * np.random.rand(k, data_dim) - np.ones((k, data_dim)))

    # preprocess train data, get constants for problem
    train_data = np.array([row[1] for row in train], dtype=np.float64)
    train_size, data_dim = train_data.shape

    # preprocess query data
    query_data = np.array([row[1] for row in query], dtype=np.float64)

    # k number of means
    k = 10

    # Init means
    means = init_centroids()

    #iterative updating of cluster centers
    e = 1e-4
    max_iters = 500
    for _ in range(max_iters):
        movements = np.zeros(k) # This will store the distances of movements of centroids
        # give labels to data closest to specific mean
        centroids = np.zeros((train_size))
        for ind, data in enumerate(train_data):
            centroids[ind] = get_nearest_centroid(data_point=data, means=means)

        # update means
        for k_i in range(k):
            current_cluster = train_data[centroids == k_i, :] # Filter data to current cluster
            new_mean = np.mean(current_cluster, axis=0) # Compute new mean of the cluster
            cluster_movement_distance = distance(means[k_i], new_mean) # Calculate distance from old mean to new mean
            means[k_i] = new_mean # Update
            movements[k_i] = cluster_movement_distance # Append cluster movements

        # If the max distance moved is greater than or equal to epsilon, it has converged
        if movements.max() <= e:
            print('Converged')
            return means
                
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
def apply_pca(train_data, query_data, n_components=2):
    
    # Extract features and labels from the training data
    train_features = [item[-1] for item in train_data]
    query_features = [item[-1] for item in query_data]
    
    # Fit PCA on the combined train and query data to ensure both are transformed in the same space
    pca = PCA(n_components=n_components)
    
    train_features = np.array(train_features)
    
    # Apply PCA
    reduced_train_features = pca.fit_transform(train_features)
    reduced_query_features = pca.transform(query_features)

    new_train_set = [(train_data[i][0], reduced_train_features[i]) for i in range(len(train_features))]
    new_query_set = [(query_data[i][0], reduced_query_features[i]) for i in range(len(query_features))]
    
    return new_train_set, new_query_set


def main(k=10, num_components = 25, metric='euclidean'):
    # tests for metrics 
    a=[1,2]
    b=[3,4]
    c=[1,2,3,4]
    d=[7,8,2,4]

    assert math.isclose(euclidean(a,b), scipy_euclidean(a,b))
    assert math.isclose(cosim(a,b), scipy_cosine(a,b))
    # print(pearson_correlation(c,d))
    # print(hamming(c,d))

    from sklearn.metrics import accuracy_score
    training_data = read_data('mnist_train.csv')
    validation_data = read_data('mnist_valid.csv')

    if num_components == -1: reduced_train_data, reduced_query_data = training_data, validation_data
    else: reduced_train_data, reduced_query_data = apply_pca(training_data, validation_data, n_components=num_components)

    from sklearn.neighbors import KNeighborsClassifier

    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X=[thing[1] for thing in reduced_train_data], y=[thing[0] for thing in reduced_train_data])
    # predicted_labels = knn.predict([thing[1] for thing in reduced_query_data])

    predicted_labels = knn(
        train=reduced_train_data, 
        query=reduced_query_data, 
        metric=metric, 
        k=k
    )

    truth_labels = [item[0] for item in reduced_query_data]
    # print("Truth labels:", truth_labels[:10])
    # print("Predicted labels:", predicted_labels[:10])
    print(f'KNN (k={k} components={num_components} with {metric} metric) Accuracy = {accuracy_score(truth_labels, predicted_labels)}')
    #show('mnist_valid.csv','pixels')

#tests k-means implementation on MNIST dataset
def test_kmeans():
    # prep training set
    training_data = read_data('mnist_train.csv')
    # prep validation set
    validation_data = read_data('mnist_valid.csv')
    # prep test set
    test_data = read_data('mnist_test.csv')

    # run kmeans, test means on val data and test data
    # pred_val_labels = kmeans(train=training_data, query=validation_data, metric='euclidean')

    pcaed_training_data, pcaed_val_data = apply_pca(train_data=training_data, query_data=validation_data, n_components=25)

    # kmeans on pcaed train and val
    means = kmeans(train=pcaed_training_data, query=pcaed_val_data, metric='euclidean')

    # pred_test_labels = kmeans(train=training_data, query=test_data, metric='euclidean')
    return 0

if __name__ == "__main__":
    # grid search for KNN
    # for k_val in [10]:
    #     for num_components_val in [50]: # -1 num_components_val means no PCA
    #         for metric_val in ['euclidean', 'cosim']:
    #             main(k=k_val, num_components=num_components_val, metric=metric_val) 


    acc = test_kmeans()
    # print(f'Accuacy obtained on MNIST for k-means = {acc}')
    