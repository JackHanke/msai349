import math
import statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import array
import numpy as np
#from scipy.spatial.distance import euclidean as scipy_euclidean, cosine as scipy_cosine

# returns Euclidean distance between vectors a and b
def euclidean(a,b):
    a = [float(i) for i in a]  # Convert to float
    b = [float(i) for i in b] 

    #calculate Euclidean distance bewteen vectors a and b
    dist = math.sqrt((math.pow((a[0]-b[0]),2))+(math.pow((a[1]-b[1]),2)))
    return(dist)
        
# returns Cosine Similarity between vectors a and b
def cosim(a,b):
    #ensures vectors have same dimensions
    if len(a)!=len(b):
        print("Dimensions of a and b not the same")
    else:
        zip(a, b) #traversing the vectors for every dimension
        numerator = sum(x * y for x, y in zip(a,b)) #compute dot product
        mod_a = math.sqrt(sum(x ** 2 for x in a)) #compute magnitude of vector a
        mod_b = math.sqrt(sum(y ** 2 for y in b)) #compute magnitudeof vector b
        denom = mod_a*mod_b 
        while denom!=0:
            #calculate cosine similarity
            dist=numerator/denom
            return(dist)
        return(0)

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
def knn(train, query, metric):
    def distance(a,b):
        if metric == 'euclidean': 
            return euclidean(a, b) 
        elif metric == 'cosim': 
            return cosim(a, b)
        else:
            return("error")
        
    predicted_labels = []
    k = 3 #number of neighbors to consider
    for q in query:
        query_features = q[-1]
        distances=[(distance(list(map(int, item[-1])), list(map(int, query_features))), item[0]) for item in train]
        nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k] #select k nearest neighbors
        nearest_labels = [(label) for _, label in nearest_neighbors]
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
        elif metric == 'cosim': # TODO rewrite in vector form! 
            return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        else: return("error")

    # get closest mean to a given data point
    def get_label(data_point, means):
        closest = float('inf')
        for k, mean in enumerate(means):
            dist = distance(data_point, mean)
            if dist < closest: closest, label = dist, k
        return label

    # preprocess train data, get constants for problem
    train_data = np.array([row[1] for row in train], dtype=np.float64)
    train_labels = np.array([row[0] for row in train], dtype=np.float64) # IGNORE !!!
    train_size, data_dim = train_data.shape

    # preprocess query data
    query_data = np.array([row[1] for row in query], dtype=np.float64)
    query_labels = np.array([row[0] for row in query], dtype=np.float64)

    # k number of means
    k = 10
    # initialize means as mean of training data plus noise
    np.random.seed(1)
    means = np.mean(train_data, axis=0)+(2*np.random.rand(k, data_dim) - np.ones((k, data_dim)))
    # means = 5*np.random.rand(k, data_dim)

    #iterative updating of cluster centers
    sum_dists = 1
    while sum_dists > 0.001:
        # give labels to data closest to specific mean
        labels = np.zeros((train_size))
        for ind, data in enumerate(train_data):
            labels[ind] = get_label(data_point=data, means=means)

        print(np.unique(labels, return_counts=True))
        # update means
        for k_i in range(k):
            mask = (labels == k_i)
            current_cluster = train_data[mask, ]
            sum_dists = 0
            if current_cluster.shape[0] > 0: 
                new_mean = np.mean(current_cluster, axis=0)
                sum_dists += distance(means[k_i], new_mean)
                print(f'dist between old and new means = {distance(means[k_i], new_mean)}')
                means[k_i] = new_mean

        # query
        query_size = query_labels.shape[0]
        pred_query_labels = np.zeros((query_size))
        for ind, data in enumerate(query_data):
            pred_query_labels[ind] = get_label(data_point=data, means=means)
        query_acc = sum(pred_query_labels == query_labels)/query_size
        print(f'>>> Query acc = {query_acc}')
        print(f'loop completed')

    return(labels)

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


def main():
    a=[1,2]
    b=[3,4]
    c=[1,2,3,4]
    d=[7,8,2,4]
    print(euclidean(a,b))
    print(cosim(c,d))
    print(pearson_correlation(c,d))
    print(hamming(c,d))
    from sklearn.metrics import accuracy_score
    training_data = read_data('mnist_train.csv')
    validation_data = read_data('mnist_valid.csv')
    reduced_train_data, reduced_query_data = apply_pca(training_data, validation_data)
    predicted_labels = knn(reduced_train_data, reduced_query_data, "cosim")
    truth_labels = [item[0] for item in reduced_query_data]
    print("Truth labels:", truth_labels[:10])
    print("Predicted labels:", predicted_labels[:10])
    print(accuracy_score(truth_labels, predicted_labels))
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
    pred_val_labels = kmeans(train=training_data, query=validation_data, metric='euclidean')
    # pred_test_labels = kmeans(train=training_data, query=test_data, metric='euclidean')
    return 0

if __name__ == "__main__":
    main()

    # acc = test_kmeans()
    # print(f'Accuacy obtained on MNIST for k-means = {acc}')
    