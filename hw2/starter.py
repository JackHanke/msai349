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
    dist = math.sqrt((math.pow((a[0]-b[0]),2))+(math.pow((a[1]-b[1]),2)))
    return(dist)
        
# returns Cosine Similarity between vectors a and b
def cosim(a,b):
    if len(a)!=len(b):
        print("Dimensions of a and b not the same")
    else:
        zip(a, b) #traversing the vectors for every dimension
        numerator = sum(x * y for x, y in zip(a,b))
        mod_a = math.sqrt(sum(x ** 2 for x in a))
        mod_b = math.sqrt(sum(y ** 2 for y in b))
        denom = mod_a*mod_b
        while denom!=0:
            dist=numerator/denom
            return(dist)
        return(0)

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
        
    predicted_labels= []
    k = 3
    #print(x for x, label in train)
    for q in query:
        #print(q[1])
        query_features = q[1]
        print(query_features)
        distances=[(distance(x, query_features), label) for x, label in train]
        nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k]
        nearest_labels = [(label) for _, label in nearest_neighbors]
        max_labels = statistics.mode(nearest_labels)
        predicted_labels.append(max_labels) 
    return(predicted_labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    def distance(a,b):
        if metric == 'euclidean': 
            delta = a-b
            return np.sqrt(np.dot(delta, delta.transpose(), axis=1), axis=1)
        elif metric == 'cosim': 
            return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        else: return("error")

    labels, k = [], 3
    train = np.repeat(train[:, :, np.newaxis], repeats=k, axis=2)
    print(f'train shape = {train.shape}')
    train_size, data_dim, _k = train.shape
    means = 256*np.random.rand(k, data_dim)

    
    # label every point
    repeated_means = np.dot(np.ones((train_size, k)), means)
    print(repeated_means.shape)
    # print(temp.shape)
    # temp = distance(train, repeated_means)
    # point_labels = np.argmin(temp, axis=0)
    # print(point_labels)
    input('uhh')

    return(labels)

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

def apply_pca(train_data, query_data, n_components=2):
    
    # Extract features and labels from the training data
    train_features = [features for features, label in train_data]
    
    # Fit PCA on the combined train and query data to ensure both are transformed in the same space
    pca = PCA(n_components=n_components)
    
    # Combine the training and query data for consistent transformation
    all_data = np.array(train_features + query_data)
    
    # Apply PCA
    all_data_reduced = pca.fit_transform(all_data)
    
    # Split the reduced data back into train and query
    reduced_train_features = all_data_reduced[:len(train_features)]
    reduced_query_features = all_data_reduced[len(train_features):]
    
    return reduced_train_features, reduced_query_features

    # # Sample data
    # #data = np.array([[2, 3, 5], [5, 8, 11], [1, 2, 3], [7, 1, 5]])

    # # Standardize the data
    # scaler = StandardScaler()
    # data_standardized = scaler.fit_transform(query)

    # # Perform PCA
    # pca = PCA(n_components=2)  # Reduce to 2 dimensions
    # data_reduced = pca.fit_transform(data_standardized)

    # print(data_reduced)

def main():
    a=[1,2]
    b=[3,4]
    c=[1,2,3,4]
    d=[7,8,2,4]
    print(euclidean(a,b))
    print(cosim(c,d))
    print(pearson_correlation(c,d))
    print(hamming(c,d))
    training_data = read_data('mnist_train.csv')
    validation_data = read_data('mnist_valid.csv')
    #reduced_training_data=feature_PCA(training_data)
    reduced_train_features, reduced_query_features = apply_pca(training_data, validation_data)
    knn(reduced_train_features, reduced_query_features, "cosim")
    #test()
    #show('mnist_valid.csv','pixels')

def test_kmeans():
    training_data = read_data('mnist_train.csv')
    validation_data = read_data('mnist_valid.csv')
    test_data = read_data('mnist_test.csv')

    # print(training_data[0])
    unlabelled_data = np.array([row[1] for row in training_data], dtype=np.float64)

    # np.delete(training_data, 0, axis=1) # remove labels from training data

    kmeans_labels = kmeans(train=unlabelled_data, query=0, metric='euclidean')
    # acc = test()
    acc = 0
    return acc

if __name__ == "__main__":
    # main()

    acc = test_kmeans()
    print(f'Accuacy obtained on MNIST for k-means = {acc}')
    