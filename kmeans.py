from numpy.random import shuffle
import math 
import copy
import numpy as np
INITIAL_MIN_VALUE = 999999

class KMeans:
    # initialize attribute for KMeans model
    def __init__(self, **kwargs):
        self.__ncluster = kwargs.get('n_clusters', 2)
        self.__max_iter = kwargs.get('max_iter', 10000)
        self.__centroid = []
        self.__old_centroid = []
        self.__cluster = []
        
    # calculate squared euclidean distance between 2 data
    def squared_distance(self, a, b):
        sum = 0
        for i in range(self.__len_attribute):
            sum += (a[i] - b[i]) * (a[i] - b[i])
        return sum
    
    # calculate euclidean distance between 2 data
    def euclidean_distance(self, a, b):
        return math.sqrt(self.squared_distance(a, b))
    
    # get data at index cluster[i][j]
    def get_data(self, i, j):
        return self.__data[self.cluster[i][j]]
    
    # check whether centroid has changed or not
    def check_convergence(self):
        same = True 
        for i in range(self.__ncluster):
            for j in range(self.__len_attribute):
                if (self.__centroid[i][j] != self.__old_centroid[i][j]):
                    same = False
        return same

    # initialize initial centroid by shuffling data and take first n_cluster data
    def get_centroid_init(self):
        data_temp = copy.deepcopy(self.__data)
        shuffle(data_temp)
        return data_temp[:self.__ncluster]
    
    # initialize empty cluster
    def initialize_cluster(self):
        for i in range(self.__ncluster):
            self.__cluster.append([])
    
    # initialize centroid by calling get_centroid_init
    def initialize_centroid(self):
        self.__centroid.extend(self.get_centroid_init())
    
    # initialize initial centroid and empty cluster
    def initialize(self):
        self.initialize_centroid()
        self.initialize_cluster()

    # calculate new centroid for a cluster at index_cluster
    def get_new_centroid(self, index_cluster):
        self.__centroid[index_cluster] = []

        # initialize with zero zalue for each centroid
        for i in range(self.__len_attribute):
            self.__centroid[index_cluster].append(0)
        
        # sum all of the data in the cluster
        for i in self.__cluster[index_cluster]:
            for j in range(self.__len_attribute):
                self.__centroid[index_cluster][j] += self.__data[i][j]

        # divide sum of data with number of data in the cluster to get the new centroid for the cluster
        for i in range(self.__len_attribute):
            self.__centroid[index_cluster][i] /= len(self.__cluster[index_cluster])
    
    # update centroid for all cluster
    def update_centroid(self):
        self.__old_centroid = copy.deepcopy(self.__centroid)
        for i in range(self.__ncluster):
            self.get_new_centroid(i)

    # assign each data to the new cluster based on the new centroid
    def assignment(self):
        for i in range(len(self.__data)):
            min_value = INITIAL_MIN_VALUE
            min_index = -1
            for j in range(self.__ncluster):
                dist = self.euclidean_distance(self.__data[i], self.__centroid[j])
                if dist < min_value:
                    min_value = dist
                    min_index = j
            self.__cluster[min_index].append(i)
    
    # change result format
    def result_cluster(self):
        cluster = [-1] * len(self.__data)
        for i in range(self.__ncluster):
            for index in self.__cluster[i]:
                cluster[index] = i
        return cluster
   
    # return current centroid
    def centroids(self):
        return self.__centroid
    
    # predict cluster for the new data
    def predict(self, new_data):
        predict_data = []
        for i in range(len(new_data)):
            min_value = INITIAL_MIN_VALUE
            min_index = -1
            # assign data to the cluster with the closest centroid
            for j in range(self.__ncluster):
                dist = self.euclidean_distance(new_data[i], self.__centroid[j])
                if dist < min_value:
                    min_value = dist
                    min_index = j
            predict_data.append(min_index)
        return predict_data

    # train model and predict cluster based on trained model
    def fit_predict(self, data):
        self.__data = data
        self.__len_attribute = len(self.__data[0]) if len(self.__data) > 0 else 0

        self.initialize()
        if (len(self.__data) == 0):
            self.__centroid = []
            return []
        else:
            for i in range(self.__max_iter):
                self.assignment()
                self.update_centroid()
                if self.check_convergence():
                    return self.result_cluster()
                else:
                    self.__cluster = []
                    self.initialize_cluster()
            return self.result_cluster()