from numpy.random import shuffle
import math 
import copy
import numpy as np
INITIAL_MIN_VALUE = 999999

class KMeans:
    def __init__(self, data, ncluster, tolerance):
        self.__data = data
        self.__ncluster = ncluster
        self.__tolerance = tolerance
        self.__len_attribute = len(self.__data[0]) if len(self.__data) > 0 else 0
        self.__centroid = []
        self.__old_centroid = []
        self.__cluster = []
        
    def squared_distance(self, a, b):
        sum = 0
        for i in range(self.__len_attribute):
            sum += (a[i] - b[i])*(a[i] - b[i])
        return sum
    
    def euclidean_distance(self, a, b):
        return math.sqrt(self.squared_distance(a, b))
    
    def get_data(self, i, j):
        return self.__data[self.cluster[i][j]]
    
    def check_convergence(self):      
        error = 0
        for i in range(self.__ncluster):
            error += self.euclidean_distance(self.__centroid[i], self.__old_centroid[i])
        return (error <= self.__tolerance)
    
    def get_centroid_init(self):
        data_temp = copy.deepcopy(self.__data)
        shuffle(data_temp)
        return data_temp[:self.__ncluster]
    
    def initialize_cluster(self):
        for i in range(self.__ncluster):
            self.__cluster.append([])
        
    def initialize_centroid(self):
        self.__centroid.extend(self.get_centroid_init())
        
    def initialize(self):
        self.initialize_centroid()
        self.initialize_cluster()

    def get_new_centroid(self, index_cluster):
        self.__centroid[index_cluster] = []
        for i in range(self.__len_attribute):
            self.__centroid[index_cluster].append(0)
        
        for i in self.__cluster[index_cluster]:
            for j in range(self.__len_attribute):
                self.__centroid[index_cluster][j] += self.__data[i][j]

        for i in range(self.__len_attribute):
            self.__centroid[index_cluster][i] /= len(self.__cluster[index_cluster])
        
    def update_centroid(self):
        self.__old_centroid = copy.deepcopy(self.__centroid)
        for i in range(self.__ncluster):
            self.get_new_centroid(i)
    
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
    
    def result_cluster(self):
        cluster = [-1] * len(self.__data)
        for i in range(self.__ncluster):
            for index in self.__cluster[i]:
                cluster[index] = i
        return cluster
   
    def centroids(self):
        return self.__centroid
    
    def fit(self):
        self.initialize()
        if (len(self.__data) == 0):
            self.__centroid = []
            return []
        else:
            while True:
                self.assignment()
                self.update_centroid()
                if self.check_convergence():
                    return self.result_cluster()
                else:
                    self.__cluster = []
                    self.initialize_cluster()