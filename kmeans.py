from numpy.random import shuffle
import math 
import copy
import numpy as np
INITIAL_MIN_VALUE = 999999

class KMeans:
    def __init__(self, data, ncluster, tolerance):
        self.data = data
        self.ncluster = ncluster
        self.tolerance = tolerance
        self.len_attribute = len(self.data[0])
        self.centroid = []
        self.old_centroid = []
        self.cluster = []
        
    def squared_distance(self, a, b):
        sum = 0
        for i in range(self.len_attribute):
            sum += (a[i] - b[i])*(a[i] - b[i])
        return sum
    
    def euclidean_distance(self, a, b):
        return math.sqrt(self.squared_distance(a, b))
    
    def get_data(self, i, j):
        return self.data[self.cluster[i][j]]
    
    def check_convergence(self):      
        error = 0
        for i in range(self.ncluster):
            error += self.euclidean_distance(self.centroid[i], self.old_centroid[i])
        return (error <= self.tolerance)
    
    def get_centroid_init(self):
        data_temp = copy.deepcopy(self.data)
        shuffle(data_temp)
        return data_temp[:self.ncluster]
    
    def initialize_cluster(self):
        for i in range(self.ncluster):
            self.cluster.append([])
        
    def initialize_centroid(self):
        self.centroid.extend(self.get_centroid_init())
        
    def initialize(self):
        self.initialize_centroid()
        self.initialize_cluster()

    def get_new_centroid(self, index_cluster):
        self.centroid[index_cluster] = []
        for i in range(self.len_attribute):
            self.centroid[index_cluster].append(0)
        
        for i in self.cluster[index_cluster]:
            for j in range(self.len_attribute):
                self.centroid[index_cluster][j] += self.data[i][j]

        for i in range(self.len_attribute):
            self.centroid[index_cluster][i] /= len(self.cluster[index_cluster])
        
    def update_centroid(self):
        self.old_centroid = copy.deepcopy(self.centroid)
        for i in range(self.ncluster):
            self.get_new_centroid(i)
    
    def assignment(self):
        for i in range(len(self.data)):
            min_value = INITIAL_MIN_VALUE
            min_index = -1
            for j in range(self.ncluster):
                dist = self.euclidean_distance(self.data[i], self.centroid[j])
                if dist < min_value:
                    min_value = dist
                    min_index = j
            self.cluster[min_index].append(i)
    
    def result_cluster(self):
        cluster = [-1] * len(self.data)
        for i in range(self.ncluster):
            for index in self.cluster[i]:
                cluster[index] = i
        return cluster
    
    def fit(self):
        self.initialize()
        while True:
            self.assignment()
            self.update_centroid()
            if self.check_convergence():
                return self.result_cluster()
            else:
                self.cluster = []
                self.initialize_cluster()