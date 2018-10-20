from sklearn.metrics.pairwise import manhattan_distances
import numpy as np
import random
from random import randint
import copy

class kmedoids:
    # initialize variables in class
    def __init__(self, data, **kwargs):
        self.__pointer_data = data
        self.__n_clusters = kwargs.get('n_clusters', 2)
        # randomize medoids if it is not defined
        medoid_indexes = kwargs.get('medoid_indexes', random.sample(range(0, len(self.__pointer_data)), self.__n_clusters))
        self.__medoid_indexes = medoid_indexes
        self.__update_medoid_indexes = []
        self.__clusters = []
        self.__update_clusters = []
        self.__memory = [0] * len(self.__pointer_data)
        self.__distance_matrix = manhattan_distances(self.__pointer_data)
        self.__current_error = 0
        self.__update_error = 0
    
    # create clusters by assigning all data to nearest medoid
    def assign_cluster(self, medoids):
        cluster = []
        for i in range(len(medoids)):
            cluster.append([])
        for i in range(len(self.__pointer_data)):
            min_dist_idx = 0 # cluster index
            min_dist = self.__distance_matrix[i][medoids[0]] # distance to certain medoid
            for j in range(1, len(medoids)):
                if min_dist > self.__distance_matrix[i][medoids[j]]:
                    min_dist_idx = j
                    min_dist = self.__distance_matrix[i][medoids[j]]
            cluster[min_dist_idx].append(i)
        return cluster
    
    # calculate absolute error using manhattan distance
    def absolute_error(self, cluster, medoids):
        error = 0
        for i in range(len(medoids)):
            for j in cluster[i]:
                error += self.__distance_matrix[medoids[i]][j]
        return error
    
    # randomize new medoids and do temporary cluster assigning
    def update_medoids(self):
        rand_cluster = randint(0, len(self.__medoid_indexes)-1)
        rand_medoid = random.choice(self.__clusters[rand_cluster])
            
        self.__update_medoid_indexes = []
        self.__update_medoid_indexes = copy.deepcopy(self.__medoid_indexes)
        self.__update_medoid_indexes[rand_cluster] = rand_medoid
        self.__update_clusters = self.assign_cluster(self.__update_medoid_indexes)
    
    # clustering iteration
    def clustering(self):
        self.__clusters = self.assign_cluster(self.__medoid_indexes)
        self.__current_error = self.absolute_error(self.__clusters, self.__medoid_indexes)
        self.update_medoids()
        self.__update_error = self.absolute_error(self.__update_clusters, self.__update_medoid_indexes)
        diff = self.__update_error - self.__current_error

        while (diff < 0) or (self.__clusters != self.__update_clusters):
            if (diff < 0): # update medoids produce smaller absolute error
                self.__medoid_indexes = []
                self.__clusters = []
                # swap medoids
                self.__medoid_indexes = copy.deepcopy(self.__update_medoid_indexes)
                self.__clusters = copy.deepcopy(self.__update_clusters)
                self.__current_error = self.__update_error
            # randomize new medoids
            self.update_medoids()
            self.__update_error = self.absolute_error(self.__update_clusters, self.__update_medoid_indexes)
            diff = self.__update_error - self.__current_error
        
        return self.__clusters
    
    # return predicted cluster
    def get_cluster(self):
        for i in range(len(self.__clusters)):
            for j in self.__clusters[i]:
                self.__memory[j] = i
        return self.__memory
    
    # return cluster medoids
    def get_medoids(self):
        medoids = []
        for i in self.__medoid_indexes:
            medoids.append(self.__pointer_data[i])
        return medoids