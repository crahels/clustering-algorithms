from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
import numpy as np

class agglomerative:
    def __init__(self, **kwargs):
        self.__pointer_data = []
        self.__clusters = []
        self.__memory = []
        self.__distance_matrix = []
        self.__n_cluster = kwargs.get('n_cluster', 2)
        self.__linkage = kwargs.get('linkage', 'single')
        self.__metric = kwargs.get('metric', 'euclidean')
        
    def preparation(self):
        if self.__metric == 'euclidean':
            self.__distance_matrix = euclidean_distances(self.__pointer_data)
        elif self.__metric == 'manhattan':
            self.__distance_matrix = manhattan_distances(self.__pointer_data)
        elif self.__metric == 'cosine':
            self.__distance_matrix = cosine_distances(self.__pointer_data)
            
        n_data = len(self.__pointer_data)
        
        for i in range(0, n_data):
            self.__memory.append([i])
            
        self.__clusters = [0] * len(self.__pointer_data)
        
    def clustering(self):
        for iteration in range(len(self.__pointer_data) - self.__n_cluster):
            cluster_a = 0
            cluster_b = 1
            if (len(self.__memory[0])==1) and (len(self.__memory[1])==1):
                min_distance = self.__distance_matrix[0][1]
            else:
                min_distance = self.cluster_distance(self.__memory[0], self.__memory[1])
    
            for i in range(len(self.__memory)):
                for j in range(len(self.__memory)):
                    if (i != j) and (i < j):
                        if (len(self.__memory[i])==1) and (len(self.__memory[j])==1):
                            dist = self.__distance_matrix[i][j]
                        else:
                            dist = self.cluster_distance(self.__memory[i], self.__memory[j])
                        if dist < min_distance:
                            cluster_a = i
                            cluster_b = j
                            min_distance = dist
            temp = self.__memory.pop(cluster_b)
            for i in range(len(temp)):
                self.__memory[cluster_a].append(temp[i])
                
    def cluster_distance(self, a, b):
        distance = 0
        if self.__linkage == 'single':
            distance = self.__distance_matrix[a[0]][b[0]]
            for i in a:
                for j in b:
                    if distance > self.__distance_matrix[i][j]:
                        distance = self.__distance_matrix[i][j]
        elif self.__linkage == 'complete':
            for i in a:
                for j in b:
                    if distance < self.__distance_matrix[i][j]:
                        distance = self.__distance_matrix[i][j]
        elif self.__linkage == 'average':
            for i in a:
                for j in b:
                    distance += self.__distance_matrix[i][j]
            distance = distance/(len(a) * len(b))
        elif self.__linkage == 'average_group':
            member_a = []
            member_b = []
            for i in a:
                member_a.append(self.__pointer_data[i])
            for i in b:
                member_b.append(self.__pointer_data[i])
            mean_a = np.array(member_a).mean(axis=0)
            mean_b = np.array(member_b).mean(axis=0)
            mean_clusters = []
            mean_clusters.append(mean_a)
            mean_clusters.append(mean_b)
            if self.__metric == 'euclidean':
                distance = euclidean_distances(mean_clusters)[0][1]
            elif self.__metric == 'manhattan':
                distance = manhattan_distances(mean_clusters)[0][1]
            elif self.__metric == 'cosine':
                distance = cosine_distances(mean_clusters)[0][1]
        return distance
    
    def fit_predict(self, data):
        self.__pointer_data = data
        self.preparation()
        self.clustering()
        for i in range(len(self.__memory)):
            for j in self.__memory[i]:
                self.__clusters[j] = i
        return self.__clusters