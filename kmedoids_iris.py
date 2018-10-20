from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, calinski_harabaz_score
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
from sklearn import datasets
from kmedoids import kmedoids

result_score = {}

def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis = 0) / np.sum(cm))

def score(**kwargs):
    cluster_score = {
        "purity": purity_score(kwargs['target'], kwargs['result']),
        "homogeneity": homogeneity_score(kwargs['target'], kwargs['result']),
        "completeness": completeness_score(kwargs['target'], kwargs['result']),
        "v-measure": v_measure_score(kwargs['target'], kwargs['result']),
    }
    
    print("Purity: %0.3f" % cluster_score['purity'])
    print("Homogeneity: %0.3f" % cluster_score['homogeneity'])
    print("Completeness: %0.3f" % cluster_score['completeness'])
    print("V-measure: %0.3f" % cluster_score['v-measure'])
    result_score[kwargs['algo_name']] = cluster_score

iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target

kmedoids_model = kmedoids(iris_data, n_clusters=3)
kmedoids_model.clustering()
predict = kmedoids_model.get_cluster()
print('Medoids: ', kmedoids_model.get_medoids())
print('Labels prediction: ', predict)
score(algo_name='Kmedoids', data=iris_data, target=iris_target, result=predict)