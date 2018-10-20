from dbscan import DBSCAN
import sklearn.cluster
from sklearn.datasets import load_iris
from sklearn import metrics
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

my_model = DBSCAN(eps=0.4, min_points=3)
my_labels = my_model.fit_predict(X, y)

sklearn_model = sklearn.cluster.DBSCAN(eps=0.4, min_samples=3)
sklearn_labels = sklearn_model.fit_predict(X, y)

print('Comparing with Scikit-Learn...')
mismatch_count = 0
for i in range(len(sklearn_labels)):
	if my_labels[i] != sklearn_labels[i]:
		print('Mismatch! My Labels: ' + str(my_labels[i]) + ' Sklearn Labels: ' + str(sklearn_labels[i]))
		mismatch_count += 1
if mismatch_count == 0:
	print('All Labels Match!')

print('Actual Labels:')
print(y)
print('Predicted Labels:')
print(my_labels)

def purity_score(labels_true, labels_pred):
	contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
	return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

print('Homogeneity Score: %.3f' % metrics.homogeneity_score(labels_true=y, labels_pred=my_labels))
print('Completeness Score: %.3f' % metrics.completeness_score(labels_true=y, labels_pred=my_labels))
print('V Measure Score: %.3f' % metrics.v_measure_score(labels_true=y, labels_pred=my_labels))
print('Purity Score: %.3f' % purity_score(labels_true=y, labels_pred=my_labels))
