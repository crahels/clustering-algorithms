from dbscan import DBSCAN
import sklearn.cluster

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

my_model = DBSCAN(eps=0.7, min_points=2)
my_labels = my_model.fit_predict(X, y)

sklearn_model = sklearn.cluster.DBSCAN(eps=0.7, min_samples=2)
sklearn_labels = sklearn_model.fit_predict(X, y)

print('Testing...')
for i in range(len(sklearn_labels)):
	if my_labels[i] != sklearn_labels[i]:
		print('Mismatch! My Labels: ' + str(my_labels[i]) + ' Sklearn Labels: ' + str(sklearn_labels[i]))
print('All Labels Match!')
