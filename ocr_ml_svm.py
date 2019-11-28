import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

# Scikit-learn digit dataset - optical recognition of handwritten digits
digits = load_digits()
X, y = digits.data, digits.target

# split dataset in train- and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# dataset given in lab
# imported csv with pandas
path = '/home/anderstask1/Documents/Kyb/MachineLearning/lab2/sklearn-lab/ocr/'

X_train = np.loadtxt(f'{path}train-data.csv', delimiter=',')
X_test = np.loadtxt(f'{path}test-data.csv', delimiter=',')
y_train = np.genfromtxt(f'{path}train-targets.csv', dtype='str', delimiter=',')
y_test = np.genfromtxt(f'{path}test-targets.csv', dtype='str', delimiter=',')

print("File load completed")

# SVM chosen as classifier

# C is the primaX_traX_traininl problem of the SVM
# rbf kernel is the Gaussian kernel
# gamma is gaussian width

# k-fold cross-validation accuracy to find gamma with best average accuracy
kf = KFold(n_splits=3, shuffle=True, random_state=42)

possible_parameters = {
    'C': [1e0, 1e1, 1e2, 1e3],
    'gamma': [1e-1, 1e-2, 1e-3, 1e-4]
}

# kernel functions determine hyperplane, rbf = radial basis function
svc = SVC(kernel='rbf')

# finding the best gamma and C values with grid search

# The GridSearchCV is itself a classifier
# we fit the GridSearchCV with the training data
# and then we use it to predict on the test set
# n_jobs is number of threads to parallelize the search over (-1 uses all processors)
clf = GridSearchCV(svc, possible_parameters, n_jobs=-1, cv=3)

print("Grid search completed")

# Train over the full training set
clf.fit(X_train, y_train)

print("Training completed")

# Evaluate on the test set
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)
