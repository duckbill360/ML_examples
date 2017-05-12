import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])    # features
Y = np.array([1, 1, 1, 2, 2, 2])    # labels

# classifier
clf = GaussianNB()
clf.fit(X, Y)
GaussianNB(priors=None)
print(clf.predict([[-0.8, -1]]))
