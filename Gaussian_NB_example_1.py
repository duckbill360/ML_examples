import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# known data
feature_lst = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
X = np.array(feature_lst)    # features
label_lst = [1, 1, 1, 2, 2, 2]
Y = np.array(label_lst)    # labels
print(len(Y), 'data points')

# classifier
clf = GaussianNB()
clf.fit(X, Y)

# test data
test_point = [[-0.8, -1]]
print(clf.predict(test_point))

# scatter plot
x, y = zip(*feature_lst)
plt.scatter(x, y)
plt.show()
