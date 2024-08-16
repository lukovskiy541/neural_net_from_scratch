from sklearn.datasets import fetch_openml
import numpy as np


mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
print(type(X))
y = mnist.target.astype(int)
print(type(y))
binary_indices = np.where((y == 0) | (y == 1))
X_binary = X.iloc[binary_indices]
y_binary = y.iloc[binary_indices]

y_binary_list = y_binary.apply(lambda x: [x]).tolist()


data_all = list(zip(X_binary.values.tolist(), y_binary_list))

data = data_all[:100]
test_data = data_all[100:200]
print(test_data)