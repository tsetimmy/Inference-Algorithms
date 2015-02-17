import pickle
import math
import numpy as np

mnist_file = "../../mnist.pkl"
f =  open(mnist_file, "rb")
print("File opened")
mnist_data = pickle.load(f)
tmp = np.zeros((mnist_data['y_test'].shape[0],10))
for k in range(mnist_data['y_test'].shape[0]):
 idx = np.mod(mnist_data['y_test'][k],10)
 tmp[k,idx] = 1
mnist_data['Y_test'] = tmp
X_train = mnist_data['X'][:50000,:]
X_valid = mnist_data['X'][50000:,:]
X_test = mnist_data['X_test']
Y_train = mnist_data['Y'][:50000]
Y_valid = mnist_data['Y'][50000:]
Y_test = mnist_data['Y_test']
print("done")
