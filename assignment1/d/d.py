import math
import numpy as np

def wx(w, xj):
 sum = 0.0
 for i in range(len(w)):
   sum += w[i] * pow(xj, i)
 return sum
  
def func (k, x, y, yhat):
  lr = 0.01
  delta = []
  w = []
  for i in range(k + 1):
    delta.append(0.0)
    w.append(np.random.uniform(-0.1, 0.1))
  for i in range(10000):
    for j in range(len(yhat)):
      yhat[j] = wx(w, x[j]);
    for j in range(len(delta)):
      delta[j] = 0.0
    for j in range(len(yhat)):
      temp = (y[j] - yhat[j])
      for l in range(len(delta)):
        delta[l] += temp * pow(x[j], l)
    for j in range(len(delta)):
      delta[j] *= (2.0/50.0)
    for j in range(len(delta)):
      w[j] = w[j] + lr * delta[j]
  return w

#Main
with open("../RegressionX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../RegressionY.txt") as f:
  y = []
  for line in f:
    y.append(float(line))

y_training = y[0:50]
x_training = x[0:50]
x_validation = x[50:100]
y_validation = y[50:100]

mean = np.mean(x_training)
std = np.std(x_training)

x_training = (x_training - mean) / std
x_validation = (x_validation - mean) / std


y_training_hat = []
for i in range (50):
  y_training_hat.append(0.0)

for k in range(1, 11):
  w = func(k, x_training, y_training, y_training_hat)
  error = 0.0
  for i in range(50):
    error += pow(y_validation[i] - wx(w, x_validation[i]), 2)
  print (error/50.0)

