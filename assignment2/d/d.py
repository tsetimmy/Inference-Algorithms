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
  for i in range(500000):
    for j in range(len(yhat)):
      yhat[j] = wx(w, x[j]);
    for j in range(len(delta)):
      delta[j] = 0.0
    for j in range(len(yhat)):
      temp = (y[j] - yhat[j])
      for l in range(len(delta)):
        delta[l] += temp * pow(x[j], l)
    for j in range(len(delta)):
      delta[j] *= (2.0/float(len(yhat)))
    for j in range(len(delta)):
      w[j] = w[j] + lr * delta[j]
  return w

#Main
with open("../ClassificationX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../ClassificationY.txt") as f:
  y = []
  for line in f:
    y.append(int(line))

y_training = y[0:50]
x_training = x[0:50]
x_validation = x[50:100]
y_validation = y[50:100]
x_test = x[100:200]
y_test = y[100:200]

mean = np.mean(x_training)
std = np.std(x_training)
x_training = (x_training - mean) / std
x_validation = (x_validation - mean) / std
x_test = (x_test - mean) /std


y_training_hat = []
for i in range(len(x_training)):
  y_training_hat.append(0.0)

for k in range(1, 2):
  w = func(k, x_training, y_training, y_training_hat)

print(w)
for k in range(1, 19):
  thresh = float(k) / float(18)
  error = 0
  for i in range(len(x_test)):
    if w[0] + w[1]*x_test[i] > thresh:
      prediction = 1
    else:
      prediction = 0

    if prediction != y_test[i]:
      error += 1
  print(float(error)/float(len(x_test)))
