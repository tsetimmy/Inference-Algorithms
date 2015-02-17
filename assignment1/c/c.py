import math
import numpy as np










#def func(k, x, xa, ya):
#  x_delta = []
#  nearest = []
#  for x_element in xa:
#    x_delta.append(math.fabs(x - x_element))
#  for i in range(k):
#    best_idx = 0
#    for j in range(50):
#      if (x_delta[best_idx] == -1 and x_delta[j] == -1):
#        continue
#      elif (x_delta[best_idx] != -1 and x_delta[j] == -1):
#        continue
#      elif (x_delta[best_idx] == -1 and x_delta[j] != -1):
#        best_idx = j
#      elif (x_delta[best_idx] != -1 and x_delta[j] != -1 and x_delta[j] < x_delta[best_idx]):
#        best_idx = j
#    nearest.append(ya[best_idx])
#    x_delta[best_idx] = -1
#  sum = 0.0
#  for n in nearest:
#    sum += n
#  return sum/k

with open("../RegressionX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../RegressionY.txt") as f:
  y = []
  for line in f:
    y.append(float(line))

lr = 0.01
y_training = y[0:50]
x_training = x[0:50]

y_training_hat = []
for i in range (50):
  y_training_hat.append(0.0)

delta0 = 0.0
delta1 = 0.0
w0 = np.random.uniform(-0.1, 0.1)
w1 = np.random.uniform(-0.1, 0.1)

for i in range (10000):
  for j in range(len(y_training_hat)):
    y_training_hat[j] = w0 + w1*x_training[j]
  delta1 = 0.0
  delta0 = 0.0
  for j in range(0, len(y_training_hat)):
    temp = (y_training[j] - y_training_hat[j])
    delta0 += temp
    delta1 += temp * x_training[j]
  delta1 *= (2.0/50.0)
  delta0 *= (2.0/50.0)
  w1 = w1 + lr*delta1
  w0 = w0 + lr*delta0

x_testing = x[100:200]
y_testing = y[100:200]
error = 0.0
for i in range(100):
  error += pow(y_testing[i] - (w0 + w1 * x_testing[i]), 2)
print (error/100.0)











#
#for k in range(1, 11):
#  error = 0.0
#  for i in range(50, 100):
#    error += pow(func(k, x[i], x[0:50], y[0:50]) - y[i], 2)
#  print error/50.0
#
#
#print "On test set:"
#error = 0.0
#for i in range(100, 200):
#  error += pow(func(3, x[i], x[0:50], y[0:50]) - y[i], 2)
#print error/100.0







#error = 0.0
#for i in range(100, 200):
  #nearest = 0
  #for j in range(50):
    #if math.fabs(x[i] - x[j]) < math.fabs(x[i] - x[nearest]):
      #nearest = j
  #error += pow(y[nearest] - y[i], 2)
#
#print error/100.0
