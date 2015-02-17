import math
#import numpy

def func(k, x, xa, ya):
  x_delta = []
  nearest = []
  for x_element in xa:
    x_delta.append(math.fabs(x - x_element))
  for i in range(k):
    best_idx = 0
    for j in range(50):
      if (x_delta[best_idx] == -1 and x_delta[j] == -1):
        continue
      elif (x_delta[best_idx] != -1 and x_delta[j] == -1):
        continue
      elif (x_delta[best_idx] == -1 and x_delta[j] != -1):
        best_idx = j
      elif (x_delta[best_idx] != -1 and x_delta[j] != -1 and x_delta[j] < x_delta[best_idx]):
        best_idx = j
    nearest.append(ya[best_idx])
    x_delta[best_idx] = -1
  sum = 0.0
  for n in nearest:
    sum += n
  return sum/k

with open("../RegressionX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../RegressionY.txt") as f:
  y = []
  for line in f:
    y.append(float(line))

for k in range(1, 11):
  error = 0.0
  for i in range(50, 100):
    error += pow(func(k, x[i], x[0:50], y[0:50]) - y[i], 2)
  print (error/50.0)


print ("On test set:")
error = 0.0
for i in range(100, 200):
  error += pow(func(3, x[i], x[0:50], y[0:50]) - y[i], 2)
print (error/100.0)







#error = 0.0
#for i in range(100, 200):
  #nearest = 0
  #for j in range(50):
    #if math.fabs(x[i] - x[j]) < math.fabs(x[i] - x[nearest]):
      #nearest = j
  #error += pow(y[nearest] - y[i], 2)
#
#print error/100.0
