import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def sigmoid (a):
  return (1.0/(1.0+np.exp(-a)))

def diff_sigmoid (a):
  #return (np.exp(-a)*pow(sigmoid(a), 2))
  return (np.exp(-a)/((1+np.exp(-a))**2))


class layer1:
  def __init__(self, init_w):
   #for i in range(5):
   self.W = [np.random.uniform(-1.0, 1.0)*2*init_w] * 5
   self.DW = np.zeros(len(self.W)) #deltas
   self.X = np.zeros(len(self.W))
   self.derivative = np.zeros(len(self.W))

class layer2:
  def __init__(self, init_w):
    self.W = [np.random.uniform(-1.0, 1.0)*2*init_w] * 5
    self.DW = np.zeros(len(self.W)) #deltas
    self.X = 0.0
    self.derivative = 0.0


#Main
with open("../ClassificationX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../ClassificationY.txt") as f:
  y = []
  for line in f:
    y.append(float(line))

y_training = y[0:50]
x_training = x[0:50]
x_validation = x[50:100]
y_validation = y[50:100]
x_test = x[100:200]
y_test = y[100:200]

#mean = np.mean(x_training)
#std = np.std(x_training)
#x_training = (x_training - mean) / std
#x_validation = (x_validation - mean) / std
#x_test = (x_test - mean) /std

lr = 0.1
#layer1 = layer(1, 5, 0.01)
#output = layer(5, 1, 0.01)
#print(layer1.W)
#print(layer1.X)
#print(output.X)
#print(output.DW)
#print(layer1.derivative)
#print(layer1.DW)
#print(output.derivative[0])
#for OKAY in range(10000):
#  for i in range(len(x_training)):
#    # forward
#    for j in range(len(layer1.X)):
#      layer1.X[j] = sigmoid(x_training[i]) * layer1.W[0][j]
#    for j in range(len(output.X)):
#      for k in range(len(layer1.X)):
#        output.X[j] += sigmoid(layer1.X[k]) * output.W[k][j]
#    # backward
#    output.derivative[0] = -2*(y_training[i] - sigmoid(x_training[i]))*diff_sigmoid(x_training[i])
#    for j in range(len(layer1.derivative)):
#      layer1.derivative[j] = diff_sigmoid(layer1.X[j]) * layer1.W[0][j] * output.derivative[0]
#    # accumulate negative error derivatives
#    for j in range(len(layer1.DW[0])):
#      layer1.DW[0][j] = layer1.DW[0][j] - layer1.derivative[j] * x_training[i]
#    for j in range(len(layer1.DW[0])):
#      output.DW[j][0] = output.DW[j][0] - output.derivative[0] * sigmoid(output.X[0])
#    # update the parameters
#    for j in range(len(layer1.W[0])):
#      layer1.W[0][j] = layer1.W[0][j] + (lr * layer1.DW[0][j])/len(x_training)
#      output.W[j][0] = output.W[j][0] + (lr * output.DW[j][0])/len(x_training)

#a = layer1(0.1/2.0)
#b = layer2(0.1/2.0)
#MSE = []
t = []


for it in range(10000):
  t.append(it)



for lol in range(1, 5):
  a = layer1(0.1/2.0)
  b = layer2(0.1/2.0)
  MSE = []
  lr = float(lol)/10.0

  for it in range(10000):
    error = 0.0
    a.DW = np.zeros(5)
    b.DW = np.zeros(5)
    #t.append(it)
    for i in range(len(x_training)):
      # forward
      tmp = 0.0
      for j in range(len(a.X)):
        a.X[j] = a.W[j] * x_training[i]
      for j in range(len(a.X)):
        tmp += b.W[j] * sigmoid(a.X[j])
        #b.X += b.W[j] * sigmoid(a.X[j])
      b.X = tmp

      error += (y_training[i] - b.X)**2

      # backward
      b.derivative = -2*(y_training[i] - b.X)
      for j in range(len(a.X)):
        a.derivative[j] = diff_sigmoid(a.X[j]) * b.W[j] * b.derivative

      # compute and accum
      for j in range(len(a.X)):
        a.DW[j] -= a.derivative[j] * x_training[i]

      for j in range(len(a.X)):
        b.DW[j] -= b.derivative * sigmoid(a.X[j])


    # update parameters
    for j in range(len(a.X)):
      a.W[j] += lr*a.DW[j]/len(x_training)

    for j in range(len(a.X)):
      b.W[j] += lr*b.DW[j]/len(x_training)

    MSE.append(error/len(x_training))


  print(MSE)
  p = plt.semilogx(t, MSE, label=str(lr))



  plt.draw()


#print(MSE)
#plt.semilogx(t, MSE)
#plt.grid(True)
#plt.show()
plt.grid(True)
plt.legend()
plt.show()
