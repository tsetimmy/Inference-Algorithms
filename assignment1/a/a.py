import math

with open("../RegressionX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../RegressionY.txt") as f:
  y = []
  for line in f:
    y.append(float(line))

error = 0.0
for i in range(100, 200):
  nearest = 0
  for j in range(50):
    #if (x[i] - x[j])^2 + (y[i] - y[j])^2 < (x[i] - x[nearest])^2 + (y[i] - y[nearest])^2:
    if math.fabs(x[i] - x[j]) < math.fabs(x[i] - x[nearest]):
      nearest = j
  error += pow(y[nearest] - y[i], 2)

print (error/100.0)

