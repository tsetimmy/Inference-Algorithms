import math

with open("../ClassificationX.txt") as f:
  x = []
  for line in f:
    x.append(float(line))

with open("../ClassificationY.txt") as f:
  y = []
  for line in f:
    y.append(int(line))

xt = x[0:50]
xv = x[50:100]
xtest = x[100:200]

yt = y[0:50]
yv = y[50:100]
ytest = y[100:200]

zerozero = 0
zeroone = 0
onezero = 0
oneone = 0
error = 0
for i in range(len(xtest)):
  nearest = 0
  for j in range(len(xt)):
    if math.fabs(xtest[i] - xt[j]) < math.fabs(xtest[i] - xt[nearest]):
      nearest = j
  #error += pow(yt[nearest] - ytest[i], 2)
  if yt[nearest] != ytest[i]:
    error += 1
  if yt[nearest] == ytest[i] and yt[nearest] == 0:
    zerozero += 1
  if yt[nearest] == ytest[i] and yt[nearest] == 1:
    oneone += 1
  if yt[nearest] != ytest[i] and yt[nearest] == 0:
    onezero += 1
  if yt[nearest] != ytest[i] and yt[nearest] == 1:
    zeroone += 1
  
print ("Classification error rate:")
print (error)

print ("Confusion matrix:")
print (zerozero)
print (zeroone)
print (onezero)
print (oneone)

print ("Total:")
print (zerozero+zeroone+onezero+oneone)
