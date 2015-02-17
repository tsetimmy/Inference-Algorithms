import math


def doit (kk, xt, xv, xtest, yt, yv, ytest, flag):
  temp = list(xt)
  xv_doit = list(xv)
  xtest_doit = list(xtest)
  yt_doit = list(yt)
  yv_doit = list(yv)
  ytest_doit = list(ytest)

  zerozero = 0
  zeroone = 0
  onezero = 0
  oneone = 0
  error = 0

  for i in range(len(xv_doit)):
    xt_doit = list(temp)
    onesc = 0
    zerosc = 0
    for k in range(kk):
      nearest = 0
      for j in range(len(xt_doit)):
        if math.fabs(xv_doit[i] - xt_doit[j]) < math.fabs(xv_doit[i] - xt_doit[nearest]):
          nearest = j
      if yt_doit[nearest] == 1:
        onesc += 1
      else:
        zerosc += 1
      xt_doit[nearest] = float("inf")
    if onesc == zerosc:
      if yv_doit[i] != 0:
        error += 1
      print ("We are in a little bit of trouble")
    elif onesc > zerosc and yv_doit[i] != 1:
      error += 1
      zeroone += 1
    elif onesc < zerosc and yv_doit[i] != 0:
      error += 1
      onezero += 1
    elif onesc > zerosc and yv_doit[i] == 1:
      oneone += 1
    elif onesc < zerosc and yv_doit[i] == 0:
      zerozero += 1
    #print("ones")
    #print(onesc)
    #print("zeros")
    #print(zerosc)

  if flag == 1:
    print ("Confusion matrix:")
    print (zerozero)
    print (zeroone)
    print (onezero)
    print (oneone)

  return error/len(xv_doit)




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

e = []

for k in range(1, 12):
  if k%2!=0:
    e.append(doit(k, xt, xv, xtest, yt, yv, ytest, 0))

for k in range(len(e)):
  print (e[k])

# Smallest is k = 3
print ("Smallest is k = 3")
print (doit(3, xt, xtest, xv, yt, ytest, yv, 1))
