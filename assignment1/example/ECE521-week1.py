# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

### ECE521 - 2015 Winter
### Example of linear regression
### Written by: Alice Gao

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# <codecell>

## make dataset with a deterministic underlying model + oberservation noise
## You can use this to make multiple dataset, e.g. training, validation, testing (and note that each data point is i.i.d.)
def make_dataset(num_data=10, noise_std=0.1):
    x = (np.random.rand(num_data)-0.5)*2
    y = 0.2*x**2 + 0.1*x + 0.5 ## true underlying model
    return x, y+np.random.randn(num_data)*noise_std  ## add noise to observed y

# <codecell>

## compute cost, given x, y and parameters
## this will be useful when you want to pick the best model based on validation set
def computer_cost(x, y, theta):
    pred = np.dot(x, theta)
    cost = np.mean(0.5*(pred-y)**2)
    return cost

# <codecell>

## Gradient Descent
## lr: learning rate
## num_epoch: number of epoches (number of times we fully present the training dataset to the algorithm)
def grad_descent(x, y, lr=0.01, num_epoch=100, disp_int=10):
    num_param = x.shape[1]
    traing_cost = np.empty(num_epoch)
    ##
    theta = np.random.randn(num_param)*0.01 ## randomly initialize the parameters from Gaussian with mean=0, std=0.01
    for i in range(num_epoch): ## for fixed number of epoches (this can be changed to 'loop until convergence' where convergence is determined by the difference in training error/cost)  
        pred = np.dot(x, theta) ## compute the prediction using current parameters
        cost = np.mean(0.5*(pred-y)**2) ## mean squared error on training set (divide by 2 so we don't have a factor of 2 in gradient)
        if np.mod(i+1, disp_int)==0 or i==0: ## for print result
            print ("Epoch %d, cost %0.4f" % (i, cost))
        ##
        grad = np.dot(x.T, (pred-y)) ## Gradient (in vector format)
        ##
        theta -= grad*lr ## update parameter
        traing_cost[i] = cost ## store the training cost at this epoch (we can later plot the training cost progress to check if the learninig rate need to be changed)
    
    ## now return the parameters after training
    return theta, traing_cost

# <codecell>

#############################################

# <codecell>

## Simple example of fitting 2nd order polynomial
## Here we're showing the training procedure only, validation and testing is left as an exercise

## generate an training dataset
x_train_tmp, y_train = make_dataset()

## append input feature x with column of 1, and expand with x^2
x_train = np.empty((len(x_train_tmp), 3))
x_train[:,0] = 1
x_train[:,1] = x_train_tmp
x_train[:,2] = x_train_tmp**2

## train
theta, train_cost = grad_descent(x_train, y_train)

# <codecell>

## plot the fitted curve, and training cost progress
x_plot = np.linspace(-1,1,num=500).reshape((500,1))
y_plot = np.dot(np.concatenate((np.ones((500,1)), x_plot, x_plot**2), axis=1), theta)

plt.figure(figsize=(8,6))
plt.plot(x_train_tmp, y_train, 'og', label='Training data')
plt.plot(x_plot, y_plot, '-c', linewidth=3, label='Fitted curve')

plt.xlabel('Input x')
plt.ylabel('Output y')

plt.title('Fitted curve and training data')
plt.legend()

plt.draw()

# <codecell>

plt.figure(figsize=(8,6))
plt.plot(train_cost)

plt.xlabel('Epoch')
plt.ylabel('Training cost')

plt.draw()
plt.show()

# <codecell>


