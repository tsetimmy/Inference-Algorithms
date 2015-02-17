
# In[1]:

### ECE521 - 2015 Winter
### Example of neural network
### Written by: Alice Gao


# In[2]:

import numpy as np


# In[3]:

from copy import copy


# In[4]:

###################################
## example of sigmoidal hidden layer implementation
## and stacking layers to make deep network
## you'll need to write your own code to implement the training, validation, and testing procedure
## as well as parameter initilization, other layer types, cost function, etc.


## Note: 
## xs is of dimension N-by-p, where N is the number of cases, and p is the number of input features
## ys is of dimension N-by-1, where N is the number of cases, and 1 is the number of outputs (only for this example)



# In[5]:

def sigmoid(a):
    return 1.0/(1.0+np.exp(-a))


# In[6]:



class Network:
    def predict(self, xs):
        return self.forward(xs)
    
    def train(self, xs, ys):
        pred = self.forward(xs) ## forward propagate xs to get prediction
        ## why are we feeding error signals w.r.t the output unit before activation (instead of after)?
        ## because it has a much simpler form! you can work out the math 
        deltas = pred - ys  ## gradient of cost w.r.t. output layer units (before activation), this only works for MSE or cross-entropy loss function
        self.backward(deltas) ## backprop to get gradient
        self.update_weights(self.lr) ## update parameter for all layers, note that this can be implemented with momentum
        
    def set_lr(self, lr):
        self.lr = lr
        
    


# In[23]:


class sigmoid_layer(Network):
    def __init__(self, ni, no, init_w, is_output_layer=False):
        self.ni = ni ## input dimension
        self.no = no ## output dimension
        self.W = (np.random.rand(ni+1,no)-0.5)*2*init_w ## initialize parameter (note matrix W includes both weight and bias)
        self.DW = np.zeros(self.W.shape) ## to store the gradient
        self.is_output_layer = is_output_layer ## backprop will be slightly different for output layer (see below)
        
    def forward(self,ys):
        N = ys.shape[0] ## number of cases
        inputs = np.ones((N, self.ni+1))
        inputs[:,1:] = ys ## this is to append a column of ones (to be multiplied by bias)
        zs = sigmoid(np.dot(inputs, self.W)) ## multiply by W, and go through nonlinearity
        self.ss = (inputs, zs) ## these are to be used in backward propagation
        return zs
    
    def backward(self, deltas):
        N = deltas.shape[0] ## number of cases
        inputs, zs = self.ss
        if self.is_output_layer:
            dzsp = deltas  ## for output layer, we directly feed in the error signal w.r.t the unit BEFORE activation
        else:
            dzsp = deltas * zs * (1.0-zs) ## backprop through sigmoid function, this only works for sigmoid units
        
        di = np.dot(dzsp, self.W.T)[:,1:] ## gradient w.r.t. layer input xs (this is useful for backprop the error signal to layers before this one)
        self.DW = np.dot(inputs.T, dzsp)/N ## gradient w.r.t. W
        return di
    
    def update_weights(self, lr):
        self.W += lr*self.DW
        
        
        


# In[8]:


## this is to stack layers and make deep net

class stacked(Network):
    def __init__(self, *nets):
        self.nets = nets
        
    def forward(self, xs):
        for i,net in enumerate(self.nets):
            xs = net.forward(xs)
        return xs
    
    def backward(self,deltas):
        for i,net in reversed(list(enumerate(self.nets))):
            deltas = net.backward(deltas)
        return deltas

    def update_weights(self, lr):
        for i,net in enumerate(self.nets):
            net.update_weights(lr)
        
        


# In[8]:




# In[33]:

## Gradient check
## It is good practice to check if your analytical computed gradient matches with the numerical one
## note this will depend on your model complexity and numerical precision

## small value to perturb the parameter
epsilon = 10**(-9)

## make some data
#XS = np.random.rand(20,10)
#YS = np.random.rand(20,1)
XS = np.array([[2,3],[4,5]])
YS = np.array([[0],[1]])

## initialize the network (simple net here) (this is actually a logistic regression model =P )
grad_check_net = sigmoid_layer(2,1,0.01, is_output_layer=True) ## the parameter matrix will be of dimension (10+1)-by-1
grad_check_net.set_lr(0.1)

## let's decide on some random element in the parameter matrix 
grad_check_idx = np.random.randint(0, 2+1)

print (grad_check_idx)

## make a copy of the parameter
W_copy = copy(grad_check_net.W)

## numerical gradient 
## (before adding epsilon)
pred_old = grad_check_net.predict(XS)
cost_old = np.mean( -YS*np.log(pred_old)-(1.0-YS)*np.log(1-pred_old) )
## (after adding epsilon)
grad_check_net.W[grad_check_idx] += epsilon
pred_new = grad_check_net.predict(XS)
cost_new = np.mean( -YS*np.log(pred_new)-(1.0-YS)*np.log(1-pred_new) )
grad_numerical = (cost_new-cost_old)/epsilon


## restore the parameter
grad_check_net.W = copy(W_copy)

## analytical gradient
grad_check_net.train(XS, YS)
grad_analytical = grad_check_net.DW[grad_check_idx,0]
#print grad_check_idx





## check, are they roughly the same?
print ("analytical gradient:")
print (grad_analytical)

print ("numerical gradient:")
print (grad_numerical)

print ("difference:")
print ("%0.4e"%np.abs(grad_analytical-grad_numerical))



# Out[33]:

#     0
#     analytical gradient:
#     0.00204410078563
#     numerical gradient:
#     0.00204425365524
#     difference:
#     1.5287e-07
# 

# In[8]:




# In[9]:

## Make a deep network


## make some data
XS = np.random.rand(100,10)
YS = np.random.rand(100,1)

## initialize the network (2 hidden layers)
print("BEGIN NETWORK")
deep_net = stacked(sigmoid_layer(10,5,0.01), sigmoid_layer(5,2,0.01), sigmoid_layer(2,1,0.01,is_output_layer=True))
deep_net.set_lr(0.1)

## this will train the network for 1 epoch, and update the parameter once
deep_net.train(XS, YS)

## you can play with the rest!


