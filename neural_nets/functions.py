import numpy as np


'''  ACTIVATION FUNCTIONS '''

## Sigmoid Actication function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## Sigmoid Actication function derivative
def sigmoid_grad(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


## TanH activation function 
def tanh(x):
    return np.tanh(x)

## TanH activation function derivatibe
def tanh_grad(x):
    return 1-np.tanh(x)**2


'''  LOSS FUNCTIONS '''

## Binary Cross Entropy Loss Function
def BCELoss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid taking the logarithm of zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip y_hat to avoid numerical instability
    loss = -((y_true * np.log(y_pred)) + ((1-y_true) * np.log(1-y_pred)))
    return np.squeeze(loss) 


## Binary Cross Entropy Loss Function Derivative
def BCELoss_grad(y_true, y_pred) -> float:
    epsilon = 1e-15  # Small constant to avoid taking the logarithm of zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip y_hat to avoid numerical instability
    loss_grad = (-y_true/y_pred) + ((1-y_true)/(1-y_pred))
    return loss_grad


## Mean Square Error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

## Mean Square Error Derivate
def mse_grad(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size