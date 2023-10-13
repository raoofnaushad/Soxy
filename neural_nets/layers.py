import numpy as np
np.random.seed(7008)


## Base Class
class Layer:
    '''
        Layer class is a based class with forward and backwards methods. 
        The child classes should define these methods to work in the nueral networks.
    '''
    def __init__(self):
        self.input = None
        self.output = None
        
    ## Computes the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError
    
    ## Computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, output_error, learning_rate):
        raise NotImplementedError
    
    
## Fully Connected Layer
class FCLayer(Layer):
    '''
        Fully connected or Linear layer class inherited from Layer class with forward and backward methods.
        input: input_size, output_size
    '''
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5 
        self.bias = np.random.rand(1, output_size) - 0.5
        
    ## Computes the output Y of a layer for a given input X
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    ## Computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, output_error, learning_rate, lmda=None):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        ## Here I want to update weights_error with regularization term
        if lmda:
            weights_error += (lmda * self.weights)  
        
        ## Update Parameter
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    

## Activation Layer
class ActivationLayer(Layer):
    '''
        Activation layer class inherited from Layer class with forward and backward methods.
        input: Activation function and Derivative of Activation function.
    '''
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    ## Computes the activated output
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    ## Computes dE/dX for a given dE/dY, learning rate is not used as there is no learnable parameters.
    def backward(self, output_error, learning_rate, lmda=None):
        return self.activation_prime(self.input) * output_error
    
    
    

