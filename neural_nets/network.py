import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class Network:
    '''
        Network class is used to, 
            - Create the model architecture.
            - Define the layers.
            - Define the loss functions.
            - Train the model.
            - Predict the input using the trained model.
    '''
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_grad = None
        self.lmda = None
        self.errors = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    ## Adding layer to the network
    def add(self, layer):
        self.layers.append(layer)
        
    ## Updating the loss and loss grad
    def assign(self, loss, loss_grad):
        self.loss = loss
        self.loss_grad = loss_grad
        
    ## Predicting the output for a given input
    def predict(self, input_data):
        length = len(input_data)
        result = []
        
        ## Iterating through each data
        for i in range(length):
            output = input_data[i]
            ## Forward propogation
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        
        return result
    
    ## Training a neural network
    def train(self, x_train, y_train, x_test, y_test, learning_rate=0.01, epochs=100, lmda = None, early_stopping=False, print_metrics=True):
        length = len(x_train)
        self.lmda = lmda

        ## Iterating through the epochs and updating parameters after each input        
        for epoch in range(epochs):
            err = 0
            for i in range(length):
                output = x_train[i]
                ## Forward Propogation
                for layer in self.layers:
                    output = layer.forward(output)

                ## Compute loss/Error 
                err = self.loss(y_train[i], output)
                ## Compute Backpropogation with gradient of error
                error = self.loss_grad(y_train[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate, self.lmda)
            
            if print_metrics:
                print(f"Model trained for epoch: {epoch+1}/{epochs} and the error is: {np.round(err, 5)}") 
            self.errors.append(np.round(err, 5))           
        
            ## Training Accuracy check
            out = self.predict(x_train)
            y_pred = [1 if each>=0.5 else 0 for each in out]
            train_acc = round(accuracy_score(y_train, y_pred)*100, 2)
            self.train_accuracies.append(train_acc)   
            if print_metrics:    
                print(f"Training Accuracy: {train_acc}")

            ## Validation Accuracy            
            out = self.predict(x_test)
            y_pred = [1 if each>=0.5 else 0 for each in out]
            val_acc = round(accuracy_score(y_test, y_pred)*100, 2)
            self.val_accuracies.append(val_acc)     
            if print_metrics:  
                print(f"Validation Accuracy: {val_acc}")   
            
            if epoch > 2 and early_stopping:
                if (val_acc <= self.val_accuracies[epoch-1]) \
                    and (val_acc <= self.val_accuracies[epoch-2]) \
                    and (val_acc <= self.val_accuracies[epoch-3]):
                    if print_metrics:  
                        print("-----"*10)         
                        print(f"Early Stopping Initiated at epoch: {epoch+1}")
                        print(f"Training Accuracy: {train_acc}")
                        print(f"Validation Accuracy: {val_acc}")   
                    break 
        if print_metrics:
            print(f"Completed Training with lr = {learning_rate} and lambda = {lmda}...")
            print(f"Training Accuracy: {train_acc}")
            print(f"Validation Accuracy: {val_acc}")              
        return train_acc, val_acc, epoch+1
            
            
    ## Displaying the neural network
    def show_network(self):
        print(self.layers)
        
    