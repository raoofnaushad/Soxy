import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def normalize_data(df):
    '''
      Function is used for normalizing the data.
      input: dataframe
      output: dataframe
    '''
    df_min_max_scaled = df.copy()
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    return df_min_max_scaled


def accuracy_metrics_ml(classifier, X_test, y_test):
    '''
      Function is used for finding the accuracy of ML models.
      input: ml classifier model object, input_data, target data
      output: None
    '''
    y_pred = classifier.predict(X_test)
    result = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    print(f"The coefficient of classifier: {classifier.coef_}")
    print(f"The intercept of classifier: {classifier.intercept_}")
    print(f"The accuracy for method-1 logistic regression is: {round(accuracy_score(y_test, y_pred)*100, 2)}%")
    print("/n")
    target_names = ['Genuine', 'Not Genuine']
    print(classification_report(y_test, y_pred,target_names=target_names))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
  
  
def plot_accuracy_graph(train_accuracies, val_accuracies, model):
    '''
      Function is used to plot the validation and training accuracy against the number of epochs
      input: train accuracies (list), validation accuracies (list), model name (str)
      output: None
    '''
    plt.plot(train_accuracies)
    plt.plot(val_accuracies)
    plt.title(f'Accuracy graph for: {model}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def kfold_cross_validation(X, y, model, lr, epochs, lmbda, early_stopping, print_metrics, num_folds=5):
    # Calculate the size of each fold
    fold_size = len(X) // num_folds
    # Initialize a list to store the cross-validation results
    cross_val_results = []
    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Loop through the folds
    for i in range(num_folds):
        # Calculate the indices for the current fold
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(X)

        # Split the data into training and validation sets
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        _, accuracy, _ = model.train(X_train, 
                                    y_train, 
                                    X_val, 
                                    y_val, 
                                    learning_rate=lr, 
                                    epochs=epochs, 
                                    lmda=lmbda, 
                                    early_stopping=early_stopping,
                                    print_metrics=print_metrics)


        # Append the accuracy to the results list
        cross_val_results.append(accuracy)

    # Calculate the average and standard deviation of the cross-validation results
    avg_accuracy = sum(cross_val_results) / num_folds
    
    return avg_accuracy, lr, lmbda
    # return None, None, None