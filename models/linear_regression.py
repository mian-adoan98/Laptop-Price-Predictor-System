# Model 1: Linear Regression 

from abc import abstractmethod, ABC

# Import dependencies to create model 
# import statsmodels
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import numpy.linalg as np_lin 

""" Model 1: Linear Regression 

    Linear regression = a machine learning supervised learning model for making predictions based on past data. In this project, 
    the laptop price predictor focuses on monitoring and predicting price values for new future laptops based on their specifications. 
    
    Despite of error like overfitting and underfitting, techniques that are useful to overcome: 
    - Regularization technique
    - Feature Selection with Correlation Analysis 
    - Gradient Descent for Model Training
    
    3 types of models will be built: 
    - Simple Linear Regression: model with one input feature predicts price value
    - Multivariate Linear Regression: model with more input features predict price value
    - Linear Regression with Regularisation: Ridge, Lasso, ElasticNet
    """
# Code implementation for building linear regression models : ----------------------------------------------------------------------------
# Implement Linear regression abstract class 
class LinearRegression(ABC): 
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate 
        self.n_iters = n_iterations 
        self.weights = None 
        self.bias = None 

    """Linear regression blueprint to create different types of LR Models based on the business objectives"""
    @abstractmethod
    def fit(self): 
        pass
    @abstractmethod
    def predict(self):
        pass

class RegularisedRegression(ABC):
    pass

# Implement Simple Linear Regression class 
# Create a linear regression model 
class SimpleLinearRegression(LinearRegression):
    def __init__(self, learning_rate, n_iterations):
        LinearRegression.__init__(self, learning_rate, n_iterations)
    
    # Train the model using Gradient descent
    def fit(self, X, y):
        # Initialize the weights and bias 
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iterate through the number of iterations 
        for i in range(self.n_iters):
            # Compute the linear model 
            ypred = self.predict(X)

            # Compute gradients for weights and bias
            dW = (1/n_samples) * np.dot(X.T, (ypred - y))
            db = (1/n_samples) * np.sum(ypred - y)

            # Update weights and bias through Gradient Descent Algorithm
            self.weights = self.weights - self.lr * dW
            self.bias = self.bias - self.lr * db

            # Print the loss function
            loss = self.loss(y, ypred)
            print(f"Iteration {i + 1}: weights = {self.weights[:3]}, bias = {self.bias}, loss = {loss}")
        
        return self.weights, self.bias

    # Implement predict function: make predictions
    def predict(self, X):
        # Make predictions based on the x values
        ypred = np.dot(X, self.weights) + self.bias
        return ypred
    
    # Visualise the predictions 
    def visualise_regression(self, x, y, labels, ypred = None, title = "Training"):
        # Visualise the predictions using scatter plot

        if ypred is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color="blue")
            plt.plot(x, ypred, color="black", linewidth=2)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(f"{title}: {labels[0]} vs {labels[1]}")
        else:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color="blue")
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(f"{title}: {labels[0]} vs {labels[1]}")
        
        plt.show()
    
# Implement MultivariateLinearRegression model of the LinearRegression blueprint
class MultivariateLinearRegression(LinearRegression): 
    pass

# Implement Ridge Regression 
class RidgeRegression(RegularisedRegression):
    pass 

# Implement Lasso Regression
class LassoRegression(RegularisedRegression):
    pass 

# Implement ElastoNet Regression
class ElastoNetRegression(RegularisedRegression): 
    pass