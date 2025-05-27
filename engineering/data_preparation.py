# Data Preparation 
import os
import pandas as pd
import numpy as np  
from abc import ABC, abstractmethod

# Import depedencies from scikit learn
from sklearn.model_selection import train_test_split

# Implement DataPreparation class 
class DataPreparator:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        self.csv_file = os.path.join(self.path, f"{self.filename}.csv")
        self.dataset = pd.read_csv(self.csv_file)

    # Method 1: splitting data into training and testig sets
    def data_splitting(self, featureX: str, featureY:str, testsize=0.25):
        # Create x and y variables from dataframe
        dataX = self.dataset[[featureX]]
        dataY = self.dataset[featureY]

        # Split data into training and testing sets
        xtrain, xtest, ytrain, ytest = train_test_split(dataX, dataY, test_size=testsize, random_state=1234)

        return (xtrain, ytrain),(xtest, ytest)

# Example code 

