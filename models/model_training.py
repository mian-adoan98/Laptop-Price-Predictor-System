## Model Training

from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd 
import matplotlib as plt 

# Implement class to train model 
class ModelTraining(ABC):
    
    @abstractmethod
    # Training the model 
    def train(self, xtrain, ytrain):
        pass

    @abstractmethod
    # Make predictions on model 
    def predict(self, xtest): 
        pass 

    