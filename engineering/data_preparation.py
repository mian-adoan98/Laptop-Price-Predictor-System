# Data Preparation 
# Import libraries for system and coding configuration
import os
from typing import Tuple

# Import libraries for developing data preparation algorithm
import pandas as pd
import numpy as np  

pd.set_option("display.max.columns", 30)
# Import depedencies from scikit learn
from sklearn.model_selection import train_test_split

# Implement DataPreparation class 
class DataPreparator:
    """DataPrepator-class: a class for preparing the extracted data from disparate of resources 
    to provide well-structure data
    
    Param: 
       - filename: name of the file from a directory 
       - path: name of local directory 
       
    Return: 
    a csv-file with extension .csv loaded as pandas dataframe 
    """

    def __init__(self, filename: str, path: str, num_samples: int):
        self.filename = filename
        self.path = path
        self.num_samples = num_samples

    # Method 1: splitting data into training and testig sets
    def data_splitting(self, featureX: str, featureY:str, testsize=0.25) -> Tuple[tuple, tuple]:
        # Create a dataset 
        dataset = self.verify_csv_ext()
        dataset = dataset.select_dtypes([int, float])

        # Create x and y variables from dataframe
        dataX = dataset[[featureX]].loc[:self.num_samples]
        dataY = dataset[featureY].loc[:self.num_samples]

        # Split data into training and testing sets
        xtrain, xtest, ytrain, ytest = train_test_split(dataX, dataY, test_size=testsize, random_state=1234)

        return (xtrain, ytrain),(xtest, ytest)
    
    # Method 2: Varify if file has a csv-extension
    def verify_csv_ext(self) -> pd.DataFrame:
        # Create constant: file with unknown extension
        file = os.path.join(self.path, self.filename)

        # Check file has a csv-extension
        if not file.endswith("csv"): 
            raise FileNotFoundError("Selected file is not a csv-file. Please specify the file with csv-extension. \n " \
            "File is not a dataset")

        print(f"File is successfully verified !")
        # Create a dataframe 
        dataset = pd.read_csv(file, index_col=0)
        return dataset
    
# Example code 
if __name__ == "__main__": 

    # Set up necessary constant for loading a dataframe 
    proj_path = "C:\Development\Projects\MachineLearning\Laptop-Price-Predictor-System"
    data_folder = os.path.join(proj_path, "dataset")
    filename = "ebay_laptop_data_cleaned.csv"

    # Create a DataPreparator object with 50, 100 and 150 samples 
    data_preparator1 = DataPreparator(filename, data_folder, 50)
    # data_preparator1 = DataPreparator(filename, proj_path, 100)
    # data_preparator1 = DataPreparator(filename, proj_path, 150)

    # Verify file is a  dataset from data preparator 1
    ds1 = data_preparator1.verify_csv_ext()
    # print(ds1)

    # Prepare data for dataset 1: Processor Speed(X) and Target Laptop Price(Y)
    train_set1, test_set1 = data_preparator1.data_splitting(featureX="Processor Speed", featureY="Price",)
    print(f"Training set: {train_set1[0]}")
    print(f"Testing set: {test_set1[0]}")