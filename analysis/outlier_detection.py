# Outlier Detection 
# Import abstract dependencies for building abstract classes
from abc import ABC, abstractmethod
from typing import Tuple, List

# Import dependencies for detecting and removing outliers
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Outlier detection: finding outliers to increase performance of the model. Outliers carries out inconsistency and iregularities of the data that 
# leads to poor performance 
# 
# #

# Implement OutlierDetector class 
class OutlierDetector(ABC): 
    """Outlier detector class: detecting outliers in different ways
    
    - methods: analyse
    - parameters: feature, pandas dataframe
    - return: analyse --> pd.Dataframe"""

    @abstractmethod
    def analyse(feature_list: list, data:pd.DataFrame) -> pd.DataFrame:
        pass 

    """ Visualising outliers with boxplot """
    @abstractmethod
    def visualise(feature: str, ):
        pass


# Implement OutlierIdentifier
class OutlierIdentifier(OutlierDetector):
    """Outlier Identifier class: identify possible outliers using boxplot and scatterplot 
        - analyse(): analyses the outliers with statistical metrics
        - visualise(): create boxplot and scatter to visualise outliers of a chosen feature 
        
        Param: 
        - filename (str): name of the dataset 
        - directory (str): the path that refers to the selected dataset

        Return:

        """
    def __init__(self, filename, directory): 
        self.filename = filename
        self.directory = directory

        self.dataset = pd.read_csv(os.path.join(self.directory, self.filename))

    # Method 1: Analysing distribution of feature data + identify number of outliers 
    def analyse(self, feature: str, num: int):
        # Find outliers
        num_outliers = self.find_outliers(feature=feature)[0]

        # Visualise boxplot
        self.visualise(feature)

        # Compute statistical metrics
        feature_metrics = self.dataset[feature].aggregate(['max', "min", "mean", "std"])
        print(f"Feature {num + 1}: {feature}")
        print(feature_metrics)
        print(f"Outlier: {num_outliers.shape[0]} -- {num_outliers}") 
    
    # Method 2: Visualise outliers with Boxplot and Scatterplot
    def visualise(self, feature: str, pos: int):
        # Subplots 
        target_var = "Price"
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        
        # Boxplot + graph details
        boxplot = sns.boxenplot(self.dataset[feature], color="black", ax=ax[0]) 
        boxplot.set_xlabel(f"{feature}(F{pos})")
        boxplot.set_ylabel(f"Measurement")
        
        # Scatterplot + graph details
        scatterplot = sns.scatterplot(data=self.dataset, x=feature, y=target_var, ax=ax[1])
        scatterplot.set_xlabel(f"{feature}")
        scatterplot.set_ylabel(target_var)

        # Title of plot 
        plt.title(f"{feature}")
        plt.show()

    # Method 3: Find outliers in a dataset
    def find_outliers(self, feature: str) -> Tuple[int, pd.DataFrame]: 
        # 
        feature_data = self.dataset[feature]

        # Compute the interquartile range
        q1 = np.percentile(feature_data, 25)
        q3 = np.percentile(feature_data, 75)
        iqr = q3 - q1 

        # Compute the upper and lower bounds 
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = feature_data[(feature_data < lower_bound)| (feature_data > upper_bound)]
        num_outliers = outliers.unique()
        
        # Identify if there are outliers or not 
        if num_outliers.shape[0] > 0:
            # Extract data samples with outliers
            num_datapoints_outliers = outliers.shape[0]
            num_datapoints_outliers
            
            # Create a outlier dataframe 
            outlier_dict = {"Feature":"Display Height", "Outlier value": num_outliers, "Outlier count": num_datapoints_outliers}
            outlier_df = pd.DataFrame(outlier_dict)
            print(f"{feature}: {num_outliers}")
            return num_outliers, outlier_df
        else: 
            return f"{feature}: no outliers" 
        
# # Implement OutlierRemover class to remove outliers from the dataset
# class OutlierRemover(OutlierDetector, OutlierIdentifier):
#     pass

# Code Example 

if __name__ == "__main__":

    # Set up the path directory --> dataset 
    path = "D:\Projectwork Platform\MEP-Machine-Learning\Laptop_Price_Prediction\dataset"
    filename = "ebay_laptop_data_phase3.csv"

    # Create an outlier identifier object 
    outlier_id = OutlierIdentifier(filename, path)
    data = outlier_id.dataset

    # print(data)

    # Identify outliers 
