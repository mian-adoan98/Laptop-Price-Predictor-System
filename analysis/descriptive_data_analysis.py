# EDA Process Step 1: Descriptive Data Analysis 
from abc import ABC, abstractmethod

# Import libraries for describing the dataset 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns 

""" This phase focuses on describing the laptop dataset based on statistical parameters (mean, std, min, max). 
    It is essential to also identify the distrubtion of the data where possible outliers and missing values can ben detected. 
    Moreover, the size of the dataset and other types of every data feature are crucial for understanding the data better. """

# Implement the DataAnalysisTemplate as an abstract class to create new data analysis classes
class DataAnalysisTemplate(ABC): 
    """This method is considered as blueprints for other class methods to analyse a given dataset. The analysis focuses on 
    the size of the dataset, how many features the data contains and how many different types of features the data exposes. """
    @abstractmethod
    def analyse(self, df: pd.DataFrame):
        # Code for making descriptive analysis of the dataset
        pass 
    
    """This is method will use visualisations to provide more insights of how the laptop dataset are composed of. 
    Visualisation as barplots would come in handy for identifying parameters like: dataset size, number of different types of the data feature 
    """

# Implement class DataSize from DataAnalysisTemplate 
class DescriptiveAnalysis(DataAnalysisTemplate): 
    # Analyse the size of the dataset
    def analyse(self, df: pd.DataFrame):
        # Print the number of laptops and properties within the dataset 
        print(f"# Laptops: {df.shape[0]}")
        print(f"# Properties: {df.shape[1]}")

    def identify_dataset(self, df:pd.DataFrame):
        """ This method provides some useful information about the dataset.
            Parameters of dataset:
            - Size of the dataset: width and height
            - Number of data Types: float, int and objects """
        
        # Print on the terminal the data information sheet
        print("Data Information" + "-"*50)
        print(df.info())

        # Print the number of feature belong to that datatype
        laptops , properties = df.shape
        print(f"Total number laptops: {laptops}")
        print(f"Total number properties: {properties}")

# Implement class: provide statistical summery 
class StatisticalSummary(DataAnalysisTemplate):
    def analyse(self, df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
        "Providing stastical summary of the dataset. How the data is distributed numerically and "
        "Categorically"

        # Numerical summary 
        if stat_type == "Numerical" or stat_type == "numerical": 
            df_numeric = df.select_dtypes([int, float])
            print("Statistical Summary: Numerical Features")
            print(f"Number of numerical features: {df_numeric.shape[-1]}")
            return df_numeric.describe()

        # Categorical summary
        elif stat_type == "Categorical" or stat_type == "categorical": 
            df_categories = df.select_dtypes([object])
            print("Statistical Summary: Categorical Features")
            print(f"Number of categorical features: {df_categories.shape[-1]}")
            return df_categories.describe()
        else: 
            raise TypeError(f"The statistical type ({stat_type}) is unknown. Specify the type correctly without type errors. ")

# Implement class UniqueValueAnalysis to find distinct categories 
class UniqueValueAnalysis(DataAnalysisTemplate):
    def analyse(self, df):
        pass


# Implement DataLoader class to load the dataset
class DataLoader:
    def __init__(self, path_dir):
        self.path_dir = path_dir

    """This is used to create different data loaders with a specific file path where multiple csv files are stored. 
    Parameters of DataLoader used to create one: 
        - File path: str
        - Csv file: str 
        
    """
    def loading(self, csv_file): 
        # Combine the csv_file with the folder path
        csv_file_path = os.path.join(self.path_dir, csv_file)

        # Check whether the csv file is there or not 
        if not os.path.exists(csv_file_path): 
            raise TypeError("CSV-file not found in the directory. Please, specify the path name to CSV-file")
        
        # Load the dataframe
        df = pd.read_csv(csv_file_path)
        return df


# Example 
if __name__ == "__main__": 
    # Initialise the path variable 
    path = "D:\Projectwork Platform\MEP-Machine-Learning\Laptop_Price_Prediction\dataset"

    # Loading the data
    data_loader = DataLoader(path_dir=path)
    df = data_loader.loading("ebay_laptop_dataset.csv")

    # Create descrptive analysis object --> identifying the dataset
    descr_analysis = DescriptiveAnalysis()
    stats_summary = StatisticalSummary()
    descr_analysis.analyse(df)
    stats_summary.analyse(df)