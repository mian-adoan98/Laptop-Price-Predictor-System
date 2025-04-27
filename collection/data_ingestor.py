# Task 1: Data Collection 

# Import libraries which are useful for ingesting data from local disk
from abc import ABC, abstractmethod

import pandas as pd 
import os 

# -------------------------------------------------------------------------------
# This phase contains a laptop price dataset which needs to be ingested by a class called DataIngestor
# DataIngestor (abstract class): a template to create data-ingestor objects for collecting and refining the dataset

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, data_path: str) -> pd.DataFrame:
        # code for ingesting and extracting data from local disk 
        pass 

# FileDataIngestor (class): a template using functionality from the DataIngestor abstract class 
# ingesting csv files useful for this project and create a complete dataframe with libraries, such as Pandas

class FileDataIngestor(DataIngestor):
    def __init__(self, filename):
        DataIngestor.__init__(self)
        self.filename = filename

    # Ingestion method for importing and creating dataset from local disk 
    def ingest(self, data_path: str) -> pd.DataFrame:
        # Ensure if the path exists in the folder
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Folder with path {data_path} does not exist. Please, specify the folder correctly.")

        # Extract data from the dataset folder
        extracted_files = os.listdir(data_path)
        csv_files = [file for file in extracted_files if file.endswith(".csv")]

        print(f"{len(csv_files)}")
        # Verify if there are csv files or not
        if len(csv_files) == 0:
            raise FileNotFoundError("No csv files found in the dataset folder")
        elif len(csv_files) > 1 and self.filename in csv_files: 
            raise ValueError("Csv files are found. Please specify which one to work with")
        
        # Create a dataframe from a selected csv file
        csv_file_path = os.path.join(data_path, csv_files[csv_files.index(f"{self.filename}.csv")])
        df = pd.read_csv(csv_file_path)

        return df

# Create DataSeperator class to seperate based on data with many unknown values and data with less unknown values 

    
# example code 
if __name__ == "__main__": 
    # Initialise the path of all data 
    dataset_path = "D:\Projectwork Platform\MEP-Machine-Learning\Laptop_Price_Prediction\dataset"
    # Create a FileDataIngestor object 
    file_ingestor = FileDataIngestor("ebay_laptop_dataset")
    df = file_ingestor.ingest(data_path=dataset_path)
    
    # Print the first 5 rows of the dataframe
    print(df.head(5))