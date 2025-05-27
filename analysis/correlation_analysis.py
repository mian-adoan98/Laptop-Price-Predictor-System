## Correlation analysis 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Implement a CorrelationAnalysis class
class CorrelationAnalysis:
    def __init__(self, dataset: pd.DataFrame):
        self.corr_data = dataset.select_dtypes([float, int]).corr()
        
    # Method 1: build correlation map 
    def visualise(self, x: int, y: int):
        # Adapt the size of the correlation plot 
        plt.figure(figsize=(x, y))
        # Visualise the correlation map 
        sns.heatmap(self.corr_data, annot=True)
        plt.xlabel("Features")
        plt.ylabel("Features")

        plt.show()

    # Method 2: show features with strong correlation index
    def feature_strong_corr(self) -> pd.DataFrame:
        # Constants for creating a high correlated dataframe
        featureX = self.corr_data["Price"].index
        featureX_vals = self.corr_data["Price"].values

        # Store features with high correlated index 
        high_corr_feartures = []
        high_corr_indices = []
        high_corr_set = {}

        low_corr_features = []

        # Iterate the corr price dataset
        for feature, value in zip(featureX, featureX_vals):
            # check if features has high correlated index 
            if value > 0.25 and value < 1:
                high_corr_feartures.append(feature)
                high_corr_indices.append(value)

            else:
                feature_couple = (feature, value)
                low_corr_features.append(feature_couple)
            
        # pd.DataFrame(high_corr_feartures)
        corr_ds_names = ["Features(HC)", "CorrelationIndex"]
        high_corr_df = self.build_corr_ds(high_corr_feartures, high_corr_indices, corr_ds_names)
        print(f"High correlated features: {np.array(high_corr_feartures).shape[0]}")
        return high_corr_df
    
    # Method 3: build a correlation dataset
    def build_corr_ds(self, namelist: list, valuelist: list, column_names: list) -> pd.DataFrame:
        # Initiate a correlation dataset
        corr_ds = pd.DataFrame()

        # Add the features into the dataset
        corr_ds[column_names[0]] = namelist
        corr_ds[column_names[1]] = valuelist

        return corr_ds

        # Example code
if __name__ == "__main__":
    print("Correlation analysis")