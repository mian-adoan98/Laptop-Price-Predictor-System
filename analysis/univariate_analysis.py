## Univariate Analysis
from abc import ABC, abstractmethod

# Import libraries for data visualisation 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# -------------------------------------------------------------------------
""" Univariate analysis 
    = analysis on one specific feature
    - these features consist in two seperate groups: numerical features and categorical features 
    - Numerical features --> Histplot: checking the distribution per feature 
    - Categorical features --> Barplot: identifying possible logical categories """

# Implement a UnivariateAnalysis abstract class 
class UnivariateAnalysis(ABC):
    @abstractmethod
    def analyse(self, df: pd.DataFrame) -> pd.DataFrame: 
        # Code template for all possible univariate analysis classes to perform analysis on one specific feature
        pass 

    @abstractmethod
    def visualise(self, df: pd.DataFrame, feature: str):
        # Code template for all possible univariate analysis classes to visualise the feature data
        pass 

# Implement NumericalAnalysis class to visualise numerical features 
class NumericalAnalysis(UnivariateAnalysis):
    # Implement analyse function --> analyse the numerical distribution 
    def analyse(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initialise constants for extracting numerical features 
        num_features = []
        num_feature_dict = {}

        # Iterations: extract numeric features
        for feature in df.columns:
            # Extract the feature data
            feature_data = df[feature]
            is_numeric = feature if pd.api.types.is_numeric_dtype(feature_data) else  0
            num_features.append(is_numeric)
            num_features = [feature for feature in num_features if feature != 0]

        # Iterations 2: compute statistical parameters of numeric features (min, max, avg)
        for feature in num_features:
            # Computing the statistical parameters of the feature
            feature_data = df[feature]
            feature_min = self.compute_stats_pm(feature_data)
            feature_max = self.compute_stats_pm(feature_data)
            feature_mean = self.compute_stats_pm(feature_data)
            feature_std = self.compute_stats_pm(feature_data)

            # Encapsulate the feature and its statistical parameter into a list
            stats_params = [feature_max, feature_min, feature_mean, feature_std]
            feature_couple = self.encapsulate_features(feature, stats_params)
            num_feature_dict.update(feature_couple)

        # Create a statistical feature dataframe
        stats_labels = ["Maximum", "Minimum", "Average", "Standard Deviation"]
        feature_df = pd.DataFrame(num_feature_dict)
        feature_df["Statistical Parameters"] = pd.Series(stats_labels)

        return feature_df
    
    def compute_stats_pm(self, feature_data: pd.DataFrame) -> list:
        # compute these statistical feature 
        feature_min = feature_data.min()
        feature_max = feature_data.max()
        feature_avg = feature_data.mean()
        feature_std = feature_data.std()

        # Format the paramters on 3 decimals
        stats_pms = [feature_min, feature_max, feature_avg, feature_std]
        stats_pms = [round(pm, 3) for pm in stats_pms]
        
        return stats_pms
    
    def encapsulate_features(self, feature_name: str, parameters: list) -> dict: 
        feature_dict = {feature_name: parameters}
        return feature_dict
    
    # Implement visualisation method --> creating histplots based on selected feature 
    def visualise(self, df, feature):
        # Create subplots 
        fig, ax = plt.subplots(1, 2, figsize=(10,6))

        # Visualise histogram + boxplot for one feature 
        sns.histplot(df[feature], bins=15, kde=True, color="black", ax=ax[0])
        plt.title(f"{feature} Distribution")
        
        # visualise boxplot to find outliers 
        sns.boxplot(df[feature], color="black")
        plt.show()

# Example code 

if __name__ == "__main__":
    # Import the dataset
    csv_file_path = "D:\\Projectwork Platform\\MEP-Machine-Learning\\Laptop_Price_Prediction\\dataset\\ebay_laptop_dataset.csv"
    df = pd.read_csv(csv_file_path)

    # Crate a statistical dataframe 
    numeric_ansis = NumericalAnalysis()
    num_stats_sum = numeric_ansis.analyse(df=df)
    print(num_stats_sum)

    numeric_ansis.visualise(df, "Price")