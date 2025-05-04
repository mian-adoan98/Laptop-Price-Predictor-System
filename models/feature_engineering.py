# Feature Engineering 
import pandas as pd 
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# implement techniques for engineerin the features
#   - categorical features: label encoder
#   - numerical features: feature scaler
# 
# #

class FeatureEngineering(ABC):
    """Transforming process: allow all ways of transforming data into demanded value
        categorical variables --> numerical variables: Label Encoding
        numerical variables --> scaled numerical variables: StandardScaler
    
        - param: feature_data 
        - response: dataframe with 1D"""
    
    @abstractmethod
    def tranform(feature_data: pd.DataFrame) -> pd.DataFrame:
        pass

# Implement FeatureScaler
class FeatureScaler(FeatureEngineering): 
    
    # Scaling the feature if features are numeric 
    def transform(self, feature_data: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
        # Choose type of scaling
        if scaler_type == "StandardScaler":
            # Scale the feautre using standard scaler
            scaler = StandardScaler()
            scaled_feature_data = scaler.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        elif scaler_type == "MinMax":
            # Scale the feature using min-max scaler
            scaler = MinMaxScaler()
            scaled_feature_data = scaler.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        else:
            raise ModuleNotFoundError("Scaler is not known. Please, specify your scaler")
        
        # Print feature has scaled
        print("Feature Scaling has succesfully accomplished")

# Implement FeatureEncoder
class FeatureEncoder(FeatureEngineering):

    # Method: encode categorical feature 
    def transform(self, feature_data: pd.DataFrame, encoder_type: str) -> pd.DataFrame:
        # Choose type of scaling
        if encoder_type == "LabelEncoder":
            # Scale the feautre using standard scaler
            scaler = LabelEncoder()
            scaled_feature_data = scaler.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        elif encoder_type == "OneHotEncoder":
            # Scale the feature using min-max scaler
            scaler = OneHotEncoder()
            scaled_feature_data = scaler.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        else:
            raise ModuleNotFoundError("Scaler is not known. Please, specify your scaler")

         # Print feature has encoded
        print("Feature encoding has successfully accomplished")

    
# Implement function: build a correlation map 
# Visualize the data distribution: target variable and features
def correlation_graph(x,dataX, dataY, y = "Price", color="blue"):
    feature = dataX[x]
    target = dataY

    plt.scatter(feature, target, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Correlation Graph: {x} vs {y}")
    plt.show()

# Example code 



