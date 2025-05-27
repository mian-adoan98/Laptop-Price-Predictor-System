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
    def tranform(feature_data: pd.DataFrame, scaler_type:str) -> pd.DataFrame:
        pass

# Implement FeatureScaler
class FeatureScaler(FeatureEngineering): 
    def __init__(self):
        FeatureEngineering.__init__(self)
        self.std_scaler = StandardScaler()
        self.mm_scaler =  MinMaxScaler()
        
    # Scaling the feature if features are numeric 
    def transform(self, feature_data: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
        # Choose type of scaling
        if scaler_type == "StandardScaler":
            # Scale the feautre using standard scaler
            scaled_feature_data = self.scaler.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        elif scaler_type == "MinMax":
            # Scale the feature using min-max scaler
            scaled_feature_data = self.mm_scaler.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        else:
            raise ModuleNotFoundError("Scaler is not known. Please, specify your scaler")
        
        # Print feature has scaled
        print("Feature Scaling has succesfully accomplished")

# Implement FeatureEncoder
class FeatureEncoder(FeatureEngineering):
    def __init__(self, feature_data: pd.DataFrame):
        self.label_encoder = LabelEncoder()
        self.oht_encoder = OneHotEncoder()

    # Method: encode categorical feature 
    def transform(self, feature_data: pd.DataFrame, encoder_type: str) -> pd.DataFrame:
        # Choose type of scaling
        if encoder_type == "LabelEncoder":
            # Scale the feautre using standard scaler
            # scaler = LabelEncoder()
            scaled_feature_data = self.label_encoder.fit_transform(feature_data).reshape(-1, 1)
            return scaled_feature_data
        elif encoder_type == "OneHotEncoder":
            # Scale the feature using min-max scaler
            scaler = OneHotEncoder()
            scaled_feature_data = self.oht_encoder.fit_transform(feature_data).reshape(-1, 1)
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



