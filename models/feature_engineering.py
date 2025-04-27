# Feature Engineering 
import pandas as pd 
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from 
# implement techniques for engineerin the features
#   - categorical features: label encoder
#   - numerical features: 
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

# Implement LabelEncoder 
class LabelEncoder()



