# Feature Engineering 

# Import libraries for building feature engineering algorithms
from abc import ABC, abstractmethod
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Import dependencies for feature engineering
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
# implement techniques for engineerin the features
#   - categorical features: label encoder
#   - numerical features: feature scaler
# 

# Implement FeatureEngineering abstract class considered as blueprint for other Feature Engineering techniques
class FeatureEngineering(ABC):
    """Transforming process: allow all ways of transforming data into demanded value
        categorical variables --> numerical variables: Label Encoding
        numerical variables --> scaled numerical variables: StandardScaler
    
        - param: feature_data 
        - response: dataframe with 1D"""
    
    @abstractmethod
    def transform(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        pass


# Implement FeatureScaler class
class FeatureScaler(FeatureEngineering): 
    def __init__(self, scaler_type:str):
        # Data fields: FeatureScaler
        self.std_scaler = StandardScaler()
        self.mm_scaler =  MinMaxScaler()
        self.normalizer = Normalizer()

        self.scaler_type = scaler_type

    # Scaling the feature if features are numeric 
    def transform(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        # Check dimension of feature 
        feature_dim = feature_data.ndim
        if feature_dim != 2: 
            print(f"Feature dimension is not 2. Please reshape the dimension to 2D")

        # Transform method: Choose type of scaling
        if self.scaler_type == "StandardScaler":
            # Scale the feautre using standard scaler
            scaled_feature_data = self.std_scaler.fit_transform(feature_data.values)

             # Print feature has scaled
            print("Feature Scaling has succesfully accomplished")
            return scaled_feature_data
        elif self.scaler_type == "MinMax":
            # Scale the feature using min-max scaler
            scaled_feature_data = self.mm_scaler.fit_transform(feature_data.values)

             # Print feature has scaled
            print("Feature Scaling has succesfully accomplished")
            return scaled_feature_data
        
        elif self.scaler_type == "Normalizer":
            # Scale the feature using min-max scaler
            scaled_feature_data = self.normalizer.fit_transform(feature_data.values)

             # Print feature has scaled
            print("Feature Scaling has succesfully accomplished")
            return scaled_feature_data


        else:
            raise ModuleNotFoundError("Scaler is not known. Please, specify your scaler")


# Implement FeatureEncoder
class FeatureEncoder(FeatureEngineering):
    def __init__(self, encoder_type: str):
        # Data fields: FeatureEncoding
        self.encoder_type = encoder_type
        self.label_encoder = LabelEncoder()
        self.oht_encoder = OneHotEncoder(sparse_output=False)

    # Method: encode categorical feature 
    def transform(self, feature_data: pd.DataFrame, feature_names: list = None) -> pd.DataFrame:
        # Check the dimension of feature
        # Choose type of scaling
        if self.encoder_type == "LabelEncoder":
            # Scale the feautre using standard scaler
            encoded_feature_data = self.label_encoder.fit_transform(feature_data).reshape(-1, 1)
            # feature_name = feature_data.name
            # Print feature has encoded
            print("Feature encoding is successfull")
            return encoded_feature_data
        elif self.encoder_type == "OneHotEncoder" and len(feature_names) != 0:
            # Scale the feature using min-max scaler
            oht_feature_encode = self.oht_encoder.fit_transform(feature_data[feature_names])
            oht_feature_names = self.oht_encoder.get_feature_names_out(feature_names)
            encoded_feature_data = pd.DataFrame(oht_feature_encode, columns=oht_feature_names)

            # Print feature has encoded
            print("Feature encoding has successfully accomplished")
            return encoded_feature_data
        else:
            raise ModuleNotFoundError("Scaler is not known. Please, specify your scaler")

    
# Implement function: build a correlation map 
# Visualize the data distribution: target variable and features
def correlation_graph(data:pd.DataFrame, x: str, y:str = "Price", color: str ="blue"):
    feature = data[x]
    target = data[y]

    plt.scatter(feature, target, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Correlation Graph: {x} vs {y}")
    plt.show()


# Visualise correlation map: 
def correlation_map(dataset:pd.DataFrame, xpos: int = 10, ypos: int = 6): 
    # Select numeric features 
    ds = dataset.select_dtypes([int, float])

    # Create correlation index map
    corr_ds = ds.corr()

    # Visualise correlation map
    plt.figure(figsize=(xpos, ypos))
    sns.heatmap(corr_ds, annot=True)
    plt.xlabel("Input Features")
    plt.ylabel("Target Features")
    plt.title("Correlation Map")
    plt.show()


# Implement function: loading dataset 
def data_loader(path: str, file_index: int) -> pd.DataFrame:
    # Create a file dictionary including all datasets (with csv-extension)
    # path = "C:\Development\Projects\MachineLearning\Laptop-Price-Predictor-System\dataset"
    file_list = os.listdir(path)
    file_dict = {i:file for i, file in enumerate(file_list)}
    print(file_dict)

    # Load the dataset 
    filename = os.path.join(path, file_list[file_index])
    ds = pd.read_csv(filename)
    return ds


# Example code 
if __name__ == "__main__":
    # Prepare a dataset 
    dataset = pd.read_csv("C:\Development\Projects\MachineLearning\Laptop-Price-Predictor-System\dataset\ebay_laptop_data_cleaned.csv")
    xfeature = dataset[["Processor Speed"]]
    
    # Build a FeatureScaler object 
    std_scaler = FeatureScaler(scaler_type="StandardScaler")
    scaled_xfeature = std_scaler.transform(xfeature)
    # print(scaled_xfeature)

    # Build a FeatureEncoder object
    label_encoder = FeatureEncoder(encoder_type= "LabelEncoder")
    cat_feature = dataset["Processor"]
    encoded_feature = label_encoder.transform(cat_feature)
    print(encoded_feature)

