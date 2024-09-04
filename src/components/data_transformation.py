import os
import sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler

#class to give the path outputs
@dataclass
class DataTransformationConfig:
    processed_train_data_path: str = os.path.join('artifacts', "processed_train.csv")
    processed_test_data_path: str = os.path.join('artifacts', "processed_test.csv")
    scaler_path: str = os.path.join('artifacts', "scaler.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info("Starting data transformation process")

        try:
            #load the datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Loaded the train and test datasets")

            #separate features and target variable from the training dataset
            X_train = train_df.drop(columns = ['count', 'id'])
            y_train = train_df['count']

            #drop column
            X_test = test_df.drop(columns = ['id'])

            logging.info("Separated features and target variable for the training dataset")

            #scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            logging.info("Applied standard scaling to the features")

            #change to dfs
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns)

            #save scaler 
            os.makedirs(os.path.dirname(self.transformation_config.scaler_path), exist_ok = True)
            pd.to_pickle(scaler, self.transformation_config.scaler_path)

            logging.info("Scaler object saved for future use")

            #save processed dfs
            X_train_scaled_df.to_csv(self.transformation_config.processed_train_data_path, index = False)
            X_test_scaled_df.to_csv(self.transformation_config.processed_test_data_path, index = False)

            logging.info("Saved the transformed datasets")

            return (X_train_scaled_df, y_train), X_test_scaled_df

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'
    
    data_transformation = DataTransformation()
    (X_train, y_train), X_test = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    logging.info(f"Data transformation completed. Processed train data shape: {X_train.shape}, Processed test data shape: {X_test.shape}") 



