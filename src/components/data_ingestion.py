import os
import sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass 
from src.exception import CustomException

#create a data ingestion class to get inputs
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv") #all outputs are stored in the artifacts folder
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #the paths will be saved in this class variable

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            train_df = pd.read_csv(r'Notebooks\Dataset\train.csv')
            test_df = pd.read_csv(r'Notebooks\Dataset\test.csv')
            logging.info("Read the Datasets")

            #save the datasets to the specified paths in artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header = True)

            logging.info("Data ingestion process completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()