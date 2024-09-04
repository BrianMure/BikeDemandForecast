import os
import sys
import joblib
import numpy as np
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join('artifacts', "models")
    kmeans_file: str = os.path.join('artifacts', "models", "kmeans_model.pkl")
    rf_file: str = os.path.join('artifacts', "models", "rf_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_dir, exist_ok = True)

    def train_kmeans(self, X_train, n_clusters = 3):
        logging.info("Starting Kmeans training")

        try:
            kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
            kmeans.fit(X_train)
            joblib.dump(kmeans, self.model_trainer_config.kmeans_file)
            logging.info("Saved KMeans Model")
            return kmeans
        except Exception as e:
            logging.error("Error occurred during KMeans training")
            raise CustomException(e, sys)

    def train_random_forest(self, X_train, y_train):
        logging.info("Training Random Forest Regressor")

        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            joblib.dump(rf, self.model_trainer_config.rf_file)
            logging.info("Saved Random Forest Model")
            return rf
        except Exception as e:
            logging.error("Error occurred during Random Forest training")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        logging.info("Evaluating model performance")

        try:
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            logging.info(f"Model evaluation completed: MSE = {mse}, R2 Score = {r2}")
            return mse, r2
        except Exception as e:
            logging.error("Error occurred during model evaluation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'

    data_transformation = DataTransformation()
    (X_train, y_train), X_test = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()

    # Train KMeans model
    kmeans_model = model_trainer.train_kmeans(X_train)

    # Use KMeans clusters as an additional feature in the Random Forest model
    X_train['cluster'] = kmeans_model.labels_
    X_test['cluster'] = kmeans_model.predict(X_test)

    # Train Random Forest model
    rf_model = model_trainer.train_random_forest(X_train, y_train)

    # Evaluate the Random Forest model
    mse, r2 = model_trainer.evaluate_model(rf_model, X_test, y_train[:len(X_test)])  # Adjust y_test to match X_test size

    logging.info(f"Final Model Performance: MSE = {mse}, R2 Score = {r2}")
