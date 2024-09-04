#reading from a db and also saving model to cloud
import os
import sys
import pandas as pd
import pickle
import joblib
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error("Error occurred while loading object")
        raise CustomException(e, sys)

def save_model(file_path, model):
    """
    Save a machine learning model using joblib.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error("Error occurred while saving model")
        raise CustomException(e, sys)

def load_model(file_path):
    """
    Load a machine learning model using joblib.
    """
    try:
        model = joblib.load(file_path)
        logging.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logging.error("Error occurred while loading model")
        raise CustomException(e, sys)

def save_data(file_path, data):
    """
    Save a DataFrame or array-like object to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error("Error occurred while saving data")
        raise CustomException(e, sys)

def load_data(file_path):
    """
    Load a CSV file into a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logging.error("Error occurred while loading data")
        raise CustomException(e, sys)