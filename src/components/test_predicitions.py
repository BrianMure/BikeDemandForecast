#load libraries
import joblib
import numpy as np
import pandas as pd
from src.components.data_transformation import DataTransformation

#load the trained Random Forest model
rf_model_path = 'artifacts/models/rf_model.pkl'
rf_model = joblib.load(rf_model_path)


#prepare the test dataset
test_data_path = 'artifacts/test.csv'
data_transformation = DataTransformation()
_, X_test = data_transformation.initiate_data_transformation('artifacts/train.csv', test_data_path)

#ass clusters to the x_test
kmeans_model_path = 'artifacts/models/kmeans_model.pkl'
kmeans_model = joblib.load(kmeans_model_path)
X_test['cluster'] = kmeans_model.predict(X_test)

#make predictions
predictions = rf_model.predict(X_test)

#round of to nearest whole number
rounded_predictions = np.round(predictions).astype(int)
print(rounded_predictions)
