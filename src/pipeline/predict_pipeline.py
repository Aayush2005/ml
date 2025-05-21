import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        # self.model_path = os.path.join('artifacts', 'model.pkl')
        # self.scaler_path = os.path.join('artifacts', 'scaler.pkl')
        # self.model = load_object(self.model_path)
        # self.scaler = load_object(self.scaler_path)
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            scaler_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            scaler = load_object(scaler_path)

            data_scaled = scaler.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        
        except Exception as e:
            raise CustomException(e, sys) from e
            logging.error("Error during prediction")

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,  
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int,
                 ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity  
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],  
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data converted to DataFrame")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
            logging.error("Error converting custom data to DataFrame")
