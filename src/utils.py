import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        model_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            grid_search = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model_name = list(models.keys())[i]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_pred)
            model_report[model_name] = test_model_score

        return model_report
    except Exception as e:
        raise CustomException(e, sys) from e
    


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e
        logging.error("Error loading object from file")