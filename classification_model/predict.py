import numpy as np
import pandas as pd

from classification_model.processing.data_management import load_pipeline
from classification_model.config import  config

pipeline_file_name = "titanic_classification_model.pkl"
_titanic_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:

    data = pd.read_json(input_data)
    prediction = _titanic_pipe.predict(data[config.FEATURES])
    response = {"prediction": prediction}

    return response