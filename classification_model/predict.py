import numpy as np
import pandas as pd

from processing.data_management import load_pipeline
from config.core import config

from classification_model import __version__ as _version

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_titanic_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:

    data = pd.read_json(input_data)
    prediction = _titanic_pipe.predict(data[config.app_config.features])
    response = {"prediction": prediction}

    return response