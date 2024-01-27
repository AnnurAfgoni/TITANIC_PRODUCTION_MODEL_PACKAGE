import pathlib

import classification_model

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "survived"

FEATURES = [
    'age'
    , 'fare'
    , 'sex'
    , 'cabin'
    , 'embarked'
    , 'title'
]