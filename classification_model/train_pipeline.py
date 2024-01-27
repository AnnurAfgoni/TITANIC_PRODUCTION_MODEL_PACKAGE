import numpy as np
from sklearn.model_selection import train_test_split

from classification_model import pipeline
from classification_model.config import config
from classification_model.processing.data_management import (
    load_dataset,
    save_pipeline
)

def run_training() -> None:

    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=0
    )

    pipeline.titanic_pipe.fit(X_train[config.FEATURES], y_train)

    save_pipeline(pipeline_to_persist=pipeline.titanic_pipe)

if __name__ == "__main__":
    run_training()