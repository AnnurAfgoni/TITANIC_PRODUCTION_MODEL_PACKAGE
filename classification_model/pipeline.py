# import model machine learning
from sklearn.linear_model import LogisticRegression

# import pipeline
from sklearn.pipeline import Pipeline

# import preprocessing
from sklearn.preprocessing import StandardScaler

# import preprocessors
import preprocessors as pp

# feature engine for imputation
from feature_engine.imputation import(
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer
)

# feature engine for encoding
from feature_engine.encoding import(
    RareLabelEncoder,
    OneHotEncoder
)

# config
from classification_model.config.core import config

# PIPELINE
titanic_pipe = Pipeline([
    # IMPUTATION
    ("categorical_imputation", CategoricalImputer(
        imputation_method="missing", 
        variables=config.model_config.categorical_variables
    )),
    ("missing_indicator", AddMissingIndicator(
        variables=config.model_config.numerical_variables
    )),
    ("median_imputation", MeanMedianImputer(
        imputation_method="median", 
        variables=config.model_config.numerical_variables
    )),

    # EXTRACT LETTER
    ("extract_letter", pp.ExtractLetterTransformer(
        variables=config.model_config.cabin
    )),

    # CATEGORICAL ENCODING
    ("rare_label_encoder", RareLabelEncoder(
        tol=0.05, n_categories=1, 
        variables=config.model_config.categorical_variables
    )),
    ("categorical_encoder", OneHotEncoder(
        drop_last=True, 
        variables=config.model_config.categorical_variables
    )),

    # scaling
    ("scaler", StandardScaler()),

    # Model
    ("Logreg", LogisticRegression(
        C=config.model_config.param_c, 
        random_state=config.model_config.random_state
    ))
])