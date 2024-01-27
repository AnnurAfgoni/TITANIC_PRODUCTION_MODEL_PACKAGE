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

# CONFIGURATION VARIABLE
NUMERICAL_VARIABLES = [
    "age",
    "fare"
]

CATEGORICAL_VARIABLES = [
    "sex",
    "cabin",
    "embarked",
    "title"
]

CABIN = [
    "cabin"
]

# PIPELINE
titanic_pipe = Pipeline([
    # IMPUTATION
    ("categorical_imputation", CategoricalImputer(
        imputation_method="missing", variables=CATEGORICAL_VARIABLES
    )),
    ("missing_indicator", AddMissingIndicator(
        variables=NUMERICAL_VARIABLES
    )),
    ("median_imputation", MeanMedianImputer(
        imputation_method="median", variables=NUMERICAL_VARIABLES
    )),

    # EXTRACT LETTER
    ("extract_letter", pp.ExtractLetterTransformer(variables=CABIN)),

    # CATEGORICAL ENCODING
    ("rare_label_encoder", RareLabelEncoder(
        tol=0.05, n_categories=1, variables=CATEGORICAL_VARIABLES
    )),
    ("categorical_encoder", OneHotEncoder(
        drop_last=True, variables=CATEGORICAL_VARIABLES
    )),

    # scaling
    ("scaler", StandardScaler()),

    # Model
    ("Logreg", LogisticRegression(C=0.0005, random_state=0))
])