import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Config section ----
fileName = "googl_stock.csv"
targets = ["Close"]
topFeatures = []
dropToAvoidLeakage = ["High", "Low"]
dateColumn = "Date"
encodingMethod = "target"  # "target", "onehot", or "ordinal"
highCardThreshold = 15
bestParams = {}

filePath = "/Users/chenyige/Documents/python/001_machineLearning/datasets/" + fileName

# ---- Custom target encoder since sklearn's is broken on Kaggle ----
class ManualTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mappings_ = {}
        self.global_mean_ = None
        self.columns_ = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        self.columns_ = X.columns
        self.global_mean_ = y.mean()
        self.mappings_ = {}
        for col in self.columns_:
            self.mappings_[col] = pd.concat(
                [X[col], pd.Series(y, index=X.index, name="target")], axis=1
            ).groupby(col)["target"].mean().to_dict()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        X = X.copy()
        for col in self.columns_:
            X[col] = X[col].map(self.mappings_[col]).fillna(self.global_mean_)
        return X.values

# ---- Load and prep ----
train_df = pd.read_csv(filePath)

for col in train_df.select_dtypes(include=["string"]).columns:
    train_df[col] = train_df[col].astype("object")

if dateColumn and dateColumn in train_df.columns:
    dt = pd.to_datetime(train_df[dateColumn])
    date_features = pd.DataFrame({
        "year": dt.dt.year,
        "day_of_week": dt.dt.dayofweek,
        "quarter": dt.dt.quarter,
        "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
        "month": dt.dt.month,
        "day_of_year": dt.dt.dayofyear,
    })
    date_features["month_sin"] = np.sin(2 * np.pi * date_features["month"] / 12)
    date_features["month_cos"] = np.cos(2 * np.pi * date_features["month"] / 12)
    train_df = pd.concat([train_df.drop(dateColumn, axis=1), date_features], axis=1)

y = train_df[targets].squeeze()
X = train_df.drop(targets, axis=1)

if dropToAvoidLeakage:
    X = X.drop([c for c in dropToAvoidLeakage if c in X.columns], axis = 1)
if topFeatures:
    X = X[topFeatures]

# ---- Identify column types ----
cat_cols = list(X.select_dtypes(include=["object", "string"]).columns)
num_cols = list(X.select_dtypes(include=["number"]).columns)

if encodingMethod == "onehot":
    cat_cols = list(X.select_dtypes(include=["object", "string"]).columns)
    high_card = [c for c in cat_cols if X[c].nunique() > highCardThreshold]
    cat_cols = [c for c in cat_cols if c not in high_card]
    X = X.drop(high_card, axis=1)

# ---- Build pipeline ----
if encodingMethod == "target":
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", ManualTargetEncoder()),
    ])
elif encodingMethod == "onehot":
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
else:
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
if cat_cols:
    transformers.append(("cat", cat_transformer, cat_cols))

preprocessor = ColumnTransformer(transformers)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", xgb.XGBRegressor()),
])

# ---- Train and evaluate ----
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)
    
if __name__ == "__main__":
    if bestParams:
        pipeline.set_params(**{f"model__{k}": v for k, v in bestParams.items()})
    
    pipeline.fit(train_X, train_y)
    
    train_pred = pipeline.predict(train_X)
    val_pred = pipeline.predict(val_X)

    print("Mean target value (train):", train_y.median())
    print("MAE (train):", mean_absolute_error(train_y, train_pred))
    print("R2 (train):", r2_score(train_y, train_pred))
    print("Mean target value (val):", val_y.median())
    print("MAE (val):", mean_absolute_error(val_y, val_pred))
    print("R2 (val):", r2_score(val_y, val_pred))
    joblib.dump(pipeline, "fitted_pipeline.pkl")