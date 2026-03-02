import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Config section ----
fileName = "googl_stock.csv"
targets = ["direction"]
dropToAvoidLeakage = ["Close", "Open", "High", "Low", "bb_mid", "bb_upper", "bb_lower", "bb_std", "ema_12", "ema_26", "ma_5", "ma_20"]
dateColumn = "Date"
encodingMethod = "target"
bestParams = {
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "reg_lambda": 10,
    "reg_alpha": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
}
highCardThreshold = 15
topFeatures = ['rsi', 'year', 'Volume', 'month_cos', 'day_of_year', 'returns', 'volume_change', 'macd_signal', 'ma_ratio', 'bb_position', 'volatility', 'month']

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

train_df["returns"] = train_df["Close"].pct_change()
train_df["ma_5"] = train_df["Close"].rolling(5).mean()
train_df["ma_20"] = train_df["Close"].rolling(20).mean()
train_df["ma_ratio"] = train_df["ma_5"] / train_df["ma_20"]
train_df["volatility"] = train_df["returns"].rolling(10).std()
train_df["volume_change"] = train_df["Volume"].pct_change()
train_df["rsi"] = 100 - (100 / (1 + train_df["returns"].rolling(14).apply(lambda x: x[x > 0].mean() / abs(x[x < 0].mean()))))
train_df["bb_mid"] = train_df["Close"].rolling(20).mean()
train_df["bb_std"] = train_df["Close"].rolling(20).std()
train_df["bb_upper"] = train_df["bb_mid"] + 2 * train_df["bb_std"]
train_df["bb_lower"] = train_df["bb_mid"] - 2 * train_df["bb_std"]
train_df["bb_position"] = (train_df["Close"] - train_df["bb_lower"]) / (train_df["bb_upper"] - train_df["bb_lower"])
train_df["ema_12"] = train_df["Close"].ewm(span=12).mean()
train_df["ema_26"] = train_df["Close"].ewm(span=26).mean()
train_df["macd"] = train_df["ema_12"] - train_df["ema_26"]
train_df["macd_signal"] = train_df["macd"].ewm(span=9).mean()

train_df = train_df.dropna()
                       
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

X = X[topFeatures]

if dropToAvoidLeakage:
    X = X.drop([c for c in dropToAvoidLeakage if c in X.columns], axis = 1)

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
    ("model", xgb.XGBClassifier()),
])

# ---- Train and evaluate ----
split_idx = int(len(X) * 0.75)
train_X, val_X = X.iloc[:split_idx], X.iloc[split_idx:]
train_y, val_y = y.iloc[:split_idx], y.iloc[split_idx:]

n_down = (train_y == 0).sum()
n_up = (train_y == 1).sum()
    
if __name__ == "__main__":
    if bestParams:
        pipeline.set_params(**{f"model__{k}": v for k, v in bestParams.items()})
    
    pipeline.fit(train_X, train_y)
    
    train_pred = pipeline.predict(train_X)
    val_pred = pipeline.predict(val_X)

    print("Train accuracy:", accuracy_score(train_y, train_pred))
    print("Val accuracy:", accuracy_score(val_y, val_pred))
    print("\n", classification_report(val_y, val_pred))
    joblib.dump(pipeline, "fitted_pipeline.pkl")