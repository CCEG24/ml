from sklearn.model_selection import RandomizedSearchCV
from xgbModel import pipeline, train_X, train_y, val_X, val_y
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

param_distributions = {
    "model__n_estimators": [100, 200, 500, 1000],
    "model__max_depth": [2, 3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__reg_alpha": [1, 5, 10, 20],
    "model__reg_lambda": [1, 5, 10, 20],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

search.fit(train_X, train_y)

cleaned = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
print("bestParams =", cleaned)
print("Best CV MAE:", -search.best_score_)

val_pred = search.predict(val_X)
print("Val MAE:", mean_absolute_error(val_y, val_pred))
print("Val R2:", r2_score(val_y, val_pred))