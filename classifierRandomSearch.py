from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgbClassifier import pipeline, train_X, train_y, val_X, val_y
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

param_distributions = {
    "model__n_estimators": [50, 100, 200, 300],
    "model__max_depth": [2, 3, 4, 5],
    "model__learning_rate": [0.001, 0.01, 0.05, 0.1],
    "model__subsample": [0.5, 0.6, 0.7, 0.8],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__reg_alpha": [1, 5, 10, 20],
    "model__reg_lambda": [1, 5, 10, 20],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=7,
)

search.fit(train_X, train_y)

cleaned = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
print("bestParams =", cleaned)
print("Best CV F1:", search.best_score_)

val_pred = search.predict(val_X)
print("Val accuracy:", accuracy_score(val_y, val_pred))
print("Val F1:", f1_score(val_y, val_pred))
print(classification_report(val_y, val_pred))