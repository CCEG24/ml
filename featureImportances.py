import matplotlib.pyplot as plt
#from xgbModel import pipeline, train_X, train_y, num_cols, cat_cols, bestParams
from xgbClassifier import pipeline, train_X, train_y, num_cols, cat_cols, bestParams

pipeline.set_params(**{f"model__{k}": v for k, v in bestParams.items()})
pipeline.fit(train_X, train_y)

xgb_model = pipeline.named_steps["model"]
feature_names = num_cols + cat_cols
importances = xgb_model.feature_importances_
sorted_idx = importances.argsort()[-20:]

top_n = 10
top_idx = importances.argsort()[-top_n:]
top_features = [feature_names[i] for i in top_idx]
print("topFeatures =", top_features)

plt.figure(figsize=(10, 8))
plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
plt.title("Feature Importances")
plt.tight_layout()
plt.show()