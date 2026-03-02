import joblib
import pandas as pd
import numpy as np
from xgbModel import ManualTargetEncoder

pipeline = joblib.load("fitted_pipeline.pkl")

areas = [
    "city of london", "barking and dagenham", "barnet", "bexley", "brent",
    "bromley", "camden", "croydon", "ealing", "enfield", "greenwich",
    "hackney", "hammersmith and fulham", "haringey", "harrow", "havering",
    "hillingdon", "hounslow", "islington", "kensington and chelsea",
    "kingston upon thames", "lambeth", "lewisham", "merton", "newham",
    "redbridge", "richmond upon thames", "southwark", "sutton",
    "tower hamlets", "waltham forest", "wandsworth", "westminster"
]

print("=== London House Price Predictor ===\n")
print("Available areas:")
for i, area in enumerate(areas, 1):
    print(f"  {i}. {area}")

area_choice = int(input("\nPick area number: ")) - 1
area = areas[area_choice]

year = int(input("Year (e.g. 2024): "))
month = int(input("Month (1-12): "))

values = {
    "area": area,
    "houses_sold": float(input("Houses sold (e.g. 100): ")),
    "no_of_crimes": float(input("Number of crimes (e.g. 500, or 0 if unknown): ")),
    "year": year,
    "day_of_week": 0,
    "quarter": (month - 1) // 3 + 1,
    "is_weekend": 0,
    "month": month,
    "day_of_year": int(pd.Timestamp(year=year, month=month, day=1).dayofyear),
    "month_sin": np.sin(2 * np.pi * month / 12),
    "month_cos": np.cos(2 * np.pi * month / 12),
}

X_new = pd.DataFrame([values])
prediction = pipeline.predict(X_new)
print(f"\nPredicted average house price in {area}: £{prediction[0]:,.0f}")