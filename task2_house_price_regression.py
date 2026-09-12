"""
QSkill Internship - Task 2 (Slab 1, Beginner)
Linear Regression model to predict house price based on rooms, size,
location, and other features.
Dataset: house_prices.csv (Rooms, SizeSqFt, LocationScore, AgeYears, Price)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("house_prices.csv")

print("=" * 60)
print("DATASET PREVIEW")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")
print("\nSummary statistics:")
print(df.describe())

features = ["Rooms", "SizeSqFt", "LocationScore", "AgeYears"]
target = "Price"

X = df[features]
y = df[target]

print(f"\nMissing values:\n{df.isnull().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Mean Absolute Error : ${mae:,.2f}")
print(f"Root Mean Sq. Error : ${rmse:,.2f}")
print(f"R^2 Score            : {r2:.4f}")

print("\nFeature importance (standardized coefficients):")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:15s}: {coef:,.2f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"Actual vs Predicted House Prices (R\u00b2 = {r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("task2_actual_vs_predicted.png", dpi=150)
print("\nSaved visualization to task2_actual_vs_predicted.png")

new_house = pd.DataFrame({
    "Rooms": [4],
    "SizeSqFt": [2200],
    "LocationScore": [8],
    "AgeYears": [5],
})

new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)[0]

print("\n" + "=" * 60)
print("EXAMPLE PREDICTION")
print("=" * 60)
print(new_house.to_string(index=False))
print(f"Predicted Price: ${predicted_price:,.2f}")
