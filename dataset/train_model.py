from pathlib import Path
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Load Dataset
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

df = pd.read_csv(BASE_DIR / "HDI_Cleaned.csv")

print("=" * 60)
print("MACHINE LEARNING MODEL BUILDING")
print("=" * 60)

print("\nDataset Shape:", df.shape)

# =====================================================
# Independent & Dependent Variables
# =====================================================

X = df[
    [
        "Life Expectancy at Birth (2021)",
        "Expected Years of Schooling (2021)",
        "Mean Years of Schooling (2021)",
        "Gross National Income Per Capita (2021)"
    ]
]

y = df["Human Development Index (2021)"]

print("\nFeatures Selected")
print(X.columns.tolist())

print("\nTarget Variable")
print(y.name)

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# Train Linear Regression Model
# =====================================================

model = LinearRegression()

print("\nTraining Linear Regression Model...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully.")

# =====================================================
# HDI Score Prediction
# =====================================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions")

prediction_df = pd.DataFrame({
    "Actual HDI": y_test.values,
    "Predicted HDI": y_pred
})

print(prediction_df.head(10))

# =====================================================
# Accuracy Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\nAccuracy Evaluation")
print("-" * 40)

print(f"Mean Absolute Error : {mae:.4f}")
print(f"Mean Squared Error  : {mse:.6f}")
print(f"Root Mean Squared Error : {rmse:.4f}")

# =====================================================
# R-Squared Analysis
# =====================================================

r2 = r2_score(y_test, y_pred)

print("\nR-Squared Analysis")
print("-" * 40)

print(f"R² Score : {r2:.4f}")

# =====================================================
# Performance Testing
# =====================================================

print("\nPerformance Summary")
print("-" * 40)

if r2 >= 0.90:
    print("Excellent Model Performance")
elif r2 >= 0.75:
    print("Good Model Performance")
elif r2 >= 0.50:
    print("Average Model Performance")
else:
    print("Model Needs Improvement")

# =====================================================
# Save Model
# =====================================================

models_dir = PROJECT_DIR / "models"
models_dir.mkdir(exist_ok=True)

model_file = models_dir / "hdi_model.pkl"

with open(model_file, "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully")
print(model_file)