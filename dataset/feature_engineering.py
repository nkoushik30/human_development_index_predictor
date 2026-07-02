from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# Load Dataset
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "HDI_Cleaned.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ----------------------------------
# Dataset Information
# ----------------------------------
print("\nOriginal Dataset Shape:", df.shape)

# ----------------------------------
# Handle Missing Values
# ----------------------------------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill numeric columns with mean
numeric_columns = df.select_dtypes(include=["number"]).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical columns with mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ----------------------------------
# Label Encoding (Country)
# ----------------------------------
encoder = LabelEncoder()

df["Country"] = encoder.fit_transform(df["Country"])

print("\nCountry column encoded successfully.")

# ----------------------------------
# Feature Selection
# ----------------------------------
X = df[
    [
        "Life Expectancy at Birth (2021)",
        "Expected Years of Schooling (2021)",
        "Mean Years of Schooling (2021)",
        "Gross National Income Per Capita (2021)",
    ]
]

y = df["Human Development Index (2021)"]

print("\nIndependent Variables (X)")
print(X.head())

print("\nDependent Variable (y)")
print(y.head())

# ----------------------------------
# Train-Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-Test Split Completed")
print("-" * 40)
print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)

print("\nPreprocessing completed successfully.")