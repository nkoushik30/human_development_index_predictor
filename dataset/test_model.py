from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Load model
with open(BASE_DIR.parent / "models" / "hdi_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load cleaned dataset
df = pd.read_csv(BASE_DIR / "HDI_Cleaned.csv")

# Select one country (e.g., first row)
sample = df.iloc[0]

X = sample[[
    "Life Expectancy at Birth (2021)",
    "Expected Years of Schooling (2021)",
    "Mean Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)"
]].values.reshape(1, -1)

prediction = model.predict(X)

print("Country       :", sample["Country"])
print("Actual HDI    :", sample["Human Development Index (2021)"])
print("Predicted HDI :", round(prediction[0], 3))