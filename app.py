from pathlib import Path
import pickle
import pandas as pd

from flask import Flask, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# -----------------------------------
# Initialize Flask
# -----------------------------------

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[]
)

# -----------------------------------
# Load Model and Dataset
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "models" / "hdi_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(BASE_DIR / "dataset" / "HDI_Cleaned.csv")

# -----------------------------------
# Home Route
# -----------------------------------

@app.route("/")
def index():
    countries = (
        df["Country"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    return render_template(
        "index.html",
        countries=countries
    )

# -----------------------------------
# About Route
# -----------------------------------

@app.route("/about")
def about():
    return render_template("about.html")

# -----------------------------------
# Prediction Route
# -----------------------------------

@app.route("/predict", methods=["POST"])
@limiter.limit("10 per hour")
def predict():
    # Prediction logic here
    pass

# -----------------------------------
# Run Flask
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)