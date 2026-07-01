import pandas as pd

df = pd.read_csv("dataset/Human Development Index - Full.csv")

print(df.info())
print(df.isnull().sum())