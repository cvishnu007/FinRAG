import pandas as pd

df = pd.read_csv("./data/raw/news/AAPL.csv")
print(df.columns)
print(df.head())
