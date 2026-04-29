import pandas as pd

eia = pd.read_pickle("data/processed/eia_custom_region_daily.pkl")
print(eia.head())
print(eia.isna().sum())
print("Rows:", len(eia))
print("Custom regions:", eia["custom_region"].unique())
print(eia.groupby("custom_region")["Demand Anomaly (MW)"].describe())


import pandas as pd

eia_daily = pd.read_pickle("data/processed/region_daily.pkl")
print("Columns:", list(eia_daily.columns))
print("Unique Region labels:", sorted(eia_daily["Region"].dropna().astype(str).unique())[:50])
print("Count:", eia_daily["Region"].dropna().nunique())