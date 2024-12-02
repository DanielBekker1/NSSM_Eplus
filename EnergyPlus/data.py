import pandas as pd
import numpy as np

df = pd.read_csv("CSV_files/dataframe_output_jan_Max_speed.csv")
df = df[:-16]
columns = ["Datetime", "Occupancy_schedule"]

df = df[columns].values

print(df.head())