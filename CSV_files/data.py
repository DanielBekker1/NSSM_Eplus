import pandas as pd
folder_path = r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\CSV_files"
# Read and preprocess the first CSV
df = pd.read_csv("CSV_files/dataframe_output_jan_Max_speed.csv")
df = df[:-16]  # Remove the last 16 rows

# Read and preprocess the second CSV
df1 = pd.read_csv(r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\CSV_files\electricity_price_15min_intervals.csv")
df1 = df1[:-16]  # Remove the last 16 rows

# Extract necessary columns
columns = ["Datetime", "Occupancy_schedule"]
df = df[columns]

# Prepare df1 with electricity price and add Datetime
elec_col = ["electricity_price"]

df1["Datetime"] = df["Datetime"].values  # Add Datetime from df into df1
df1["Electricity_price"] = df1[elec_col]
# Save the new CSV files

df.to_csv(f"{folder_path}/Occupancy.csv", index=False)
df1.to_csv(f"{folder_path}/Elec_price_21_days.csv", index=False)