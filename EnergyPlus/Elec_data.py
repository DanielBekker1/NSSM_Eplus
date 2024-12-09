import openpyxl
import pandas as pd

file_path = r'C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\CSV_files\2024_DK1_Elspotprices.xlsx'  # Replace with your file path


data = pd.read_excel(file_path)
data = data.drop(columns=['HourUTC', 'PriceArea', 'SpotPriceEUR'])
data.rename(columns={'HourDK': 'Datetime'}, inplace=True)
# Ensure the 'Datetime' column is in datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Filter data for the date range
start_date = '2024-01-01 00:00:00'
end_date = '2024-01-22 00:00:00'
filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]
expanded_data = pd.DataFrame(
    {
        'Datetime': pd.date_range(
            start=filtered_data['Datetime'].min(), 
            end=filtered_data['Datetime'].max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15), 
            freq='15min'  # Use 'min' for minutes instead of 'T'
        )
    }
)

# Map the price to each 15-minute interval
expanded_data['PriceDKK'] = expanded_data['Datetime'].apply(
    lambda x: filtered_data.loc[
        filtered_data['Datetime'] == x.floor('h'), 'SpotPriceDKK'  # Use 'h' for hours instead of 'H'
    ].values[0]
)
expanded_data.rename(columns={'PriceDKK': 'Electricity_price'}, inplace=True)
filtered_data = expanded_data[1:-19]

# Save the filtered data to a new Excel file
filtered_file_path = r'C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\CSV_files\ElectricityPrice_Jan.csv'
filtered_data.to_csv(filtered_file_path, index=False)








