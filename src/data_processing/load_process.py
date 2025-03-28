import os
import pandas as pd

# Define the input and output directories
input_dir = r"c:\mgr\EPF-Thesis\data\load"
output_dir = r"c:\mgr\EPF-Thesis\data\processed_data"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the input directory
# Initialize an empty DataFrame to store all processed data
all_data = pd.DataFrame()

for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename.startswith("LOAD"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")

        # Read the CSV file
        df = pd.read_csv(input_path, sep=';', decimal=',')
        
        # Standardize the Date format and adjust Hour
        df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce') - 1  # Convert Hour to numeric and adjust to 0-based hour
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')  # Convert Date to datetime format
        
        # Combine Date and Hour into a single datetime column
        df['Datetime'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
        df = df.rename(columns={'Actual Total Load': 'Load'})
        df = df[['Datetime', 'Load']]

        # Append the processed data to the common DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)
    elif filename.endswith(".csv"):
        # Handle files with 15-minute interval data
        input_path = os.path.join(input_dir, filename)
        df = pd.read_csv(input_path, sep=';', decimal=',')
        
        # Parse the date and time range
        df['Date'] = pd.to_datetime(df['Doba handlowa'], format='%Y-%m-%d')
        df['Start_Time'] = pd.to_timedelta(df['OREB [Jednostka czasu od-do]'].str.split(' - ').str[0] + ':00')
        df['Datetime'] = df['Date'] + df['Start_Time']
        
        # Rename and select relevant columns
        df = df.rename(columns={'Rzeczywiste zapotrzebowanie KSE [MW]': 'Load'})
        df = df[['Datetime', 'Load']]
        
        # Resample to hourly data by averaging 15-minute intervals
        df['Load'] = pd.to_numeric(df['Load'], errors='coerce')  # Ensure Load is numeric
        df = df.set_index('Datetime').resample('h').mean().reset_index()  # Resample to hourly data
        
        # Append the processed data to the common DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)

# Remove rows where Load is '-'
all_data = all_data[all_data['Load'] != '-']

# Save the combined data to a single CSV file
common_csv_path = os.path.join(output_dir, "load.csv")
all_data.to_csv(common_csv_path, index=False, sep=';', decimal='.')
print(f"All data combined and saved to: {common_csv_path}")

# Check for missing hours in the combined data
start_time = pd.Timestamp('2016-01-01 00:00:00')
end_time = pd.Timestamp('2024-12-31 23:00:00')
expected_hours = pd.date_range(start=start_time, end=end_time, freq='h')

# Find missing hours
all_data.set_index('Datetime', inplace=True)
missing_hours = expected_hours.difference(all_data.index)

# Rename 'Datetime' column to 'date'
all_data.reset_index(inplace=True)
all_data.rename(columns={'Datetime': 'date'}, inplace=True)
all_data.set_index('date', inplace=True)

# Print missing hours if any
if not missing_hours.empty:
    print("Missing hours:")
    print(missing_hours)
else:
    print("No missing hours found.")