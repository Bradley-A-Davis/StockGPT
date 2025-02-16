import pandas as pd

# Input and output file names
input_file = "Pre Data/AAPL_1min_sample.csv"  # Change this to your actual file name
output_file = "Post Data\AAPL2Week.txt"

# Read the file into a DataFrame
df = pd.read_csv(input_file, names=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"], skiprows=1)

# Convert necessary columns to numeric values
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df = df.dropna()

# Calculate percentage change in close price
df["percent_change"] = df["close"].pct_change() * 100

# Drop the first row (NaN value from pct_change)
df = df.dropna()

# Round percent_change to the nearest ten-thousandth
df["percent_change"] = df["percent_change"].round(4)

# Save only the percentage changes to a new file
df[["percent_change"]].to_csv(output_file, index=False, header=False)

print(f"Percentage changes saved to {output_file}")
