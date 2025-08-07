import pandas as pd

# Load the dataset
file_path = "data/yield_df.csv"
df = pd.read_csv(file_path)

# Get unique values for 'Area' and 'Item'
areas = sorted(df['Area'].unique().tolist())
items = sorted(df['Item'].unique().tolist())

print("Areas:")
print(areas)

print("\nItems:")
print(items)
