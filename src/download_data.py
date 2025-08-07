import kagglehub
import pandas as pd

# Define dataset details
dataset_handle = "mrigaankjaswal/crop-yield-prediction-dataset"
file_name = "yield_df.csv"
output_path = "data/yield_df.csv"

# Load the dataset
print("Downloading dataset...")
df = kagglehub.load_dataset(
    kagglehub.KaggleDatasetAdapter.PANDAS,
    dataset_handle,
    file_name,
)

# Save the dataset to a CSV file
print(f"Saving dataset to {output_path}...")
df.to_csv(output_path, index=False)

print("Dataset downloaded and saved successfully.")
