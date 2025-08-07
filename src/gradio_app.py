import gradio as gr
import pandas as pd
import pickle
import json

# Load the model and columns
with open('src/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('src/model_columns.json', 'r') as f:
    model_columns = json.load(f)

# Load raw data to get unique values for dropdowns
df_raw = pd.read_csv('data/yield_df.csv')
areas = sorted(df_raw['Area'].unique().tolist())
items = sorted(df_raw['Item'].unique().tolist())

def predict_yield(year, area, item, avg_rainfall, pesticides, avg_temp):
    # Create a dictionary from the inputs
    input_dict = {
        'Year': year,
        'Area': area,
        'Item': item,
        'average_rain_fall_mm_per_year': avg_rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': avg_temp
    }

    # Convert to a DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encode the categorical features
    # This needs to be done carefully to match the training columns
    input_df = pd.get_dummies(input_df)

    # Reindex to match the model's columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)

    return f"{prediction[0]:.2f} hg/ha"

# Define the Gradio interface
inputs = [
    gr.Number(label="Year", value=2022),
    gr.Dropdown(label="Area (Country)", choices=areas, value="India"),
    gr.Dropdown(label="Item (Crop)", choices=items, value="Maize"),
    gr.Number(label="Average Rainfall (mm/year)", value=1000),
    gr.Number(label="Pesticides (tonnes)", value=5000),
    gr.Number(label="Average Temperature (°C)", value=20)
]

output = gr.Textbox(label="Predicted Yield")

interface = gr.Interface(
    fn=predict_yield,
    inputs=inputs,
    outputs=output,
    title="Crop Yield Prediction",
    description="Predict the crop yield based on various agricultural and environmental factors."
)

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
