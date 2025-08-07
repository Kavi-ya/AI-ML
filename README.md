# Crop Yield Prediction

This project is an end-to-end AI/ML solution for predicting crop yield. It includes a machine learning model trained on a real-world dataset and a web application to provide a user-friendly interface for predictions.

## Features

- **Machine Learning Model:** A Linear Regression model trained on the "Crop Yield Prediction Dataset" from Kaggle.
- **Web Application:** A Flask-based web app that allows users to input data and get a crop yield prediction.
- **Jupyter Notebook:** A detailed notebook (`Crop_Yield_Prediction.ipynb`) that walks through the entire data science workflow, from data loading and EDA to model training and evaluation.

## Project Structure

```
.
├── Crop_Yield_Prediction.ipynb
├── README.md
├── data
│   └── yield_df.csv
├── requirements.txt
├── src
│   ├── app.py
│   ├── download_data.py
│   ├── model.pkl
│   └── model_columns.json
├── static
│   └── style.css
└── templates
    └── index.html
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kavi-ya/AI-ML.git
    cd AI-ML
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Web Application

1.  **Start the Flask server:**
    ```bash
    python src/app.py
    ```

2.  **Access the application:**
    Open your web browser and go to `http://localhost:5000`.

3.  **Use the application:**
    - Fill in the form with the required data.
    - Click the "Predict Yield" button to get the prediction.

## How to Use the Jupyter Notebook

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:**
    In the Jupyter interface, open the `Crop_Yield_Prediction.ipynb` file.

3.  **Run the cells:**
    You can run the cells in the notebook sequentially to see the entire machine learning workflow.
