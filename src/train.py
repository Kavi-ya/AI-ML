import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import json

# Load the dataset
file_path = "data/yield_df.csv"
df = pd.read_csv(file_path)

# Preprocessing
# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)

# Split data into features (X) and target (y)
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Save the column list
model_columns = X.columns.tolist()
with open('src/model_columns.json', 'w') as f:
    json.dump(model_columns, f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# Model Evaluation
print("Evaluating the model...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Save the model
model_path = "src/model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
