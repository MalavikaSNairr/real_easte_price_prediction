import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
df = pd.read_csv("data/bangalore.csv")

# Extract numeric BHK value from 'size' (e.g., '3 BHK' → 3, '4 Bedroom' → 4)
df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float)

# Remove rows where sqft is not a single number (e.g., "2100 - 2850")
df = df[df['total_sqft'].apply(lambda x: str(x).replace('.', '').isdigit())]
df['total_sqft'] = df['total_sqft'].astype(float)

# Drop rows with missing values in important columns
df.dropna(subset=['total_sqft', 'bath', 'price', 'bhk'], inplace=True)

# Convert price from lakhs to INR
df['price'] = df['price'] * 1e5

# Select features and target
X = df[['total_sqft', 'bath', 'bhk']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
os.makedirs('model', exist_ok=True)
with open("model/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Print model performance
print("Model R² score:", model.score(X_test, y_test))
