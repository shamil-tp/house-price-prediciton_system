import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('house_price_prediction_dataset.csv')

# 2. Define Features (X) and Target (y)
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'AgeYears']]
y = df['Price']

# 3. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on the test set
y_pred = model.predict(X_test)

print("--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Display Learned Coefficients (Weights)
print("\n--- Model Coefficients ---")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:.2f}")
print(f"Intercept: ${model.intercept_:.2f}")

# 6. Predict price for a new custom house
# Sample: 2000 sqft, 3 beds, 2 baths, 10 years old
sample_house = pd.DataFrame([{
    'SquareFeet': 2000,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'AgeYears': 10
}])

predicted_price = model.predict(sample_house)[0]
print(f"\nPredicted Price for Sample House: ${predicted_price:,.2f}")