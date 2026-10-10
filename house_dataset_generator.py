import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 30 rows of realistic dataset
n_samples = 100

# Square Feet: 800 to 3500 sqft
sqft = np.random.randint(800, 3500, size=n_samples)

# Bedrooms: 1 to 5, correlated with sqft
bedrooms = np.clip((sqft / 600).astype(int) + np.random.randint(-1, 2, size=n_samples), 1, 5)

# Bathrooms: 1 to 4, correlated with bedrooms
bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.choice([-0.5, 0, 0.5], size=n_samples), 1), 1.0, 4.0)

# Age in Years: 1 to 50 years
age_years = np.random.randint(1, 51, size=n_samples)

# Price logic (Base price + sqft factor + bed/bath factor - age penalty + noise)
price = (
    100000 
    + (sqft * 180) 
    + (bedrooms * 15000) 
    + (bathrooms * 20000) 
    - (age_years * 1200) 
    + np.random.randint(-15000, 15000, size=n_samples)
)

# Create DataFrame
df = pd.DataFrame({
    'SquareFeet': sqft,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'AgeYears': age_years,
    'Price': price
})

# Sort by SquareFeet for better readability
df = df.sort_values(by='SquareFeet').reset_index(drop=True)

# Save to CSV
df.to_csv('house_price_prediction_dataset.csv', index=False)