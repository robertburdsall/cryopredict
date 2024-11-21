import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('datasets/ice_data.csv')

# Train a Linear Regression model on historical data (up to 2020)
X = data['Year'].values.reshape(-1, 1)  # Feature: Year
y = data['Ice Concentration'].values  # Target: Ice Concentration

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for future years (1995 to 2025)
future_years = np.arange(1995, 2026).reshape(-1, 1)
predicted_ice_concentration = model.predict(future_years)

# Plot observed vs. predicted ice concentration
plt.figure(figsize=(12, 8))

# Plot observed ice concentration
plt.plot(data['Year'], data['Ice Concentration'], label='Observed Ice Concentration', color='blue', linewidth=2)

# Plot predicted ice concentration (from 1995 to 2025)
plt.plot(future_years, predicted_ice_concentration, label='Predicted Ice Concentration', color='red', linestyle='--', linewidth=2)

# Add labels and title
plt.title("Observed and Predicted Ice Concentration for Iceland")
plt.xlabel("Year")
plt.ylabel("Ice Concentration (%)")
plt.legend()
plt.grid(True)
plt.show()

# Correlation Heatmap for emissions data (CO2, CH4, N2O) and Ice Concentration
# Assuming emissions_data.csv contains the appropriate columns: CO2, CH4, N2O
emissions_data = pd.read_csv('datasets/emissions_data.csv')

# Combine emissions and ice data into one dataframe for correlation analysis
combined_data = emissions_data[['Year', 'CO2', 'CH4', 'N2O']].merge(data[['Year', 'Ice Concentration']], on='Year')

# Calculate correlation matrix
correlation_matrix = combined_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Greenhouse Gases and Ice Concentration")
plt.show()
