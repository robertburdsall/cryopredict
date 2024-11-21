import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
# Replace 'data.csv' with your actual datasets for temperature and emissions
temperature_data = pd.read_csv('datasets/temperature_data.csv')  # Columns: 'Year', 'Temperature'
emissions_data = pd.read_csv('datasets/emissions_data.csv')  # Columns: 'Year', 'CO2', 'CH4', 'N2O', ... for iceland
ice_data = pd.read_csv('datasets/ice_data.csv')  # Columns: 'Year', 'Region', 'Ice Concentration' for iceland

# Merge data on 'Year'
data = pd.merge(temperature_data, emissions_data, on='Year')
data = pd.merge(data, ice_data, on=['Year', 'Region'])

# 2. Determine Greenhouse Gas Impact
X = data[['CO2', 'CH4', 'N2O']]  # Greenhouse gases as features
y = data['Temperature']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=0.1),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

# Feature importance from Random Forest
feature_importance = models['Random Forest'].feature_importances_

# Visualize feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x=X.columns, y=feature_importance, palette="viridis")
plt.title("Feature Importance of Greenhouse Gases on Temperature")
plt.ylabel("Importance")
plt.xlabel("Greenhouse Gas")
plt.show()

# 3. Use Dominant Gas to Project Ice Melting
dominant_gas = X.columns[np.argmax(feature_importance)]  # Most impactful gas

# Filter data for the dominant gas and ice concentration
ice_X = data[['Year', dominant_gas]]
ice_y = data['Ice Concentration']

# Train-test split for ice melting projection
ice_X_train, ice_X_test, ice_y_train, ice_y_test = train_test_split(
    ice_X, ice_y, test_size=0.2, random_state=42)

# Train a Random Forest model for ice melting prediction
ice_model = RandomForestRegressor(n_estimators=100, random_state=42)
ice_model.fit(ice_X_train, ice_y_train)

# Predict future ice concentration
future_years = np.arange(data['Year'].max() + 1, data['Year'].max() + 11)
future_emissions = np.linspace(ice_X[dominant_gas].min(), ice_X[dominant_gas].max(), len(future_years))
future_data = pd.DataFrame({'Year': future_years, dominant_gas: future_emissions})
future_ice_predictions = ice_model.predict(future_data)

# Visualize ice melting projections
plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Ice Concentration'], label='Observed Ice Concentration', color='blue')
plt.plot(future_years, future_ice_predictions - 8, label='Projected Ice Concentration', color='red', linestyle='--')
plt.title(f"Ice Melting Projection (Using {dominant_gas})")
plt.xlabel("Year")
plt.ylabel("Ice Concentration")
plt.legend()
plt.show()

# 4. Compare Models
# Visualize R2 scores
r2_scores = [r2 for _, r2 in results.values()]
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=r2_scores, palette="coolwarm")
plt.title("Model Comparison (R2 Scores)")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))

# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Greenhouse Gases, Temperature, and Ice Concentration")
plt.show()


# Residuals Plot for Random Forest (Temperature Prediction)
y_pred_rf = models['Random Forest'].predict(X_test)
residuals = y_test - y_pred_rf

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_rf, y=residuals, color='purple')
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Residuals Plot: Random Forest Model")
plt.xlabel("Predicted Temperature")
plt.ylabel("Residuals (Actual - Predicted)")
plt.show()

# Combined Observed and Projected Ice Concentration
plt.figure(figsize=(12, 8))
plt.plot(data['Year'], data['Ice Concentration'], label='Observed Ice Concentration', color='blue', linewidth=2)
plt.plot(future_years, future_ice_predictions, label='Projected Ice Concentration', color='red', linestyle='--', linewidth=2)
plt.title(f"Observed and Projected Ice Concentration for Iceland (Using {dominant_gas})")
plt.xlabel("Year")
plt.ylabel("Ice Concentration (%)")
plt.legend()
plt.grid(True)
plt.show()

# Table of Model Metrics
import matplotlib.table as tbl

metrics_table = pd.DataFrame(
    {model: {'MSE': mse, 'R2': r2} for model, (mse, r2) in results.items()}
).T

# Display as a table in the console
print("\nModel Performance Metrics:")
print(metrics_table)

# Graphical Table
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=metrics_table.values,
    colLabels=metrics_table.columns,
    rowLabels=metrics_table.index,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(metrics_table.columns))))
plt.title("Model Performance Metrics")
plt.show()

# Scatterplot of Dominant Gas vs Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[dominant_gas], y=data['Temperature'], color='green')
plt.title(f"Impact of {dominant_gas} on Temperature")
plt.xlabel(f"{dominant_gas} Emissions (Metric Tons)")
plt.ylabel("Average Temperature (°C)")
plt.grid(True)
plt.show()

# Distribution of Ice Concentration
plt.figure(figsize=(10, 6))
sns.histplot(data['Ice Concentration'], kde=True, color='blue', bins=15)
plt.title("Distribution of Ice Concentration")
plt.xlabel("Ice Concentration (%)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
