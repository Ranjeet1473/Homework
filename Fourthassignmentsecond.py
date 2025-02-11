import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('weight-height.csv')

# Display first few rows of the data
print(df.head())

# Scatter plot of Height vs Weight
plt.scatter(df['Height'], df['Weight'], alpha=0.5)
plt.title('Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

# Prepare data for Random Forest Regression
X = df[['Height']]  # Independent variable (Height)
y = df['Weight']    # Dependent variable (Weight)

# Step 1: Create and fit a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 2: Make predictions using the Random Forest model
y_pred = model.predict(X)

# Step 3: Plot the results
plt.scatter(df['Height'], df['Weight'], alpha=0.5, label='Actual Data')
plt.scatter(df['Height'], y_pred, color='red', alpha=0.5, label='Predictions')
plt.title('Height vs Weight with Random Forest Predictions')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

# Step 4: Calculate RMSE and R2 score
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Print the results
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
