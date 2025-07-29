import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from cma import CMAEvolutionStrategy
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Downloads/Python files/PulauLangkawi_C4.csv')
X = data[['Year', 'Month', 'Day', 'MaxTemp', 'MinTemp', 'MeanTemp']]
y = data['ET']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM model
def svm_model(solution):
    C, gamma = solution  # Unpack the solution vector
    model = SVR(C=C, gamma=max(0.0001, min(10.0, gamma)), kernel='rbf')  # Clip gamma within valid range
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate D1 and KGE
    mean_observed = y_test.mean()
    d1 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((np.abs(y_pred - mean_observed) + np.abs(y_test - mean_observed)) ** 2))
    
    # Kling-Gupta Efficiency (KGE)
    r = np.corrcoef(y_test, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_test)
    beta = np.mean(y_pred) / np.mean(y_test)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    return mse, mae, rmse, d1, kge

# Define the objective function for CMA-ES
def cma_objective(solution):
    solution = np.clip(solution, bounds[:, 0], bounds[:, 1])
    return svm_model(solution)[0]  # Optimizing for MSE

# Define the parameter bounds
bounds = np.array([[0.001, 100], [0.0001, 10]])

# Run CMA-ES optimization
es = CMAEvolutionStrategy(x0=[50, 0.001], sigma0=0.5)
es.optimize(cma_objective)

# Get the best solution and score
best_solution = es.best.x
best_mse, best_mae, best_rmse, best_d1, best_kge = svm_model(best_solution)

# Train the SVM model with the best parameters
C, gamma = best_solution
model = SVR(C=C, gamma=max(0.0001, min(10.0, gamma)), kernel='rbf')
model.fit(X_train, y_train)

# Evaluate the SVM model
y_pred = model.predict(X_test)

# Calculate Standard Deviations
std_obs = np.std(y_test)
std_sim = np.std(y_pred)

# Calculate Correlation (Pearson's r)
correlation = np.corrcoef(y_test, y_pred)[0, 1]
mse = np.mean((y_test - y_pred) ** 2)  # Direct calculation for MSE
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate the General Performance Index (GPI)
gpi = np.sqrt((mse / best_mse) * (mae / best_mae) * (best_rmse / rmse) * (r2 / best_kge))

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label='Predicted vs. Observed')
plt.xlabel('Observed ET0')
plt.ylabel('Simulated ET0')
plt.title('Scatter Plot: Observed vs. Simulated ET0')

# Trendline
trendline_x = np.linspace(min(y_test), max(y_test), 100)
trendline_y = trendline_x
plt.plot(trendline_x, trendline_y, color='red', label='Trendline')
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()

# Print metrics
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("Willmott's Index of Agreement (D1):", best_d1)
print("Kling-Gupta Efficiency (KGE):", best_kge)
print("General Performance Index (GPI):", gpi)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)
print("Standard Deviation (Observed):", std_obs)
print("Correlation Coefficient (r):", correlation)
print("Standard Deviation (Simulated):", std_sim)