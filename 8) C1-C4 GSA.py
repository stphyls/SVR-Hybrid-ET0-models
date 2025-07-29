import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Downloads/Python files/Labuan_C4.csv')
X = data[['Year', 'Month', 'Day', 'MaxTemp', 'MinTemp', 'MeanTemp']]
y = data['ET']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM model and evaluation metrics
def svm_model(solution):
    C, gamma = solution  # Unpack solution vector
    model = SVR(C=C, gamma=np.clip(gamma, 0.0001, 10.0), kernel='rbf')  # Clip gamma value within valid range
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # D1 (Willmott's index) and KGE
    mean_observed = y_test.mean()
    d1 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((np.abs(y_pred - mean_observed) + np.abs(y_test - mean_observed)) ** 2))
    r = np.corrcoef(y_test, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_test)
    beta = np.mean(y_pred) / np.mean(y_test)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    return mse, mae, rmse, d1, kge

# Define GSA algorithm
def gsa_optimize(func, bounds, num_agents=10, max_iterations=50):
    dimensions = len(bounds)
    agents = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_agents, dimensions))
    velocities = np.zeros((num_agents, dimensions))
    best_solution = None
    best_fitness = float('inf')
    best_metrics = None  # To store all metrics of the best solution
    
    for iteration in range(max_iterations):
        # Evaluate fitness for each agent
        fitness_and_metrics = [func(agent) for agent in agents]
        fitness = np.array([metrics[0] for metrics in fitness_and_metrics])  # Use MSE as fitness

        # Update best solution
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_solution = agents[min_fitness_idx].copy()
            best_metrics = fitness_and_metrics[min_fitness_idx]  # Capture all metrics of best agent

        # Calculate gravitational constant (G), masses, and forces
        G = 100 * (1 - iteration / max_iterations)  # Decreases over iterations
        masses = fitness.max() - fitness + 1e-10  # Inverse to make lower fitness agents heavier
        masses /= masses.sum()  # Normalize masses
        
        # Update velocities and positions
        for i in range(num_agents):
            force = np.zeros(dimensions)
            for j in range(num_agents):
                if i != j:
                    distance = np.linalg.norm(agents[i] - agents[j]) + 1e-10
                    direction = (agents[j] - agents[i]) / distance
                    force += np.random.rand() * G * masses[j] * direction / distance

            # Update velocity and position
            velocities[i] = np.random.rand() * velocities[i] + force
            agents[i] = np.clip(agents[i] + velocities[i], bounds[:, 0], bounds[:, 1])

        print(f"Iteration {iteration + 1}/{max_iterations} - Best MSE: {best_fitness}")

    return best_solution, best_fitness, best_metrics

# Define the parameter bounds
bounds = np.array([[0.001, 100], [0.0001, 10]])

# Run GSA optimization
best_solution, best_mse, best_metrics = gsa_optimize(svm_model, bounds)
best_mae, best_rmse, best_d1, best_kge = best_metrics[1], best_metrics[2], best_metrics[3], best_metrics[4]

# Train the SVM model with the best parameters
C, gamma = best_solution
model = SVR(C=C, gamma=np.clip(gamma, 0.0001, 10.0), kernel='rbf')
model.fit(X_train, y_train)

# Evaluate the SVM model
y_pred = model.predict(X_test)

# Calculate Standard Deviations
std_obs = np.std(y_test)
std_sim = np.std(y_pred)

# Calculate Correlation (Pearson's r)
correlation = np.corrcoef(y_test, y_pred)[0, 1]

mse = np.mean((y_test - y_pred) ** 2)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# General Performance Index (GPI)
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