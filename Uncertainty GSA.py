import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import FeatureUnion, make_pipeline
import matplotlib.pyplot as plt
import time

# Configuration identifier
config = 'C4'
print(f"\n===== RUNNING UNCERTAINTY ANALYSIS FOR {config} CONFIGURATION WITH SVR-GSA =====")
start_time = time.time()

# Define feature sets for each configuration (only C1 for now)
feature_sets = {
    'C1': ['MaxTemp', 'MinTemp', 'MeanTemp'],
    'C2': ['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation'],
    'C3': ['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation', 'WS'],
    'C4': ['MaxTemp', 'MinTemp', 'MeanTemp', 'Radiation', 'WS', 'Humidity']
}
columns_to_impute = feature_sets[config] + ['ET']

# Step 1: Inspect raw data before any processing
print("\n1. RAW DATA INSPECTION")
try:
    raw_data = pd.read_csv('Downloads/Python files/Labuan_C4.csv')
    raw_data = raw_data.drop(columns=['Unnamed: 10', 'Unnamed: 11'])
    print("Raw data shape:", raw_data.shape)
    print("Raw data columns:", raw_data.columns.tolist())
    print("Raw data types:\n", raw_data.dtypes)
    print("Raw data missing values:\n", raw_data.isna().sum())
    
    print("\nET value statistics:")
    print("ET min:", raw_data['ET'].min(), "ET max:", raw_data['ET'].max())
    print("ET mean:", raw_data['ET'].mean(), "ET median:", raw_data['ET'].median())
    print("ET quartiles:", raw_data['ET'].quantile([0.25, 0.50, 0.75]).tolist())
    
    print("\nFirst 5 rows:")
    print(raw_data.head())
    print("\nLast 5 rows:")
    print(raw_data.tail())
except Exception as e:
    print(f"Error loading raw data: {e}")

# Step 2: Verify filtering logic
print("\n2. FILTERING VERIFICATION")
filtered_data = raw_data[raw_data['Year'] >= 2000]
print("Filtered data shape:", filtered_data.shape)
print("Filtered data missing values:\n", filtered_data.isna().sum())

# Step 3: Create datetime column and handle duplicates
print("\n3. DATETIME CONVERSION")
try:
    filtered_data['Date'] = pd.to_datetime(filtered_data[['Year', 'Month', 'Day']])
    data = filtered_data.groupby('Date').mean().reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index()
    
    print("Date range:", data.index.min(), "to", data.index.max())
    print("Total days in dataset:", len(data))
    print("Unique days in dataset:", len(data.index.unique()))
    
    dupes = data.index.duplicated()
    if any(dupes):
        print(f"WARNING: Found {sum(dupes)} duplicate dates in the index after aggregation")
except Exception as e:
    print(f"Error during datetime conversion: {e}")
    data = filtered_data.copy()

# Step 4: Examine missing value handling
print("\n4. MISSING VALUE HANDLING")
print("Before interpolation - missing values by column:")
print(data[columns_to_impute].isna().sum())

print("\nApplying interpolation...")
data_imputed = data.copy()
try:
    data_imputed[columns_to_impute] = data_imputed[columns_to_impute].interpolate(method='linear')
    data_imputed[columns_to_impute] = data_imputed[columns_to_impute].ffill().bfill()
    print("After interpolation - missing values by column:")
    print(data_imputed[columns_to_impute].isna().sum())
except Exception as e:
    print(f"Error during interpolation: {e}")
    data_imputed = data.copy()

data = data_imputed

# Step 5: Resampling to monthly data
print("\n5. MONTHLY RESAMPLING")
try:
    data_monthly = data.resample('ME').mean()
    print("Data shape after resampling:", data_monthly.shape)
    print("Missing values after resampling:\n", data_monthly.isna().sum())
    
    non_nan_months = data_monthly['ET'].count()
    print(f"Non-NaN months: {non_nan_months}")
    
    print("\nMonthly ET statistics:")
    print("Min:", data_monthly['ET'].min(), "Max:", data_monthly['ET'].max())
    print("Mean:", data_monthly['ET'].mean(), "Median:", data_monthly['ET'].median())
    
    plt.figure(figsize=(12, 6))
    data_monthly['ET'].plot()
    plt.title(f'Monthly ET Values ({config} Configuration)')
    plt.ylabel('ET')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig(f'{config}_monthly_ET_timeseries.png')
    plt.close()
    print(f"Monthly ET time series plot saved to '{config}_monthly_ET_timeseries.png'")
except Exception as e:
    print(f"Error during monthly resampling: {e}")
    data_monthly = data.copy()

# Step 6: Feature Engineering with Correlation Analysis
print("\n6. FEATURE ENGINEERING")
try:
    if len(data_monthly) < 2:
        raise ValueError("Dataset has insufficient rows (< 2) to apply lag feature.")
    
    data_monthly['Month'] = data_monthly.index.month
    data_monthly['Month_sin'] = np.sin(2 * np.pi * data_monthly['Month'] / 12)
    data_monthly['Month_cos'] = np.cos(2 * np.pi * data_monthly['Month'] / 12)
    data_monthly['ET_lag1'] = data_monthly['ET'].shift(1)
    
    # Correlation analysis
    print("\nFeature Correlations with ET:")
    for col in feature_sets[config] + ['ET_lag1', 'Month_sin', 'Month_cos']:
        valid_indices = ~data_monthly[col].isna() & ~data_monthly['ET'].isna()
        if valid_indices.sum() > 2:
            corr = np.corrcoef(data_monthly.loc[valid_indices, col], 
                              data_monthly.loc[valid_indices, 'ET'])[0, 1]
            print(f"Correlation between {col} and ET: {corr:.4f}")
    
    data_monthly['ET_lag1'] = data_monthly['ET_lag1'].fillna(data_monthly['ET'].mean())
    
    # Define features based on configuration
    engineered_features = ['Month', 'ET_lag1', 'Month_sin', 'Month_cos']
    X = data_monthly[feature_sets[config] + engineered_features]
    y = data_monthly['ET']
    
    print("\nFeature set validation:")
    for col in X.columns:
        print(f"Feature {col}: {len(X[col])} values, {X[col].isna().sum()} missing")
except Exception as e:
    print(f"Error during feature engineering: {e}")
    X = data_monthly[feature_sets[config]]
    y = data_monthly['ET']

# Step 7: Feature Imputation and Transformation
print("\n7. FEATURE IMPUTATION AND TRANSFORMATION")
try:
    print("Missing values in X before imputation:")
    print(X.isna().sum())
    print("Missing values in y before imputation:", y.isna().sum())
    
    transformer = FeatureUnion(
        transformer_list=[
            ('features', SimpleImputer(strategy='mean')),
            ('indicators', MissingIndicator())
        ]
    )
    X_transformed = transformer.fit_transform(X)
    
    indicator_features = [f"missing_{col}" for col in X.columns if np.any(X[col].isna())]
    X_columns = list(X.columns) + indicator_features
    X = pd.DataFrame(X_transformed, columns=X_columns, index=X.index)
    
    y_imputer = SimpleImputer(strategy='mean')
    y = pd.Series(y_imputer.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)
    
    print("\nMissing values in X after imputation:", X.isna().sum().sum())
    print("Missing values in y after imputation:", y.isna().sum())
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
except Exception as e:
    print(f"Error during feature imputation: {e}")
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    X_scaled = X

# Step 8: Train/Test Split Validation
print("\n8. TRAIN/TEST SPLIT VALIDATION")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train ET stats - min:", y_train.min(), "max:", y_train.max(), "mean:", y_train.mean())
print("Test ET stats - min:", y_test.min(), "max:", y_test.max(), "mean:", y_test.mean())

# Step 9: Define SVR-GSA Model and Optimization
print(f"\n===== PROCEEDING TO MODEL TRAINING FOR {config} CONFIGURATION WITH SVR-GSA =====")

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
final_model = SVR(C=C, gamma=np.clip(gamma, 0.0001, 10.0), kernel='rbf')
final_model.fit(X_train, y_train)

# Step 10: Initial Model Validation
print("\n10. INITIAL MODEL VALIDATION")
initial_preds = final_model.predict(X_test)
initial_mse = mean_squared_error(y_test, initial_preds)
initial_r2 = r2_score(y_test, initial_preds)
initial_mae = mean_absolute_error(y_test, initial_preds)

print("Initial model performance with optimized SVR-GSA:")
print(f"MSE: {initial_mse:.4f}")
print(f"RMSE: {np.sqrt(initial_mse):.4f}")
print(f"MAE: {initial_mae:.4f}")
print(f"R²: {initial_r2:.4f}")

# Step 11: Bootstrap Uncertainty Analysis
print("\n11. BOOTSTRAP UNCERTAINTY ANALYSIS")
n_bootstraps = 1000
bootstrap_predictions = np.zeros((len(X_test), n_bootstraps))

print(f"Running bootstrap with {n_bootstraps} iterations...")
for i in range(n_bootstraps):
    if i % 10 == 0:
        print(f"Bootstrap iteration {i}/{n_bootstraps}")
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[indices].copy()
    y_boot = y_train.iloc[indices].copy()
    model = SVR(C=C, gamma=np.clip(gamma, 0.0001, 10.0), kernel='rbf')
    model.fit(X_boot, y_boot)
    bootstrap_predictions[:, i] = model.predict(X_test)

# Calculate confidence intervals
mean_predictions = np.mean(bootstrap_predictions, axis=1)
lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=1)
upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=1)

y_test_original = y_test.values

# Calculate statistical metrics
y_pred = mean_predictions
mse = mean_squared_error(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

mean_observed = np.mean(y_test_original)
d1 = 1 - (np.sum((y_test_original - y_pred) ** 2) / np.sum((np.abs(y_pred - mean_observed) + np.abs(y_test_original - mean_observed)) ** 2))

epsilon = 1e-10
std_pred = np.std(y_pred) + epsilon
std_obs = np.std(y_test_original) + epsilon
r = np.corrcoef(y_test_original, y_pred)[0, 1]
alpha = std_pred / std_obs
beta = np.mean(y_pred) / np.mean(y_test_original)
kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

best_mse = mse
best_mae = mae
best_rmse = rmse
best_kge = max(0.01, kge)
gpi = np.sqrt((mse / best_mse) * (mae / best_mae) * (best_rmse / rmse) * (max(0.01, r2) / best_kge))

print(f"\n12. FINAL MODEL PERFORMANCE ({config} CONFIGURATION WITH SVR-GSA)")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"D1 (Willmott's Index): {d1:.4f}")
print(f"KGE: {kge:.4f}")
print(f"GPI: {gpi:.4f}")

# Bootstrap metrics
bootstrap_metrics = {
    'MSE': [], 'MAE': [], 'RMSE': [], 'R2': [], 'D1': [], 'KGE': []
}

for i in range(n_bootstraps):
    y_pred_boot = bootstrap_predictions[:, i]
    bootstrap_metrics['MSE'].append(mean_squared_error(y_test_original, y_pred_boot))
    bootstrap_metrics['MAE'].append(mean_absolute_error(y_test_original, y_pred_boot))
    bootstrap_metrics['RMSE'].append(np.sqrt(bootstrap_metrics['MSE'][-1]))
    bootstrap_metrics['R2'].append(r2_score(y_test_original, y_pred_boot))
    
    d1_boot = 1 - (np.sum((y_test_original - y_pred_boot) ** 2) / 
                   np.sum((np.abs(y_pred_boot - mean_observed) + np.abs(y_test_original - mean_observed)) ** 2))
    bootstrap_metrics['D1'].append(d1_boot)
    
    std_pred_boot = np.std(y_pred_boot) + epsilon
    r_boot = np.corrcoef(y_test_original, y_pred_boot)[0, 1]
    alpha_boot = std_pred_boot / std_obs
    beta_boot = np.mean(y_pred_boot) / np.mean(y_test_original)
    kge_boot = 1 - np.sqrt((r_boot - 1) ** 2 + (alpha_boot - 1) ** 2 + (beta_boot - 1) ** 2)
    bootstrap_metrics['KGE'].append(kge_boot)

print(f"\n13. BOOTSTRAP METRIC UNCERTAINTY ({config} CONFIGURATION WITH SVR-GSA)")
for metric, values in bootstrap_metrics.items():
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)
    mean_val = np.mean(values)
    print(f"{metric}: Mean = {mean_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Step 14: Plotting
print("\n14. GENERATING FINAL VISUALIZATIONS")
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, 'r-', label='Actual ET')
plt.plot(mean_predictions, 'bo', label='Predicted ET with 95% CI')
plt.fill_between(range(len(mean_predictions)), lower_bound, upper_bound, color='blue', alpha=0.2)
plt.xlabel('Test Sample Index')
plt.ylabel('ET')
plt.legend()
plt.title(f'Predictions with 95% Confidence Intervals - {config} Configuration (Monthly Data) with SVR-GSA')
plt.grid(True)
plt.ylim(0, 8)
plt.savefig(f'{config}_final_predictions_with_CI_SVR-GSA.png')
plt.show()

avg_ci_width = np.mean(upper_bound - lower_bound)
rel_ci_width = avg_ci_width / np.mean(y_test_original) * 100

print(f"\n15. UNCERTAINTY SUMMARY ({config} CONFIGURATION WITH SVR-GSA)")
print(f"Average 95% CI Width: {avg_ci_width:.4f}")
print(f"Relative CI Width (%): {rel_ci_width:.2f}%")

# Export results
results_df = pd.DataFrame({
    'Actual_ET': y_test_original,
    'Predicted_ET': mean_predictions,
    'CI_Lower': lower_bound,
    'CI_Upper': upper_bound
})
results_df.to_csv(f'{config}_uncertainty_results_SVR-GSA.csv', index=False)
print(f"Results exported to '{config}_uncertainty_results_SVR-GSA.csv'")

metrics_df = pd.DataFrame({
    'Configuration': [config],
    'MSE': [mse],
    'MAE': [mae],
    'RMSE': [rmse],
    'R2': [r2],
    'D1': [d1],
    'KGE': [kge],
    'GPI': [gpi],
    'Avg_CI_Width': [avg_ci_width],
    'Rel_CI_Width': [rel_ci_width]
})
metrics_df.to_csv(f'{config}_performance_metrics_SVR-GSA.csv', index=False)
print(f"Performance metrics saved to '{config}_performance_metrics_SVR-GSA.csv'")

elapsed_time = time.time() - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")