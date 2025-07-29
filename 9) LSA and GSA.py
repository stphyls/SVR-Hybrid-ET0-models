import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import fast_sampler
from SALib.analyze import fast

# Load data
data = pd.read_csv('Downloads/Python files/Kuching_C4.csv')
X = data[['MaxTemp', 'MinTemp', 'MeanTemp', 'Humidity', 'Radiation','WS']]
y = data['ET']  

# Define the ETo calculation function first
def calculate_eto(params):
    # Unpack parameters
    max_temp, min_temp, mean_temp, rel_humidity, solar_rad, wind_speed = params
    
    # Constants
    albedo = 0.23
    Gsc = 0.082  # MJ m-2 min-1
    sigma = 4.903e-9  # MJ K-4 m-2 day-1
    G = 0  # Assume soil heat flux is negligible for daily calculations
    
    # Calculations
    es = 0.6108 * np.exp(17.27 * mean_temp / (mean_temp + 237.3))
    ea = es * rel_humidity / 100
    delta = 4098 * es / (mean_temp + 237.3)**2
    gamma = 0.665e-3 * 101.3  # Assumed atmospheric pressure is 101.3 kPa
    Rn = (1 - albedo) * solar_rad - sigma * ((max_temp + 273.16)**4 + (min_temp + 273.16)**4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * solar_rad / (Gsc * 24 * 60) - 0.35)
    
    # FAO-56 Penman-Monteith Equation
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (mean_temp + 273)) * wind_speed * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_speed)
    eto = numerator / denominator
    
    return max(0, eto)  # ETo should not be negative


# One-at-a-Time (OAT) Sensitivity Analysis
def oat_sensitivity_analysis(X, y, feature_names):
    results = {}
    sensitivities = {}
    
    # Define parameter-specific ranges
    parameter_ranges = {
        'MaxTemp': [-20, -10, 0, 10, 20],    
        'MinTemp': [-20, -10, 0, 10, 20],    
        'MeanTemp': [-20, -10, 0, 10, 20],   
        'Humidity': [-15, -7.5, 0, 7.5, 15], 
        'Radiation': [-20, -10, 0, 10, 20],  
        'WS': [-30, -15, 0, 15, 30]          
    }
    
    for feature in feature_names:
        baseline_value = X[feature].mean()
        sensitivity = []
        changes = []
        percentages = parameter_ranges[feature]
        
        # Calculate baseline ETo
        baseline_y = y.mean()
        
        # Calculate actual values for each percentage change
        for pct in percentages:
            change = baseline_value * (pct/100)
            new_value = baseline_value + change
            changes.append(change)
            
            # Create modified dataset
            X_modified = X.copy()
            X_modified[feature] = new_value
            
            # Calculate modified ETo using calculate_eto function
            # We need to calculate new ETo for each row in X_modified
            modified_etos = []
            for _, row in X_modified.iterrows():
                modified_eto = calculate_eto(row.values)
                modified_etos.append(modified_eto)
            
            # Calculate sensitivity as mean absolute difference from baseline
            delta_y = np.mean(np.abs(np.array(modified_etos) - baseline_y))
            sensitivity.append(delta_y)
        
        # Calculate average sensitivity for this feature
        avg_sensitivity = np.mean(sensitivity)
        sensitivities[feature] = avg_sensitivity
        
        results[feature] = {
            'changes': changes,
            'percentages': percentages,
            'sensitivity': sensitivity,
            'average_sensitivity': avg_sensitivity,
            'baseline': baseline_value
        }
    
    return results, sensitivities

# Perform OAT sensitivity analysis
print("\n=== LOCAL SENSITIVITY ANALYSIS (One-at-a-Time Method) ===")
print("-------------------------------------------------------")
oat_results, oat_sensitivities = oat_sensitivity_analysis(X, y, X.columns)

# Sort and print OAT sensitivities
sorted_oat = sorted(oat_sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nOAT Sensitivity Rankings:")
print("-------------------------")
for feature, sensitivity in sorted_oat:
    print(f"{feature:20} : {sensitivity:.6f}")

# Visualize OAT results
# Print detailed OAT results
print("\nDetailed OAT Results:")
print("-------------------")
for feature, result in oat_results.items():
    print(f"\n{feature}:")
    print(f"Baseline value: {result['baseline']:.2f}")
    print("Percentage Change | Absolute Change | Sensitivity")
    print("-" * 50)
    for pct, change, sens in zip(result['percentages'], result['changes'], result['sensitivity']):
        print(f"{pct:>15}% | {change:>14.2f} | {sens:>10.4f}")

# Visualize OAT results
plt.figure(figsize=(15, 10))
for i, (feature, result) in enumerate(oat_results.items(), 1):
    plt.subplot(3, 2, i)
    plt.scatter(result['percentages'], result['sensitivity'], alpha=0.6)
    plt.plot(result['percentages'], result['sensitivity'], '--', alpha=0.3)
    plt.title(f'OAT Sensitivity to {feature}')
    plt.xlabel('Percentage Change (%)')
    plt.ylabel('Mean Absolute Change in ETo')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at 0% change
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels
    for x, y in zip(result['percentages'], result['sensitivity']):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()

# GLOBAL SENSITIVITY ANALYSIS (eFAST)
print("\n=== GLOBAL SENSITIVITY ANALYSIS (eFAST Method) ===")
print("-----------------------------------------------")

# Define the problem for eFAST
problem = {
    'num_vars': len(X.columns),
    'names': X.columns.tolist(),
    'bounds': [[X[col].min(), X[col].max()] for col in X.columns]
}

# Generate samples
param_values = fast_sampler.sample(problem, 1000)

# Run model for all samples
Y = np.array([calculate_eto(sample) for sample in param_values])

# Perform analysis
Si = fast.analyze(problem, Y, print_to_console=True)

# Process eFAST results for plotting
sorted_Si = sorted(zip(Si['names'], Si['S1']), key=lambda x: x[1], reverse=True)
sorted_ST = sorted(zip(Si['names'], Si['ST']), key=lambda x: x[1], reverse=True)


# Visualize eFAST results with scatter plots
plt.figure(figsize=(12, 5))

# First-Order Indices
plt.subplot(1, 2, 1)
names, S1_values = zip(*sorted_Si)
plt.scatter(range(len(names)), S1_values, s=100)
plt.plot(range(len(names)), S1_values, '--', alpha=0.3)  # Adding a light connecting line
plt.xticks(range(len(names)), names, rotation=45)
plt.title('First-Order Sensitivity Indices')
plt.ylabel('Sensitivity Index')
plt.grid(True, alpha=0.3)

# Total-Order Indices
plt.subplot(1, 2, 2)
names, ST_values = zip(*sorted_ST)
plt.scatter(range(len(names)), ST_values, s=100, c='orange')
plt.plot(range(len(names)), ST_values, '--', alpha=0.3)  # Adding a light connecting line
plt.xticks(range(len(names)), names, rotation=45)
plt.title('Total-Order Sensitivity Indices')
plt.ylabel('Sensitivity Index')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Add an additional comparison scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(names)), S1_values, label='First-Order (Si)', s=100)
plt.scatter(range(len(names)), ST_values, label='Total-Order (ST)', s=100)
plt.plot(range(len(names)), S1_values, '--', alpha=0.3)
plt.plot(range(len(names)), ST_values, '--', alpha=0.3)
plt.xticks(range(len(names)), names, rotation=45)
plt.title('Comparison of First-Order and Total-Order Sensitivity Indices')
plt.ylabel('Sensitivity Index')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print eFAST results
print("\neFAST First-Order Indices:")
for name, S1 in zip(Si['names'], Si['S1']):
    print(f"{name}: {S1:.4f}")

print("\neFAST Total-Order Indices:")
for name, ST in zip(Si['names'], Si['ST']):
    print(f"{name}: {ST:.4f}")

# Calculate correlations (for comparison)
y_series = pd.Series(y, index=X.index)
correlations = X.corrwith(y_series)
print("\nCorrelations with ETo:")
print(correlations)

# Basic statistics
print("\nBasic Statistics:")
print(X.describe())