# ================================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================================================================
# 2. LOAD THE DATASET
# ================================================================================
# Make sure the file name matches your file path perfectly
data = pd.read_csv("kc_house_data - kc_house_data.csv")

# ================================================================================
# 3. BASIC EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================================
print("======= DATA OVERVIEW =======")
print("First 5 Rows:")
print(data.head())

print("\nData Shape (Rows, Columns):", data.shape)

print("\nData Information:")
print(data.info())

print("\nMissing Values Count:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

# --- EDA Visualizations ---
# Chart 1: Target Variable (Price) Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['price'], bins=50, kde=True, color='blue')
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.ticklabel_format(style='plain', axis='x') # Keeps numbers fully expanded, not scientific notation
plt.show()

# Chart 2: Feature vs Price Relationship (Sqft Living vs Price)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['sqft_living'], y=data['price'], alpha=0.5, color='darkgreen')
plt.xlabel("Sqft Living (Square Feet)")
plt.ylabel("Price")
plt.title("Price vs Sqft Living")
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# =================================================================================
# 4. DATA PREPROCESSING
# =================================================================================
# Dropping non-numeric or irrelevant columns
data = data.drop(['id', 'date'], axis=1, errors='ignore')
data = data.select_dtypes(include=[np.number])
data = data.dropna()

# Splitting Features (X) and Target (y)
X = data.drop('price', axis=1)
y = data['price']

# Train - Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================================================================================
# 5. MODEL BUILDING & EVALUATION
# ==================================================================================

# Dictionary to smoothly loop through all models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# Empty list to store metrics for the final comparison chart
results = []

print("======= MODEL TRAINING & EVALUATION =======")
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Append results to the list
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })
    
    print(f"\n[{name}]")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")

# Convert results into a DataFrame for easy plotting
df_results = pd.DataFrame(results)

# =================================================================================
# 6. VISUALIZATION: MODEL COMPARISON
# =================================================================================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="R2 Score", data=df_results, palette='viridis',hue='Model',legend=False)
plt.ylim(0, 1.0)
plt.ylabel('R² Score (Higher is Better)')
plt.title('Model Comparison: Which Model Predicts Best?')

# Display the exact R2 values on top of the bars
for index, row in df_results.iterrows():
    plt.text(index, row['R2 Score'] + 0.02, f"{row['R2 Score']:.4f}", ha='center', fontweight='bold')

plt.show()

# =================================================================================
# 7. AUTOMATED CONCLUSION
# =================================================================================
best_model = df_results.loc[df_results['R2 Score'].idxmax()]['Model']
print(f"\n======= CONCLUSION =======")
print(f"Based on the R² Score and Error metrics, the BEST model for this dataset is: {best_model}")