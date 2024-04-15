import warnings
import numpy as np

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_excel('Ipl_Batsman.xlsx')

# Convert categorical variables to numerical using LabelEncoder
label_encoder = LabelEncoder()
df['AGE'] = label_encoder.fit_transform(df['AGE'])
df['PLAYING ROLE'] = label_encoder.fit_transform(df['PLAYING ROLE'])

# Drop rows with missing values
df.dropna(inplace=True)

# Drop the 'Sl.NO.' column
df.drop('Sl.NO.', axis=1, inplace=True)

# One-hot encode 'COUNTRY' and 'TEAM' columns
df = pd.get_dummies(df, columns=['COUNTRY', 'TEAM'])

# Separate features and target variable
X = df.drop(['PLAYER NAME', 'SOLD PRICE'], axis=1)
y = df['SOLD PRICE']

# Calculate correlation analysis between features and target variable
correlations = X.corrwith(y)

# Plotting a decreasing bar chart for correlation
plt.figure(figsize=(10, 6))
sorted_correlation = correlations.abs().sort_values(ascending=False)
plt.bar(sorted_correlation.index, sorted_correlation)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Correlation with Target', fontsize=14)
plt.title('Correlation of Features with Target Variable')
plt.xticks(rotation=45, ha='right')
plt.show()

# Select the top 10 features correlated with the target variable
top_features = sorted_correlation[:10].index

# Generate all combinations of 6 features from the top 10
combinations_6_features = list(combinations(top_features, 6))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store R2 scores and feature combinations
train_accuracies = []
test_accuracies = []
combination_scores = []

# Train models using each combination and calculate R2 score for both training and testing sets
for combination in combinations_6_features:
    # Splitting the dataset into training and testing sets
    X_train_subset = X_train[list(combination)]
    X_test_subset = X_test[list(combination)]

    # Train the model using Linear Regression
    model = LinearRegression()
    model.fit(X_train_subset, y_train)

    # Predictions on training set
    y_pred_train = model.predict(X_train_subset)
    train_accuracy = r2_score(y_train, y_pred_train)
    train_accuracies.append(train_accuracy)

    # Predictions on testing set
    y_pred_test = model.predict(X_test_subset)
    test_accuracy = r2_score(y_test, y_pred_test)
    test_accuracies.append(test_accuracy)

    # Store combination and its R2 score
    combination_scores.append((combination, test_accuracy))

# Sort combinations based on testing set accuracy
sorted_combinations = sorted(combination_scores, key=lambda x: x[1], reverse=True)

# Print combination with the best R2 score
best_combination = sorted_combinations[0]
print("Best Combination (Features):", best_combination[0])
print("R2 Score (Testing Set):", best_combination[1])

# Print R2 score for top 5 best feature combinations
print("\nTop 5 Feature Combinations:")
for i in range(5):
    print("Features:", sorted_combinations[i][0])
    print("R2 Score (Testing Set):", sorted_combinations[i][1])
    print()

# Extract best combination for further analysis
best_combination_features = best_combination[0]

# Train the model using the best combination
X_train_best_subset = X_train[list(best_combination_features)]
X_test_best_subset = X_test[list(best_combination_features)]

model_best = LinearRegression()
model_best.fit(X_train_best_subset, y_train)

# Predictions on training set using the best combination
y_pred_train_best = model_best.predict(X_train_best_subset)

# Predictions on testing set using the best combination
y_pred_test_best = model_best.predict(X_test_best_subset)

# Calculate R2 score for the best combination
best_combination_train_accuracy = r2_score(y_train, y_pred_train_best)
best_combination_test_accuracy = r2_score(y_test, y_pred_test_best)

print("\nBest Combination R2 Score (Training Set):", best_combination_train_accuracy)
print("Best Combination R2 Score (Testing Set):", best_combination_test_accuracy)

# Plot actual vs predicted values for training set using the best combination
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train_best, label='Actual vs Predicted (Training Set)', color='blue')
plt.plot(y_train, y_train, color='red', label='Ideal Prediction Line')  # Ideal prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Training Set) - Best Combination')
plt.legend()
plt.show()

# Plot actual vs predicted values for testing set using the best combination
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test_best, label='Actual vs Predicted (Testing Set)', color='green')
plt.plot(y_test, y_test, color='red', label='Ideal Prediction Line')  # Ideal prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Testing Set) - Best Combination')
plt.legend()
plt.show()
