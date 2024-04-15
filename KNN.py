import warnings
import numpy as np


warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression


df = pd.read_excel('Ipl_Batsman.xlsx')


label_encoder = LabelEncoder()
df['AGE'] = label_encoder.fit_transform(df['AGE'])
df['PLAYING ROLE'] = label_encoder.fit_transform(df['PLAYING ROLE'])


df.dropna(inplace=True)


df.drop('Sl.NO.', axis=1, inplace=True)


df = pd.get_dummies(df, columns=['COUNTRY', 'TEAM'])


X = df.drop(['PLAYER NAME', 'SOLD PRICE'], axis=1)
y = df['SOLD PRICE']


mutual_info = mutual_info_regression(X, y)


sorted_indices = np.argsort(mutual_info)[::-1]
sorted_features = X.columns[sorted_indices]
sorted_mutual_info = mutual_info[sorted_indices]


plt.figure(figsize=(10, 6))
plt.bar(sorted_features, sorted_mutual_info, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Mutual Information')
plt.title('Mutual Information between Features and Target Variable (Decreasing Order)')
plt.xticks(rotation=90)
plt.show()


top_features_idx = sorted_indices[:10]
top_features = X.columns[top_features_idx]


combinations_6_features = list(combinations(top_features, 6))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train_accuracies = []
test_accuracies = []
combination_scores = []


for combination in combinations_6_features:
    
    X_train_subset = X_train[list(combination)]
    X_test_subset = X_test[list(combination)]

    
    model = KNeighborsRegressor()
   
    model.fit(X_train_subset, y_train)

    
    y_pred_train = model.predict(X_train_subset)
    train_accuracy = r2_score(y_train, y_pred_train)
    train_accuracies.append(train_accuracy)

    
    y_pred_test = model.predict(X_test_subset)
    test_accuracy = r2_score(y_test, y_pred_test)
    test_accuracies.append(test_accuracy)

    
    combination_scores.append((combination, test_accuracy))


sorted_combinations = sorted(combination_scores, key=lambda x: x[1], reverse=True)


best_combination = sorted_combinations[0]
print("Best Combination (Features):", best_combination[0])
print("R2 Score (Testing Set):", best_combination[1])


print("\nTop 5 Feature Combinations:")
for i in range(5):
    print("Features:", sorted_combinations[i][0])
    print("R2 Score (Testing Set):", sorted_combinations[i][1])
    print()


best_combination_features = best_combination[0]


X_train_best_subset = X_train[list(best_combination_features)]
X_test_best_subset = X_test[list(best_combination_features)]

model_best = KNeighborsRegressor()
model_best.fit(X_train_best_subset, y_train)


y_pred_train_best = model_best.predict(X_train_best_subset)


y_pred_test_best = model_best.predict(X_test_best_subset)


best_combination_train_accuracy = r2_score(y_train, y_pred_train_best)
best_combination_test_accuracy = r2_score(y_test, y_pred_test_best)

print("\nBest Combination R2 Score (Training Set):", best_combination_train_accuracy)
print("Best Combination R2 Score (Testing Set):", best_combination_test_accuracy)


plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train_best, label='Actual vs Predicted (Training Set)', color='blue')
plt.plot(y_train, y_train, color='red', label='Ideal Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Training Set) - Best Combination')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test_best, label='Actual vs Predicted (Testing Set)', color='green')
plt.plot(y_test, y_test, color='red', label='Ideal Prediction Line')  
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Testing Set) - Best Combination')
plt.legend()
plt.show()
