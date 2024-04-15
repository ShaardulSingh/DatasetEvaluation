import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('Ipl_Batsman.xlsx')

label_encoder = LabelEncoder()
df['AGE'] = label_encoder.fit_transform(df['AGE'])
df['PLAYING ROLE'] = label_encoder.fit_transform(df['PLAYING ROLE'])

df.dropna(inplace=True)

df.drop('Sl.NO.', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['COUNTRY', 'TEAM'])

X = df.drop(['PLAYER NAME', 'SOLD PRICE'], axis=1)
y = df['SOLD PRICE']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

model = RandomForestRegressor(n_estimators=100,
                              random_state=42)
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
feature_names = X.columns

sorted_indices = feature_importances.argsort()[::-1]
sorted_features = feature_names[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

plt.figure(figsize=(12, 8))
plt.bar(sorted_features, sorted_importances)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance Scores (Decreasing Order)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

top_features = sorted_features[:15]

combinations_6_features = list(combinations(top_features, 6))

train_accuracies = []
test_accuracies = []
for combination in combinations_6_features:
  model_subset_features = RandomForestRegressor(n_estimators=100,
                                                random_state=42)
  model_subset_features.fit(X_train[list(combination)], y_train)

  y_pred_train = model_subset_features.predict(X_train[list(combination)])
  train_accuracy = r2_score(y_train, y_pred_train)
  train_accuracies.append(train_accuracy)

  y_pred_test = model_subset_features.predict(X_test[list(combination)])
  test_accuracy = r2_score(y_test, y_pred_test)
  test_accuracies.append(test_accuracy)

sorted_combinations = [
    comb for _, comb in sorted(zip(test_accuracies, combinations_6_features),
                               reverse=True)
]
sorted_test_accuracies = sorted(test_accuracies, reverse=True)

best_combination = sorted_combinations[0]

best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train[list(best_combination)], y_train)

y_pred_train_best = best_model.predict(X_train[list(best_combination)])

y_pred_test_best = best_model.predict(X_test[list(best_combination)])

top_5_combinations = sorted_combinations[:5]
top_5_test_accuracies = sorted_test_accuracies[:5]

print("Top 5 Combinations of Features:")
for i, combination in enumerate(top_5_combinations, 1):
  model_subset_features = RandomForestRegressor(n_estimators=100,
                                                random_state=42)
  model_subset_features.fit(X_train[list(combination)], y_train)

  y_pred_test = model_subset_features.predict(X_test[list(combination)])
  test_accuracy = r2_score(y_test, y_pred_test)

  print(f"Combination {i}: {combination}")
  print(f"R2 Score (Testing Set): {test_accuracy}")
  print(f"R2 Score (Training Set): {train_accuracy}")

  print()

plt.figure(figsize=(8, 6))
plt.scatter(y_train,
            y_pred_train_best,
            label='Actual vs Predicted (Training Set)',
            color='blue')
plt.plot(y_train, y_train, color='red',
         label='Ideal Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Training Set) - Best Combination')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test,
            y_pred_test_best,
            label='Actual vs Predicted (Testing Set)',
            color='green')
plt.plot(y_test, y_test, color='red',
         label='Ideal Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Testing Set) - Best Combination')
plt.legend()
plt.show()
