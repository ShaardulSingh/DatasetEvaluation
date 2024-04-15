import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('Ipl_Batsman.xlsx')

df_numeric = df.select_dtypes(include=['number'])

correlations = df_numeric.corr()['SOLD PRICE'].abs().sort_values(
    ascending=False)

top_15_features = correlations.index[1:16]

combinations_6_features = list(combinations(top_15_features, 6))

test_accuracies = []
y_pred_test_best = None
X_train_best = None
y_train_best = None

for combination in combinations_6_features:
  X = df[list(combination)]
  y = df['SOLD PRICE']

  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42)

  model = LinearRegression()
  model.fit(X_train, y_train)

  y_pred_test = model.predict(X_test)
  test_accuracy = r2_score(y_test, y_pred_test)
  test_accuracies.append(test_accuracy)

  if test_accuracy == max(test_accuracies):
    y_pred_test_best = y_pred_test
    X_train_best = X_train
    y_train_best = y_train

sorted_combinations = [
    comb for _, comb in sorted(zip(test_accuracies, combinations_6_features),
                               reverse=True)
]
sorted_test_accuracies = sorted(test_accuracies, reverse=True)

print("Top 5 Combinations of Features:")
for i, (combination, accuracy) in enumerate(
    zip(sorted_combinations[:5], sorted_test_accuracies[:5]), 1):
  print(f"Combination {i}: {combination}")
  print(f"R2 Score (Testing Set): {accuracy}")
  print()

model_best = LinearRegression()
model_best.fit(X_train_best, y_train_best)

y_pred_train_best = model_best.predict(X_train_best)

plt.figure(figsize=(8, 6))
plt.scatter(y_train_best,
            y_pred_train_best,
            label='Actual vs Predicted (Training Set)',
            color='blue')
plt.plot(y_train_best,
         y_train_best,
         color='red',
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