
# KNN Classification using Iris CSV Dataset

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load CSV dataset
df = pd.read_csv("Iris.csv")

# Display dataset
print("First five rows:")
print(df.head())

# Step 3: Remove unnecessary column (if present)
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Step 4: Separate features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Step 5: Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: KNN for different K values
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracy_list = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_list.append(accuracy)

    print(f"K = {k}, Accuracy = {accuracy * 100:.2f}%")

# Step 9: Find best K
best_index = np.argmax(accuracy_list)
best_k = k_values[best_index]

print("\nBest K Value:", best_k)
print("Best Accuracy:", accuracy_list[best_index] * 100, "%")

# Step 10: Plot K vs Accuracy
plt.plot(k_values, accuracy_list, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Classification on Iris Dataset")
plt.grid()
plt.show()

# Step 11: Check overfitting using best K
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

train_pred = final_knn.predict(X_train)
test_pred = final_knn.predict(X_test)

print("\nTraining Accuracy:",
      accuracy_score(y_train, train_pred) * 100, "%")

print("Testing Accuracy:",
      accuracy_score(y_test, test_pred) * 100, "%")

# Step 12: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# Step 13: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))
