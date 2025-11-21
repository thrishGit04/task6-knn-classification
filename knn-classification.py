# ==========================================
# Task 6 – K-Nearest Neighbors (KNN) Classification
# ==========================================

# ---- 1. Imports ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sqlite3
import os

# For nicer plots in Colab
plt.rcParams["figure.figsize"] = (7, 5)

# ==========================================
# 2. Load dataset (Iris.csv)
# ==========================================
df = pd.read_csv("Iris.csv")

print("First 5 rows of Iris dataset:")
display(df.head())
print("\nShape:", df.shape)


if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Features and target
feature_cols = [col for col in df.columns if col != "Species"]
X = df[feature_cols].values
y = df["Species"].values

print("\nFeature columns:", feature_cols)
print("Target classes:", np.unique(y))


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==========================================
# 3. Train / Test Split & Normalization
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# ==========================================
# 4. Experiment with different K values
# ==========================================
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# ---- Log results into SQLite database.sqlite ----
db_path = "database.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS knn_results (
    k INTEGER,
    train_accuracy REAL,
    test_accuracy REAL
)
""")

cur.execute("DELETE FROM knn_results")

for k, tr_acc, te_acc in zip(k_values, train_accuracies, test_accuracies):
    cur.execute(
        "INSERT INTO knn_results (k, train_accuracy, test_accuracy) VALUES (?, ?, ?)",
        (int(k), float(tr_acc), float(te_acc))
    )

conn.commit()
conn.close()

print("\nLogged KNN results into database.sqlite (table: knn_results).")

# ---- Plot K vs Accuracy ----
plt.figure()
plt.plot(k_values, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(k_values, test_accuracies, marker="s", label="Test Accuracy")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN: Effect of K on Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# ---- Choose best K based on highest test accuracy ----
best_index = int(np.argmax(test_accuracies))
best_k = k_values[best_index]
print(f"\nBest K based on test accuracy: {best_k}")
print(f"Train Accuracy at K={best_k}: {train_accuracies[best_index]:.4f}")
print(f"Test  Accuracy at K={best_k}: {test_accuracies[best_index]:.4f}")

# ==========================================
# 5. Final model with best K & evaluation
# ==========================================
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

y_test_pred_best = knn_best.predict(X_test_scaled)

print("\n=== Classification Report (Test, best K) ===")
print(classification_report(y_test, y_test_pred_best, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_test_pred_best)
print("Confusion Matrix:\n", cm)

# ---- Confusion matrix heatmap ----
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title(f"Confusion Matrix (K={best_k})")
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# ==========================================
# 6. Visualize Decision Boundaries (2 features)
# ==========================================

try:
    f1_name = [c for c in df.columns if "Petal" in c][0]
    f2_name = [c for c in df.columns if "Petal" in c][1]
except:
    f1_name = feature_cols[-2]
    f2_name = feature_cols[-1]

X_2 = df[[f1_name, f2_name]].values
y_2 = y_encoded

scaler_2 = StandardScaler()
X_2_scaled = scaler_2.fit_transform(X_2)

knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_2_scaled, y_2)

# Create mesh grid
x_min, x_max = X_2_scaled[:, 0].min() - 0.5, X_2_scaled[:, 0].max() + 0.5
y_min, y_max = X_2_scaled[:, 1].min() - 0.5, X_2_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn_2d.predict(grid_points).reshape(xx.shape)

# ---- Plot decision boundary ----
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.4)

# Plot training points
for class_index, class_name in enumerate(label_encoder.classes_):
    plt.scatter(
        X_2_scaled[y_2 == class_index, 0],
        X_2_scaled[y_2 == class_index, 1],
        label=class_name,
        edgecolor="black"
    )

plt.xlabel(f1_name + " (standardized)")
plt.ylabel(f2_name + " (standardized)")
plt.title(f"KNN Decision Boundary (K={best_k}, features: {f1_name}, {f2_name})")
plt.legend()
plt.grid(True)
plt.show()

print("\nTask 6 – KNN Classification pipeline completed")
