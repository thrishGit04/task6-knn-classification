![Python](https://img.shields.io/badge/Language-Python-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/ML%20Library-ScikitLearn-orange.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-green.svg)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-yellow.svg)
![Dataset](https://img.shields.io/badge/Dataset-Iris-purple.svg)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-98%25-brightgreen.svg)
![Status](https://img.shields.io/badge/Task-Completed-success.svg)

---

# ⭐ Task 6 — K-Nearest Neighbors (KNN) Classification

This repository contains **Task 6** of my AIML Internship project.  
The goal is to understand, implement, and evaluate the **K-Nearest Neighbors (KNN)** algorithm on the classic **Iris dataset**, including feature normalization, K-value experimentation, decision boundary visualization, and logging results into SQLite.

---

## 📚 Table of Contents

1. [Repository Structure](#-repository-structure)  
2. [Objective](#-objective)  
3. [Data Preprocessing](#-data-preprocessing-steps)  
4. [Model Training Pipeline](#-model-training-knn-classificationpy)  
5. [Visualizations](#-generated-visualizations)  
6. [Evaluation Metrics](#-evaluation-metrics)  
7. [How to Run](#-how-to-run-the-project)  
8. [Dataset](#-dataset)  
9. [Author](#-author)

---

## 📁 Repository Structure

```text
├── Iris.csv                        # Original dataset (uploaded)
├── knn-classification.py           # Complete training & visualization script (single-run)
├── database.sqlite                 # SQLite DB storing KNN experiment results
├── README.md                       # Documentation (this file)
└── outputs/
    ├── confusion_matrix.png        # Confusion matrix visualization
    ├── decision_boundary.png       # 2D KNN decision boundary (PetalLength vs PetalWidth)
    ├── empty                       # Placeholder auto-created by Colab
    ├── iris_processed.csv          # Cleaned & preprocessed Iris dataset
    ├── k_vs_accuracy.png           # K vs accuracy line plot
    └── knn_k_results.csv           # Results of K-value experimentation
```

---

## 🎯 Objective

This task focuses on implementing and understanding:

* **K-Nearest Neighbors (KNN)** for multiclass classification
* **Feature normalization** using `StandardScaler`
* **Effect of K on model performance**
* **Confusion matrix and classification metrics**
* **Decision boundary visualization in 2D**
* **Storing experiment results in an SQLite database**

---

## 🧹 Data Preprocessing Steps

The following steps are applied to `Iris.csv`:

1. Load the original Iris dataset.
2. Drop the `Id` column (if present) and check for missing values.
3. Separate features and target (`Species`).
4. Encode the target labels using **LabelEncoder**.
5. Split data into **train** and **test** sets (80/20, stratified).
6. Normalize numerical features with **StandardScaler**.
7. Save the cleaned dataset as `outputs/iris_processed.csv`.

---

## 🤖 Model Training (`knn-classification.py`)

The script implements a full KNN classification workflow:

### 1️⃣ Normalization & Label Encoding

* All numerical features are standardized.
* `Species` is mapped to numeric class indices.

### 2️⃣ KNN Training & K-Value Experimentation

* Trains KNN models for multiple values of **K**:
  `1, 3, 5, 7, 9, 11, 13, 15`
* For each K, computes:

  * **Train accuracy**
  * **Test accuracy**
* Stores results in:

  * `outputs/knn_k_results.csv`
  * `database.sqlite` → table **`knn_results`**

### 3️⃣ Best K Selection

* Automatically selects the K that achieves the **highest test accuracy**.
* Prints the best K along with corresponding train and test accuracies.

### 4️⃣ Final Evaluation

For the best K:

* Trains a final KNN model.
* Generates:

  * **Classification report** (precision, recall, F1-score, support)
  * **Confusion matrix** (printed + plotted)

### 5️⃣ Decision Boundary Visualization

* Uses two features (PetalLength and PetalWidth, or the last two numeric features) to create a 2D space.
* Plots:

  * Colored decision regions predicted by KNN.
  * Standardized feature points for each class.
* Saves the plot as `outputs/decision_boundary.png`.

---

## 📊 Generated Visualizations

All visual outputs are stored in the `outputs/` directory.

### ✔ K vs Accuracy Plot

**File:** `k_vs_accuracy.png`
Shows how both train and test accuracy change with different values of K.
Useful for observing overfitting (very small K) vs underfitting (very large K).

### ✔ Confusion Matrix

**File:** `confusion_matrix.png`
Heatmap that compares predicted vs actual classes to see where the model is correct or confused.

### ✔ Decision Boundary Plot

**File:** `decision_boundary.png`
2D visualization of KNN decision regions over PetalLength & PetalWidth, with each Iris class shown as separate clusters.

---

## 🧪 Evaluation Metrics

The project evaluates the KNN classifier using:

* **Accuracy (train & test)**
* **Confusion matrix**
* **Classification report** (per-class precision, recall, F1-score, support)

These metrics provide a clear understanding of how well KNN performs on the Iris dataset and how performance varies with different K values.

---

## 🚀 How to Run the Project

### Option 1 — Google Colab (Recommended)

1. Upload the following files to your Colab session:

   * `Iris.csv`
   * `knn-classification.py`
   * *(optional)* `database.sqlite` (if you want to reuse/inspect previous logs)
2. Run the script:

```python
!python knn-classification.py
```

All generated artifacts (processed CSV, plots, and results) will appear in the **`outputs/`** folder.

---

### Option 2 — Local Machine

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Run the script:

```bash
python knn-classification.py
```

Outputs will be created in the `outputs/` directory in the project root.

---

## 📝 Dataset

**Iris Dataset**

A classic multiclass classification dataset containing measurements of iris flowers:

* **Classes:** Setosa, Versicolor, Virginica
* **Features:** Sepal length, sepal width, petal length, petal width

This dataset is widely used to demonstrate machine learning algorithms such as KNN.

---

## ✨ Author

**Thrishool M S**

AIML Internship — *Task 6: K-Nearest Neighbors (KNN) Classification*
