# ⭐ **Task 6 — K-Nearest Neighbors (KNN) Classification**

This repository contains **Task 6** of my AIML Internship project.

The objective of this task is to understand, implement, and evaluate the **K-Nearest Neighbors (KNN)** classification algorithm using the **Iris dataset**.
The project includes feature normalization, K-value experimentation, model evaluation, decision boundary visualization, and logging results into an SQLite database.

---

## 📁 **Repository Structure**

```
├── Iris.csv                        # Original dataset (uploaded)
├── knn-classification.py           # Complete training & visualization script (single-run)
├── database.sqlite                 # SQLite DB storing KNN experiment results
├── README.md                       # Documentation (this file)
└── outputs/
    ├── confusion_matrix.png        # Confusion matrix visualization
    ├── decision_boundary.png       # 2D KNN decision boundary (PetalLength vs PetalWidth)
    ├── empty                       # Placeholder auto-created by Colab
    ├── iris_processed.csv          # Duplicate processed CSV saved inside folder
    ├── k_vs_accuracy.png           # K vs accuracy line plot
    └── knn_k_results.csv           # Results of K-value experimentation
```

---

## 🎯 **Objective**

This task focuses on understanding and applying:

* **K-Nearest Neighbors (KNN)** for classification
* **Feature normalization** using StandardScaler
* **Effect of K on model accuracy**
* **Confusion matrix & classification metrics**
* **2D decision boundary visualization**
* **Logging experimental results into SQLite**

---

## 🧹 **Data Preprocessing Steps**

Steps performed on the raw `Iris.csv` dataset:

1. Loaded the dataset and removed the `Id` column.
2. Verified class distributions and checked for missing values.
3. Split the dataset into training and testing splits (80/20).
4. Standardized all numerical features using **StandardScaler**.
5. Encoded the categorical target (`Species`) using **LabelEncoder**.
6. Saved the cleaned dataset as `iris_processed.csv`.

---

## 🤖 **Model Training (knn-classification.py)**

The script performs the full KNN workflow:

### **1. Normalization & Label Encoding**

* Standardizes all numerical features.
* Converts species names into numeric class labels.

### **2. KNN Training & Comparison**

* Trains KNN models for **multiple K values**:
  `{1, 3, 5, 7, 9, 11, 13, 15}`
* Calculates:

  * Train accuracy
  * Test accuracy
* Saves the results in:

  * `knn_k_results.csv`
  * `database.sqlite` → table **knn_results**

### **3. Best K Selection**

* Automatically selects the best K based on **highest test accuracy**.

### **4. Final Model Evaluation**

* Generates:

  * **Classification Report**
  * **Confusion Matrix**
  * **Confusion matrix visualization (PNG)**

### **5. Decision Boundary Visualization**

* Uses two features:

  * PetalLength
  * PetalWidth
* Creates a **color-coded decision region plot**, saved as:

  * `decision_boundary.png`

---

## 📊 **Generated Visualizations**

All stored in the `outputs/` folder:

### ✔ **K vs Accuracy Plot**

`k_vs_accuracy.png`
Shows how accuracy changes as K increases.

### ✔ **Confusion Matrix**

`confusion_matrix.png`
Displays model predictions vs actual labels.

### ✔ **2D Decision Boundary**

`decision_boundary.png`
Visual representation of how KNN separates classes in 2D space.

---

## 🧪 **Evaluation Metrics**

The following metrics were computed:

* **Accuracy (train & test)**
* **Confusion Matrix**
* **Classification Report**

  * Precision
  * Recall
  * F1-score

These metrics help assess the reliability and performance of the KNN classifier.

---

## 🚀 **How to Run the Project**

### **Option 1 — Google Colab (Recommended)**

Upload:

* `Iris.csv`
* `knn-classification.py`
* *(optional)* `database.sqlite`

Run:

```python
!python knn-classification.py
```

All visualizations and processed files will appear inside the **outputs/** folder.

---

### **Option 2 — Local Machine**

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Run:

```bash
python knn-classification.py
```

---

## 📝 **Dataset**

**Iris Dataset** (Fisher’s Iris dataset)
A foundational multiclass dataset with three flower species:

* Setosa
* Versicolor
* Virginica

Includes measurements of sepal & petal dimensions.

---

## ✨ **Author**

**Thrishool M S**

AIML Internship — *Task 6: K-Nearest Neighbors Classification*
