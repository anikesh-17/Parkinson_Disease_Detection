# Parkinson's Disease Prediction using SVM

This project builds a machine learning model to predict Parkinson's Disease using vocal features. The model is trained using a **Support Vector Machine (SVM)** with a linear kernel.

---

## 📌 Dataset Overview

- **Total Samples:** 195  
- **Total Features:** 23 (after removing `name`)  
- **Target Variable:**  
  - `1` → Parkinson’s Positive  
  - `0` → Healthy  

The dataset contains biomedical voice measurements such as jitter, shimmer, RPDE, DFA, and PPE.

---

## 📊 Exploratory Data Analysis (EDA)

- No missing values.
- Data is imbalanced:  
  - **147 positive cases**  
  - **48 healthy cases**

Basic statistics and feature distributions were analyzed to understand data patterns.

---

## 🔧 Data Pre‑Processing

### 1. Feature–Target Split
```python
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']
```

### 2. Train–Test Split
```python
train_test_split(X, Y, test_size=0.2, random_state=2)
```

### 3. Standardization
StandardScaler is applied to normalize feature values.

---

## 🤖 Model Training — Support Vector Machine (SVM)

```python
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
```

---

## ✅ Model Performance

| Dataset | Accuracy |
|--------|----------|
| Training Data | **88.46%** |
| Test Data | **87.17%** |

The model performs well without overfitting.

---

## 🔮 Making Predictions

The system accepts 22 input features and returns either:

- **"The Person has Parkinson's"**
- **"The Person does not have Parkinson's Disease"**

Example:
```python
prediction = model.predict(std_data)
```

---

## 📁 Files Used

- `parkinsons.csv` — dataset  
- `parkinsons_prediction.ipynb` — notebook containing full code  

---

## 🚀 How to Run the Project

1. Install dependencies:
```bash
pip install numpy pandas scikit-learn
```

2. Run the notebook or script:
```bash
python parkinsons_prediction.py
```

---

## 📜 License

This project is for educational purposes only.

---

## ✨ Author

Developed by **Anikesh Sharma**  
For machine learning practice and project development.
