# 🚦 Road Accident Risk Prediction: Physics-Informed AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-Stacking%20Ensemble-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

### 📋 Project Overview
This project aims to predict the **risk of traffic accidents** (0-1 probability) based on telematic data. 
Unlike traditional models that only use raw data, this solution incorporates **Physics-Informed Feature Engineering** to capture real-world driving dynamics (e.g., centrifugal force) and utilizes a **Stacking Ensemble** architecture for maximum robustness.

---

### 🧠 The Strategy: "Physics Meets Data Science"

#### 1. Feature Engineering (Domain Knowledge)
We didn't just feed the model raw numbers. We created synthetic features based on road safety physics:
* **Centrifugal Force Proxy:** Calculated as `Speed Limit × Curvature`. High speed on a sharp curve exponentially increases risk.
* **Visibility Hazard:** Interaction between `Lighting` (Night/Dim) and `Speed`.
* **Accident Density:** Historical accident frequency relative to the road's speed limit.

#### 2. Model Architecture (Stacking Ensemble)
We used a 2-Level Stacking approach to minimize variance and bias:
* **Level 0 (Base Learners):**
    * 🚀 **LightGBM:** Optimized for speed and gradient boosting efficiency.
    * 🛡️ **XGBoost:** Provides stability and regularization.
    * 🐈 **CatBoost:** Specifically tuned for handling categorical features (Road Type, Weather).
* **Level 1 (Meta-Learner):**
    * 🧠 **Ridge Regression:** Learns the optimal weights from the base learners' predictions to generate the final output.

---

### 🛠️ Tech Stack
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (StackingRegressor, RidgeCV)
* **Boosting Libraries:** XGBoost, LightGBM, CatBoost
* **Evaluation Metric:** Mean Absolute Error (MAE)

---

### 📊 Key Findings
* **Speed is not the only enemy:** Speed alone has a moderate correlation, but **Speed combined with Curvature** is the highest predictor of accident risk.
* **The "Dark" Factor:** Driving at night significantly amplifies the risk of other factors (e.g., wet roads), captured successfully by the model's interaction terms.
* **Stacking Superiority:** The Stacking model outperformed individual models (XGB/LGBM) by effectively blending their strengths.

---

### 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/AI-Road-Safety-Risk-Prediction.git](https://github.com/YourUsername/AI-Road-Safety-Risk-Prediction.git)
2. Install dependencies
   ```bash
    pip install pandas xgboost lightgbm catboost scikit-learn
3. Run the notebook
   ```bash
   Road_Accident_Prediction.ipynb
4. Dataset
   The dataset used in this project is too large for GitHub.
   You can download it directly from the source:
   (https://www.kaggle.com/competitions/dstc-kaggle-practice-2025)
