# 🌙 AI Sleep Health & Early Warning System

An AI-powered clinical decision support system that predicts sleep disorder risk from lifestyle and health data, built with Streamlit.

---

## 🎯 About the Project

This project predicts whether an individual has **Insomnia**, **Sleep Apnea**, or a **Healthy** sleep profile based on their lifestyle habits and vital signs.

**Approach:**
- **KMeans clustering** groups individuals into sleep profiles (unsupervised learning)
- The cluster label is used as an additional feature for the **XGBoost classifier** (supervised learning)
- **SHAP values** explain why the model made each prediction
- A **Streamlit** web app provides a clinical-ready interface

---

## 📁 Project Structure

```
.
├── proje_uyku_english.ipynb    # Full analysis and model training
├── app.py                      # Streamlit web app
├── requirements.txt            # Required libraries
├── README.md                   # Project documentation
├── data/                       # Dataset (not tracked by git)
│   └── Sleep_health_and_lifestyle_dataset.csv
├── models/                     # Trained models (not tracked by git)
│   ├── xgboost_model.pkl
│   ├── label_encoder.pkl
│   ├── kmeans_model.pkl
│   └── scaler.pkl
└── outputs/                    # Generated plots (not tracked by git)
```

---

## 🗂️ Dataset

**Sleep Health and Lifestyle Dataset** — Kaggle  
🔗 https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

| Detail | Value |
|---|---|
| Samples | 374 individuals |
| Features | 13 variables |
| Target | Sleep Disorder (None / Insomnia / Sleep Apnea) |

---

## 🔬 Analysis Pipeline

| Section | Content |
|---|---|
| 1 | Data loading and exploration |
| 2 | Data cleaning and preprocessing (BP split, encoding) |
| 3 | Exploratory Data Analysis — 8 visualizations |
| 4 | KMeans clustering (k=4 via Elbow method) |
| 5 | 4-model comparison + overfitting check |
| 6 | Model explainability with SHAP |

---

## 📊 Model Results

| Model | Accuracy |
|---|---|
| **XGBoost** | **91%** |
| Random Forest | 88% |
| KNN | 88% |
| Logistic Regression | 89% |

---

## 🚀 Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sleep-analysis-system.git
cd sleep-analysis-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
```
data/Sleep_health_and_lifestyle_dataset.csv
```

### 4. Run the notebook to generate models
```bash
jupyter notebook proje_uyku_english.ipynb
```
> Running all cells will save 4 model files to the `models/` folder.

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `Streamlit` `Matplotlib` `Seaborn`

---

## 👤 Developer

GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
