Titanic Survival Prediction 🚢
Predict survival of Titanic passengers using Machine Learning with comprehensive EDA, Feature Engineering, and Model Deployment.

Tech Stack
Python: Core programming language

Pandas, NumPy: Data manipulation and cleaning

Matplotlib, Seaborn: Exploratory Data Analysis (EDA)

Scikit-learn: Logistic Regression, Random Forest

XGBoost: Gradient Boosted Trees

Streamlit: Interactive web app deployment

Project Structure
bash
Copy
Edit
titanic_survival_prediction/
│
├── data/
│   ├── raw/          # Original Titanic dataset
│   ├── processed/    # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb     # Model training & evaluation
│   └── 04_final_model.ipynb  # Final tuned model
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning scripts
│   ├── feature_engineering.py # Custom features
│   ├── model.py               # Model training & evaluation
│
├── models/
│   └── final_model.pkl        # Best performing model
│
├── app/
│   └── streamlit_app.py       # Deployment script
│
├── reports/
│   └── figures/               # EDA visualizations
│
└── requirements.txt           # Dependencies
Features
EDA: Insights on passenger demographics and survival factors

Feature Engineering: Title extraction, family features, and encoding

Modeling: Logistic Regression, Random Forest, XGBoost (best model)

Deployment: Streamlit app to predict survival interactively

How to Run Locally
1. Clone repository
bash
Copy
Edit
git clone https://github.com/MUHAMMADWAHAJASAD/titanic_survival_prediction.git
cd titanic_survival_prediction
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run Streamlit app
bash
Copy
Edit
streamlit run app/streamlit_app.py
Results
Best Model: XGBoost

Accuracy: ~85% test accuracy

Key Feature: Passenger Title significantly improves prediction

Deployment
Hosted via Streamlit Cloud:
https://titanicsurvivalprediction-efwmffpbrd2uoszpd9uhoa.streamlit.app/

Author
Muhammad Wahaj Asad

https://github.com/MUHAMMADWAHAJASAD

LinkedIn www.linkedin.com/in/wahaj-asad-9a1092206

