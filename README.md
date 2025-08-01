# Titanic Survival Prediction 🚢

A comprehensive machine learning project that predicts passenger survival on the Titanic using advanced data analysis, feature engineering, and multiple ML algorithms with interactive web deployment.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen)](https://titanicsurvivalprediction-efwmffpbrd2uoszpd9uhoa.streamlit.app/)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning techniques. The project includes comprehensive exploratory data analysis (EDA), advanced feature engineering, multiple ML model implementations, and an interactive web application for real-time predictions.

**Key Highlights:**
- 🔍 In-depth exploratory data analysis with statistical insights
- ⚙️ Advanced feature engineering including title extraction and family groupings
- 🤖 Multiple ML algorithms: Logistic Regression, Random Forest, and XGBoost
- 📊 Model comparison and hyperparameter tuning
- 🚀 Interactive Streamlit web application
- 📈 85% accuracy achieved with optimized XGBoost model

## ✨ Features

- **Comprehensive EDA**: Deep dive into passenger demographics, survival patterns, and feature correlations
- **Smart Feature Engineering**: 
  - Title extraction from passenger names
  - Family size categorization
  - Cabin deck analysis
  - Age group binning
- **Multiple ML Models**: Implementation and comparison of various algorithms
- **Interactive Web App**: User-friendly interface for survival predictions
- **Model Persistence**: Trained models saved for deployment and reuse
- **Visualization**: Rich charts and graphs for data insights

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Web Framework** | Streamlit |
| **Development** | Jupyter Notebooks |

## 📁 Project Structure

```
titanic_survival_prediction/
│
├── 📂 data/
│   ├── 📂 raw/                    # Original Titanic dataset
│   └── 📂 processed/              # Cleaned & engineered features
│
├── 📂 notebooks/
│   ├── 📓 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 📓 02_preprocessing.ipynb  # Data cleaning & feature engineering
│   ├── 📓 03_modeling.ipynb      # Model training & evaluation
│   └── 📓 04_final_model.ipynb   # Final optimized model
│
├── 📂 src/
│   ├── 🐍 data_preprocessing.py   # Data cleaning utilities
│   ├── 🐍 feature_engineering.py # Custom feature creation
│   └── 🐍 model.py               # Model training & evaluation
│
├── 📂 models/
│   └── 💾 final_model.pkl        # Best performing model
│
├── 📂 app/
│   └── 🌐 streamlit_app.py       # Web application
│
├── 📂 reports/
│   └── 📂 figures/               # Generated visualizations
│
├── 📄 requirements.txt            # Project dependencies
└── 📄 README.md                  # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**

   git clone https://github.com/MUHAMMADWAHAJASAD/titanic_survival_prediction.git
   cd titanic_survival_prediction
  

2. **Create virtual environment** (recommended)
   `
   python -m venv titanic_env
   source titanic_env/bin/activate  # On Windows: titanic_env\Scripts\activate
   

3. **Install dependencies**
   
   pip install -r requirements.txt
   

## 💻 Usage

### Running the Web Application

streamlit run app/streamlit_app.py

### Exploring the Notebooks

jupyter notebook notebooks/


### Training Models

python src/model.py


## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 81.2% | 0.78 | 0.74 | 0.76 |
| Random Forest | 83.7% | 0.82 | 0.79 | 0.80 |
| **XGBoost** | **85.1%** | **0.84** | **0.81** | **0.82** |

### Key Insights
- **Most Important Features**: Passenger title, fare, age, and passenger class
- **Title Feature Impact**: Extracting titles (Mr., Mrs., Miss, etc.) improved accuracy by ~3%
- **Family Features**: Family size and being alone significantly affect survival probability

## 🌐 Deployment

The application is deployed on Streamlit Cloud and accessible at:
**[Live Demo](https://titanicsurvivalprediction-efwmffpbrd2uoszpd9uhoa.streamlit.app/)**

### Local Deployment
For local deployment, simply 
streamlit run app/streamlit_app.py


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Muhammad Wahaj Asad**

- 🐙 GitHub: [@MUHAMMADWAHAJASAD](https://github.com/MUHAMMADWAHAJASAD)
- 💼 LinkedIn: [wahaj-asad-9a1092206](https://www.linkedin.com/in/wahaj-asad-9a1092206)
- 📧 Email: [muhammadwahaj34]

---

⭐ **Star this repository if you found it helpful!**

