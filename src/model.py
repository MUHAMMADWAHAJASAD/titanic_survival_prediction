import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import os

def save_model(model, filepath):
    """
    Save trained model to disk
    """
    joblib.dump(model, filepath)

# ------------------------
# Load Model
# ------------------------
def load_model(filepath):
    """
    Load trained model from disk
    """
    return joblib.load(filepath)

def split_data(df, target_col="Survived", test_size=0.2, random_state=52):
    """
    Split dataframe into train/test sets
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=52)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model with accuracy and classification report
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    results = {
        "train_acc": accuracy_score(y_train, train_preds),
        "test_acc": accuracy_score(y_test, test_preds),
        "conf_matrix": confusion_matrix(y_test, test_preds),
        "report": classification_report(y_test, test_preds, output_dict=True)
    }

    return results


def save_model(model, filename="../models/final_model.pkl"):
    """
    Save trained model to models/ folder
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model
    """
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model    
