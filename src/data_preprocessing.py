import pandas as pd
import os

def load_data(filename="Titanic-Dataset.csv", data_dir="../data/raw"):
    """
    Load dataset from raw folder
    """
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath)


def clean_data(df):
    """
    Basic cleaning:
    - Drop PassengerId, Ticket, Cabin
    - Fill missing Embarked and Fare
    - Encode Sex and Embarked
    """
    df = df.copy()

    # Drop irrelevant columns
    df.drop(columns=["PassengerId", "Ticket", "Cabin"], inplace=True, errors="ignore")

    # Fill missing Embarked
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Fill missing Fare
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Encode Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Encode Embarked
    embark_mapping = {"S": 0, "C": 1, "Q": 2}
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map(embark_mapping)

    return df
