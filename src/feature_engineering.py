import pandas as pd

def extract_title(df):
    """
    Extract Title (Mr, Miss, etc.) from Name and encode
    """
    df = df.copy()

    # Extract title
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Group rare titles
    df["Title"] = df["Title"].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare'
    )
    df["Title"] = df["Title"].replace(['Mlle', 'Ms'], 'Miss')
    df["Title"] = df["Title"].replace('Mme', 'Mrs')

    # Encode titles numerically
    title_mapping = {title: idx for idx, title in enumerate(df["Title"].unique())}
    df["Title"] = df["Title"].map(title_mapping)

    return df


def impute_age_by_group(df):
    """
    Fill missing Age using median Age per Title + Pclass + Sex group
    """
    df = df.copy()
    df["Age"] = df.groupby(["Title", "Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    return df


def create_family_features(df):
    """
    Add FamilySize = SibSp + Parch + 1
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df):
    """
    Add IsAlone = 1 if FamilySize == 1 else 0
    """
    df = df.copy()
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1
    return df
