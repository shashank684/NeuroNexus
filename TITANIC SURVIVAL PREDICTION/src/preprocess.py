import pandas as pd

def preprocess_data(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Pclass'] = df['Pclass'].astype(int)
    features = ['Pclass', 'Sex', 'Age', 'Fare']
    return df[features]
