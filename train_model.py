import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model():

    data = pd.read_csv("dataset/heart.csv")

    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    model = LogisticRegression()

    model.fit(X_train, y_train)


    predictions = model.predict(X_test)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Model Accuracy:", accuracy)

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    os.makedirs(
        "models",
        exist_ok=True
    )


    pickle.dump(
        model,
        open(
            "models/heart_disease_model.pkl",
            "wb"
        )
    )


    pickle.dump(
        scaler,
        open(
            "models/scaler.pkl",
            "wb"
        )
    )


if __name__ == "__main__":
    train_model()
