import pandas as pd
import joblib
from model import create_model


def main():
    df = pd.read_csv("data/colors.csv")

    X = df[["R", "G", "B"]]
    y = df["label"]

    model = create_model()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("Model trained and saved")


if __name__ == "__main__":
    main()
