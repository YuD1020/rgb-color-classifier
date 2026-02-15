import joblib
import sys
import pandas as pd


def main():
    model = joblib.load("model.pkl")

    r = int(sys.argv[1])
    g = int(sys.argv[2])
    b = int(sys.argv[3])

    input_df = pd.DataFrame([[r, g, b]], columns=["R", "G", "B"])

    prediction = model.predict(input_df)
    print("Prediction:", prediction[0])


if __name__ == "__main__":
    main()
