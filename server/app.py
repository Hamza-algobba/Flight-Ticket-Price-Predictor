import csv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict, update_model
from preprocess import preprocess
import pandas as pd

app = Flask(__name__)
CORS(app)

BATCH_SIZE = 2


def append_data_to_file(data, filepath):
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def retrain_model_with_new_data(filepath):
    data_df = pd.read_csv(filepath)

    processed_data = preprocess(
        data_df.drop(["predicted_price", "actual_price"], axis=1)
    )
    actual_prices = data_df["actual_price"]  # these are the ground truth for retraining
    update_model(processed_data, actual_prices)

    os.remove(filepath)


def validate_input(data):
    required_fields = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "class",
        "stops",
        "days_left",
        "duration",
    ]
    if not all(field in data for field in required_fields):
        return False
    return True


@app.route("/submit_prices", methods=["POST"])
def submit_prices():
    data = request.json
    if (
        not validate_input(data)
        or "predicted_price" not in data
        or "actual_price" not in data
    ):
        return jsonify({"error": "Missing or invalid input fields"}), 400
    try:
        append_data_to_file(data, "data_points.csv")

        # Check if it's time to retrain
        if (
            sum(1 for line in open("data_points.csv")) - 1 >= BATCH_SIZE
        ):  # We subtract 1 for header
            retrain_model_with_new_data("data_points.csv")

        return jsonify({"message": "Data submitted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.json
    if not validate_input(data):
        return jsonify({"error": "Missing or invalid input fields"}), 400
    try:
        data_df = pd.DataFrame([data])
        processed_data = preprocess(data_df)
        prediction = predict(processed_data)

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
