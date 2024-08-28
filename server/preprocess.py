import pickle
import pandas as pd


def load_preprocessors():
    with open("./model_files/scaler_days_left.pkl", "rb") as file:
        scaler_days_left = pickle.load(file)
    with open("./model_files/scaler_duration.pkl", "rb") as file:
        scaler_duration = pickle.load(file)
    with open("./model_files/scaler_stops.pkl", "rb") as file:
        scaler_stops = pickle.load(file)

    return scaler_days_left, scaler_duration, scaler_stops


def one_hot_encode(data_point):
    # Define all possible categories for each feature
    airlines = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]
    cities = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    times = ["Evening", "Early_Morning", "Morning", "Afternoon", "Night", "Late_Night"]
    classes = ["Economy", "Business"]

    # Columns that need to be one-hot encoded
    columns_to_encode = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "class",
    ]

    # Separate the columns to be one-hot encoded from those that are not
    encode_df = data_point[columns_to_encode]
    remainder_df = data_point.drop(columns=columns_to_encode)

    # One-hot encode the necessary columns
    one_hot_encoded = pd.get_dummies(
        encode_df,
        columns=columns_to_encode,
        prefix=[
            "airline",
            "source_city",
            "destination_city",
            "departure_time",
            "arrival_time",
            "class",
        ],
    )

    # Define all potential one-hot encoded columns
    all_columns = (
        [f"airline_{airline}" for airline in airlines]
        + [f"source_city_{city}" for city in cities]
        + [f"destination_city_{city}" for city in cities]
        + [f"departure_time_{time}" for time in times]
        + [f"arrival_time_{time}" for time in times]
        + [f"class_{c}" for c in classes]
    )

    # Reindex the DataFrame to include all columns, filling missing ones with 0s
    one_hot_encoded = one_hot_encoded.reindex(columns=all_columns, fill_value=0)

    # Merge the one-hot encoded columns back with the remainder of the DataFrame
    final_encoded_df = pd.concat(
        [remainder_df.reset_index(drop=True), one_hot_encoded.reset_index(drop=True)],
        axis=1,
    )

    return final_encoded_df


def map_stops(data_point):
    stops_mapping = {"zero": 0, "one": 1, "two_or_more": 2}
    data_point["stops"] = data_point["stops"].map(stops_mapping)
    return data_point


def scale_standard(data_point, scaler_days_left, scaler_stops):
    data_point["days_left"] = scaler_days_left.transform(data_point[["days_left"]])
    data_point["stops"] = scaler_stops.transform(data_point[["stops"]])
    return data_point


def scale_robust(data_point, scaler_duration):
    data_point["duration"] = scaler_duration.transform(data_point[["duration"]])
    return data_point


def preprocess(data_point):
    scaler_days_left, scaler_duration, scaler_stops = load_preprocessors()

    data_point = one_hot_encode(data_point)
    data_point = map_stops(data_point)
    data_point = scale_standard(data_point, scaler_days_left, scaler_stops)
    data_point = scale_robust(data_point, scaler_duration)

    # Define the expected feature order
    expected_feature_order = [
        "stops",
        "duration",
        "days_left",
        "airline_AirAsia",
        "airline_Air_India",
        "airline_GO_FIRST",
        "airline_Indigo",
        "airline_SpiceJet",
        "airline_Vistara",
        "source_city_Bangalore",
        "source_city_Chennai",
        "source_city_Delhi",
        "source_city_Hyderabad",
        "source_city_Kolkata",
        "source_city_Mumbai",
        "destination_city_Bangalore",
        "destination_city_Chennai",
        "destination_city_Delhi",
        "destination_city_Hyderabad",
        "destination_city_Kolkata",
        "destination_city_Mumbai",
        "departure_time_Afternoon",
        "departure_time_Early_Morning",
        "departure_time_Evening",
        "departure_time_Late_Night",
        "departure_time_Morning",
        "departure_time_Night",
        "arrival_time_Afternoon",
        "arrival_time_Early_Morning",
        "arrival_time_Evening",
        "arrival_time_Late_Night",
        "arrival_time_Morning",
        "arrival_time_Night",
        "class_Business",
        "class_Economy",
    ]

    # Reorder the columns
    data_point = data_point[expected_feature_order]

    return data_point
