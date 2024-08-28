import numpy as np
import pandas as pd
import pickle


# def load_model():
#     with open("./model_files/model.pkl", "rb") as file:
#         model = pickle.load(file)
#     return model


def load_model():
    try:
        with open("./model_files/model_retrained.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        with open("./model_files/model.pkl", "rb") as file:
            model = pickle.load(file)
    return model


def predict(input_data):
    """Assumes input_data is already preprocessed."""
    model = load_model()
    input_data = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_data)
    return prediction[0]


def update_model(input_data, actual_prices):
    model = load_model()
    # model.fit(input_data, actual_prices)

    with open("./model_files/model_retrained.pkl", "wb") as file:
        pickle.dump(model, file)
