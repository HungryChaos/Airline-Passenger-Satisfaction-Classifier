import joblib
import pandas as pd
import numpy as np


# ---------------- LOAD SAVED OBJECTS ----------------
saved = joblib.load("project/models/model.joblib")

model = saved["model"]
ct = saved["transformer"]
le = saved["label_encoder"]
selected_features = saved["selected_features"]


# ---------------- FEATURE ENGINEERING ----------------
def feature_engineering(df):

    rating_cols = [
        'wifi','convenience','booking','gate','food','boarding',
        'seat','entertainment','onboard_service','leg_room',
        'baggage','checkin','inflight_service','cleanliness'
    ]

    # log transforms
    df["dep_delay_log"] = np.log1p(df["dep_delay"])
    df["arr_delay_log"] = np.log1p(df["arr_delay"])

    df.drop(["dep_delay", "arr_delay"], axis=1, inplace=True)

    # engineered features
    df["wifi_low"] = (df["wifi"] <= 2).astype(int)

    df["rating_std"] = df[rating_cols].std(axis=1)

    df["premium_customer"] = (
        (df["customer_type"] == "Loyal Customer") &
        (df["class"].isin(["Business", "Business Plus"]))
    ).astype(int)

    df["comfort_score"] = df[["seat","leg_room","cleanliness"]].mean(axis=1)

    df["digital_experience"] = df[["wifi","entertainment"]].mean(axis=1)

    df["service_quality"] = df[
        ["onboard_service","inflight_service","checkin","boarding"]
    ].mean(axis=1)

    df["booking_experience"] = df[
        ["booking","convenience"]
    ].mean(axis=1)

    df["has_significant_delay"] = (
        df["dep_delay_log"] > np.log1p(10)
    ).astype(int)

    df["overall_service_mean"] = df[rating_cols].mean(axis=1)

    return df


# ---------------- PREDICTION FUNCTION ----------------
def predict_satisfaction(input_data):

    # convert dict → dataframe
    df = pd.DataFrame([input_data])

    # feature engineering
    df = feature_engineering(df)

    # transform using saved transformer
    transformed = ct.transform(df)

    transformed_df = pd.DataFrame(
        transformed,
        columns=ct.get_feature_names_out()
    )

    # keep only selected features
    final_input = transformed_df[selected_features]

    # predict
    pred = model.predict(final_input)

    # convert back label
    result = le.inverse_transform(pred)

    return result[0]


# ---------------- TEST ----------------
if __name__ == "__main__":

    sample = {
        "age": 45,
        "distance": 300,

        "gender": "Male",
        "customer_type": "disloyal Customer",
        "travel_type": "Personal Travel",
        "class": "Eco",

        "wifi": 1,
        "convenience": 1,
        "booking": 1,
        "gate": 1,
        "food": 1,
        "boarding": 1,
        "seat": 1,
        "entertainment": 1,
        "onboard_service": 1,
        "leg_room": 1,
        "baggage": 1,
        "checkin": 1,
        "inflight_service": 1,
        "cleanliness": 1,

        "dep_delay": 120,
        "arr_delay": 90
    }

    prediction = predict_satisfaction(sample)

    print("Prediction:", prediction)