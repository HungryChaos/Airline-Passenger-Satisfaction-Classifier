import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
from src.predict import predict_satisfaction


st.title("Airline Customer Satisfaction Prediction")

st.write("Enter passenger details below")


# Basic passenger info
age = st.number_input("Age", min_value=1, max_value=100, value=25)
distance = st.number_input("Flight Distance", min_value=0, value=1000)

gender = st.selectbox("Gender", ["Male", "Female"])

customer_type = st.selectbox(
    "Customer Type",
    ["Loyal Customer", "disloyal Customer"]
)

travel_type = st.selectbox(
    "Travel Type",
    ["Business travel", "Personal Travel"]
)

flight_class = st.selectbox(
    "Class",
    ["Business", "Eco", "Eco Plus"]
)


# Ratings (1–5)
wifi = st.slider("Wifi Service", 1, 5, 3)
convenience = st.slider("Convenience", 1, 5, 3)
booking = st.slider("Booking Service", 1, 5, 3)
gate = st.slider("Gate Location", 1, 5, 3)
food = st.slider("Food Service", 1, 5, 3)
boarding = st.slider("Boarding", 1, 5, 3)
seat = st.slider("Seat Comfort", 1, 5, 3)
entertainment = st.slider("Entertainment", 1, 5, 3)
onboard_service = st.slider("Onboard Service", 1, 5, 3)
leg_room = st.slider("Leg Room", 1, 5, 3)
baggage = st.slider("Baggage Handling", 1, 5, 3)
checkin = st.slider("Checkin Service", 1, 5, 3)
inflight_service = st.slider("Inflight Service", 1, 5, 3)
cleanliness = st.slider("Cleanliness", 1, 5, 3)

dep_delay = st.number_input("Departure Delay", min_value=0, value=0)
arr_delay = st.number_input("Arrival Delay", min_value=0, value=0)


if st.button("Predict Satisfaction"):

    sample = {
        "age": age,
        "distance": distance,

        "gender": gender,
        "customer_type": customer_type,
        "travel_type": travel_type,
        "class": flight_class,

        "wifi": wifi,
        "convenience": convenience,
        "booking": booking,
        "gate": gate,
        "food": food,
        "boarding": boarding,
        "seat": seat,
        "entertainment": entertainment,
        "onboard_service": onboard_service,
        "leg_room": leg_room,
        "baggage": baggage,
        "checkin": checkin,
        "inflight_service": inflight_service,
        "cleanliness": cleanliness,

        "dep_delay": dep_delay,
        "arr_delay": arr_delay
    }

    result = predict_satisfaction(sample)

    st.success(f"Prediction: {result}")