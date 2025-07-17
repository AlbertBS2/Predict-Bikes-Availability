# Frontend code for the Streamlit app
import streamlit as st
import datetime
import requests


st.title("Do we need more bikes in Washington DC?")

## Features selection
# Month selector
with st.expander("Month"):
    months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_str = st.radio(
        "",
        months_list,
        index=datetime.date.today().month - 1,
        horizontal=True
    )
    month = months_list.index(month_str) + 1

# Day of week selector
with st.expander("Day"):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_str = st.radio(
        "",
        days,
        horizontal=True,
        index=datetime.date.today().weekday()+1,
    )
    day_of_week = days.index(day_str)

hour_of_day = st.slider(
    "Hour of the day",
    min_value=0,
    max_value=23
)

temp = st.number_input("Temperature (ºC)")
dew = st.number_input("Dew")
humidity = st.number_input("Humidity")
snowdepth = st.checkbox("Snow")
precip = st.checkbox("Rain")
windspeed = st.number_input("Wind speed")
cloudcover = st.number_input("Cloud cover")
visibility = st.number_input("Visibility")

# Create a dictionary with the input data
input_data = {
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "month": month,
    "weekday": 1 if day_of_week < 5 else 0,
    "summertime": 1 if month in [5, 6, 7, 8] else 0,
    "temp": temp,
    "dew": dew,
    "humidity": humidity,
    "precip": 1 if precip else 0,
    "snowdepth": 1 if snowdepth else 0,
    "windspeed": windspeed,
    "cloudcover": cloudcover,
    "visibility": visibility
}

if st.button("Predict"):

    # Send a POST request to the model API endpoint to obtain the prediction
    #response = requests.post("http://localhost:80/predict", json=input_data) # If running without Docker
    response = requests.post("http://ml-api:80/predict", json=input_data) # If running with Docker
    prediction = response.json()["prediction"]
    proba_0, proba_1 = response.json()["predict_proba"]
    
    # Display prediction
    st.subheader("Prediction")
    if prediction == 0:
        st.write(f"Low bike demand ({(proba_0 * 100):.2f}%)")
    else:
        st.write(f"High bike demand ({(proba_1 * 100):.2f}%)")
