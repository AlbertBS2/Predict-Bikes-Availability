# Frontend code for the Streamlit app
import streamlit as st
import datetime
import requests
import openmeteo_requests


st.set_page_config(
    page_title="Spot Prediction",
    page_icon="🎯",
)

st.title("Spot Prediction of Bikes Availability in Washington DC")

## Feature inputs
day = st.date_input("Day")
day_of_week = day.weekday()
month = day.month

hour_of_day = st.slider(
    "Hour of the day",
    min_value=0,
    max_value=23,
    value=datetime.datetime.now().hour+1,
)

# Setup the Open-Meteo API client
openmeteo = openmeteo_requests.Client()

# Make the Open-Meteo API call with Washington DC params
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 38.89511,
	"longitude": -77.03637,
	"hourly": ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "rain", "snow_depth", "visibility", "cloud_cover", "wind_speed_180m"],
	#"timezone": "America/New_York",
	"start_date": str(day),
	"end_date": str(day),
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# Process hourly data
hourly = response.Hourly()
temp = hourly.Variables(0).Values(hour_of_day)
dew = hourly.Variables(1).Values(hour_of_day)
humidity = hourly.Variables(2).Values(hour_of_day)
precip = hourly.Variables(3).Values(hour_of_day)
snowdepth = hourly.Variables(4).Values(hour_of_day)
visibility = hourly.Variables(5).Values(hour_of_day)
cloudcover = hourly.Variables(6).Values(hour_of_day)
windspeed = hourly.Variables(7).Values(hour_of_day)

# Display the input fields with the fetched data
temp = st.number_input("Temperature (ºC)", value=temp, disabled=True)
dew = st.number_input("Dew", value=dew, disabled=True)
humidity = st.number_input("Humidity", value=humidity, disabled=True)
snowdepth = st.checkbox("Snow", value=snowdepth, disabled=True)
precip = st.checkbox("Rain", value=precip, disabled=True)
windspeed = st.number_input("Wind speed", value=windspeed, disabled=True)
cloudcover = st.number_input("Cloud cover", value=cloudcover, disabled=True)
visibility = st.number_input("Visibility", value=visibility, disabled=True)

# Create a dictionary with the input data
input_data = {
    "hour_of_day": [hour_of_day],
    "day_of_week": [day_of_week],
    "month": [month],
    "weekday": [1 if day_of_week < 5 else 0],
    "summertime": [1 if month in [5, 6, 7, 8] else 0],
    "temp": [temp],
    "dew": [dew],
    "humidity": [humidity],
    "precip": [1 if precip else 0],
    "snowdepth": [1 if snowdepth else 0],
    "windspeed": [windspeed],
    "cloudcover": [cloudcover],
    "visibility": [visibility]
}

if st.button("Predict"):

    # Send a POST request to the model API endpoint to obtain the prediction
    #response = requests.post("http://localhost:80/predict", json=input_data) # If running without Docker
    response = requests.post("http://ml-api:80/predict", json=input_data) # If running with Docker

    prediction = response.json()["prediction"][0]
    proba_0, proba_1 = response.json()["predict_proba"][0]
    
    # Display prediction
    st.subheader("Prediction")
    if prediction == 0:
        st.write(f"Low bike demand ({(proba_0 * 100):.2f}%)")
    else:
        st.write(f"High bike demand ({(proba_1 * 100):.2f}%)")
