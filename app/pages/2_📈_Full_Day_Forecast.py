# Frontend code for the Streamlit app
import streamlit as st
import requests
import openmeteo_requests
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Full Day Forecast",
    page_icon="📈",
)

st.title("Full Day Forecast of Bikes Availability in Washington DC")

# Feature inputs
day = st.date_input("Day")
day_of_week = day.weekday()
month = day.month

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
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy().tolist()
hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy().tolist()
hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy().tolist()
hourly_rain = hourly.Variables(3).ValuesAsNumpy().tolist()
hourly_snow_depth = hourly.Variables(4).ValuesAsNumpy().tolist()
hourly_visibility = hourly.Variables(5).ValuesAsNumpy().tolist()
hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy().tolist()
hourly_wind_speed_180m = hourly.Variables(7).ValuesAsNumpy().tolist()

# Create a dictionary with the input data
input_data = {
    "hour_of_day": [i for i in range(24)],
    "day_of_week": [day_of_week] * 24,
    "month": [month] * 24,
    "weekday": [1 if day_of_week < 5 else 0] * 24,
    "summertime": [1 if month in [5, 6, 7, 8] else 0] * 24,
    "temp": hourly_temperature_2m,
    "dew": hourly_dew_point_2m,
    "humidity": hourly_relative_humidity_2m,
    "precip": [1 if rain else 0 for rain in hourly_rain],
    "snowdepth": [1 if snow else 0 for snow in hourly_snow_depth],
    "windspeed": hourly_wind_speed_180m,
    "cloudcover": hourly_cloud_cover,
    "visibility": hourly_visibility
}

if st.button("Predict"):

    # Send a POST request to the model API endpoint to obtain the prediction
    #response = requests.post("http://localhost:80/predict", json=input_data) # If running without Docker
    response = requests.post("http://ml-api:80/predict", json=input_data) # If running with Docker

    predictions = response.json()["prediction"]
    probabilities = response.json()["predict_proba"]

    # Plot probability of high bike demand against hour of the day
    hours = range(24)
    high_prob_values = [proba[1] for proba in probabilities]

    fig, ax = plt.subplots()
    ax.plot( 
        hours,
        high_prob_values
    )

    ax.fill_between(
        hours,
        high_prob_values,
        color='skyblue',
        alpha=0.5
    )

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)

    ax.set_title("Probability of High Bike Demand by Hour")
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Probability of High Bike Demand (%)")

    st.pyplot(fig)

    # Display prediction in a table
    df_pred = pd.DataFrame({
        "Hour": [f"{i}:00" for i in range(24)],
        "Prediction": predictions,
        "Probability": probabilities
    })

    st.dataframe(df_pred, hide_index=True)
