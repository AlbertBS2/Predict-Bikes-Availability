# Frontend code for the Streamlit app
import streamlit as st
import datetime
import requests
import openmeteo_requests
import pandas as pd


st.title("Do we need more bikes in Washington DC?")

# ## Features selection
# # Month selector
# with st.expander("Month"):
#     months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
#                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#     month_str = st.radio(
#         "",
#         months_list,
#         index=datetime.date.today().month - 1,
#         horizontal=True
#     )
#     month = months_list.index(month_str) + 1

# # Day of week selector
# with st.expander("Day"):
#     days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
#     day_str = st.radio(
#         "",
#         days,
#         horizontal=True,
#         index=datetime.date.today().weekday()+1,
#     )
#     day_of_week = days.index(day_str)

# hour_of_day = st.slider(
#     "Hour of the day",
#     min_value=0,
#     max_value=23
# )

# temp = st.number_input("Temperature (ºC)")
# dew = st.number_input("Dew")
# humidity = st.number_input("Humidity")
# snowdepth = st.checkbox("Snow")
# precip = st.checkbox("Rain")
# windspeed = st.number_input("Wind speed")
# cloudcover = st.number_input("Cloud cover")
# visibility = st.number_input("Visibility")



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


# st.write(input_data)



if st.button("Predict"):

    # Send a POST request to the model API endpoint to obtain the prediction
    response = requests.post("http://localhost:80/predict", json=input_data) # If running without Docker
    #response = requests.post("http://ml-api:80/predict", json=input_data) # If running with Docker
    prediction = response.json()["prediction"]
    proba_0, proba_1 = response.json()["predict_proba"]
    
    # Display prediction
    st.subheader("Prediction")
    if prediction == 0:
        st.write(f"Low bike demand ({(proba_0 * 100):.2f}%)")
    else:
        st.write(f"High bike demand ({(proba_1 * 100):.2f}%)")






# # Setup the Open-Meteo API client
# openmeteo = openmeteo_requests.Client()

# # Make the Open-Meteo API call with Washington DC params
# url = "https://api.open-meteo.com/v1/forecast"
# params = {
# 	"latitude": 38.89511,
# 	"longitude": -77.03637,
# 	"hourly": ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "rain", "snow_depth", "visibility", "cloud_cover", "wind_speed_180m"],
# 	#"timezone": "America/New_York",
# 	"start_date": str(day),
# 	"end_date": str(day),
# }
# responses = openmeteo.weather_api(url, params=params)
# response = responses[0]

# # Process hourly data. The order of variables needs to be the same as requested.
# hourly = response.Hourly()
# hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
# hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
# hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
# hourly_rain = hourly.Variables(3).ValuesAsNumpy()
# hourly_snow_depth = hourly.Variables(4).ValuesAsNumpy()
# hourly_visibility = hourly.Variables(5).ValuesAsNumpy()
# hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
# hourly_wind_speed_180m = hourly.Variables(7).ValuesAsNumpy()

# hourly_data = {"date": pd.date_range(
# 	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
# 	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
# 	freq = pd.Timedelta(seconds = hourly.Interval()),
# 	inclusive = "left"
# )}

# hourly_data["temperature_2m"] = hourly_temperature_2m
# hourly_data["dew_point_2m"] = hourly_dew_point_2m
# hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
# hourly_data["rain"] = hourly_rain
# hourly_data["snow_depth"] = hourly_snow_depth
# hourly_data["visibility"] = hourly_visibility
# hourly_data["cloud_cover"] = hourly_cloud_cover
# hourly_data["wind_speed_180m"] = hourly_wind_speed_180m

# hourly_dataframe = pd.DataFrame(data = hourly_data)