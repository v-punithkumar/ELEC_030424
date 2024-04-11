import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

def main():
    st.title("Max Demand Analysis and Prediction")

    # Read Excel file
    excel_file = "combined_data_2013_to_2023.xlsx"
    xls = pd.ExcelFile(excel_file)

    # List of hardcoded states
    states = ["Punjab", "Haryana", "Rajasthan", "Delhi", "UP", "Uttarakhand", "HP", "J&K(UT) & Ladakh(UT)", "Chandigarh",
              "Railways_NR ISTS", "Chhattisgarh", "Gujarat", "MP", "Maharashtra", "Goa", "DNHDDPDCL", "AMNSIL", "BALCO",
              "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu", "Puducherry", "Bihar", "DVC", "Jharkhand",
              "Odisha", "West Bengal", "Sikkim", "Railways_ER ISTS", "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
              "Mizoram", "Nagaland", "Tripura"]

    # Select state for prediction
    selected_state = st.selectbox("Select State", states)

    # Get historical data for selected state
    historical_data = get_historical_data_for_state(xls, selected_state)

    # Print historical data for selected state
    st.write("Historical Data for " + selected_state + ":")
    st.write(historical_data)
    selected_column = historical_data.iloc[:, 3]  # Selecting all rows and the fourth column
    column_dtype = selected_column.dtype  # Getting the datatype of the selected column
    st.write("Datatype of the selected column:", column_dtype)

    # Train ARIMA model
    arima_model = train_arima_model(historical_data)

    # Train SARIMA model
    sarima_model = train_sarima_model(historical_data)

    # Train LSTM model
    lstm_model, scaler = train_lstm_model(historical_data)

    # Predict maximum demand for the next 3 months
    predicted_demand_arima = predict_demand_arima(arima_model)
    predicted_demand_sarima = predict_demand_sarima(sarima_model)
    predicted_demand_lstm = predict_demand_lstm(lstm_model, scaler, historical_data)

    # Display predicted demand
    st.write("Predicted Maximum Demand using ARIMA for the Next 3 Months:")
    st.write(predicted_demand_arima)
    st.write("Predicted Maximum Demand using SARIMA for the Next 3 Months:")
    st.write(predicted_demand_sarima)
    st.write("Predicted Maximum Demand using LSTM for the Next 3 Months:")
    st.write(predicted_demand_lstm)

def get_historical_data_for_state(xls, state):
    # Initialize an empty list to store data frames with dates
    data_with_dates = []

    # Iterate over each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        if state in df.values:
            # Add a new column with the date from the sheet name
            df['Date'] = pd.to_datetime(sheet_name, format='%d-%m-%y')
            # Append the modified DataFrame to the list
            data_with_dates.append(df[df.apply(lambda row: state in row.values, axis=1)])

    # Concatenate all DataFrames with dates
    state_data = pd.concat(data_with_dates)

    return state_data

def train_arima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Train ARIMA model
    arima_model = ARIMA(numeric_data, order=(5,1,0))
    arima_model_fit = arima_model.fit()

    return arima_model_fit

def train_sarima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Train SARIMA model
    sarima_model = SARIMAX(numeric_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()

    return sarima_model_fit

def train_lstm_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Remove NaN values
    numeric_data = numeric_data[~np.isnan(numeric_data)]

    # Reshape the data to have two dimensions
    numeric_data = np.array(numeric_data).reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(numeric_data)

    # Prepare data for LSTM model
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape data for LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    return model, scaler

def predict_demand_arima(model):
    # Predict maximum demand for the next 3 months using ARIMA
    forecast = model.forecast(steps=90)
    predicted_demand = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), value) for i, value in enumerate(forecast)]

    return predicted_demand

def predict_demand_sarima(model):
    # Predict maximum demand for the next 3 months using SARIMA
    forecast = model.forecast(steps=90)
    predicted_demand = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), value) for i, value in enumerate(forecast)]

    return predicted_demand

def predict_demand_lstm(model, scaler, data):
    # Get today's date
    today = datetime.date.today()

    # Initialize an empty list to store the predicted demands for each day
    predicted_demand = []

    # Iterate over each day in the next three months
    for i in range(90):
        next_day = today + datetime.timedelta(days=i)
        # Prepare the input data for prediction
        input_data = np.array([[next_day.year, next_day.month, next_day.day]])
        # Predict the maximum demand for the next day
        predicted_value_scaled = model.predict(input_data)
        # Inverse transform the scaled prediction to get the original scale
        predicted_value = scaler.inverse_transform(predicted_value_scaled)
        predicted_demand.append((next_day.strftime("%Y-%m-%d"), predicted_value[0][0]))

    return predicted_demand

if __name__ == "__main__":
    main()
