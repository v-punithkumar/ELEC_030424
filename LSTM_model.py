import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

    # Train LSTM model
    lstm_model, scaler = train_lstm_model(historical_data)

    # Buttons for prediction duration
    prediction_duration = st.radio("Select Prediction Duration", ["1 Month", "3 Months", "1 Year"])

    if st.button("Predict"):
        if prediction_duration == "1 Month":
            predicted_demand_lstm = predict_demand_lstm(lstm_model, scaler, historical_data, 30)
        elif prediction_duration == "3 Months":
            predicted_demand_lstm = predict_demand_lstm(lstm_model, scaler, historical_data, 90)
        elif prediction_duration == "1 Year":
            predicted_demand_lstm = predict_demand_lstm(lstm_model, scaler, historical_data, 365)

        # Display predicted demand using LSTM
        st.write(f"Predicted Maximum Demand using LSTM for the Next {prediction_duration}:")
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
    model.fit(X_train, y_train, epochs=17, batch_size=32)

    return model, scaler

def predict_demand_lstm(model, scaler, data, days):
    # Get today's date
    today = datetime.date.today()

    # Initialize an empty list to store the predicted demands for each day
    predicted_demand = []

    # Iterate over each day in the next 'days'
    for i in range(days):
        next_day = today + datetime.timedelta(days=i)
        # Prepare the input data for prediction
        input_data = np.array([[next_day.year, next_day.month, next_day.day]])
        # Predict the maximum demand for the next day
        predicted_value_scaled = model.predict(input_data)
        # Inverse transform the scaled prediction to get the original scale
        predicted_value = scaler.inverse_transform(predicted_value_scaled)
        predicted_demand.append((next_day.strftime("%Y-%m-%d"), round(predicted_value[0][0])))

    return predicted_demand

if __name__ == "__main__":
    main()
