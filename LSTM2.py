import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from itertools import product

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

    # Checkbox to select all states
    select_all_states = st.checkbox("Select All States")

    if select_all_states:
        selected_states = states
    else:
        # Select states for prediction
        selected_states = st.multiselect("Select States", states)

    if not selected_states:
        st.warning("Please select at least one state.")
        return

    # Get historical data for selected states
    historical_data = pd.concat([get_historical_data_for_state(xls, state) for state in selected_states])

    # Wait for user to select the prediction interval
    button_choice = st.radio("Select Prediction Interval", ("Next 1 Month", "Next 3 Months", "Next 1 Year"), index=0)

    if button_choice == "Next 1 Month":
        predicted_demand_lstm = predict_demand_lstm(historical_data, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_lstm = predict_demand_lstm(historical_data, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_lstm = predict_demand_lstm(historical_data, steps=365)
    else:
        predicted_demand_lstm = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using LSTM for the {button_choice}:")

    if predicted_demand_lstm:
        # Sort states based on maximum demand
        sorted_states = sorted(predicted_demand_lstm.items(), key=lambda x: max(x[1]), reverse=True)

        # Plotting the predicted demand
        fig = go.Figure()

        for state, demand in sorted_states:
            dates = pd.date_range(start=datetime.date.today(), periods=len(demand))
            fig.add_trace(go.Scatter(x=dates, y=demand, mode='lines+markers', name=state))

        fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                          xaxis_title='Date',
                          yaxis_title='Max Demand')

        st.plotly_chart(fig)

    input_date = st.text_input("Enter a date (Format: DD-MM-YY)")

    if input_date:
        try:
            date_obj = datetime.datetime.strptime(input_date, "%d-%m-%y")
            week_start_date = date_obj - datetime.timedelta(days=date_obj.weekday())
            week_end_date = week_start_date + datetime.timedelta(days=6)

            st.write(f"Week starting from: {week_start_date.strftime('%Y-%m-%d')} to {week_end_date.strftime('%Y-%m-%d')}")

            # Check if historical data exists for the provided week
            week_data = historical_data[(historical_data['Date'] >= week_start_date) & (historical_data['Date'] <= week_end_date)]

            if not week_data.empty:
                max_demand_day = determine_max_demand_day_for_week(week_data)
                st.write("Date with Maximum Demand within the Week:")
                st.write(max_demand_day)
            else:
                st.warning(f"No historical data found for the week from {week_start_date} to {week_end_date}. Applying prediction method...")

        except ValueError:
            st.error("Please provide a valid date in the format DD-MM-YY.")

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

def predict_demand_lstm(data, steps):
    predicted_demand = {}
    for state in data['State'].unique():
        state_data = data[data['State'] == state]['Demand'].values
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        state_data_normalized = scaler.fit_transform(state_data.reshape(-1, 1))

        # Prepare the data for LSTM
        X, y = [], []
        for i in range(len(state_data_normalized) - steps):
            X.append(state_data_normalized[i:i+steps])
            y.append(state_data_normalized[i+steps])

        X = np.array(X)
        y = np.array(y)

        # Reshape the data for LSTM (samples, timesteps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Fit the model
        model.fit(X, y, epochs=100, batch_size=32, verbose=0)

        # Predict maximum demand for the specified number of steps
        forecast = model.predict(X[-1].reshape(1, steps, 1))

        # Inverse transform the forecasted values
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

        # Generate dates for the forecasted steps
        forecast_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, steps+1)]

        predicted_demand[state] = list(zip(forecast_dates, forecast.flatten()))

    return predicted_demand

def determine_max_demand_day_for_week(data):
    max_demand_day = None
    max_demand = 0

    for date, value in zip(data['Date'], data['Demand']):
        if value > max_demand:
            max_demand = value
            max_demand_day = date.strftime("%Y-%m-%d")

    return max_demand_day

if __name__ == "__main__":
    main()
