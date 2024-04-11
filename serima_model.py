import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    

    # Train SARIMA model
    sarima_model = train_sarima_model(historical_data)

    # Predict maximum demand for the next 3 months using SARIMA
    predicted_demand_sarima = predict_demand_sarima(sarima_model)

    # Display predicted demand
    st.write("Predicted Maximum Demand using SARIMA for the Next 3 Months:")
    st.write(predicted_demand_sarima)

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

def train_sarima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Train SARIMA model
    sarima_model = SARIMAX(numeric_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()

    return sarima_model_fit

def predict_demand_sarima(model):
    # Predict maximum demand for the next 3 months using SARIMA
    forecast = model.forecast(steps=90)
    predicted_demand = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()
