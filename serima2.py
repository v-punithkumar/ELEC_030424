""" import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
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

    # Option to select individual states or all states
    select_all_states = st.checkbox("Select All States")
    if select_all_states:
        selected_states = states
    else:
        selected_states = st.multiselect("Select States", states)

    if not selected_states:
        st.warning("Please select at least one state.")
        return

    # Get historical data for selected states
    historical_data = pd.concat([get_historical_data_for_state(xls, state) for state in selected_states])

    # Print historical data for selected states
    st.write("Historical Data for Selected States:")
    # st.write(historical_data)

    button_choice = st.radio("Select Prediction Interval", ("Next 1 Month", "Next 3 Months", "Next 1 Year"))

    # Train SARIMA model for each selected state
    sarima_models = {state: train_sarima_model(get_historical_data_for_state(xls, state)) for state in selected_states}

    if button_choice == "Next 1 Month":
        predicted_demand_sarima = predict_demand_sarima(sarima_models, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_sarima = predict_demand_sarima(sarima_models, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_sarima = predict_demand_sarima(sarima_models, steps=365)
    else:
        predicted_demand_sarima = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using SARIMA for the {button_choice}:")
    # st.write(predicted_demand_sarima)

    if predicted_demand_sarima:
        # Display predicted demand
        st.write(f"Predicted Maximum Demand using SARIMA for the {button_choice}:")
        # st.write(predicted_demand_sarima)

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

    # Grid search for SARIMA parameters
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    p_values = range(0, 3)  # Change the range as needed
    d_values = range(0, 3)  # Change the range as needed
    q_values = range(0, 3)  # Change the range as needed
    seasonal_p_values = range(0, 3)  # Change the range as needed
    seasonal_d_values = range(0, 3)  # Change the range as needed
    seasonal_q_values = range(0, 3)  # Change the range as needed
    seasonal_periods = [12]  # Change the seasonal period as needed

    for order in product(p_values, d_values, q_values):
        for seasonal_order in product(seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_periods):
            try:
                model = SARIMAX(numeric_data, order=order, seasonal_order=seasonal_order)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue

    st.write(f"Best SARIMA Order for {data.columns[3]}: {best_order}")
    st.write(f"Best SARIMA Seasonal Order for {data.columns[3]}: {best_seasonal_order}")

    # Train SARIMA model with the best order
    sarima_model = SARIMAX(numeric_data, order=best_order, seasonal_order=best_seasonal_order)
    sarima_model_fit = sarima_model.fit()

    return sarima_model_fit

def predict_demand_sarima(models, steps):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using SARIMA
        forecast = model.forecast(steps=steps)
        predicted_demand[state] = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()
 """


import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import plotly.graph_objs as go
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

    # Train SARIMA models with grid search
    sarima_models = {state: train_sarima_model(get_historical_data_for_state(xls, state)) for state in selected_states}

    if button_choice == "Next 1 Month":
        predicted_demand_sarima = predict_demand_sarima(sarima_models, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_sarima = predict_demand_sarima(sarima_models, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_sarima = predict_demand_sarima(sarima_models, steps=365)
    else:
        predicted_demand_sarima = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using SARIMA for the {button_choice}:")

    if predicted_demand_sarima:
        # Plotting the predicted demand
        fig = go.Figure()

        for state, demand in predicted_demand_sarima.items():
            if state in selected_states:
                dates, values = zip(*demand)
                fig.add_trace(go.Scatter(x=values, y=[state]*len(values), mode='markers', name=state, 
                                         hoverinfo='text', hovertext=[f"State: {state}<br>Max Demand: {value}" for value in values]))

        fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                          xaxis_title='Max Demand',
                          yaxis_title='State')

        st.plotly_chart(fig)

        # Provide option to switch between line and bar graph
        graph_type = st.radio("Select Graph Type", ("Line Graph", "Bar Graph"), index=0)

        if graph_type == "Bar Graph":
            fig_bar = go.Figure()

            for state, demand in predicted_demand_sarima.items():
                if state in selected_states:
                    dates, values = zip(*demand)
                    fig_bar.add_trace(go.Bar(x=values, y=[state]*len(values), name=state, 
                                             hoverinfo='text', hovertext=[f"State: {state}<br>Max Demand: {value}" for value in values]))

            fig_bar.update_layout(title=f'Predicted Maximum Demand for {button_choice} (Bar Graph)',
                                  xaxis_title='Max Demand',
                                  yaxis_title='State')

            st.plotly_chart(fig_bar)

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
                # Apply prediction method
                week_start_date_str = week_start_date.strftime("%d-%m-%y")
                week_end_date_str = week_end_date.strftime("%d-%m-%y")
                st.warning(f"No historical data found for the week from {week_start_date_str} to {week_end_date_str}. Applying prediction method...")

                # Predict maximum demand for the week
                week_demand_prediction = predict_demand_for_week(sarima_models, week_start_date, week_end_date)
                max_demand_day = max(week_demand_prediction, key=lambda x: x[1])[0]
                st.write("Predicted Date with Maximum Demand within the Week:")
                st.write(max_demand_day)

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

def train_sarima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Grid search for SARIMA parameters
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    p_values = range(0, 3)  # Change the range as needed
    d_values = range(0, 3)  # Change the range as needed
    q_values = range(0, 3)  # Change the range as needed
    P_values = range(0, 3)  # Change the range as needed
    D_values = range(0, 3)  # Change the range as needed
    Q_values = range(0, 3)  # Change the range as needed
    m_values = [12]  # Change the seasonal period as needed

    for (p, d, q, P, D, Q, m) in product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
        try:
            model = SARIMAX(numeric_data, order=(p, d, q), seasonal_order=(P, D, Q, m))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, m)
        except:
            continue

    st.write(f"Best SARIMA Order: {best_order}, Seasonal Order: {best_seasonal_order}")

    # Train SARIMA model with the best order and seasonal order
    sarima_model = SARIMAX(numeric_data, order=best_order, seasonal_order=best_seasonal_order)
    sarima_model_fit = sarima_model.fit()

    return sarima_model_fit

def predict_demand_sarima(models, steps):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using SARIMA
        forecast = model.forecast(steps=steps)
        predicted_demand[state] = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

def determine_max_demand_day_for_week(data):
    max_demand_day = None
    max_demand = 0

    for date, value in zip(data['Date'], data['Demand']):
        if value > max_demand:
            max_demand = value
            max_demand_day = date.strftime("%Y-%m-%d")

    return max_demand_day

def predict_demand_for_week(models, week_start_date, week_end_date):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using SARIMA
        forecast = model.forecast(steps=(week_end_date - week_start_date).days + 1)
        predicted_demand[state] = [((week_start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()
