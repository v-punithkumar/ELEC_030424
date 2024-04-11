""" import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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
    

    # Train ARIMA model
    arima_model = train_arima_model(historical_data)

    # Predict maximum demand for the next 3 months
    predicted_demand_arima = predict_demand_arima(arima_model)

    # Display predicted demand
    st.write("Predicted Maximum Demand using ARIMA for the Next 3 Months:")
    st.write(predicted_demand_arima)

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

def predict_demand_arima(model):
    # Predict maximum demand for the next 3 months using ARIMA
    forecast = model.forecast(steps=90)
    predicted_demand = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()
 """
""" 
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plotly.graph_objs as go

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

    # Select states for prediction
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

    # Train ARIMA model
    arima_models = {state: train_arima_model(get_historical_data_for_state(xls, state)) for state in selected_states}

    if button_choice == "Next 1 Month":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=365)
    else:
        predicted_demand_arima = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using ARIMA for the {button_choice}:")
    # st.write(predicted_demand_arima)

    if predicted_demand_arima:
        # Plotting the predicted demand
        graph_type = st.checkbox("Show as Bar Graph", value=False)
        if graph_type:
            # Plot as a bar graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Bar(x=dates, y=values, name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')
        else:
            # Plot as a line graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')

        st.plotly_chart(fig)

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

def predict_demand_arima(models, steps):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using ARIMA
        forecast = model.forecast(steps=steps)
        predicted_demand[state] = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()

 """
""" import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plotly.graph_objs as go

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

    # Select states for prediction
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

    # Train ARIMA model
    arima_models = {state: train_arima_model(get_historical_data_for_state(xls, state)) for state in selected_states}

    if button_choice == "Next 1 Month":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=365)
    else:
        predicted_demand_arima = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using ARIMA for the {button_choice}:")
    # st.write(predicted_demand_arima)

    if predicted_demand_arima:
        # Plotting the predicted demand
        graph_type = st.checkbox("Show as Bar Graph", value=False)
        if graph_type:
            # Plot as a bar graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Bar(x=dates, y=values, name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')
        else:
            # Plot as a line graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')

        st.plotly_chart(fig)

    input_date = st.text_input("Enter a date (Format: DD-MM-YY)")

    if input_date:
        try:
            date_obj = datetime.datetime.strptime(input_date, "%d-%m-%y")
            week_start_date = date_obj - datetime.timedelta(days=date_obj.weekday())
            week_end_date = week_start_date + datetime.timedelta(days=6)

            if week_start_date.year < 2000:
                st.error("Please provide a valid date after the year 2000.")
                return

            if week_end_date.year > 2023:
                st.error("Please provide a date within the range of available data (before 2024).")
                return

            st.write(f"Week starting from: {week_start_date.strftime('%Y-%m-%d')} to {week_end_date.strftime('%Y-%m-%d')}")
            max_demand_day = determine_max_demand_day_for_week(predicted_demand_arima, week_start_date, week_end_date)
            st.write("Date with Maximum Demand within the Week:")
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

def train_arima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Train ARIMA model
    arima_model = ARIMA(numeric_data, order=(5,1,0))
    arima_model_fit = arima_model.fit()

    return arima_model_fit

def predict_demand_arima(models, steps):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using ARIMA
        forecast = model.forecast(steps=steps)
        predicted_demand[state] = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

def determine_max_demand_day_for_week(predicted_demand_arima, week_start_date, week_end_date):
    max_demand_day = None
    max_demand = 0

    for state, demand in predicted_demand_arima.items():
        for date, value in demand:
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
            if week_start_date <= date_obj <= week_end_date:
                if value > max_demand:
                    max_demand = value
                    max_demand_day = date

    return max_demand_day

if __name__ == "__main__":
    main()
 """


""" import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plotly.graph_objs as go

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

    # Select states for prediction
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

    # Train ARIMA model
    arima_models = {state: train_arima_model(get_historical_data_for_state(xls, state)) for state in selected_states}

    if button_choice == "Next 1 Month":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=365)
    else:
        predicted_demand_arima = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using ARIMA for the {button_choice}:")
    # st.write(predicted_demand_arima)

    if predicted_demand_arima:
        # Plotting the predicted demand
        graph_type = st.checkbox("Show as Bar Graph", value=False)
        if graph_type:
            # Plot as a bar graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Bar(x=dates, y=values, name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')
        else:
            # Plot as a line graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')

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
                # Apply prediction method
                week_start_date_str = week_start_date.strftime("%d-%m-%y")
                week_end_date_str = week_end_date.strftime("%d-%m-%y")
                st.warning(f"No historical data found for the week from {week_start_date_str} to {week_end_date_str}. Applying prediction method...")

                # Predict maximum demand for the week
                week_demand_prediction = predict_demand_for_week(arima_models, week_start_date, week_end_date)
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

def train_arima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Train ARIMA model
    arima_model = ARIMA(numeric_data, order=(5,1,0))
    arima_model_fit = arima_model.fit()

    return arima_model_fit

def predict_demand_arima(models, steps):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using ARIMA
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
        # Predict maximum demand for the specified number of days using ARIMA
        forecast = model.forecast(steps=(week_end_date - week_start_date).days + 1)
        predicted_demand[state] = [((week_start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()
 """

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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

      # Select states for prediction
    states_selection = st.multiselect("Select States", ["All States"] + states)

    selected_states = states if "All States" in states_selection else states_selection

    if not selected_states:
        st.warning("Please select at least one state.")
        return

    # Get historical data for selected states
    historical_data = pd.concat([get_historical_data_for_state(xls, state) for state in selected_states])

    # Print historical data for selected states
    st.write("Historical Data for Selected States:")
    # st.write(historical_data)

    button_choice = st.radio("Select Prediction Interval", ("Next 1 Month", "Next 3 Months", "Next 1 Year"))

    # Train ARIMA models with grid search
    arima_models = {state: train_arima_model(get_historical_data_for_state(xls, state)) for state in selected_states}

    if button_choice == "Next 1 Month":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=30)
    elif button_choice == "Next 3 Months":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=90)
    elif button_choice == "Next 1 Year":
        predicted_demand_arima = predict_demand_arima(arima_models, steps=365)
    else:
        predicted_demand_arima = []

    # Display predicted demand
    st.write(f"Predicted Maximum Demand using ARIMA for the {button_choice}:")
    # st.write(predicted_demand_arima)

    if predicted_demand_arima:
        # Plotting the predicted demand
        graph_type = st.checkbox("Show as Bar Graph", value=False)
        if graph_type:
            # Plot as a bar graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Bar(x=dates, y=values, name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')
        else:
            # Plot as a line graph
            fig = go.Figure()
            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=state))

            fig.update_layout(title=f'Predicted Maximum Demand for {button_choice}',
                              xaxis_title='Date',
                              yaxis_title='Predicted Maximum Demand',
                              xaxis=dict(tickangle=-45),
                              hovermode='x')

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
                # Apply prediction method
                week_start_date_str = week_start_date.strftime("%d-%m-%y")
                week_end_date_str = week_end_date.strftime("%d-%m-%y")
                st.warning(f"No historical data found for the week from {week_start_date_str} to {week_end_date_str}. Applying prediction method...")

                # Predict maximum demand for the week
                week_demand_prediction = predict_demand_for_week(arima_models, week_start_date, week_end_date)
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

def train_arima_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Grid search for ARIMA parameters
    best_aic = np.inf
    best_order = None

    p_values = range(0, 3)  # Change the range as needed
    d_values = range(0, 3)  # Change the range as needed
    q_values = range(0, 3)  # Change the range as needed

    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(numeric_data, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except:
            continue

    st.write(f"Best ARIMA Order for {data.name}: {best_order}")

    # Train ARIMA model with the best order
    arima_model = ARIMA(numeric_data, order=best_order)
    arima_model_fit = arima_model.fit()

    return arima_model_fit

def predict_demand_arima(models, steps):
    predicted_demand = {}
    for state, model in models.items():
        # Predict maximum demand for the specified number of days using ARIMA
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
        # Predict maximum demand for the specified number of days using ARIMA
        forecast = model.forecast(steps=(week_end_date - week_start_date).days + 1)
        predicted_demand[state] = [((week_start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

if __name__ == "__main__":
    main()
