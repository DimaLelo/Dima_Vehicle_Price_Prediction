import streamlit as st
import pandas as pd
import numpy as np
import joblib



# Page configuration


st.set_page_config(
    page_title="Kenya Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)



# Load saved files


model = joblib.load("vehicle_price_model.pkl")
preprocessor = joblib.load("vehicle_preprocessor.pkl")
encoding_data = joblib.load("make_model_encoding.pkl")

make_model_means = encoding_data["means"]
global_mean = encoding_data["global_mean"]



# Title


st.title("🚗 Kenya Used Car Price Predictor")

st.write(
    "Enter the vehicle details below to estimate "
    "the expected selling price."
)

st.divider()



# Inputs

make = st.text_input(
    "Car Make",
    placeholder="e.g. Toyota"
)

model_name = st.text_input(
    "Car Model",
    placeholder="e.g. Corolla"
)

condition = st.selectbox(
    "Condition",
    ["Foreign Used", "Local Used", "New"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)

county = st.text_input(
    "County",
    placeholder="e.g. Nairobi"
)

yom = st.number_input(
    "Year of Manufacture",
    min_value=1960,
    max_value=2026,
    value=2018,
    step=1
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=1000000,
    value=80000,
    step=1000
)


st.divider()



# Prediction


if st.button("Predict Vehicle Price", type="primary"):

    if not make or not model_name or not county:

        st.warning(
            "Please enter the Make, Model and County."
        )

    else:

        # Feature engineering
        current_year = 2026

        car_age = current_year - yom

        log_mileage = np.log1p(mileage)

        mileage_per_year = mileage / max(car_age, 1)

        # Make + Model
        make_model = (
            str(make).strip()
            + "_"
            + str(model_name).strip()
        )

        # Target encoding
        make_model_te = make_model_means.get(
            make_model,
            global_mean
        )

        # Model input
        input_data = pd.DataFrame({
            "Car_Age": [car_age],
            "Log_Mileage": [log_mileage],
            "Mileage_Per_Year": [mileage_per_year],
            "MakeModel_TE": [make_model_te],
            "Condition": [condition],
            "Transmission": [transmission],
            "County": [county]
        })

        # Preprocess
        input_processed = preprocessor.transform(
            input_data
        )

        # Predict
        predicted_price = model.predict(
            input_processed
        )[0]

        # Display result
        st.success(
            f"Estimated Vehicle Price: "
            f"KES {predicted_price:,.0f}"
        )

        st.info(
            "Prediction generated using the "
            "XGBoost SMOTE-NC model."
        )
