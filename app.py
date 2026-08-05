import streamlit as st
import pandas as pd
import numpy as np
import joblib



# PAGE CONFIGURATION

st.set_page_config(
    page_title="Kenya Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)



# LOAD TRAINED MODEL AND PREPROCESSING FILES

@st.cache_resource
def load_models():

    model = joblib.load("vehicle_price_model.pkl")

    preprocessor = joblib.load(
        "vehicle_preprocessor.pkl"
    )

    encoding_data = joblib.load(
        "make_model_encoding.pkl"
    )

    return model, preprocessor, encoding_data


model, preprocessor, encoding_data = load_models()



# TARGET ENCODING DATA

make_model_means = encoding_data["means"]
global_mean = encoding_data["global_mean"]



# PAGE TITLE

st.title("🚗 Kenya Used Car Price Predictor")

st.write(
    "Enter the details of a used vehicle to estimate "
    "its selling price in Kenyan Shillings (KES)."
)

st.divider()



# VEHICLE INFORMATION


st.subheader("Vehicle Information")


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
    [
        "Foreign Used",
        "Local Used",
    ]
)


transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual"
    ]
)


county = st.text_input(
    "County",
    placeholder="e.g. Nairobi"
)



# VEHICLE AGE AND MILEAGE


st.subheader("Vehicle Details")


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



# PREDICTION

if st.button(
    "Predict Vehicle Price",
    type="primary",
    use_container_width=True
):

  
    # Validate required inputs
    

    if not make.strip():

        st.warning("Please enter the car make.")

        st.stop()


    if not model_name.strip():

        st.warning("Please enter the car model.")

        st.stop()


    if not county.strip():

        st.warning("Please enter the county.")

        st.stop()


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    

    CURRENT_YEAR = 2026

    car_age = CURRENT_YEAR - yom

    log_mileage = np.log1p(mileage)

    mileage_per_year = (
        mileage / max(car_age, 1)
    )


  
    # CREATE MAKE + MODEL FEATURE
    
    make_clean = make.strip()

    model_clean = model_name.strip()

    make_model = (
        make_clean
        + "_"
        + model_clean
    )


    
    # MAKE + MODEL TARGET ENCODING
   

    make_model_te = make_model_means.get(
        make_model,
        global_mean
    )


    
    # CREATE INPUT DATAFRAME
    

    input_data = pd.DataFrame({

        "Car_Age": [
            car_age
        ],

        "Log_Mileage": [
            log_mileage
        ],

        "Mileage_Per_Year": [
            mileage_per_year
        ],

        "MakeModel_TE": [
            make_model_te
        ],

        "Condition": [
            condition
        ],

        "Transmission": [
            transmission
        ],

        "County": [
            county.strip()
        ]
    })


   
    # APPLY THE SAVED PREPROCESSOR
    

    try:

        input_processed = preprocessor.transform(
            input_data
        )

    except Exception as e:

        st.error(
            "There was a problem processing the vehicle "
            "information."
        )

        st.exception(e)

        st.stop()


    
    # MODEL PREDICTION
   

    try:

        predicted_log_price = model.predict(
            input_processed
        )[0]

        predicted_price = np.expm1(
            predicted_log_price
        )

    except Exception as e:

        st.error(
            "There was a problem generating the prediction."
        )

        st.exception(e)

        st.stop()


    
    
# SAFETY CHECK
    
    # -------------------------------------------

    if not np.isfinite(predicted_price):

        st.error(
            "The model returned an invalid prediction."
        )

        st.stop()


    if predicted_price < 0:

        predicted_price = 0
   
    # DISPLAY PREDICTION
    
    st.success(
        f"Estimated Vehicle Price: "
        f"KES {predicted_price:,.0f}"
    )


    # SHOW INPUT SUMMARY
    

    st.divider()

    st.subheader("Prediction Details")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Make:** {make_clean}"
        )

        st.write(
            f"**Model:** {model_clean}"
        )

        st.write(
            f"**Condition:** {condition}"
        )

        st.write(
            f"**Transmission:** {transmission}"
        )

    with col2:

        st.write(
            f"**County:** {county.strip()}"
        )

        st.write(
            f"**Year:** {yom}"
        )

        st.write(
            f"**Mileage:** {mileage:,.0f} km"
        )

        st.write(
            f"**Vehicle Age:** {car_age} years"
        )


    st.info(
        "The prediction is generated using the "
        "XGBoost (SMOTE-NC) model trained in the "
        "vehicle price prediction notebook."
    )
