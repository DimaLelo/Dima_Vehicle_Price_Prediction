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



# VISUAL IDENTITY — "NAIROBI DUSK"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    :root{
        --bg:            #12181a;
        --surface:       #1a2225;
        --surface-2:     #212b2e;
        --border:        #2c3639;
        --text:          #f2eee4;
        --text-muted:    #93a3a1;
        --accent:        #e8a33d;
        --accent-soft:   #e8a33d33;
        --accent-2:      #2f8f7a;
        --accent-2-soft: #2f8f7a2e;
    }

    html, body, [class*="css"]{
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp{
        background:
            radial-gradient(circle at 15% 0%, #1c2528 0%, transparent 45%),
            repeating-linear-gradient(
                115deg,
                #ffffff05 0px, #ffffff05 1px,
                transparent 1px, transparent 34px
            ),
            var(--bg);
    }

    /* headings use the display face */
    h1, h2, h3{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    h1{
        font-weight: 700 !important;
        font-size: clamp(1.8rem, 4vw, 2.4rem) !important;
        background: linear-gradient(100deg, var(--text) 35%, var(--accent) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h1 + div p{
        color: var(--text-muted) !important;
        font-size: 1.02rem !important;
    }

    .hero-badge{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid var(--accent);
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        margin-bottom: 0.7rem;
    }

    /* section headings styled with a left tick, like a route marker */
    h2, h3{
        font-weight: 600 !important;
        padding-left: 0.7rem;
        border-left: 3px solid var(--accent);
        margin-top: 1.6rem !important;
    }

    /* signature: dashed road-line divider, replacing st.divider() */
    hr{
        border: none !important;
        height: 0 !important;
        margin: 1.6rem 0 !important;
        background: none !important;
        position: relative;
    }
    hr::before{
        content: "";
        display: block;
        height: 2px;
        background-image: repeating-linear-gradient(
            90deg,
            var(--accent) 0px, var(--accent) 10px,
            transparent 10px, transparent 20px
        );
        opacity: 0.55;
    }

    /* eyebrow tag helper */
    .eyebrow{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: var(--accent-2);
        background: var(--accent-2-soft);
        border: 1px solid var(--accent-2);
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        margin-bottom: 0.4rem;
    }

    /* inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div{
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    label[data-testid="stWidgetLabel"] p{
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* dropdown popover (select options list) — not covered by the
       closed-state selector since it renders in a portal */
    ul[data-baseweb="menu"]{
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    ul[data-baseweb="menu"] li{
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }
    ul[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] li[aria-selected="true"]{
        background-color: var(--accent-2-soft) !important;
    }

    /* number input steppers */
    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"]{
        background-color: var(--surface-2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    button[data-testid="stNumberInputStepUp"]:hover,
    button[data-testid="stNumberInputStepDown"]:hover{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* primary button */
    div[data-testid="stButton"] button[kind="primary"]{
        background: var(--accent) !important;
        color: #191008 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.01em;
        padding: 0.65rem 0 !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px var(--accent-soft);
    }

    /* result banner */
    div[data-testid="stAlertContentSuccess"]{
        background: var(--accent-2-soft) !important;
        border: 1px solid var(--accent-2) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlertContentInfo"]{
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlertContentWarning"],
    div[data-testid="stAlertContentError"]{
        border-radius: 10px !important;
    }

    /* price display card */
    .price-card{
        background: var(--surface);
        border: 1px solid var(--accent);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 0.6rem 0 1rem 0;
        animation: rise 0.35s ease;
    }
    .price-label{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        color: var(--text-muted);
        text-transform: uppercase;
    }
    .price-value{
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(1.7rem, 6vw, 2.4rem);
        font-weight: 600;
        color: var(--accent);
        margin-top: 0.2rem;
    }
    @keyframes rise{
        from{ opacity: 0; transform: translateY(6px); }
        to{ opacity: 1; transform: translateY(0); }
    }

    /* summary card */
    .detail-card{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .detail-card p{ margin-bottom: 0.35rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)



# LOAD TRAINED MODEL AND PREPROCESSING FILES

@st.cache_resource
def load_models():
    """Load the trained XGBoost + SMOTE-NC model and preprocessing pipeline."""
    
    model = joblib.load("vehicle_price_model.pkl")
    preprocessor = joblib.load("vehicle_preprocessor.pkl")
    encoding_data = joblib.load("make_model_encoding.pkl")
    
    return model, preprocessor, encoding_data


model, preprocessor, encoding_data = load_models()



# TARGET ENCODING DATA
# Matches the notebook's make_model target encoding dictionary

make_model_means = encoding_data["means"]
global_mean = encoding_data["global_mean"]



# VALID INPUT CATEGORIES
# Derived from the fitted preprocessor to ensure consistency with training data.
# Condition, Transmission, and County come from the OneHotEncoder; 
# Make/Model come from the target encoding dictionary keys.

@st.cache_data
def get_category_options(_preprocessor, _make_model_means):
    """Extract valid categories from the trained preprocessing pipeline."""
    
    cat_encoder = _preprocessor.named_transformers_["cat"]
    condition_options = sorted(cat_encoder.categories_[0].tolist())
    transmission_options = sorted(cat_encoder.categories_[1].tolist())
    county_options = sorted(cat_encoder.categories_[2].tolist())
    
    # Build Make → Models mapping from target encoding dictionary
    make_to_models = {}
    for key in _make_model_means.keys():
        if "_" not in key:
            continue
        make_key, model_key = key.split("_", 1)
        make_to_models.setdefault(make_key, set()).add(model_key)
    
    make_options = sorted(make_to_models.keys())
    make_to_models = {m: sorted(models) for m, models in make_to_models.items()}
    
    return condition_options, transmission_options, county_options, make_options, make_to_models


(
    condition_options,
    transmission_options,
    county_options,
    make_options,
    make_to_models,
) = get_category_options(preprocessor, make_model_means)



# PAGE TITLE

st.markdown('<span class="hero-badge">NAIROBI · KES PRICING MODEL</span>', unsafe_allow_html=True)
st.title("🚗 Kenya Used Car Price Predictor")

st.write(
    "Enter the details of a used vehicle to estimate "
    "its selling price in Kenyan Shillings (KES)."
)

st.markdown("<hr>", unsafe_allow_html=True)



# VEHICLE INFORMATION
# Make, Model, Condition, Transmission, and County are categorical inputs
# extracted from the training data categories, so each is a constrained dropdown.

st.markdown('<span class="eyebrow">STEP · IDENTIFY THE CAR</span>', unsafe_allow_html=True)
st.subheader("Vehicle Information")

col_a, col_b = st.columns(2)

with col_a:
    make = st.selectbox(
        "Car Make",
        options=make_options,
        index=None,
        placeholder="Select a make"
    )

with col_b:
    models_for_make = make_to_models.get(make, []) if make else []
    model_name = st.selectbox(
        "Car Model",
        options=models_for_make,
        index=None,
        placeholder="Select a make first" if not make else "Select a model",
        disabled=not make
    )

col_c, col_d = st.columns(2)

with col_c:
    condition = st.selectbox(
        "Condition",
        options=condition_options,
        index=None,
        placeholder="Select a condition"
    )

with col_d:
    transmission = st.selectbox(
        "Transmission",
        options=transmission_options,
        index=None,
        placeholder="Select a transmission"
    )

county = st.selectbox(
    "County",
    options=county_options,
    index=None,
    placeholder="Select a county"
)



# VEHICLE AGE AND MILEAGE
# Year of Manufacture and Mileage are numeric inputs that feed into
# Car_Age and Log_Mileage features in the notebook.

st.markdown('<span class="eyebrow">STEP · USAGE DETAILS</span>', unsafe_allow_html=True)
st.subheader("Vehicle Details")

col_e, col_f = st.columns(2)

with col_e:
    yom = st.number_input(
        "Year of Manufacture",
        min_value=1960,
        max_value=2026,
        value=2018,
        step=1,
        format="%d"
    )

with col_f:
    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        max_value=1000000,
        value=80000,
        step=1000,
        format="%d"
    )


st.markdown("<hr>", unsafe_allow_html=True)



# PREDICTION

if st.button(
    "Predict Vehicle Price",
    type="primary",
    use_container_width=True
):

    # Validate required inputs
    
    if not make:
        st.warning("Please select the car make.")
        st.stop()
    
    if not model_name:
        st.warning("Please select the car model.")
        st.stop()
    
    if not condition:
        st.warning("Please select the condition.")
        st.stop()
    
    if not transmission:
        st.warning("Please select the transmission.")
        st.stop()
    
    if not county:
        st.warning("Please select the county.")
        st.stop()
    
    
    # --------------------------------------------------------
    # FEATURE ENGINEERING — matches the notebook exactly
    
    CURRENT_YEAR = 2026
    
    car_age = CURRENT_YEAR - yom
    log_mileage = np.log1p(mileage)
    mileage_per_year = mileage / max(car_age, 1)
    
    
    # CREATE MAKE + MODEL FEATURE
    # Both come from dropdowns, so they are guaranteed to be valid.
    
    make_model = make + "_" + model_name
    
    
    # MAKE + MODEL TARGET ENCODING
    # Falls back to global mean if the exact make-model wasn't in training.
    
    make_model_te = make_model_means.get(make_model, global_mean)
    
    
    # CREATE INPUT DATAFRAME
    # Feature order must match training: numeric first, then categorical.
    
    input_data = pd.DataFrame({
        "Car_Age": [car_age],
        "Log_Mileage": [log_mileage],
        "Mileage_Per_Year": [mileage_per_year],
        "MakeModel_TE": [make_model_te],
        "Condition": [condition],
        "Transmission": [transmission],
        "County": [county]
    })
    
    
    # APPLY THE SAVED PREPROCESSOR
    # Handles categorical one-hot encoding and passthrough of numerics.
    
    try:
        input_processed = preprocessor.transform(input_data)
    except Exception as e:
        st.error("There was a problem processing the vehicle information.")
        st.exception(e)
        st.stop()
    
    
    # MODEL PREDICTION
    # XGBoost + SMOTE-NC model predicts log-transformed price.
    
    try:
        predicted_log_price = model.predict(input_processed)[0]
        predicted_price = np.expm1(predicted_log_price)
    except Exception as e:
        st.error("There was a problem generating the prediction.")
        st.exception(e)
        st.stop()
    
    
    # SAFETY CHECK
    
    if not np.isfinite(predicted_price):
        st.error("The model returned an invalid prediction.")
        st.stop()
    
    if predicted_price < 0:
        predicted_price = 0
    
    
    # DISPLAY PREDICTION

    st.markdown(
        f"""
        <div class="price-card">
            <div class="price-label">Estimated Selling Price</div>
            <div class="price-value">KES {predicted_price:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    # SHOW INPUT SUMMARY

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">SUMMARY</span>', unsafe_allow_html=True)
    st.subheader("Prediction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="detail-card">
                <p><b>Make:</b> {make}</p>
                <p><b>Model:</b> {model_name}</p>
                <p><b>Condition:</b> {condition}</p>
                <p><b>Transmission:</b> {transmission}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="detail-card">
                <p><b>County:</b> {county}</p>
                <p><b>Year:</b> {yom}</p>
                <p><b>Mileage:</b> {mileage:,.0f} km</p>
                <p><b>Vehicle Age:</b> {car_age} years</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "The prediction is generated using the tuned XGBoost + SMOTE-NC model "
        "trained in the vehicle price prediction notebook. This model achieves "
        "13.78% MAPE on the test set and is optimized for both common and rare "
        "vehicle segments."
    )
