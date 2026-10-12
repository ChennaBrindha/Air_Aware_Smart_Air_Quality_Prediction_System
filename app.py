
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="AirAware | AQI Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4faff, #e8f5ff);
    color: #17375e;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dceafa;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

.brand {
    font-size: 40px;
    font-weight: 800;
    color: #2166a5;
    letter-spacing: -1px;
}

.brand span {
    color: #39a96b;
}

.subtitle {
    font-size: 17px;
    color: #506b8b;
}

.panel {
    background: #ffffff;
    border: 1px solid #dceafa;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 25px rgba(58, 108, 160, 0.08);
}

.panel-title {
    font-size: 22px;
    font-weight: 750;
    color: #17375e;
    margin-bottom: 16px;
}

/* Readable labels */
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] {
    color: #17375e !important;
    font-weight: 600 !important;
}

/* Readable input values */
.stApp input,
.stApp [data-baseweb="input"] input {
    color: #172b4d !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #172b4d !important;
    opacity: 1 !important;
    caret-color: #2166a5 !important;
}

/* Input containers */
.stApp [data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #cbdcf0 !important;
    border-radius: 10px !important;
}

/* Placeholder */
.stApp input::placeholder {
    color: #7186a3 !important;
    -webkit-text-fill-color: #7186a3 !important;
    opacity: 1 !important;
}

/* Number input buttons */
.stApp [data-testid="stNumberInput"] button {
    background-color: #edf5ff !important;
    color: #17375e !important;
    border-color: #cbdcf0 !important;
}

/* Buttons */
div.stFormSubmitButton > button,
div.stButton > button {
    background: linear-gradient(100deg, #42b76c, #2671c8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 48px;
}

div.stFormSubmitButton > button p,
div.stButton > button p {
    color: white !important;
}

/* Result gauge */
.aqi-circle {
    width: 205px;
    height: 205px;
    border-radius: 50%;
    margin: 18px auto;
    border: 12px solid #d9f2ec;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #fbffff;
}

.aqi-number {
    font-size: 45px;
    font-weight: 800;
    color: #17375e;
}

.aqi-label {
    font-size: 14px;
    color: #6682a3;
}

.category-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 2px;
    border-bottom: 1px solid #edf3fa;
    color: #385a80;
}

.dot {
    display: inline-block;
    height: 12px;
    width: 12px;
    border-radius: 50%;
    margin-right: 9px;
}

.footer {
    text-align: center;
    color: #6682a3;
    padding: 26px 0 8px;
}

/* Sidebar readability */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #244466;
}

</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open("aqi_model.pkl", "rb") as file:
        data = pickle.load(file)

    return data["model"], data["feature_names"]


model, feature_names = load_model()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        '<div class="brand">🌿 Air<span>Aware</span></div>',
        unsafe_allow_html=True
    )

    st.caption("Clean Air • Healthier Tomorrow")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "ℹ️ About", "⚙️ How It Works", "✉️ Contact"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### 🌱 Our Mission")
    st.write("Helping people understand air quality through machine learning.")
    st.divider()
    st.caption("Better Air for a Brighter Future")


# ---------- AQI CATEGORIES ----------
categories = [
    (50, "Good", "#36ad72", "0–50"),
    (100, "Satisfactory", "#a7d85c", "51–100"),
    (200, "Moderate", "#ffd34e", "101–200"),
    (300, "Poor", "#ff9b45", "201–300"),
    (400, "Very Poor", "#ef5350", "301–400"),
    (500, "Severe", "#8650bc", "401–500")
]


# ---------- HOME PAGE ----------
if page == "🏠 Home":

    st.markdown("""
    <div style="padding:10px 0 24px">
        <div class="brand">🌿 Air<span>Aware</span></div>
        <div style="font-size:30px;font-weight:750;color:#17375e">
            Smart Air Quality Prediction System
        </div>
        <div class="subtitle">
            Understand your air quality with machine learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.25, 0.9], gap="large")

    # ---------- INPUT FORM ----------
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">🌬️ Input Air Quality Data</div>',
            unsafe_allow_html=True
        )

        with st.form("aqi_form"):
            station_id = st.text_input(
                "Station ID",
                placeholder="Enter a valid training station ID"
            )

            st.markdown("**Particulate Matter**")

            c1, c2 = st.columns(2)

            with c1:
                pm25 = st.number_input(
                    "PM2.5 (µg/m³)", min_value=0.0, value=20.0
                )

            with c2:
                pm10 = st.number_input(
                    "PM10 (µg/m³)", min_value=0.0, value=40.0
                )

            st.markdown("**Gases and Other Pollutants**")

            c1, c2 = st.columns(2)

            with c1:
                no = st.number_input("NO", min_value=0.0, value=5.0)
                nox = st.number_input("NOx", min_value=0.0, value=15.0)
                co = st.number_input("CO", min_value=0.0, value=0.5)
                o3 = st.number_input("O3", min_value=0.0, value=10.0)
                toluene = st.number_input(
                    "Toluene", min_value=0.0, value=2.0
                )

            with c2:
                no2 = st.number_input("NO2", min_value=0.0, value=10.0)
                nh3 = st.number_input("NH3", min_value=0.0, value=5.0)
                so2 = st.number_input("SO2", min_value=0.0, value=5.0)
                benzene = st.number_input(
                    "Benzene", min_value=0.0, value=1.0
                )

            submitted = st.form_submit_button(
                "✨ Predict AQI",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PREDICTION PANEL ----------
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if "aqi_prediction" not in st.session_state:
            st.session_state.aqi_prediction = None

        if submitted:
            if not station_id.strip():
                st.warning("Please enter a Station ID.")
            else:
                try:
                    values = {
                        "StationId": int(station_id),
                        "PM2.5": pm25,
                        "PM10": pm10,
                        "NO": no,
                        "NO2": no2,
                        "NOx": nox,
                        "NH3": nh3,
                        "CO": co,
                        "SO2": so2,
                        "O3": o3,
                        "Benzene": benzene,
                        "Toluene": toluene
                    }

                    input_data = pd.DataFrame(
                        [[values[name] for name in feature_names]],
                        columns=feature_names
                    )

                    prediction = float(model.predict(input_data)[0])
                    st.session_state.aqi_prediction = prediction

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        prediction = st.session_state.aqi_prediction

        if prediction is None:
            st.markdown("""
            <div class="aqi-circle">
                <div style="font-size:28px">🌿</div>
                <div class="aqi-label">YOUR AQI</div>
                <div class="aqi-number">--</div>
            </div>
            <div style="text-align:center;color:#6682a3">
                Enter measurements and click Predict AQI
            </div>
            """, unsafe_allow_html=True)

        else:
            score = max(0, min(500, prediction))
            category = "Severe"
            category_color = "#8650bc"

            for upper, name, color, limits in categories:
                if score <= upper:
                    category = name
                    category_color = color
                    break

            st.markdown(f"""
            <div class="aqi-circle" style="border-color:{category_color}66">
                <div class="aqi-label">🌿 YOUR AQI</div>
                <div class="aqi-number">{prediction:.1f}</div>
                <div style="font-weight:750;color:{category_color}">
                    {category}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(score / 500)
            st.caption("Predicted numeric AQI from your trained model.")

        st.divider()
        st.markdown("**AQI Categories (India)**")

        for upper, name, color, limits in categories:
            st.markdown(
                f'<div class="category-row">'
                f'<span><span class="dot" style="background:{color}"></span>'
                f'{limits}</span><span>{name}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">🌿 AirAware | A cleaner, healthier tomorrow</div>',
        unsafe_allow_html=True
    )


# ---------- ABOUT ----------
elif page == "ℹ️ About":
    st.title("About AirAware")
    st.write(
        "AirAware is a machine-learning project that predicts numeric AQI "
        "from station and pollutant measurements."
    )
    st.info("Predictions depend on the quality of the input data and training model.")


# ---------- HOW IT WORKS ----------
elif page == "⚙️ How It Works":
    st.title("How AirAware Works")

    st.markdown("""
    1. Enter a valid station ID.
    2. Enter pollutant measurements in the units used during training.
    3. Click **Predict AQI**.
    4. Review the predicted AQI and its category.
    """)

    st.warning(
        "Use the same StationId encoding and preprocessing used when training the model."
    )


# ---------- CONTACT ----------
elif page == "✉️ Contact":
    st.title("Contact")
    st.write("AirAware — Smart Air Quality Prediction System")
    st.write("For project questions, contact your project team.")
