import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import plotly.express as px

st.set_page_config(
    page_title="Water Intelligence System",
    page_icon="🌊",
    layout="wide"
)

df = pd.read_csv("Gujarat_Water_Intelligence_Dataset_100K (1).csv")

st.markdown("""
<style>
.stApp {
        background-color: #f5f9fc;
        color: #1a1a1a;
    }
      p, span, label, div {
        color: #1a1a1a;
    }
     h1 {
        color: #0b5ed7 !important;
    }

    h2 {
        color: #0b5ed7 !important;
    }

    h3 {
        color: #084298 !important;
    }
    .nav-container {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    }
     div.stButton > button {
        background-color: #0b5ed7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #084298;
        color: white;
        border: none;
    
    }

div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 2px solid #0b5ed7 !important;
    border-radius: 8px !important;
}


div[data-baseweb="input"] input {
    color: #1a1a1a !important;
    background-color: #ffffff !important;
    font-size: 16px !important;
}

/* Input box when selected */
div[data-baseweb="input"]:focus-within {
    border: 2px solid #084298 !important;
    box-shadow: 0 0 5px rgba(11, 94, 215, 0.3) !important;
}


button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    color: #0b5ed7 !important;
    background-color: #ffffff !important;
}


div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 2px solid #0b5ed7 !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] button {
    display: none !important;
}
div[data-testid="stNumberInput"] input:focus {
    caret-color: #0b5ed7 !important;
}
</style>
""", unsafe_allow_html=True)


#load dataset 
df = pd.read_csv("Gujarat_Water_Intelligence_Dataset_100K (1).csv")

menu = option_menu(
    menu_title="💧 Water Intelligence System",
    options=["Home",
        "Water Shortage Risk",
        "Water Consumption",
        "Leakage Detection",
        "dashboard"
    ],
      icons=[
        "house",
        "droplet",
        "bar-chart",
        "water",
        "bar-chart"
    ],
     menu_icon="water",
    default_index=0,
    orientation="horizontal",

    styles={
         
        "container": {
            "padding": "5px",
            "background-color": "#ffffff"
        },
    

        "menu-title": {
            "color": "#0b5ed7",
            "font-size": "22px",
            "font-weight": "bold"
        },
   
        "icon": {
            "color": "#0b5ed7",
            "font-size": "18px"
        },
        
       
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "5px",
            "color": "#000000",
            "--hover-color": "#e8f1ff"
        },
        
        "nav-link-selected": {
            "background-color": "#0b5ed7",
            "color": "#ffffff"
        }
    }
)
if menu=="Home":
    st.header("💧 Smart Water Management")

    st.write("""
    The Water Intelligence System is an AI-powered platform designed to
    support smart and sustainable water resource management.

    The system uses environmental, reservoir, groundwater, consumption,
    flow and pressure data to identify potential water-related challenges
    and generate data-driven predictions.

    By combining Machine Learning and Data Analytics, the platform helps
    communities, authorities and organizations make informed decisions
    about water conservation, resource planning and infrastructure management.
    """)
    st.subheader(" Water Intelligence at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💧 Prediction Modules",
            value="3"
        )

    with col2:
        st.metric(
            label="🤖 ML Technology",
            value="AI + ML"
        )

    with col3:
        st.metric(
            label="🌍 Coverage",
            value="Gujarat"
        )

    with col4:
        st.metric(
            label="📈 Data-Driven",
            value="100%"
        )

    st.divider()

    st.subheader(" AI Prediction Modules")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success("💧 Water Shortage Risk")

        st.write(
            """
            Predict the possibility of a water shortage by analyzing
            rainfall, temperature, reservoir levels, groundwater levels,
            population and water stress indicators.

            The system provides early warnings to help communities and
            authorities prepare for potential water scarcity.
            """
        )

        st.markdown("**Prediction:** Low Risk | Medium Risk | High Risk")

    with col2:

        st.success("📊 Water Consumption")

        st.write(
            """
            Estimate water consumption in Million Litres per Day (MLD)
            using population, rainfall, temperature and environmental
            factors.

            This prediction supports better water supply planning and
            efficient management of future water demand.
            """
        )

        st.markdown("**Output:** Predicted Water Consumption (MLD)")

    with col3:

        st.success("🚰 Leakage Detection")

        st.write(
            """
            Detect potential water leakage in distribution systems by
            analyzing expected flow, actual flow, water pressure and
            reservoir outflow.

            Early detection can reduce water loss and improve the
            efficiency of water distribution networks.
            """
        )

        st.markdown("**Detection:** Leakage | No Leakage")
    st.subheader(" How the System Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1️⃣")
        st.markdown("**Collect Data**")
        st.write(
            "Collect environmental, reservoir, groundwater,consumption and sensor data."
        )

    with col2:
        st.markdown("### 2️⃣")
        st.markdown("**Analyze Data**")
        st.write(
            "Process and analyze the collected data to identify important water-related patterns."
        )

    with col3:
        st.markdown("### 3️⃣")
        st.markdown("**ML Prediction**")
        st.write(
            "Machine Learning models analyze the data and generate predictions for water-related risks."
        )

    with col4:
        st.markdown("### 4️⃣")
        st.markdown("**Take Action**")
        st.write(
            "Use the insights to support conservation, planning and better water resource management."
        )
    st.subheader(" Key Benefits")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 💧")
        st.markdown("**Water Conservation**")
        st.write(
            "Encourages efficient and responsible use of water resources."
        )

    with col2:
        st.markdown("### 🔍")
        st.markdown("**Early Detection**")
        st.write(
            "Identifies potential water shortage and leakage risks early."
        )

    with col3:
        st.markdown("### 📊")
        st.markdown("**Data Insights**")
        st.write(
            "Converts complex water data into understandable insights."
        )

    with col4:
        st.markdown("### 🌱")
        st.markdown("**Sustainability**")
        st.write(
            "Supports long-term and sustainable water resource planning."
        )
elif menu == "Water Shortage Risk":

    st.title("💧 Water Shortage Risk Prediction")
    x = df[
        [
            "population",
            "rainfall_mm",
            "temperature_c",
            "humidity_percent",
            "reservoir_level_percent",
            "reservoir_inflow_mld",
            "reservoir_outflow_mld",
            "groundwater_level_m",
            "water_stress_index"
        ]
    ]

    y = df["water_shortage_risk"]
    @st.cache_resource
    def load_shortage_model():
        with open("shortage.pkl", "rb") as f:
            return pickle.load(f)

    shortage_model = load_shortage_model()
    st.subheader(" Predict Water Shortage Risk")

    col1, col2, col3 = st.columns(3)

    with col1:

        population = st.number_input(
            "Population",
           
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            
        )

        temperature = st.number_input(
            "Temperature (°C)"
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            
        )

        reservoir_level = st.number_input(
            "Reservoir Level (%)",
            
        )

        reservoir_inflow = st.number_input(
            "Reservoir Inflow (MLD)",
            
        )
        

    with col3:

        reservoir_outflow = st.number_input(
            "Reservoir Outflow (MLD)",
            
        )

        groundwater = st.number_input(
            "Groundwater Level (m)"
        )

        stress_index = st.number_input(
            "Water Stress Index",
            min_value=0.0
        )

    if st.button(
        " Predict Shortage Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                population,
                rainfall,
                temperature,
                humidity,
                reservoir_level,
                reservoir_inflow,
                reservoir_outflow,
                groundwater,
                stress_index
            ]],
            columns=x.columns
        )

        prediction = shortage_model.predict(input_data)
        

        st.success(
            f"💧 Predicted Water Shortage Risk: {prediction[0]}"
        )
elif menu == "Water Consumption":

    st.title("📊 Water Consumption Prediction")

    x = df[
        [
            "population",
            "rainfall_mm",
            "temperature_c",
            "humidity_percent",
            "reservoir_level_percent",
            "groundwater_level_m"
        ]
    ]

    y = df["water_consumption_mld"]
    @st.cache_resource
    def load_Demand_model():
        with open("demand.pkl", "rb") as f:
            return joblib.load("demand.pkl")

    Demand_model = load_Demand_model()
    
    

    st.subheader(" Predict Water Consumption")

    col1, col2 = st.columns(2)

    with col1:

        population = st.number_input(
            "Population",
            
        )

        temperature = st.number_input(
            "Temperature (°C)"
        )

        humidity = st.number_input(
            "Humidity (%)",
            
        )

    with col2:

        rainfall_mm = st.number_input(
            "Rainfall(mm)",
            
        )

        reservoir_level = st.number_input(
            "Reservoir Level (%)",
           
        )

        groundwater = st.number_input(
            "Groundwater Level (m)"
        )

    if st.button(
        " Predict Water Consumption",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                population,
                rainfall_mm,
                temperature,
                humidity,
                reservoir_level,
                groundwater
            ]],
            columns=x.columns
        )
        prediction = Demand_model.predict(input_data)

            

        st.success(
            f"📊 Predicted Water Consumption: "
            f"{prediction[0]:.2f} MLD"
        )
elif menu == "Leakage Detection":

    st.title("🚰 Water Leakage Detection")



    x = df[
        [
            "expected_flow_lpm",
            "actual_flow_lpm",
            "water_pressure_psi",
            "reservoir_outflow_mld"
        ]
    ]

    y = df["leakage_detected"]

    @st.cache_resource
    def load_leakage_model():
        with open("leakage.pkl", "rb") as f:
            return pickle.load(f)

    leakage_model = load_leakage_model()
    

    st.subheader("Detect Water Leakage")

    col1, col2 = st.columns(2)

    with col1:

        expected_flow = st.number_input(
            "Expected Flow (LPM)",
            min_value=0.0
        )

        actual_flow = st.number_input(
            "Actual Flow (LPM)",
            min_value=0.0
        )

    with col2:

        pressure = st.number_input(
            "Water Pressure (PSI)",
            min_value=0.0
        )

        reservoir_outflow = st.number_input(
            "Reservoir Outflow (MLD)",
            min_value=0.0
        )

    if st.button(
        "🚰 Detect Leakage",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                expected_flow,
                actual_flow,
                pressure,
                reservoir_outflow
            ]],
            columns=x.columns
        )
        prediction = leakage_model.predict(input_data)
        

        

        if prediction[0] == 1:

            st.error(
                "⚠️ Water Leakage Detected!"
            )

        else:

            st.success(
                "✅ No Water Leakage Detected."
            )

elif menu == "dashboard":

    st.title("💧 Water Intelligence Dashboard")

    # ---------------- CSS ---------------- #

    st.markdown("""
    <style>

    .card{
        background:white;
        padding:20px;
        border-radius:12px;
        box-shadow:0px 2px 8px rgba(0,0,0,0.15);
        text-align:center;
        margin-bottom:20px;
    }

    .number{
        font-size:28px;
        color:#0b5ed7;
        font-weight:bold;
    }

    .title{
        font-size:18px;
        color:black;
    }

    </style>
    """, unsafe_allow_html=True)

    # ---------------- District ---------------- #

    selected_district = st.selectbox(
        "📍 Select District",
        sorted(df["district"].dropna().unique())
    )

    district_df = df[df["district"] == selected_district]

    st.divider()

    # ---------------- KPI Cards ---------------- #

    c1, c2, c3, = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
        <div class="title">Population</div>
        <div class="number">{int(district_df["population"].mean())}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
        <div class="title">Avg Rainfall</div>
        <div class="number">{district_df["rainfall_mm"].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
        <div class="title">Water Consumption</div>
        <div class="number">{district_df["water_consumption_mld"].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)

   
    # ---------------- Charts ---------------- #

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Water Consumption")

        fig1 = px.bar(
            district_df.head(20),
            x="taluka",
            y="water_consumption_mld",
            color="water_consumption_mld",
            title="Water Consumption by Taluka"
        )

        fig1.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        st.subheader("💧 Water Shortage Risk")

        risk = district_df["water_shortage_risk"].value_counts().reset_index()

        risk.columns = ["Risk", "Count"]

        fig2 = px.pie(
            risk,
            names="Risk",
            values="Count",
            title="Water Shortage Risk"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------ #

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("🌧 Rainfall vs Groundwater")

        fig3 = px.scatter(
            district_df,
            x="rainfall_mm",
            y="groundwater_level_m",
            color="water_shortage_risk",
            title="Rainfall vs Groundwater Level"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col4:

        st.subheader("🚰 Leakage Detection")

        leak = district_df["leakage_detected"].value_counts().reset_index()

        leak.columns = ["Leakage", "Count"]

        fig4 = px.pie(
            leak,
            names="Leakage",
            values="Count",
            title="Leakage Status"
        )

        st.plotly_chart(fig4, use_container_width=True)


  # ---------------- Temperature ---------------- #

    st.subheader("🌡 Temperature Distribution")

    fig5 = px.histogram(
        district_df,
        x="temperature_c",
        nbins=20,
        title="Temperature Distribution"
    )

    st.plotly_chart(fig5, use_container_width=True)

   
