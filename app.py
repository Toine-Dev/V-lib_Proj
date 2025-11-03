import streamlit as st
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static
from datetime import datetime
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os

CSV_PATH = "comptage-velo-donnees-compteurs (2).csv"
PICKLE_PATH = "comptage_velo_df.pkl"

def _create_pickle_from_csv(csv_path=CSV_PATH, pickle_path=PICKLE_PATH):
    try:
        df = pd.read_csv(csv_path, sep=";")
        df.to_pickle(pickle_path)
        return df
    except Exception as e:
        st.error(f"Failed to create pickle from CSV: {e}")
        return None

# Create pickle only if it doesn't exist
if not os.path.exists(PICKLE_PATH):
    _create_pickle_from_csv()

# Load the dataframe from pickle, fall back to creating it from CSV if load fails
try:
    cached_df = pd.read_pickle(PICKLE_PATH)
except Exception:
    cached_df = _create_pickle_from_csv()

# Accessor to use elsewhere in the app
def get_cached_dataframe():
    return cached_df


# Set page configuration
st.set_page_config(
    page_title="Vélib Paris Analysis",
    # page_layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def get_season_from_date(date):
    year = date.year
    spring = pd.Timestamp(f'{year}-03-20')
    summer = pd.Timestamp(f'{year}-06-21')
    autumn = pd.Timestamp(f'{year}-09-22')
    winter = pd.Timestamp(f'{year}-12-21')

    if spring <= date < summer:
        return 'spring'
    elif summer <= date < autumn:
        return 'summer'
    elif autumn <= date < winter:
        return 'autumn'
    else:
        return 'winter'

def is_rush_hour(hour):
    return (7 <= hour < 10) or (17 <= hour < 20)

def is_night(hour, season):
    night_hours = {
        'winter': {'start': 17, 'end': 8},
        'spring': {'start': 20, 'end': 6},
        'summer': {'start': 22, 'end': 5},
        'autumn': {'start': 19, 'end': 7},
    }
    
    nh = night_hours.get(season.lower())
    if nh is None:
        return False
    
    start, end = nh['start'], nh['end']
    return hour >= start or hour < end

# Create the pages in the sidebar
pages = ["Overview", "Data Analysis", "Model & Predictions"]
page = st.sidebar.selectbox("Choose a page", pages)

if page == "Overview":
    st.title("Vélib Paris Analysis")
    st.write("Welcome to the Vélib Paris Analysis Dashboard!")
    # Add content for overview page

elif page == "Data Analysis":
    st.title("Data Analysis")
    # Add content for data analysis page

elif page == "Model & Predictions":
    st.title("Model Training & Predictions")
    
    # Load the model and feature names
    @st.cache_resource
    def load_model_and_features():
        try:
            model = RandomForestRegressor()
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            return model, feature_names
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None

    model, feature_names = load_model_and_features()
    
    if model is not None:
        # Model Performance Section
        st.header("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Mean Absolute Error", value="12.45")  # Your actual MAE
        with col2:
            st.metric(label="RMSE", value="18.32")  # Your actual RMSE

        # Feature Importance Plot
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        sns.barplot(data=feature_imp.head(10), x='Importance', y='Feature')
        plt.title('Top 10 Most Important Features')
        st.pyplot(fig)
        
        # Interactive Prediction Section
        st.header("Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_date = st.date_input("Select Date")
            pred_time = st.time_input("Select Time")
        with col2:
            temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0)
            is_holiday = st.checkbox("Is Holiday?")
        with col3:
            is_precipitation = st.checkbox("Is Raining?")
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 50.0, 10.0)

        if st.button("Predict"):
            # Combine date and time
            datetime_pred = pd.Timestamp.combine(pred_date, pred_time)

            # Convert to timezone-aware UTC timestamp.
            # If naive, assume local system timezone then convert to UTC.
            if datetime_pred.tz is None:
                local_tz = datetime.now().astimezone().tzinfo
                datetime_pred = datetime_pred.tz_localize(local_tz)
            datetime_pred = datetime_pred.tz_convert('UTC')

            # if datetime_pred > 

            # Create feature vector
            season = get_season_from_date(datetime_pred)
            hour = datetime_pred.hour
            
            # Create dummy variables for season and hour
            season_dummies = pd.get_dummies([season], prefix='saison').iloc[0]
            hour_dummies = pd.get_dummies([hour], prefix='heure').iloc[0]
            
            # Create feature dictionary
            features = {
                'nuit': int(is_night(hour, season)),
                'vacances': int(is_holiday),
                'heure_de_pointe': int(is_rush_hour(hour)),
                'pluie': int(is_precipitation),
                'apparent_temperature': temperature,
                **season_dummies.to_dict(),
                **hour_dummies.to_dict()
            }

            features = pd.read_csv()
            
            # Create DataFrame with all features
            pred_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = model.predict(pred_df)[0]
            
            st.success(f"Predicted bicycle count: {prediction:.0f}")

        # Heatmap Section
        st.header("Traffic Heatmap")
        
        # Load location data
        @st.cache_data
        def load_location_data():
            df = pd.read_csv("comptage-velo-donnees-compteurs (2).csv", sep=";")
            df[["latitude", "longitude"]] = df["Coordonnées géographiques"].str.split(",", expand=True)
            df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
            df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
            # Remove any rows with NaN values
            df = df.dropna(subset=["latitude", "longitude"])
            locations_df = df[["latitude", "longitude"]].drop_duplicates()
            # Add a dummy weight column for the heatmap (1 for each location)
            locations_df['weight'] = 1
            return locations_df

        locations = load_location_data()
        
        # Create the map
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles='CartoDB positron')
        
        # Add the heatmap layer with proper formatting
        HeatMap(
            data=locations[['latitude', 'longitude', 'weight']].values.tolist(),
            radius=30,
            max_zoom=13
        ).add_to(m)
        
        # Display the map
        folium_static(m)

        # Additional insights
        st.header("Model Insights")
        st.write("""
        Key findings from the model:
        - Rush hours significantly impact bicycle traffic
        - Weather conditions (temperature, precipitation) affect usage patterns
        - Seasonal variations show higher usage during spring/summer months
        - Location-specific patterns vary across Paris
        """)