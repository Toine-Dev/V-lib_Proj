import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import pickle
from MyFunctions import get_season_from_date, is_rush_hour, is_night, is_vacances, query_weather_api, encode_features, sort_indexes

# Load and preprocess data
def preprocess_data(up_to_date=None):
    # Load data
    # df = pd.read_csv("comptage-velo-donnees-compteurs (2).csv", sep=";")
    df = pd.read_pickle("comptage_velo_df.pkl")
    print('Data loaded successfully.')

    if up_to_date:
        df = df[pd.to_datetime(df['date_et_heure_de_comptage'], utc=True) <= pd.to_datetime(up_to_date, utc=True)]

    # delete_columns = ['Lien vers photo du site de comptage', 'ID Photos', 'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1', 'url_sites', 'type_dimage']
    # df = df.drop(delete_columns, axis = 1)

    # Apply to all columns of the DataFrame
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]

    #enlever les NA
    df = df.dropna()

    # Parse the datetime column correctly
    df['date_et_heure_de_comptage'] = pd.to_datetime(df['date_et_heure_de_comptage'], utc=True)

    # Remove the timezone (make tz-naive to avoid time zone issues (mismatches))
    df['date_et_heure_de_comptage'] = df['date_et_heure_de_comptage'].dt.tz_convert(None)
    
    # Extract temporal features
    df['heure'] = df['date_et_heure_de_comptage'].dt.hour
    df['mois'] = df['date_et_heure_de_comptage'].dt.month
    df['jour'] = df['date_et_heure_de_comptage'].dt.day
    df['nom_jour'] = df['date_et_heure_de_comptage'].dt.day_name(locale='fr_FR.UTF-8')

    
    df['saison'] = df['date_et_heure_de_comptage'].apply(get_season_from_date)
    df['vacances'] = df['date_et_heure_de_comptage'].apply(is_vacances)
    df['heure_de_pointe'] = df['date_et_heure_de_comptage'].apply(is_rush_hour)
    df['nuit'] = df.apply(is_night, axis=1)

    df[["latitude", "longitude"]] = df["coordonnées_géographiques"].str.split(",", expand=True)

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    # df = df.drop(columns=["coordonnées_géographiques"])

    # df_unique = df.drop_duplicates(subset=["nom_du_site_de_comptage"])

    df_weather = query_weather_api()
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    df_weather['time'] = df_weather['time'].dt.tz_convert(None)

    df_merged = pd.merge(df, df_weather, how="left", left_on="date_et_heure_de_comptage", right_on="time").drop(columns=["time"])

    df_merged['pluie'] = (df_merged['rain'] > 0)
    df_merged['vent'] = (df_merged['wind_speed_10m'] > 30)
    df_merged['neige'] = (df_merged['snowfall'] > 0)
    # df_merged["mois"] = df_merged["date_et_heure_de_comptage"].dt.month

    # Dynamically aggregate: "mean" for comptage_horaire, "first" for others
    agg_dict = {col: "first" for col in df_merged.columns if col not in ["identifiant_du_site_de_comptage", "date_et_heure_de_comptage", "comptage_horaire"]}
    agg_dict["comptage_horaire"] = "mean"

    df_merged = df_merged.groupby(
        ["identifiant_du_site_de_comptage", "date_et_heure_de_comptage"],
        as_index=False
    ).agg(agg_dict)


    # Removing unnecessary columns for modeling
    df_merged = df_merged.drop(columns=["latitude", "longitude","date_d'installation_du_site_de_comptage", \
                                        "identifiant_technique_compteur", "mois_annee_comptage", "identifiant_du_compteur", \
                                        "nom_du_site_de_comptage", "jour", "nom_du_compteur", "snowfall", "rain", "vent", \
                                        "wind_speed_10m", 'lien_vers_photo_du_site_de_comptage', 'id_photos', \
                                        'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1', 'url_sites', 'type_dimage', \
                                        "coordonnées_géographiques"])
    
    # I am not going to train one model per identifiant_du_site_de_comptage as it would be too computationally complex
    # Therefore, I will compute some statistics per site and use them as additional features (static features, i.e. not changing over time) first
    # to capture site-specific usage patterns and make up for the lack of per-site models
    # Next, I will compute time-varying features per site (e.g., rolling averages) to capture temporal patterns specific to each site with the 
    # goal of improving model accuracy without the overhead of multiple models

    site_stats = (
        df_merged.groupby('identifiant_du_site_de_comptage')['comptage_horaire']
        .agg(['mean', 'std', 'max', 'min'])
        .rename(columns={
            'mean': 'site_mean_usage',
            'std': 'site_usage_variability',
            'max': 'site_max_usage',
            'min': 'site_min_usage'
        })
    )

    df_merged = df_merged.merge(site_stats, on='identifiant_du_site_de_comptage', how='left')

    # Time-varying features per site
    df_merged = df_merged.sort_values(['identifiant_du_site_de_comptage', 'date_et_heure_de_comptage'])

    df_merged['lag_1'] = df_merged.groupby('identifiant_du_site_de_comptage')['comptage_horaire'].shift(1)
    df_merged['lag_24'] = df_merged.groupby('identifiant_du_site_de_comptage')['comptage_horaire'].shift(24)
    df_merged['rolling_mean_24'] = (
        df_merged.groupby('identifiant_du_site_de_comptage')['comptage_horaire']
        .shift(1).rolling(24).mean()
    )

    df_merged = df_merged.dropna() # Again because of lag and rolling features

    # Create a copy of df_merged to avoid modifying the original
    df_encoded = df_merged.copy()
    df_encoded = encode_features(df_encoded)

    df_encoded = sort_indexes(df_encoded)

    # # Create encoded features
    # df_encoded = pd.get_dummies(df, columns=['heure', 'mois'])
    
    # Select features for modeling
    features = [col for col in df_encoded.columns if col not in ['comptage_horaire', 'date_et_heure_de_comptage']]
    
    return df_encoded, features

# Train the model
def train_model(X, y):
    # Define time series split
    tscv = TimeSeriesSplit(n_splits=5, test_size=5000)  

    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\nFold {fold}: Train={len(X_train)}, Val={len(X_val)}")

        model = LGBMRegressor(n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds)

        print(f"MAE={mae:.2f}, RMSE={rmse:.2f}")
        results.append({'fold': fold, 'MAE': mae, 'RMSE': rmse})

    cv_results = pd.DataFrame(results)
    print("\nAverage CV performance:")
    print(cv_results[['MAE', 'RMSE']].mean())

    # You can specify `test_size` or let it infer from n_splits
    # model = RandomForestRegressor(n_estimators=100, random_state=42)

    # model.fit(X, y)
    return model

if __name__ == "__main__":
    # Get preprocessed data
    print("HEYYYYYYY!!!")
    df_encoded, features = preprocess_data()
    print('Data preprocessed successfully.')
    
    # Prepare features and target
    X = df_encoded[features].reset_index(drop=True)
    y = df_encoded['comptage_horaire'].reset_index(drop=True)
    
    # Train mode
    print("Training model...")
    model = train_model(X, y)
    
    # Save the model
    print("Saving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    print("Saving feature names...")
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(features, f)