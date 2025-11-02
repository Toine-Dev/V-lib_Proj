import pandas as pd
import requests

# Function to determine the season from a date
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


# Define function to test if date falls in a holiday
def is_vacances(date):
    # Define vacation periods inside the function
    vacances_periods = [
        ('2024-10-19', '2024-11-05'),  # Toussaint
        ('2024-12-21', '2025-01-07'),  # Noël
        ('2025-02-15', '2025-03-04'),  # Hiver
        ('2025-04-12', '2025-04-29'),  # Printemps
        ('2025-05-29', '2025-06-01'),  # Ascension + pont (29, 30, 31)
        ('2025-07-05', '2025-09-02'),  # Summer begins 5 July to 1 Sept
    ]
    
    # Convert to datetime timestamps
    vacances_intervals = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in vacances_periods
    ]
    
    # Check if date falls in any vacation period
    for start, end in vacances_intervals:
        if start <= date < end:
            return True
    return False

# Function to classify rush hour
def is_rush_hour(dt):
    hour = dt.hour
    return (7 <= hour < 10) or (17 <= hour < 20)




def is_night(row):
    # Example rough night hours per season (24h format)
    night_hours = {
        'winter':    {'start': 17, 'end': 8},
        'spring':{'start': 20.5, 'end': 6},   # 20:30
        'summer':      {'start': 22, 'end': 5},
        'autumn':  {'start': 19, 'end': 7},
    }

    season = row['saison'].lower()
    dt = row['date_et_heure_de_comptage']
    hour = dt.hour + dt.minute/60  # fractional hour
    
    nh = night_hours.get(season)
    if nh is None:
        # if season is unknown, consider not night
        return False
    
    start, end = nh['start'], nh['end']
    
    # Since all seasons cross midnight, we only need this check
    return hour >= start or hour < end

def query_weather_api():
    # API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=48.8575&longitude=2.3514&start_date=2024-08-01&end_date=2025-10-07&hourly=rain,snowfall,apparent_temperature,wind_speed_10m"
    # Send GET request
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # print(data)
        # records = data.get("temperature_2m", [])
        # temperature = data.get("temperature_2m", [])
        records = data["hourly"]
        # Convert list of dicts to DataFrame
        df_weather = pd.DataFrame(records)
        return df_weather
        
        # # Show first few rows
        # print(df_weather)
    else:
        print("Error:", response.status_code)


def encode_features(df):
    # 1. Encode boolean columns to 0/1 (they're already boolean but let's make them int)
    boolean_cols = ['nuit', 'vacances', 'heure_de_pointe', 'pluie', 'neige']
    for col in boolean_cols:
        df[col] = df[col].astype(int)

    # 2. Create dummy variables for categorical columns
    # Saison (keep n-1 categories to avoid multicollinearity)
    saison_dummies = pd.get_dummies(df['saison'], prefix='saison', drop_first=True)
    df = pd.concat([df, saison_dummies], axis=1)

    # Nom_jour (keep n-1 categories to avoid multicollinearity)
    jour_dummies = pd.get_dummies(df['nom_jour'], prefix='jour', drop_first=True)
    df = pd.concat([df, jour_dummies], axis=1)

    # Mois (keep n-1 categories to avoid multicollinearity)
    mois_dummies = pd.get_dummies(df['mois'], prefix='mois', drop_first=True)
    df = pd.concat([df, mois_dummies], axis=1)

    # Heure (keep n-1 categories to avoid multicollinearity)
    heure_dummies = pd.get_dummies(df['heure'], prefix='heure', drop_first=True)
    df = pd.concat([df, heure_dummies], axis=1)

    # 3. Drop the original categorical columns
    df = df.drop(['saison', 'nom_jour', "mois", "heure"], axis=1)

    return df

def sort_indexes(df):
    # 1️⃣ Ensure your datetime column is a datetime type
    # This is necessary for TimeSeriesSplit to work properly
    df['date_et_heure_de_comptage'] = pd.to_datetime(df['date_et_heure_de_comptage'])

    # 2️⃣ Sort by time (and optionally station if you have multiple)
    # Sorting by both ensures chronological order *within each site*
    df = df.sort_values(
        by=['date_et_heure_de_comptage', 'identifiant_du_site_de_comptage'],
        ascending=[True, True]
    ).reset_index(drop=True)

    # 3️⃣ Optional sanity checks
    print("Earliest timestamp:", df['date_et_heure_de_comptage'].min())
    print("Latest timestamp:", df['date_et_heure_de_comptage'].max())
    print("Number of rows:", len(df))
    print("Number of stations:", df['identifiant_du_site_de_comptage'].nunique())

    # Check if there are duplicates in time per site (which can mess up lags)
    dupes = (
        df
        .groupby(['identifiant_du_site_de_comptage', 'date_et_heure_de_comptage'])
        .size()
        .reset_index(name='count')
        .query('count > 1')
    )

    if len(dupes) > 0:
        print(f"⚠️ Found {len(dupes)} duplicated (site, datetime) entries — consider cleaning.")
    else:
        print("✅ No duplicated timestamps per site found.")

    # Optional: ensure the data is evenly spaced (e.g., hourly)
    # This will help you spot missing hours per site.
    check_freq = (
        df
        .groupby('identifiant_du_site_de_comptage')['date_et_heure_de_comptage']
        .diff()
        .value_counts()
        .head()
    )
    print("\nMost common time difference between observations:")
    print(check_freq)
    
    return df
