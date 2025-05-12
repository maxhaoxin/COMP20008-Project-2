import os
import pandas as pd
import numpy as np

accident_df = pd.read_csv('datasets/accident.csv', usecols=[
    'ACCIDENT_NO', 'SEVERITY','ACCIDENT_DATE',
    'LIGHT_CONDITION', 'ROAD_GEOMETRY','ROAD_GEOMETRY_DESC'])

atmospheric_df = pd.read_csv('datasets/atmospheric_cond.csv', usecols=[
    'ACCIDENT_NO','ATMOSPH_COND','ATMOSPH_COND_DESC'])

road_df = pd.read_csv('datasets/road_surface_cond.csv', usecols=[
    'ACCIDENT_NO','SURFACE_COND', 'SURFACE_COND_DESC'])

vehicle_df = pd.read_csv('datasets/filtered_vehicle.csv', usecols=[
    'ACCIDENT_NO','VEHICLE_YEAR_MANUF', 'VEHICLE_TYPE','TOTAL_NO_OCCUPANTS'])

person_df = pd.read_csv('datasets/person.csv', usecols=[
    'ACCIDENT_NO','PERSON_ID', 'INJ_LEVEL','AGE_GROUP','SEATING_POSITION'])

accident_merged = accident_df.merge(atmospheric_df, on='ACCIDENT_NO', how='left').merge(road_df, on='ACCIDENT_NO', how='left')

accident_merged["ACCIDENT_YEAR"] = pd.to_datetime(accident_merged["ACCIDENT_DATE"]).dt.year

##Aggregating Vehicle DataFrame
vehicle_df = vehicle_df.merge(accident_merged[["ACCIDENT_NO", "ACCIDENT_YEAR"]], on="ACCIDENT_NO", how="left")

# Step 1: Compute medians
median_vehicle_year = vehicle_df["VEHICLE_YEAR_MANUF"].median()
median_accident_year = vehicle_df["ACCIDENT_YEAR"].median()

# Step 2: Impute missing values
vehicle_df["VEHICLE_YEAR_MANUF"] = vehicle_df["VEHICLE_YEAR_MANUF"].fillna(median_vehicle_year)
vehicle_df["ACCIDENT_YEAR"] = vehicle_df["ACCIDENT_YEAR"].fillna(median_vehicle_year)


# Step 3: Recompute vehicle age
vehicle_df["VEHICLE_AGE"] = vehicle_df["ACCIDENT_YEAR"] - vehicle_df["VEHICLE_YEAR_MANUF"]

#Manual Mode
def custom_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else None

vehicle_type_mode = vehicle_df.groupby("ACCIDENT_NO")["VEHICLE_TYPE"].apply(custom_mode)

vehicle_agg = vehicle_df.groupby("ACCIDENT_NO").agg({
    "VEHICLE_AGE": "mean",
    "TOTAL_NO_OCCUPANTS": "sum"
}).reset_index()

vehicle_agg["VEHICLE_TYPE"] = vehicle_type_mode.values

accident_merged = accident_merged.merge(vehicle_agg, on="ACCIDENT_NO", how="left")

##Aggregating Person DataFrame

#Replacing NA in SEATING_POSITION with NA_val
person_df["SEATING_POSITION"] = person_df["SEATING_POSITION"].fillna("NA_val")

#Compute Age Mode and Seat Mode manually
age_mode = person_df.groupby("ACCIDENT_NO")["AGE_GROUP"].apply(custom_mode)
seat_mode = person_df.groupby("ACCIDENT_NO")["SEATING_POSITION"].apply(custom_mode)

person_agg = person_df.groupby("ACCIDENT_NO").agg({
    "PERSON_ID": "count",
    "INJ_LEVEL": "min"
}).rename(columns={
    "PERSON_ID": "NUM_PEOPLE",
    "INJ_LEVEL": "SEVERE_INJURED"
})

# Add mode columns
person_agg["AGE_GROUP_MODE"] = age_mode
person_agg["SEATING_MODE"] = seat_mode

accident_merged = accident_merged.merge(person_agg, on="ACCIDENT_NO", how="left")
accident_merged.isnull().sum()

#Dropping null value
accident_merged = accident_merged.dropna(subset=["VEHICLE_AGE", "TOTAL_NO_OCCUPANTS", "VEHICLE_TYPE"])
accident_merged.shape

# Get object (categorical/textual) columns
categorical_cols = accident_merged.select_dtypes(include=['object', 'float64', 'int64']).columns

# Show unique values for each categorical column
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(accident_merged[col].unique())

# Define filters for each column
invalid_atmosphere = "Not known"
invalid_surface = "Unk."
invalid_vehicle_types = [18, 99]
invalid_age_groups = ["Unknown"]
invalid_seating = ["NA_val", "NK"]
invalid_roadgeom = "Unknown"

# Apply all filters
accident_clean = accident_merged[
    (accident_merged["ATMOSPH_COND_DESC"] != invalid_atmosphere) &
    (accident_merged["SURFACE_COND_DESC"] != invalid_surface) &
    (~accident_merged["VEHICLE_TYPE"].isin(invalid_vehicle_types)) &
    (~accident_merged["AGE_GROUP_MODE"].isin(invalid_age_groups)) &
    (~accident_merged["SEATING_MODE"].isin(invalid_seating)) &
    (~accident_merged["ROAD_GEOMETRY_DESC"].isin(invalid_roadgeom))
]

accident_merged.shape
accident_clean.shape

categorical_cols2 = accident_clean.select_dtypes(include=['object', 'float64', 'int64']).columns

# Show unique values for each categorical column
for col in categorical_cols2:
    print(f"\n--- {col} ---")
    print(accident_clean[col].unique())
    
### Encoding Categorical Variable ###

# Reversing Severity Mapping
severity_ord = {4: 0, 3: 1, 2: 2, 1: 3}
accident_clean["SEVERITY_ORD"] = accident_clean["SEVERITY"].map(severity_ord)

severity_map = {0:'Non injury accident', 1:'Other injury accident',
                2: 'Serious injury accident', 3:'Fatal accident'}
accident_clean["SEVERITY_DESC"] = accident_clean["SEVERITY_ORD"].map(severity_map)

# Ordinal Age Variable
age_ord = {
    "0-4": 0, "5-12": 1, "13-15": 2, "16-17": 3, "18-21": 4,
    "22-25": 5, "26-29": 6, "30-39": 7, "40-49": 8,
    "50-59": 9, "60-64": 10, "65-69": 11, "70+": 12
}
accident_clean.loc[:, "AGE_CODE"] = accident_clean["AGE_GROUP_MODE"].map(age_ord)

# Binary Road Surface Variable
road_map = {"Dry": 0, "Wet": 1, "Icy": 1, "Muddy": 1, "Snowy": 1}
accident_clean.loc[:, "ROAD_BINARY"] = accident_clean["SURFACE_COND_DESC"].map(road_map)

# Binary Seating Map Variable
seat_map = { "D": 0, "LF": 0, "CF": 0,
    "OR": 1, "CR": 1, "LR": 1, "RR": 1, "PL": 1}
accident_clean.loc[:, "SEAT_BINARY"] = accident_clean["SEATING_MODE"].map(seat_map)

# Categorical Seating Map Variable
seat_map2 = { "D": 1, "LF": 2, "CF": 3,
    "OR": 4, "CR": 5, "LR": 6, "RR": 7, "PL": 8}
accident_clean.loc[:, "SEAT_CAT"] = accident_clean["SEATING_MODE"].map(seat_map2)

# Binary Atmospheric  Variable
atm_map = { "Clear": 0, "Raining": 1, "Strong winds": 1,
    "Fog": 1, "Dust": 1, "Smoke": 1, "Snowing": 1}
accident_clean.loc[:, "ATM_BINARY"] = accident_clean["ATMOSPH_COND_DESC"].map(atm_map)

# Road Risk Scoring
road_risk_map = { "Dry": 0, "Wet": 1,  "Muddy": 2, "Icy": 3, "Snowy": 3}
accident_clean.loc[:, "ROAD_RISK"] = accident_clean["SURFACE_COND_DESC"].map(road_risk_map)

# Atmospheric Risk Scoring
atm_risk_map = { "Clear": 0, "Raining": 1, "Strong winds": 1,
    "Fog": 2, "Dust": 2, "Smoke": 2, "Snowing": 3}
accident_clean.loc[:, "ATM_RISK"] = accident_clean["ATMOSPH_COND_DESC"].map(atm_risk_map)

# Weather Risk Scoring
accident_clean.loc[:, "WEATHER_RISK"] = accident_clean["ROAD_RISK"] + accident_clean["ATM_RISK"]

### Normalize Variable ###

# Normalize Weather Risk
accident_clean["WEATHER_RISK_NORM"] = (
    accident_clean["WEATHER_RISK"] - accident_clean["WEATHER_RISK"].min()
) / (accident_clean["WEATHER_RISK"].max() - accident_clean["WEATHER_RISK"].min())

# Normalize Vehicle Age
accident_clean["VEHICLE_AGE_NORM"] = (
    accident_clean["VEHICLE_AGE"] - accident_clean["VEHICLE_AGE"].min()
) / (accident_clean["VEHICLE_AGE"].max() - accident_clean["VEHICLE_AGE"].min())

# Normalize Occupant
accident_clean["TOTAL_OCCUPANTS_NORM"] = (
    accident_clean["TOTAL_NO_OCCUPANTS"] - accident_clean["TOTAL_NO_OCCUPANTS"].min()
) / (accident_clean["TOTAL_NO_OCCUPANTS"].max() - accident_clean["TOTAL_NO_OCCUPANTS"].min())

### Detecting Outlier ###
def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

out_age = iqr_outliers(accident_clean["VEHICLE_AGE"])
print("Outliers in VEHICLE_AGE:", out_age.count())

out_occupant = iqr_outliers(accident_clean["TOTAL_NO_OCCUPANTS"])
print("Outliers in TOTAL_NO_OCCUPANTS:", out_occupant.count())

accident_merged.to_csv('datasets/accident_merged.csv', index=False)
accident_clean.to_csv('datasets/accident_clean.csv', index=False)

accident_clean.shape
