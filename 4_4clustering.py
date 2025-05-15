import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import os

def conditions_data(df):
    conditions = (
        df.groupby(['SURFACE_COND_DESC', 'ATMOSPH_COND_DESC', 'ROAD_GEOMETRY_DESC', 'AGE_GROUP_MODE','LIGHT_CONDITION'])
        .size()
        .reset_index(name='condition')
    )
    return conditions


def clustering():
    df = pd.read_csv('datasets/accident_clean.csv')
    
    crash_counts = conditions_data(df)
    
    grouped = df.groupby([
        'SURFACE_COND_DESC',
        'ATMOSPH_COND_DESC',
        'ROAD_GEOMETRY_DESC',
        'AGE_GROUP_MODE',
        'LIGHT_CONDITION'
    ])
    
    aggregated = grouped.agg({
        'VEHICLE_AGE_NORM': 'mean',
        'TOTAL_OCCUPANTS_NORM': 'mean',
        'ACCIDENT_NO': 'count',
        'ROAD_RISK': 'mean',
        'ATM_RISK': 'mean',
        'WEATHER_RISK': 'mean'       
    }).reset_index().rename(columns={'ACCIDENT_NO': 'CRASH_COUNT'})
    
    features = ['VEHICLE_AGE_NORM', 'TOTAL_OCCUPANTS_NORM', 'CRASH_COUNT', 'ROAD_RISK', 'ATM_RISK', 'WEATHER_RISK']
    feature_data = aggregated[features]

    scaler = StandardScaler()
    normalised_features = scaler.fit_transform(feature_data)
    
    kmeans = KMeans(n_clusters=7, random_state=42)
    aggregated['CLUSTER'] = kmeans.fit_predict(normalised_features)
    
    #sns.countplot(x='CLUSTER', hue = 'AGE_GROUP_MODE', data=aggregated)
    #plt.title('Cluster Distribution by Age Group')
    #plt.show()


    for cluster_id in range(7):
        cluster_data = aggregated[aggregated['CLUSTER'] == cluster_id]
        top10 = cluster_data.sort_values(by='CLUSTER', ascending=False).head(10)
        top10.to_csv(f'4_4cluster_{cluster_id}.csv', index=False)
        
    return()

clustering()