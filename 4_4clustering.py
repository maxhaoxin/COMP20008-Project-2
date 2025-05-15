import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import os

def conditions_data(df):
    conditions = (
        df.groupby([
            'SURFACE_COND_DESC', 'ATMOSPH_COND_DESC', 'ROAD_GEOMETRY_DESC', 'AGE_GROUP',
            'LIGHT_CONDITION'
    ])
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
        'AGE_GROUP',
        'LIGHT_CONDITION'
    ])
    
    aggregated = grouped.agg({
        'VEHICLE_AGE_NORM': 'mean',
        'TOTAL_OCCUPANTS_NORM': 'mean',
        'ACCIDENT_NO': 'count',
        'WEATHER_RISK': 'mean',
        'SPEED_ZONE_BINARY': 'mean',
        'ATMOSPH_COND_RISK': 'mean',
        'SURFACE_COND_RISK': 'mean'  
    }).reset_index().rename(columns={'ACCIDENT_NO': 'CRASH_COUNT'})
    
    features = aggregated[['VEHICLE_AGE_NORM', 'TOTAL_OCCUPANTS_NORM', 'CRASH_COUNT', 'WEATHER_RISK', 'SPEED_ZONE_BINARY', 'ATMOSPH_COND_RISK', 'SURFACE_COND_RISK']]
    
    feature_data = features

    scaler = StandardScaler()
    normalised_features = scaler.fit_transform(feature_data)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    aggregated['CLUSTER'] = kmeans.fit_predict(normalised_features)
    
    #sns.countplot(x='CLUSTER', hue = 'AGE_GROUP',
    #  data=aggregated)
    #plt.title('Cluster Distribution by Age Group')
    #plt.show()


    for cluster_id in range(4):
        cluster_data = aggregated[aggregated['CLUSTER'] == cluster_id]
        top10 = cluster_data.sort_values(by='CLUSTER', ascending=False).head(10)
        top10.to_csv(f'4_4cluster_{cluster_id}.csv', index=False)
        
    return()

clustering()