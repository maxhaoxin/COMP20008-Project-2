import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import glob
import os  

df = pd.read_csv('datasets/accident_clean.csv')

def task4_4elbow():
    #print('aaaa')
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

    scaler = StandardScaler()
    normalised_features = scaler.fit_transform(features)

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalised_features)
        sse.append(kmeans.inertia_)
        
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared errors (SSE)')
    plt.title('Elbow Method for Optimal k')  
    plt.tight_layout()
    plt.show()
    #print("umum")
    return ()

task4_4elbow()
#print('hi')