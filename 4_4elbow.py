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
    grouped = df.groupby(['SURFACE_COND_DESC', 'ATMOSPH_COND_DESC', 'ROAD_GEOMETRY_DESC', 'AGE_GROUP_MODE','LIGHT_CONDITION'])
    aggregated = grouped.agg({
        'SEVERITY_ORD': 'mean',
        'SEVERE_INJURED': 'mean',
        'VEHICLE_AGE': 'mean',
        'NUM_PEOPLE': 'mean',
        'ACCIDENT_NO': 'count'        
    }).reset_index().rename(columns={'ACCIDENT_NO': 'CRASH_COUNT'})
    
    features = aggregated[['SEVERITY_ORD', 'SEVERE_INJURED', 'VEHICLE_AGE', 'NUM_PEOPLE', 'CRASH_COUNT']]

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