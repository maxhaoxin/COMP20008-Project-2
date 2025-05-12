import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


# dataframes
dfAccident = pd.read_csv('accident_clean.csv')
dfFilteredVehicle = pd.read_csv('filtered_vehicle.csv')
dfPerson = pd.read_csv('person.csv')


# ----------severity vs light conditions----------
hm1_data = dfAccident.groupby(['LIGHT_CONDITION','SEVERITY_DESC']).size().unstack(fill_value=0)

sns.heatmap(hm1_data, annot=True, fmt = 'd', cmap= 'Reds')

plt.xlabel('Severity')
plt.ylabel('Light Conditions')
plt.title('Severity vs Light Conditions')

plt.tight_layout()
plt.savefig('severity_VS_lightConditions.png')
plt.close()

x1 = dfAccident[['LIGHT_CONDITION']] 
y1 = dfAccident['SEVERITY_ORD'] 

# get MI
mi1 = mutual_info_classif(x1, y1, discrete_features=True)

print("MI for light condition vs severity:", mi1[0])

# ----------severity vs road geometry----------
hm2_data = dfAccident.groupby(['ROAD_GEOMETRY_DESC','SEVERITY_DESC']).size().unstack(fill_value=0)

sns.heatmap(hm2_data,  annot=True, fmt = 'd', cmap= 'Blues')
plt.xlabel('Severity')
plt.ylabel('Road Geometry')
plt.title('Severity vs Road Geometry')

plt.tight_layout()
plt.savefig('severity_VS_roadGeometry.png')
plt.close()

x2 = dfAccident[['ROAD_GEOMETRY']] 
y2 = dfAccident['SEVERITY_ORD'] 

# get MI
mi2 = mutual_info_classif(x2, y2, discrete_features=True)

print("MI for road geometry vs severity:", mi2[0])

# ----------severity vs Atmosphere Condition----------
hm3_data = dfAccident.groupby(['ATMOSPH_COND_DESC','SEVERITY_DESC']).size().unstack(fill_value=0)

sns.heatmap(hm3_data,  annot=True, fmt = 'd', cmap= 'Greens')
plt.xlabel('Severity')
plt.ylabel('Atmosphere Condition')
plt.title('Severity vs Atmosphere Condition')

plt.tight_layout()
plt.savefig('severity_VS_atmoCond.png')
plt.close()


x3 = dfAccident[['ATMOSPH_COND']] 
y3 = dfAccident['SEVERITY_ORD'] 

# get MI
mi3 = mutual_info_classif(x3, y3, discrete_features=True)

print("MI for atmosphere condition vs severity:", mi3[0])

# ----------severity vs Surface Condition----------
hm4_data = dfAccident.groupby(['SURFACE_COND_DESC','SEVERITY_DESC']).size().unstack(fill_value=0)

sns.heatmap(hm4_data,  annot=True, fmt = 'd', cmap= 'Purples')
plt.xlabel('Severity')
plt.ylabel('Surface Condition')
plt.title('Severity vs Surface Condition')

plt.tight_layout()
plt.savefig('severity_VS_surfCond.png')
plt.close()


x4 = dfAccident[['SURFACE_COND']] 
y4 = dfAccident['SEVERITY_ORD'] 

# get MI
mi4 = mutual_info_classif(x4, y4, discrete_features=True)

print("MI for surface condition vs severity:", mi4[0])

# ----------level of damage vs vehicle type----------
hm5_data = dfFilteredVehicle.groupby(['VEHICLE_TYPE_DESC','LEVEL_OF_DAMAGE']).size().unstack(fill_value=0)

sns.heatmap(hm5_data,  annot=False, fmt = 'd', cmap= 'Oranges')
plt.xlabel = ('Vehicle Type')
plt.ylabel = ('Level of Damage')
plt.title('Level of Damage vs Vehicle Type')

plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.savefig('lvlDmg_VS_vehicleType.png')
plt.close()


x5 = dfFilteredVehicle[['VEHICLE_TYPE']] 
y5 = dfFilteredVehicle['LEVEL_OF_DAMAGE'] 

# get MI
mi5 = mutual_info_classif(x5, y5, discrete_features=True)

print("MI for level of damage vs vehicle type:", mi5[0])


# ----------Injury Level vs Sex----------
hm6_data = dfPerson.groupby(['SEX','INJ_LEVEL_DESC']).size().unstack(fill_value=0)

sns.heatmap(hm6_data,  annot=True, fmt = 'd', cmap= 'Oranges')
plt.xlabel = ('Sex')
plt.ylabel = ('Injury Level')
plt.title('Injury Level vs Sex')

plt.tight_layout()
plt.savefig('injLvl_VS_sex.png')
plt.close()

dfPerson['SEX_NUM'] = dfPerson['SEX'].map({'M': 0, 'F': 1, 'U': 2})

dfPersonFull = dfPerson.dropna(subset=['SEX_NUM', 'INJ_LEVEL'])
x6 = dfPersonFull[['SEX_NUM']] 
y6 = dfPersonFull['INJ_LEVEL'] 

# get MI
mi6 = mutual_info_classif(x6, y6, discrete_features=True)

print("MI for injury level vs sex:", mi6[0])

# ----------Severity vs Age group----------
hm7_data = dfAccident.groupby(['AGE_GROUP_MODE','SEVERITY_DESC']).size().unstack(fill_value=0)

sns.heatmap(hm7_data,  annot=True, fmt = 'd', cmap= 'Oranges')
plt.xlabel = ('Age Group')
plt.ylabel = ('Severity')
plt.title('Severity vs Age Group')

plt.tight_layout()
plt.savefig('severity_VS_ageGroup.png')
plt.close()


#dfPersonFull = dfPerson.dropna(subset=['AGE_GROUP_NUM', 'INJ_LEVEL'])
x7 = dfAccident[['AGE_CODE']] 
y7 = dfAccident['SEVERITY_ORD'] 

# get MI
mi7 = mutual_info_classif(x7, y7, discrete_features=True)

print("MI for injury level vs age group:", mi7[0])
