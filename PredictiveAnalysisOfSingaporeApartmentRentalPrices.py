#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from collections import defaultdict


# In[2]:


pg = pd.read_csv('/Users/krishnamchapa/BITS/Sem3/DataScience/PredictiveAnalysisOfSingaporeRentalPrices/apartment_rental_prices.csv')


# In[3]:


print(pg.shape)


# In[4]:


pg['Price'].idxmax()


# In[5]:


pg.sort_values(by='Price', ascending=True).head()


# In[6]:


pg['Price'].mean()


# In[7]:


pg['Size in sqm'] = pg['Floor Size'].apply(lambda x: round(x * 0.092903, 1))


# In[8]:


pg.head()


# In[9]:


appCoords = pg[['Geo Latitude', 'Geo Longitude']].to_numpy()


# In[10]:


kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(appCoords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([appCoords[cluster_labels == n] for n in range(num_clusters)])
clusters


# In[11]:


clustersWithProperties = {}
for i, cluster in clusters.items():
    for propertyCoords in cluster:
        singleResidence = pg[(pg['Geo Latitude'] == propertyCoords[0]) & (pg['Geo Longitude'] == propertyCoords[1])]
        for j, unit in singleResidence.iterrows():
            residenceNameCleansed = ''.join(e for e in unit['Name'] if e.isalnum() and not e.isdigit())
            if i in clustersWithProperties:
                if residenceNameCleansed in clustersWithProperties.get(i):
                    clustersWithProperties[i][residenceNameCleansed].append(unit['Price'])
                else:
                    clustersWithProperties[i][residenceNameCleansed] = [unit['Price']]
            else:
                clustersWithProperties[i] = {residenceNameCleansed: [unit['Price']]}


# In[12]:


averagePricePerCluster = []
for i, cluster in clustersWithProperties.items():
    numberOfUnits = 0
    totalRentalPricePerCluster = 0
    for propertyName, prices in cluster.items():
        totalRentalPricePerCluster += sum(prices) 
        numberOfUnits += len(prices)
    averagePricePerCluster.append([i, round(totalRentalPricePerCluster / numberOfUnits)])


# In[13]:


dfAveragePrice = pd.DataFrame(averagePricePerCluster, columns=['Cluster No', 'Average Price'])


# In[14]:


path='/Users/krishnamchapa/BITS/Sem3/DataScience/PredictiveAnalysisOfSingaporeRentalPrices/average_apartment_rental_price.csv'
dfAveragePrice.to_csv(path)


# In[15]:


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return centermost_point
centermost_points = clusters.map(get_centermost_point)
dfCentermostPoints = pd.DataFrame(centermost_points.array, columns=['Latitude', 'Longitude'])


# In[16]:


path='/Users/krishnamchapa/BITS/Sem3/DataScience/PredictiveAnalysisOfSingaporeRentalPrices/property_clusters_coordinates.csv'
dfCentermostPoints.to_csv(path)


# In[17]:


fig, ax = plt.subplots(figsize=[10, 6])
pg_scatter = ax.scatter(pg['Geo Latitude'], pg['Geo Longitude'], c='k', alpha=0.9, s=3)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([pg_scatter], ['Property Guru Set'], loc='upper right')
plt.show()


# In[ ]:





# In[ ]:




