#!/usr/bin/env python
# coding: utf-8

# # library

# In[94]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# # memasukkan data

# In[95]:


air = pd.read_csv('E:\Clustering\AirQualityItaly.csv')
air.head()


# # hapus variabel tidak terpakai

# In[96]:


air = air.drop(['Date','Time','CO(GT)','PT08.S1(CO)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','AH'], axis = 1)
air


# In[97]:


air.info()


# # visualisasi data

# In[98]:


plt.scatter(air.NMHC, air.RH, s = 10, c = 'c', marker = '.', alpha = 1)
plt.show()


# # menentukan klaster

# In[99]:


air_x = air.iloc[:, 0:2]
air_x.head()


# In[100]:


x_array = np.array(air_x)
print(x_array)


# In[101]:


scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled


# # klaster = 5

# In[129]:


kmeans = KMeans(n_clusters = 5, init='k-means++', max_iter=100, n_init=10, random_state=123)


# In[130]:


c_kmeans = kmeans.fit_predict(x_scaled)


# In[131]:


air["Kelas"] = c_kmeans


# In[132]:


print(air)


# # sentroid 5 klaster

# In[76]:


print(kmeans.cluster_centers_)


# In[77]:


air['kluster'] = kmeans.labels_


# # visualisasi sentroid 5 klaster

# In[78]:


output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = air.kluster, marker = '.', alpha = 1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', label='Centroids', s=150, alpha=1 , marker='o');
plt.title('Hasil Klustering K-Means Kualitas Udara')
plt.xlabel('Tingkat Kadar NHCM')
plt.ylabel('Tingkat Kadar RH')
plt.colorbar (output)
plt.legend()
plt.show()


# In[ ]:




