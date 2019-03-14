#!/usr/bin/env python
# coding: utf-8

# In[294]:


from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
#X, y = make_blobs(n_samples=100,n_features=2,centers=3, cluster_std=0.5,shuffle=True, random_state=0)
X, y = make_moons(n_samples=200,noise=0.05,random_state=0)


# In[295]:


plt.scatter(X[:, 0], X[:, 1], s=15, marker = 'o')
plt.show()


# In[296]:


inertia = dict()
silhouette_avg = dict()
for i in range(2,11):
    kmeans_temp = KMeans(n_clusters=i, random_state=0).fit(X)
    k_pred = kmeans_temp.fit_predict(X)
    inertia[i] = kmeans_temp.inertia_
    silhouette_avg[i] = silhouette_score(X, k_pred)
K_I = min(inertia, key=inertia.get)
K_S = min(silhouette_avg, key=silhouette_avg.get)
print(K_I)
print(K_S)
inertia_list = sorted(inertia.items())
x, y = zip(*inertia_list)
plt.plot(x, y)
plt.show()

silhouette_list = sorted(silhouette_avg.items())
x, y = zip(*silhouette_list)
plt.plot(x, y)
plt.show()


# In[297]:


kmeans_i = KMeans(n_clusters=K_I, random_state=0).fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans_i.labels_, cmap='rainbow')


# In[298]:


kmeans_s = KMeans(n_clusters=K_S, random_state=0).fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans_s.labels_, cmap='rainbow')


# In[299]:


#For blobs dataset, set eps = 0.5. For moons dataset, set eps = 0.3.
dbscan = DBSCAN(eps=0.3, min_samples=2).fit(X)
plt.scatter(X[:,0],X[:,1], c=dbscan.labels_, cmap='rainbow')


# In[300]:


# For both blobs and moons, DBSCAN worked better than Kmeans.


# In[ ]:




