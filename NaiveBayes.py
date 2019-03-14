#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100,centers = 2, cluster_std=2)  #2


# In[5]:


import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], s=15, marker = 'o', c=y)   #3
plt.show()


# In[61]:


from sklearn.model_selection import train_test_split                   #4  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)    


# In[62]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)          #5


# In[63]:


y_pred_test = clf.predict(X_test) #6
print(y_pred_test)


# In[64]:


from sklearn.metrics import accuracy_score
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test))  

