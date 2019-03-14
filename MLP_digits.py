#!/usr/bin/env python
# coding: utf-8

# In[227]:


from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[10]) 
plt.show()


# In[228]:


plt.matshow(digits.images[3]) 
plt.show()
plt.matshow(digits.images[4]) 
plt.show()


# In[229]:


print(digits.images[10])


# In[230]:


n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
print(digits.target[10])
y = digits.target


# In[231]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[232]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(learning_rate_init= 0.001, random_state=1, solver='lbfgs')
# After trying out different parameters and different values for them, I found that
# this combination gave over 95% accuracy

mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))


# In[ ]:




