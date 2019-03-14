#!/usr/bin/env python
# coding: utf-8

# In[16]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

categories = ['rec.autos', 'rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
corpus_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'), categories=categories,shuffle=True, random_state = 42)
corpus_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'), categories=categories,shuffle=True, random_state= 50)


# In[17]:


y_train= corpus_train.target
y_test = corpus_test.target


# In[18]:


#Pre-processing - I have pre-processed the text using TF-IDF and removing stop-words.
my_stop_words = text.ENGLISH_STOP_WORDS
vectorizer = TfidfVectorizer(stop_words = my_stop_words)
X_train = vectorizer.fit_transform(corpus_train.data)
X_test = vectorizer.transform(corpus_test.data)


# In[19]:


clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)


# In[20]:


print('Accuracy of test: %.2f' %accuracy_score (y_test, y_pred_test)) 


# In[ ]:




