#!/usr/bin/env python
# coding: utf-8

# Рассмотрим регрессионную модель$$
# y_i = \sum_{j = 1}^m w_j x_{ij} + \epsilon_i
# $$
# 
# Анализ заключается в проверке следующих гипотез:
# 
# $E\epsilon_i = 0$
# $D\epsilon_i = \sigma^2$
# $\epsilon_i \sim N(0, \sigma)$
# все $\epsilon_i$ - независимы, где$$
#   \epsilon_i = y_i - f_i
# $$

# Пример хорошей модели

# In[1]:



import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[2]:


N = 1000
M = 10

x = np.linspace(0, 6 * np.pi, N)
e = np.random.randn(N)

y = 5 * x

yp = y + e

plt.scatter(np.arange(N), y)
plt.scatter(np.arange(N), yp)


# In[3]:



model = sm.OLS(yp, x)
results = model.fit()


# In[4]:



results.summary()


# ### Пример плохой модели

# In[5]:


y = 4 * x


# In[6]:


yp = y + np.sin(x) * (4 * np.random.randn(N) + 5)


# In[7]:


plt.scatter(np.arange(N), y)
plt.scatter(np.arange(N), yp)


# In[8]:


model = sm.OLS(yp, x)
results = model.fit()


# In[9]:



results.summary()


# In[ ]:




