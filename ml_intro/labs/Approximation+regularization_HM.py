#!/usr/bin/env python
# coding: utf-8

# In[100]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import scipy.linalg as la
import scipy.interpolate as interp
import scipy.integrate as integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy

default_dpi = mpl.rcParamsDefault['figure.dpi']
factor = 1
mpl.rcParams['figure.dpi'] = default_dpi * factor


# Подготовка данных

# In[101]:


x_left = -3
x_right = 5
step = 20

x = np.linspace(x_left, x_right, step)
y = 5 + 2 * x


# In[102]:


plt.plot(x,y, 'ro')


# Зашумление данных

# In[103]:


noise = 1.5 * np.random.rand(len(x))


# In[104]:


x = x + noise


# In[105]:


plt.plot(x,y, 'ro')


# Добавление outliers

# In[106]:


y[5] = 14
y[12] = -4


# In[107]:


plt.plot(x,y, 'ro')


# Построение аппромаксионного полинома

# Функция для построения аппрмаксионного полинома простым методом наименьших квадратов

# In[166]:


m = 2


# In[167]:


def get_funct(degree, val):
    return val ** degree


# In[168]:


def sql_least(x, y ,m):
    #[строки, столбцы]
    n = len(x)
    koef = np.zeros((n,m + 1))
    
    for i in range(0 , m +  1):
        koef[:, i] = get_funct(i, x)
        
    koef_T = koef.T
    G = koef_T @ koef
    z = koef_T @ y
    
    return np.linalg.solve(G,z)
        


# In[169]:


get_funct(2,x)


# In[170]:


a_result_least_simple = sql_least(x,y,2)


# In[171]:


a_result_least_simple


# In[172]:


y_hat = np.zeros(len(x))


# In[173]:


y_hat


# In[174]:


for i in range(m+1):
    y_hat = y_hat + x * a_result_least_simple[i] ** i


# In[175]:


y_hat


# In[176]:


plt.plot(x, y_hat)
plt.plot(x,y,'ro')


# Можем видеть, что нижний выброс утащил за собой всю линию, смещая ее вниз

# Борьба с выбросами. Регуляризация

# In[177]:


np.eye(2) * 2 + [[1,1],[1,1]]


# In[306]:


def sql_least_reg(x,y,m,alpha):
    
    n = len(x)
    koef = np.zeros((n, m + 1))
    
    for i in range(0, m+1):
        koef[:, i ] = get_funct(i , x)
        
    koef_t = koef.T
    regAlpha = np.eye(m + 1) * alpha
    G = koef_t @ koef + regAlpha
    z = koef_t @ y
    return np.linalg.solve(G,z)


# In[315]:


a_result_reg = sql_least_reg(x,y,2,1/10)


# In[316]:


y_hat_reg = np.zeros(len(x))


# In[317]:


for i in range(0, m+1):
    y_hat_reg = y_hat_reg + x * a_result_reg[i] ** i


# In[318]:


plt.plot(x,y,'ro')
plt.plot(x,y_hat_reg, color = 'green')


# Построение робастной аппрокмасимации

# In[ ]:


def sql_least_robust(x,y,m):
    n = len(x)
    koef = np.zeros((n, m + 1))
    
    for i in range(0, m+1):
        koef[:, i ] = get_funct(i , x)
        
    koef_t = koef.T
    regAlpha = np.eye(m + 1) * alpha
    G = koef_t @ koef + regAlpha
    z = koef_t @ y
    a = np.linalg.solve(G,z)
    
    E = np.eye (n)
    yr = y @ E 
    
    eps = 0.001
    
    for i in range(10):
            
        ait = a  
          
        PhiA = Phi @ a
    
        res = Phi @ a - yr        
          
        for i in range (0, n):
            #print ('it*', it,  y[10], PhiA[i]-dist)
            if (res[i]>dist):
                yr[i] = PhiA[i]-dist
                #print ('it+', it,  y[10], PhiA[i]-dist)
            if (res[i]<-dist):
                yr[i] = PhiA[i]+dist
                #print ('it-', it,  y[10], PhiA[i]-dist)
                    
        z = PhiT @ yr
        a = np.linalg.solve(G, z)
        
        eit = a - ait
        epsit = (sum(eit**2))**0.5/len(eit)
        
        print ('iteration', it, a, epsit)
        
        if (epsit<=eps):
            print ()
            break 
                
    return a
        
        
    
    
    
    
    

