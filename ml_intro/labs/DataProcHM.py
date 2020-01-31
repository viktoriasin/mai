#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


x = np.array([-1, 0, 1, 2, 3, 4,5,6,7])
y = np.array([-0.5, 0, 0.5, 0.86603, 1, 0.86603, 0.784, 0.578, 1])
plt.plot (x,y, 'o')


# # Аппроксимация данных полиномами

# In[5]:


PolyCoef1 = np.polyfit(x, y, 2)
print (PolyCoef1)


# In[6]:


PolyCoef1 = np.polyfit(x, y, 3)
print (PolyCoef1)


# In[19]:


def get_appr_function(x,y,power):
    PolyCoef = np.polyfit(x, y, power)
    
    p = 0.
    for k in range(0, power+1):
        p += PolyCoef[power-k]*x**k
    
    return p 

def get_appr_function_plot(x,y,x_plot,power):
    PolyCoef = np.polyfit(x, y, power)
    
    p = 0.
    for k in range(0, power+1):
        p += PolyCoef[power-k]*x_plot**k
    
    return p 
    


# In[16]:


p1 = get_appr_function(x, y, 1)

ErrorLocal = y - p1
ErrorGlobal = (np.sum ((ErrorLocal)**2)/len(x))**0.5

print ("Local Error " , ErrorLocal)
print()
print ("Global Error " ,ErrorGlobal,"\n",ErrorGlobal/(max(y)-min(y)))


# In[22]:


x_plot = np.arange(-1, 7.01, 0.01)

y_plot1 = get_appr_function_plot(x, y, x_plot, 1)
plt.grid()
plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot1)


# In[23]:


x_plot = np.arange(-1, 7.01, 0.01)

y_plot1 = get_appr_function_plot(x, y, x_plot, 2)
plt.grid()
plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot1)


# In[24]:


x_plot = np.arange(-1, 7.01, 0.01)

y_plot1 = get_appr_function_plot(x, y, x_plot, 3)
plt.grid()
plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot1)


# In[25]:


x_plot = np.arange(-1, 7.01, 0.01)

y_plot1 = get_appr_function_plot(x, y, x_plot, 5)
plt.grid()
plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot1)


# # Матрично векторная формулировка

# In[26]:


n = len(x) # n - размер массива данных
m = 1   # m - степень полинома (количество базисных функций - 1)

def SqLeast (x, y, m):
    n = len(x)
    Phi = np.zeros ((n,m+1))
    for k in range (0, m+1):
        Phi[:,k] = BasicFunctions (k, x)
    PhiT = Phi.T
    G = PhiT @ Phi
    #print (Phi)
    #print (G)
    z = PhiT @ y
    #print (z)
    a = np.linalg.solve(G, z)
    #print(a)
    return a

def BasicFunctions (num, x):
    PhiFun = x**num
    return PhiFun  

PolynomCoefs = SqLeast (x, y, 1)
print (PolynomCoefs)


# In[27]:


PolynomCoefs = SqLeast (x, y, 2)
print (PolynomCoefs)


# In[ ]:




