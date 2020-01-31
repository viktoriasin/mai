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


# # interp1d

# In[4]:


import matplotlib.pyplot as plt
from scipy import interpolate
x = np.arange(0, 10) * np.pi
y = np.sin(x)
f = interpolate.interp1d(x, y)


# In[12]:


xnew = np.arange(0, 10) * np.pi
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()


# In[15]:


x = np.linspace(0, 10, num=11, endpoint=True)
y = np.exp(-x**2/9.0)
f = interpolate.interp1d(x, y)
f2 = interpolate.interp1d(x, y, kind='cubic')


# In[16]:


xnew = np.linspace(0, 10, num=41, endpoint=True)

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()


# # lagrange

# Используем интерполяционный многочлен Лагранжа

# In[17]:


x_d = np.arange (-1, 1.01, 0.01)


# In[22]:


x_f = np.arange(-1.0, 1.25, 0.25)
y_f = 1/(1+55*x_f*x_f*x_f)
poly_f = interp.lagrange(x_f, y_f)
print (poly_f)
x_plot = x_d
plt.plot(x_f, y_f, 'v')
plt.plot(x_plot, poly_f(x_plot));


# Таже самая функция, используем кубическую интерполяцию

# Кубический сплайн — гладкая функция, область определения которой разбита на конечное число отрезков, на каждом из которых она совпадает с некоторым кубическим многочленом (полиномом)

# In[23]:


f2 = interpolate.interp1d(x_f, y_f, kind='cubic')
x_new =  np.arange(-1.0, 1.25, 0.25)
plt.plot(x_f, y_f, 'v')
plt.plot(x_f, f2(x_new));


# # Spline

# In[27]:


cubic = interp.CubicSpline(x_f, y_f, bc_type='natural')

x_new =  np.arange(-1.0, 1.25, 0.25)
plt.grid()
plt.plot(x_f, y_f, 'v')
plt.plot(x_new, cubic(x_new))


# In[28]:


cubic = interp.CubicSpline(x_f, y_f, bc_type='clamped')

x_new =  np.arange(-1.0, 1.25, 0.25)
plt.grid()
plt.plot(x_f, y_f, 'v')
plt.plot(x_new, cubic(x_new))


# In[29]:


x_f = np.arange(-1.0, 1.25, 0.25)
y_f = 1/(1+55*x_f*x_f*x_f*x_f)
poly_f = interp.lagrange(x_f, y_f)
print (poly_f)
x_plot = x_d
plt.plot(x_f, y_f, 'v')
plt.plot(x_plot, poly_f(x_plot));


# In[30]:


cubic = interp.CubicSpline(x_f, y_f, bc_type='clamped')

x_new =  np.arange(-1.0, 1.25, 0.25)
plt.grid()
plt.plot(x_f, y_f, 'v')
plt.plot(x_new, cubic(x_new))


# In[ ]:




