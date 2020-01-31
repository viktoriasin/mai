#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import scipy.linalg as la
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.fftpack as fft
import scipy.signal as signal
import matplotlib as mpl
import matplotlib.pyplot as plt

default_dpi = mpl.rcParamsDefault['figure.dpi']
factor = 1.5
mpl.rcParams['figure.dpi'] = default_dpi * factor


# In[10]:


x = np.linspace(0, np.pi * 2, 101)
y = 0. + 4.*np.sin(x * 4) + np.cos(x * 5 ) 
n = len(x)

plt.plot(x, y);


# In[21]:


def dft(x,signal):
    L = max(x) - min(x)
    a = np.zeros ((len(x)))
    b = np.zeros ((len(x)))
    A = np.zeros ((len(x)))

    CosTerm = np.zeros ((len(x)))
    SinTerm = np.zeros ((len(x)))

    n = len(x)

    dx = L / (n-1)
    a[0] = 1./L * np.sum (signal [:(n-1)]) * dx
    b[0] = 0.



    for k in range (1, n-1):

        CosTerm = np.cos(2.* np.pi * k * x / L) * dx
        SinTerm =  np.sin(2.* np.pi * k * x / L) * dx

        a[k]  = 2./L * np.sum (signal[:(n-1)]*CosTerm [:(n-1)])
        b[k]  = 2./L * np.sum (signal[:(n-1)]*SinTerm [:(n-1)])

        A = (a**2+b**2)**(0.5)
    return A, a, b

def idf(x,signal, a,b):
    L = max(x) - min(x)


    n = len(x)

    dx = L / (n-1)
    m = int(n/2)

    yf = np.zeros (n)

    for k in range (0, m):

        yf += a[k]*np.cos(2.* np.pi * k * x / L) + b[k]*np.sin(2.* np.pi * k * x / L)
    return yf


# In[17]:


amps, a_s, b_s = dft(x,y)


# In[18]:


plt.plot (amps[0:n], 'o')
plt.figure ()
plt.grid ()
plt.plot (amps[0:50]);


# Как и ожидалось, график на все значения получился симметричным. Также как и ожидалось, получились две ярко выраженные гармоники, которые соответствуют нашему синусу и косинусу в функции исходного сигнала

# In[22]:


y_idf = idf(x,y,a_s,b_s)


# При помощи обратного преобразования Фурье, восстановим наши исходные данные:

# In[24]:


plt.figure ()
plt.plot (x,y, 'v')
plt.plot (x,y_idf);

