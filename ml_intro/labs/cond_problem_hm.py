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


# In[20]:


# x(n), y(n) - массив данных
# m - степень полинома (количество базисных функций - 1)

def SqLeast (x, y, m):
    global G, Phi
    n = len(x)
    Phi = np.zeros ((n,m+1))
    for k in range (0, m+1):
        Phi[:,k] = BasicFunctions (k, x)
    PhiT = Phi.T
    G = PhiT @ Phi
    z = PhiT @ y
    a = np.linalg.solve(G, z)
    return a

def SqLeastReg (x, y, m, alpha):
    global G
    n = len(x)
    Phi = np.zeros ((n,m+1))
    for k in range (0, m+1):
        Phi[:,k] = BasicFunctions (k, x)
    PhiT = Phi.T
    
    RegAlpha = np.eye (m+1) * alpha
        
    G = PhiT @ Phi + RegAlpha 
    z = PhiT @ y
    a = np.linalg.solve(G, z)
    return a

def BasicFunctions (num, x):
    PhiFun = x**num
    return PhiFun


# In[40]:


xleft = 0.0
xright = 2.0
n = 15
x = np.linspace(xleft,xright,n)
yf = np.zeros ((n))
y = np.zeros ((n))

coefs = np.array ([50, 100, 200.0002, -100,-0.001,-15])
m = len(coefs)- 1

print ('Коэффициенты точные', coefs)

for i in range (0, m+1):
    yf += coefs[i]*x**i
plt.plot (x,yf, 'o')


# In[45]:


Delta = 109 * np.random.randn(len(x))
y = yf + Delta
plt.plot (x, y, '*')


# Можем видеть, что число обучлосвленности матрицы коэффициентов, составленных при помощи обычной МНК велико, что говорит о том, что матрицы неустойчива к возмущениям

# In[46]:


PolynomCoefs = SqLeast (x, y, m)
print ('Коэффициенты МНК', PolynomCoefs)

print ('Gmatrix',G)
cond = np.linalg.cond (G)
print (cond)


# In[47]:


PolynomValue = np.zeros(len(x))
for i in range (0, m+1):
    PolynomValue += PolynomCoefs[i]*x**i
        
plt.plot (x, y, 'o')
plt.plot (x, PolynomValue)


# При добавлении регуляризации число обусловленности стало меньше

# In[48]:


CoefsReg = SqLeastReg (x, y, m, 0.01)
print ('Коэффициенты МНК с регуляризацией', CoefsReg)
cond = np.linalg.cond (G)
print (cond)


# In[49]:



PolynomValue = np.zeros(len(x))
for i in range (0, m+1):
    PolynomValue += CoefsReg[i]*x**i
plt.plot (x, PolynomValue)


# In[50]:


PhiInv = np.linalg.pinv(Phi, rcond = 1e-2)
CoefsSVD = PhiInv @ y
print ('коэффициенты МНК SVD', CoefsSVD )


# Можем видеть, что луше всего себя проявили коэффициенты, полученные после SVD

# In[51]:



U, s, Vh = np.linalg.svd(Phi)
print ('s', s)
#smat = np.diag(s)

#plt.subplot (2,1,1)
plt.plot (coefs)
plt.plot (CoefsReg, '*')
plt.plot (PolynomCoefs, 'v')
plt.plot (CoefsSVD, 'o')
#plt.plot (coef2, '-')
#plt.subplot (2,1,2)
#plt.plot (coefs)
#plt.plot (coef2)


# In[ ]:




