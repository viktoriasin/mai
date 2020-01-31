#!/usr/bin/env python
# coding: utf-8

# In[100]:

#ЛАБОРАТОРНЫЕ РАБОТЫ ПО КУРСУ ВВЕДЕНИЕ В МЕТОДЫ ОБРАБОТКИ И АНАЛИЗА ДАННЫХ
#ВЫПОЛНИЛА СИНДИЦКАЯ м80-103м-19


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


# Исходные данные

# In[263]:


x = np.linspace(0, np.pi * 4, 101)
y = 0. + 2.*np.sin(x * 2) + np.tan(x * 5 ) #+ np.sin(x * 25.5)

n = len(x)

#print (x/(np.pi * 2))

yd = np.zeros(len(x))

plt.plot(x, y);
Delta = 1.0 * np.random.randn(len(x))
yd = y + Delta
y_av = np.sum(y)/len(x)

plt.figure ()

plt.plot (x, yd, 'o');


# In[264]:


n = len(x)


# In[265]:


n


# In[266]:


kernel = np.ones(5)


# In[267]:


kernel


# In[268]:


kernel = kernel/sum(kernel)


# In[269]:


kernel


# In[270]:


kernel_padded = np.pad(kernel, n)[n:]


# In[271]:


np.roll(kernel_padded,-len(kernel)+1)


# In[272]:


len(kernel_padded)


# In[273]:


kernel


# In[274]:


def convolve(signal, window):
    n= len(signal)
    n_w = len(window)
    kernel = np.pad(window, n - n_w + n)[n - n_w + n:]
    kernel_rolled = np.roll(kernel, -n_w // 2)
    convolved = np.zeros(n)
    

    for i in range(0,n-1):
        convolved[i] = 0.

        for k in range(0,n-1):

            convolved[i] = convolved[i] + signal[k]*kernel_rolled[i-k]
 
    
    return convolved
            
    
    
    
    


# In[275]:


y_c = convolve(yd,kernel)


# In[276]:


plt.plot(x,y)
plt.plot (x, yd, 'o');
plt.plot (x, y_c[:len(x)]);


# In[277]:


kernel2 = np.ones(3)


# In[278]:


kernel2 /= sum(kernel2)


# In[279]:


kernel2


# In[280]:


y_c2 = convolve(yd,kernel2)


# In[281]:


plt.plot(x,y)
plt.plot (x, yd, 'o');
plt.plot (x, y_c[:len(x)]);


# In[282]:


def make_kernel(kernel):
    kernel /= sum(kernel)
    return kernel


# In[283]:


plt.plot(x,y)
plt.plot (x, yd, 'o');
plt.plot (x, convolve(yd,make_kernel(np.ones(2))));
plt.plot (x, convolve(yd,make_kernel(np.ones(5))));


# Вывод: чем меньше размер окна, тем менее сглаживается

# Попробуем синусоидальное окно

# In[304]:


kernel_sin = np.arange(-2,5//2)
kernel_sin = np.pi * kernel_sin / (len(kernel_sin ) - 1)


# In[305]:


kernel_sin


# In[306]:


kernel_sin = np.apply_along_axis(np.sin,0,kernel_sin)


# In[307]:


plt.plot(x,y)
plt.plot (x, yd, 'o');
plt.plot (x, convolve(yd,kernel_sin),label='sinusoid');
plt.plot (x, convolve(yd,make_kernel(np.ones(5))),label='average')
plt.legend()


# # Автокоррелляция

# In[320]:


def serial_correlation(signal,lag=1):
    
    n = len(signal)
    avg = np.sum(signal) / n
    y1 = signal[lag:]
    y2 = signal[:n-lag]
    corr  = 0.
    for i in range(0,n-lag-1):
        corr = corr + (y1[i] - avg)*(y2[i] - avg)/(n-lag-1)
    return corr
    


# In[321]:


serial_correlation([1,1,2,2,3,3,4,4],lag=1)


# In[324]:


def auto_corr(signal):
    lags = np.arange(len(signal)//2)
    corrs = [serial_correlation(signal,lag) for lag in lags]
    corrs /= corrs[0]
    return lags, corrs
    


# In[325]:


auto_corr([1,1,2,2,3,3,4,4])


# In[326]:


x = np.linspace(0, np.pi*4,101)


# In[338]:


y = 0. + np.sin(x*2) * np.sin(x*2)


# In[339]:


plt.plot(x, y);


# In[341]:


lags,corrs = auto_corr(y)


# In[343]:


plt.plot(lags/101,corrs)


# Вывод коррелиционная функция синусоид представляет собой косинусоиду

# # Оконное преобразование Фурье

# In[369]:


yv[(xv>np.pi*2) & (xv<np.pi*4)]


# In[383]:


xv = np.linspace(0, np.pi * 9, 15)
nv = len(xv)
Lv = max(xv) - min(xv)
yv = 0. + np.exp(np.cos(xv*2))
yv[xv > np.pi* 4] = 0
yv[xv > np.pi* 6] = 1. * np.cos(xv [xv > np.pi* 6]* 11)
yv[(xv>np.pi*2) & (xv<np.pi*4)] = 0

yv[xv<np.pi*2] =1. * np.sin(xv [xv <np.pi*2]*2)
Bv = np.zeros ((nv))

dxv = Lv / (nv-1)

xwv = np.linspace(-Lv, Lv, len(Winv))

Winv = np.zeros ((2*nv-1))

WinSizev = np.pi/2 #L #/3

Winv[abs(xwv)<WinSizev] = 1.






plt.plot(xv/np.pi, yv)
plt.title('Исходные данные');


# In[384]:


nfv = 15 #частоты для которых будем строить спектрограмму

a2v = np.zeros((nv,nfv))
b2v = np.zeros((nv,nfv))
A2v = np.zeros((nv,nfv))

Freqv = np.zeros ((len(xv)))
Shiftv = np.zeros ((len(xv)))

CosTermv = np.zeros ((len(xv)))
SinTermv = np.zeros ((len(xv)))

for k in range (0,nfv):
    
    Freqv [k] = k

    for s in range (0,nv-1):
        
        Shiftv[s] = s
    
        a2v[s,k]=0.
        b2v[s,k]=0.
       
        for i in range (0,nv-1):
    
            a2v[s,k] +=  2./ WinSizev /2 * yv[i] * Winv[i-s+nv] * np.cos(2*np.pi*xv[i]/Lv * k) * dxv 
            b2v[s,k] +=  2./ WinSizev /2 * yv[i] * Winv[i-s+nv] * np.sin(2*np.pi*xv[i]/Lv * k) * dxv
        
                                   
A2v = (a2v**2+b2v**2)**0.5 


plt.figure()
plt.contourf(xv/np.pi, Freqv[0:15], A2v.T, 100, cmap='rainbow')
plt.colorbar();


# На спектрограмме там, где в исходном ряде нулевые значения, можем наблюдать низкие амплитуды. Наоборот, там, где высокие значения в исходном ряде, можем наблюдать преобладание высоких амплитуд.

# In[ ]:




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


# In[29]:


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

def sql_least_robust(x,y,m,alpha=0.01):
    n = len(x)
    koef = np.zeros((n, m + 1))
    
    for i in range(0, m+1):
        koef[:, i ] = get_funct(i , x)
    dist = 1.5
    dist = dist
        
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
                    
        z = koef_t @ yr
        a = np.linalg.solve(G, z)
        
        eit = a - ait
        epsit = (sum(eit**2))**0.5/len(eit)
        
        print ('iteration', i, a, epsit)
        
        if (epsit<=eps):
            print ()
            break 
                
    return a
        
        
    
    
    
    
    

def BasicFunctions (num, x):
    PhiFun = x**num
    return PhiFun


# In[4]:


xleft = -3
xright = 6
n = 24
x = np.linspace(xleft,xright,n)
yf = np.zeros ((n))
y = np.zeros ((n))
yf = -3. + 3*x **2
plt.plot (x,yf, 'o')


# In[6]:


Delta = 0.5 * np.random.randn(len(x))
y = yf + Delta
y_av = np.sum(y)/len(x)

y [10] = 80.
y [15] = 70.

plt.plot (x, y, 'o')

print (y[10])


# ### Построим аппроксимационный полином

# Можем видеть, что две точки портят аппроксимацию

# In[16]:


m = 2

PolynomCoefs = SqLeast (x, y, m)
print ('Коэффициенты', PolynomCoefs)

cond = np.linalg.cond (G)
condPhi = np.linalg.cond (Phi)
EigG = np.linalg.eigvals (G)
print ('Числа обусловленности матриц G и Phi', cond, condPhi)

PolynomValue = np.zeros(len(x))
for i in range (0, m+1):
    PolynomValue += PolynomCoefs[i]*x**i
    
plt.plot (x, y, 'o')
plt.plot (x, PolynomValue)


# ### Робастная аппроксимация

# In[19]:


def get_funct(degree, val):
    return val ** degree


# In[30]:


m = 2

PolynomCoefs = sql_least_robust (x, y, m)
print ('Коэффициенты', PolynomCoefs)

cond = np.linalg.cond (G)
condPhi = np.linalg.cond (Phi)
EigG = np.linalg.eigvals (G)
print ('Числа обусловленности матриц G и Phi', cond, condPhi)

PolynomValue = np.zeros(len(x))

for i in range (0, m+1):
    PolynomValue += PolynomCoefs[i]*x**i
    
plt.plot (x, y, 'o')
plt.plot (x, PolynomValue)


# Робастная регуляризация проявила себя лучше

# In[ ]:




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

#!/usr/bin/env python
# coding: utf-8

# # 1.Описательная статистика

# ## Основные характеристики

# In[1]:


import numpy as np
import random
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

default_dpi = mpl.rcParamsDefault['figure.dpi']
factor = 1
mpl.rcParams['figure.dpi'] = default_dpi * factor

n = 1000
X=np.random.sample(n)
#X=np.random.sample((n, 2))

#print (X)

print ('_'*40)
print (len(X))
print(np.mean(X))


# **Среднее значение** $$\frac{\sum_{i=0}^{n} X_i}{n} = \bar{X} = \mu = \mathbb{M}[X] = \mathbb{E}[X]$$

# In[2]:



N = len(X)
print(N)

Mu = sum(X)/N
print(Mu)
print ('_'*40)
print(np.mean(X))


# **Максимальное и минимальное значение**

# In[3]:


Max_x = max(X)
Min_x = min(X)

print('max =', Max_x, '\nmin = ',  Min_x)


# **Дисперсия** $$\sigma^2=\frac{(\sum_{i=1}^{n}(X_i-\mu)^2)}{(n-1)} = \mathbb{D}[X] = Var(X) = \mathbb{M}[(X - \mathbb{M}[X])^2]$$

# In[4]:


Sigma = (sum((X-Mu)**2))/(N-1)
print(Sigma)
print ('_'*40)

print(np.var(X))
print ('_'*40)

print(np.std(X))
print ('_'*40)

print((np.std(X))**2)


# **Коэффициент вариации**

# In[5]:


kv = (np.sqrt(Sigma)/Mu)*100 

print (kv)


# **Медиана**

# In[6]:


#X = np.array([10, 12, 40, 500, 30, 80, 60, 700, 20, 13, 15, 40, 40, 35, 20, 10, 10])
#mu = np.mean(X)
#print (mu)

print('-'*40)
med = np.percentile(X, 50)
med1=np.median(X)
print (med)
print (med1)


# **Мода**

# In[7]:


#X = np.array([10, 12, 40, 500, 30, 80, 60, 700, 20, 13, 15, 40, 40, 35, 20, 10, 10])
st.mode(X)


# **Гистограммы**

# In[8]:



plt.hist(X, bins=10);


# In[9]:


Y = np.random.randn(1000)
plt.hist(Y, bins=10);


# **Ассиметрия и эксцесс**

# Коэффицие́нт эксце́сса в теории вероятностей — мера остроты пика распределения случайной величины

# In[10]:


print (st.skew(X))
print (st.skew(Y))
#W = np.array([0.2, -0.2, 0.5, 0.8, 0.6, 0.98, -0.2, -0.1, 0.9, 0.5, 0.58, 0.1, 0.4, 0.5])
#plt.hist(W, bins=10) 
#print (st.skew(W))

print('_ '*40)
print(st.kurtosis(Y))


# # 2. Функции плотности вероятности  и функции распределения 

# In[11]:


domain = np.linspace(np.max(Y), np.min(Y), 1000)

#print(domain)
kde1 = st.gaussian_kde(Y)

plt.hist(Y, bins=10, density = 1.0) 
plt.plot(domain, kde1(domain));


# In[12]:


domain = np.linspace(np.max(X), np.min(X), 1000)

#print(domain)
kde1 = st.gaussian_kde(X)

plt.hist(X, bins=10, density = 1.0) 
plt.plot(domain, kde1(domain));


# ## Определение
# $$F(x) = P(X<x)$$ $f(x) = F'(x)$ $$F(x) = P(X\le x) = \int_{-\infty}^x f(x) dx$$ или для интервала $$P(a\le X \le b) = \int_a^b f(x) dx = F(b) - F(a)$$ 

# ### Непрерывные СВ

# In[13]:


Contin = st.uniform(loc=0, scale=1)
Z = Contin.rvs(size=10000)
plt.hist(Z, bins=400, density = 1.0); 

grid = np.linspace(np.min(Z), np.max(Z), 100)
plt.plot(grid, Contin.pdf(grid), color = 'red');
plt.figure()
plt.plot(grid, Contin.cdf(grid), color = 'red');


# In[14]:


Contin = st.norm(loc=2, scale=4)
Z_1 = Contin.rvs(size=10000)
plt.hist(Z_1, bins=400, density = 1.0) ;

grid = np.linspace(np.min(Z_1), np.max(Z_1), 100)
plt.plot(grid, Contin.pdf(grid), color = 'black');

Contin = st.norm(loc=0, scale=10)
Z_2 = Contin.rvs(size=10000)
plt.hist(Z_2, bins=400, density = 1.0) ;

grid = np.linspace(np.min(Z_2), np.max(Z_2), 100)
plt.plot(grid, Contin.pdf(grid), color = 'red');

plt.figure()

plt.plot(grid, Contin.cdf(grid), color = 'red');

Contin = st.norm(loc=2, scale=4)
Z_1 = Contin.rvs(size=10000)
grid = np.linspace(np.min(Z_1), np.max(Z_1), 100)
plt.plot(grid, Contin.cdf(grid), color = 'black');


# In[15]:


Contin = st.t(5, 0, 1)
Z_1 = Contin.rvs(size=10000)
plt.hist(Z_1, bins=400, density = 1.0);

grid = np.linspace(np.min(Z_1), np.max(Z_1), 100)
plt.plot(grid, Contin.pdf(grid), color = 'red');

plt.figure()
plt.plot(grid, Contin.cdf(grid), color = 'red');


# In[16]:


Contin = st.laplace(0, 1)
Z_1 = Contin.rvs(size=10000)
plt.hist(Z_1, bins=400, density = 1.0);

grid = np.linspace(np.min(Z_1), np.max(Z_1), 100)
plt.plot(grid, Contin.pdf(grid), color = 'red');

plt.figure()
plt.plot(grid, Contin.cdf(grid), color = 'red');


# ### Дискретные СВ

# In[17]:


Discr = st.bernoulli(0.7)
Z_1 = Discr.rvs(size=100)
print (Z_1)

plt.vlines(Z_1,0, Discr.pmf(Z_1), color = 'red');

grid = np.linspace(0.0, 1.5, 10)
plt.figure()
plt.plot(grid, Discr.cdf(grid), color = 'red');


# In[18]:


Discr = st.binom(5, 0.3)
Z_1 = Discr.rvs(size=100)
print (Z_1)

plt.vlines(Z_1, 0, Discr.pmf(Z_1));

grid = np.linspace(0.0, 5.5, 10)
plt.figure()
plt.plot(grid, Discr.cdf(grid), color = 'red');


# # 3.Моменты СВ, Квантиль, ЦПТ, Репрезентативная выборка
# k-м начальным моментом СВ $$v_k = \mathbb{M}[X^k]$$
# k-м центральным моментом СВ $$\mu_k = \mathbb{M}[(X - \mathbb{M}X)^k]$$
# Репрезентативная выборка $$P(|\widehat{p}-p|<\epsilon)=1-\alpha$$
# точность - $\epsilon$, уровень доверия - $\alpha$
# 
# Квантиль уровня-$\alpha$ $$F(x_\alpha) = \alpha$$

# ### Пример
# Выведем формулу, определяющую необходимый размер выборки для случая биноминального распределения для заданной погрешностью $\epsilon$ и вероятностью $\alpha$
# Возможны 2 случая:
# 1) Выборка формируется с возвращением (т.е. возможны ситуации, когда одно и тоже наблюдение попадет в нашу выборку m раз)
# 2) Без возвращения (когда наблюдение может попасть в выборку только один раз)
# 1. Для $n$ наблюдений будем иметь $x$ наблюдений со значениием 1 и $(n-x)$. Тогда:
# $$P(X=K)=C^k_np^k(1-p)^{n-k}$$
# $$X \sim Bi(n,p)$$
# $$M[X]=np$$
# $$D[X]=np(1-p)=npq$$
# 
# Исходная задача может быть представлена как: $\widehat{p}-p \sim N(0, \frac{pq}{n})$ в силу центральной предельной теоремы (для случая с большим достаточно большим n).
# Тогда:
# $1-\alpha=P(|\widehat{p}-p|<\epsilon)=2*\Phi_0(\frac{\epsilon*\sqrt{n}}{\sqrt{pq}})$, где $\Phi_0$ - функция Лапласса
# $$\Phi_0(\frac{\epsilon*\sqrt{n}}{\sqrt{pq}}) = 1/2 - \alpha/2$$
# Отсюда следует, что:
# $$\frac{\epsilon*\sqrt{n}}{\sqrt{pq}} = Z_{1-\alpha/2}$$, тогда
# $$n=(\frac{Z_{1-\alpha/2}}{\epsilon})^2pq$$
# Поскольку $p$ и $q$ из предыдущей формулы нам не известны по условию, мы можешь задать их $p=q=1/2$, как наихудший случай

# In[19]:


n1 = 1459371 #Решка
n2 = 1184786 #Орел
n = n1+n2 #Общее количество бросков

alpha = 0.95 
z = st.norm 

eps = z.ppf(alpha/2+z.cdf(0))/(2.0*np.sqrt(n)) #радиус доверительного интервала

print(float(n1)/n)
LowI = float(n1)/n-eps #Нижняя граница для вероятности
UpperI= float(n1)/n+eps #Верхняя граница для вероятности
print ('Доверительный интервал:[',LowI,',',UpperI,']')


# # 4.Определение доверительного интервала для неизвестного параметра нормального распределения.

# ## Основные определения
# Пусть есть выборка $X_1, X_2, ... , X_n$ из распределения $F(X, \theta)$, $\theta \in \Theta \subset R^1$. Говорят, что для параметра $\Theta$ построен доверительный интервал уровня надежности $1-\alpha$, где $0 < \alpha < 1$ и найдены такие статистики $T_1(X_1, ..., X_n)$ и $T_2(X_1, ..., X_n)$, что  $T_1(X_1, ..., X_n) < T_2(X_1, ..., X_n)$ и $P(T_1(X_1, ..., X_n) < \theta < T_2(X_1, ..., X_n)) = 1 - \alpha$,
# Интервал со случайными концами $(\ T_1(X_1, ..., X_n), T_2(X_1, ..., X_n)\ )$ называется доверительным интервалом для параметра $\theta$ уровня надежности $1-\alpha$
# 
# Случайная функция $G(X,\theta)$ называется центральной статистикой, если она удовлетворяет 2 условиям:
#     $$1)$$ $G(X,\theta)$ строгомонотонна и непрерывна по $\theta$ 
#     $$2)$$ Функция распределения $G$ $F_{G}$ не зависит от $\theta$. Т.е. $F_{G}(X)$
# 
# Дальнейший ход определения доверительных интервалов для параметра $\theta$ будет заключаться в построении неравенства 
# $P(F_{G}(X)<Z_{1-\alpha})=1-\alpha$, либо $P(F_{G}(X)>Z_{\alpha})=1-\alpha$, либо $P(Z_{\alpha/2}<F_{G}(X)<Z_{1-\alpha/2})=1-\alpha$ (взависимости от того, какой тип доверительного интервала (правый односторонний, левый односторонний или двусторонний соответственно) нам нужен) и преобразовании его к виду из определения доверительного интервала с целью нахождения его границ. Отдельно стоит отметить, что $Z_\alpha$ - в данном случае это квартиль уровня $\alpha$ распределения $F_{G}(X)$.

# ## Одновыборочный случай

# ### 1. Неизвестное мат. ожидание при известной дисперсии
# Дано: $X_1, ..., X_n \sim N(\theta, \sigma^2)$
# 
# В качестве оценки для мат. ожидания будем использовать выборочное среднее: $\widehat{\theta} = 1/n * \sum_{i=0}^{n} X_i = \bar{X}$
# 
# 
# центральная статистика $$\frac{(\widehat{\theta} - \theta)*\sqrt{n}}{\sigma} \sim N(0, 1)$$
# Тогда доверительный интервал можно построить след. образом: $$P(Z_{\alpha/2}<\frac{(\widehat{\theta} - \theta)*\sqrt{n}}{\sigma}<Z_{1-\alpha/2})=1-\alpha$$, тогда в силу $z_{\alpha/2} = - z_{1-\alpha/2}$ для нормального распределения, получаем
# $$P(\widehat{\theta} - \frac{Z_{1-\alpha/2}*\sigma}{\sqrt{n}} <\theta<\frac{Z_{1-\alpha/2}*\sigma}{\sqrt{n}} + \widehat{\theta} )=1-\alpha$$
# Тогда доверительный интервал для $\sigma$ в соответствии с определением дов. интервала определяется:
# $$(\ \bar{X} - \frac{Z_{1-\alpha/2}*\sigma}{\sqrt{n}}\ ,\ \bar{X} + \frac{Z_{1-\alpha/2}*\sigma}{\sqrt{n}}\ )$$

# ### 2. Неизвестная дисперсия при известном мат. ожидании
# 
# Дано: $X_1, ..., X_n \sim N(\mu, \theta_2^2)$
# 
# Для начала построим оценку для дисперсии. Возьмем выборочную дисперсию: $$\widehat{\theta}^2=S^2=1/n*\sum_{i=1}^{n}(X_i-\mu)^2$$
# Данная оценка имеет распределение очень похожее на $\chi^2$. Отнормируем слогаемые на среднеквадратичное отклонение и умножим сумму на n, тогда получим след. статистику: $$\sum_{i=1}^{n}(\frac{(X_i-\mu)}{\theta_2})^2 \sim \chi^2(n)$$, где $n$ - кол-во степеней свободы распределения $\chi^2$
# 
# Данная статистика будет являться центральной статистикой (по определению). Соответственно, можем найти границы доверительного интервала для $\theta_2$. Для этого построим неравенство:
# $$P(\chi^2_{\alpha/2, n} < \sum_{i=1}^{n}(\frac{(X_i-\mu)}{\theta_2})^2 < \chi^2_{1-\alpha/2, n})=1-\alpha$$, где $\chi^2_{\alpha, n}$ - квантиль уровня $\alpha$ распределения $\chi^2$ с $n$ степенями свободы.
# $$(\frac{\sum_{i=1}^{n}(X_i-\mu)^2}{\chi^2_{1-\alpha/2, n}},\ \frac{\sum_{i=1}^{n}(X_i-\mu)^2}{\chi^2_{\alpha/2, n}})$$

# ### 3. Неизвестная дисперсия при неизвестном мат. ожидании
# 
# Дано: $X_1, ..., X_n \sim N(\theta_1, \theta_2^2)$
# 
# $$(\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{\chi^2_{1-\alpha/2, n-1}},\ \frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{\chi^2_{\alpha/2, n-1}})$$

# ### 4. Неизвестное мат. ожидание при неизвестной дисперсии
# 
# Дано: $X_1, ..., X_n \sim N(\theta_1, \theta_2^2)$
# 
# $$P(t_{\alpha/2, n-1}<\frac{(\bar{X} - \theta_1)*\sqrt{n}}{S}<t_{1-\alpha/2, n-1})=1-\alpha$$
# Тогда доверительный интервал будет задаваться: 
# $$(\bar{X} - \frac{t_{1-\alpha/2, n-1}*S}{\sqrt{n}}, \frac{t_{1-\alpha/2, n-1}*S}{\sqrt{n}} + \bar{X} )$$

# **Работа с данными:** https://www2.stat.duke.edu/courses/Spring01/sta114/data/andrews.html (7. Stanford Heart Transplant Data)

# In[20]:


#наиболее простой способ считать данные из таблицы 
Z = np.loadtxt('T07.1');
print(Z.shape)


# In[21]:


class DataSet(object):
    
    def __init__(self, id, day, life, age, titr):
        """Constructor"""
        self.id = id
        self.day = day
        self.life = life
        self.age = age
        self.titr = titr


FILE = open('T07.1', 'r')
FILE.seek(0)

n = sum(1 for line in FILE);
FILE.seek(0)

DS = [DataSet(0,0, 0, 0, 0) for x in range(n)];

DS_day = []
DS_life = []
DS_age = []
DS_titr = []


i=0     
for line in FILE:
    S=line.split(' ')
    S=[x for x in S if x]
    #print(S)
    
    DS[i].id = int(S[3]) 
    DS[i].day = float(S[4])
    DS[i].life = int(S[5])
    DS[i].age = float(S[6])
    DS[i].titr = float(S[7])
    
    DS_day.append(float(S[4]));
    DS_life.append(int(S[5]));
    DS_age.append(float(S[6]))
    DS_titr.append(float(S[7]))
    
    i=i+1;
    
    
FILE.close()

N = float(n)
print(N)

mu = 0
for i in range(n):
    mu=mu+DS[i].age
    
print(mu/N)
print(np.mean(DS_age))
print(type(DS))


# In[22]:


plt.hist(DS_age, bins=20, density = 1.0);


# In[23]:


plt.hist(DS_age, bins=20, density = 1.0);

#alfa = 0.05 
mu = np.mean(DS_age)
print(mu)
print(np.var(DS_age))
print('-'*40)
St = (sum((DS_age-mu)**2))


LowI = St/st.chi2.ppf(0.975, 183) #Нижняя граница 
UpperI= St/st.chi2.ppf(0.025, 183) #Верхняя граница 

print ('Доверительный интервал Sigm^2:[',LowI,',',UpperI,']')


# **Выбросы в данных**

# In[24]:


plt.hist(DS_titr, bins=20, density = 1.0);


# In[25]:


print(DS_titr)
print('\n')
for i in range(DS_titr.count(-9999)):
    DS_titr.remove(-9999) 
    
print(DS_titr)
#функции которые будем использовать
#list.remove()
#list.count() 


# In[26]:


plt.hist(DS_titr, bins=20, density = 1.0);


# След. занятие.
# 
# # Проверка статистических гипотез
# ## Методика проверки статистических гипотез
# 
# ### 1. Достигаемый уровень значимости (p-value)
# ### 2. Критерии согласия
# ### 3. Критерии сдвига
# ### 4. Критерии нормальности
# ### 5. Коэффициент корреляции Пирсона. Статистическая проверка наличия корреляции. 

# In[ ]:





# In[ ]:





# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # Проверка статистических гипотез. Параметрические критерии.
# ## Методика проверки статистических гипотез
# 
# #### 1. Сформулировать нулевую гипотезу $H_0$ и альтернативную - $H_1$ (Гипотеза $H_0$ формулируется исходя из требований прикладной задачи. Иногда альтернатива не формулируется в явном виде; тогда предполагается, что $H_1$ означает «не $H_0$»).
# #### 2. Сформулировать статистические допущения о распределении выборки.
# #### 3. Выбрать тестовую статистику, которая отражает характер допущений.
# #### 4. Зафиксировать уровень значимости $1 \le \alpha \le 0$.
# #### 5. Рассчитать наблюдаемое значение выбранной статистики.
# #### 6. Сдалать вывод о гипотезе $H_0$:
#    6.1 На основе критической области $\Omega_\alpha$
#    
#    6.2 На основе достигаемого уровня значимости (p-value)
# 

# ## p-value
# Достигаемый уровень значимости (p-value) — это наименьшая величина уровня значимости, при которой нулевая гипотеза отвергается для данного значения статистики критерия T:$$p(T) = min \{\alpha: T \in \Omega_\alpha \}$$

# ## Критические области
# Пусть $x_\alpha$ - квантиль уровня-$\alpha$ функции распределения $F$ статистики $T$: $F(x_\alpha) = \alpha$. Здесь $F(t) = P(T<t)$
# 1. Левосторонняя критическая область:
# 
# Определяется интервалом: $\Omega_\alpha \in \{-\infty, x_\alpha \}$
# 
# p-value: $P(T) = F(T)$
# 
# 2. Правосторонняя критическая область:
# 
# Определяется интервалом: $\Omega_\alpha \in \{x_{1-\alpha}, \infty \}$
# 
# p-value: $P(T) = 1-F(T)$
# 
# 3. Двухсторонняя критическая область:
# 
# Определяется интервалом: $\Omega_\alpha \in \{-\infty, x_{\alpha/2} \}U\{x_{1-\alpha/2}, \infty \}$
# 
# p-value: $P(T) = min\{2F(T), 2(1-F(T))\}$

# ## Ошибки первого и второго рода
# 
# 1. Ошибка первого рода («ложная тревога»)  — когда нулевая гипотеза отвергается, хотя на самом деле она верна. Вероятность ошибки первого рода:$$\alpha = P\{T \in \Omega_\alpha | H_0\}$$
# 2. Ошибка второго рода («пропуск цели»)  — когда нулевая гипотеза принимается, хотя на самом деле она не верна. Вероятность ошибки второго рода:$$\beta = P\{T \notin \Omega_\alpha | H_1\}$$
# Мощность критерия $1-\beta(H) = P\{T \in \Omega_\alpha | H\}$ - вероятность отклонить гипотезу $H_0$, если на самом деле верна альтернативная гипотеза $H$. (Мощность критерия является числовой функцией от альтернативной гипотезы $H$)

# In[4]:


import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

default_dpi = mpl.rcParamsDefault['figure.dpi']
factor = 1
mpl.rcParams['figure.dpi'] = default_dpi * factor

X = np.random.randn(400)

X = 5.0+X*2

plt.hist(X, bins=10, density = 1);


# # T-test
# 
# Дано: $X_1, ..., X_n \sim N$ 
# 
# $$H_0: \mu = a $$ $$H_1: \mu > a$$
# 
# $$\frac{\sqrt{n}*(\bar{X}-a)}{S} \sim t(n-1)$$
# 

# In[8]:


print(len(X))
X_mean = np.mean(X);
print(X_mean)
Std = np.std(X);
print(Std)
print('_'*40)

alpha = 0.05;
a = 5.5;

St = np.sqrt(len(X))*(X_mean-a)/Std
print('Stat = ', St)
Krit = st.t.ppf(0.95, len(X)-1)
print('\nkrit = ', Krit)
print('p-value = ', st.t.sf(St, len(X)-1))


# $$H_0: \mu = a $$ $$H_1: \mu < a$$

# In[9]:


Krit = st.t.ppf(0.05, len(X)-1)
print('krit = ', Krit)
print('p-value = ', st.t.cdf(St, len(X)-1))


# $$H_0: \mu = a $$ $$H_1: \mu \neq a$$

# In[10]:



print('krit = [', st.t.ppf(0.025, len(X)-1),',',st.t.ppf(0.975, len(X)-1),']')

print('p-value = ', min(2.0*st.t.cdf(St, len(X)-1), 2.0*(1.0-st.t.cdf(St, len(X)-1))))


# In[11]:


st.ttest_1samp(X, a)


# In[12]:


#Построим доверительный интервал
X_mean-(Std*st.t.ppf(0.975, len(X)-1))/np.sqrt(len(X)), X_mean+(Std*st.t.ppf(0.975, len(X)-1))/np.sqrt(len(X))


# ## Двувыборочный случай
# 
# ### Критерий сдвига
# 
# 1. Неизвестная разница мат. ожиданий при неизвестных, но не равных дисперсиях
# 
# Дано: $$X_1, ..., X_{n_1} \sim N(\mu_1, \sigma_1^2)$$
# $$Y_1, ..., Y_{n_2} \sim N(\mu_2, \sigma_2^2)$$
# $X$ и $Y$  - независимы, $\sigma_1 != \sigma_2$ 
# $$H_0: \mu_1-\mu_2 = \theta $$ $$H_1: \mu_1-\mu_2 \neq \theta$$
# Центральная статистика имеет вид: $$\frac{\bar{X}-\bar{Y}-\theta}{\sqrt{S_1^2/n_1+S_2^2/n_2}} \sim t(k)$$.
# $$k=\frac{(s_1^2/n_1+s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1-1)+(s_2^2/n_2)^2/(n_2-1)}$$
# 

# In[13]:


def df(X_1, X_2):
    n_1 = len(X_1);
    n_2 = len(X_2);
    s_1 = (sum((X_1-np.mean(X_1))**2))/(n_1-1.0)#np.var(X_1);
    s_2 = (sum((X_2-np.mean(X_2))**2))/(n_2-1.0)#np.var(X_2);
    
    k = (((s_1/n_1)+(s_2/n_2))**2)/((((s_1/n_1)**2)/(n_1-1)) + (((s_2/n_2)**2)/(n_2-1)));
    
    return k;

Y = np.random.randn(22000);
X = np.random.randn(17000);
X = 4.0*X+1.0

plt.hist(Y, bins=10, density = 1);
plt.hist(X, bins=10, density = 1);


# In[14]:


theta = 1.0;

x_mean = np.mean(X);
S_1 = (sum((X-np.mean(X))**2))/(len(X)-1.0);
print('x_mean = ', x_mean, 'S1^2 = ', S_1, 'n1 = ', len(X));

y_mean = np.mean(Y);
S_2 = (sum((Y-np.mean(Y))**2))/(len(Y)-1.0);
print('y_mean = ', y_mean, 'S2^2 = ', S_2, 'n2 = ', len(Y));

St = (x_mean-y_mean-theta)/np.sqrt((S_1/len(X))+(S_2/len(Y)))
print('St = ', St);

k = int(df(X, Y));
print('k = ', k);

print('_'*40);
print('Krit1 = ', st.t.ppf(0.025, k));
print('Krit2 = ', st.t.ppf(0.975, k));


# 2. Неизвестная разница мат. ожиданий при неизвестных, но равных дисперсиях
# 
# Дано: $$X_1, ..., X_{n_1} \sim N(\mu_1, \sigma_1^2)$$
# $$Y_1, ..., Y_{n_2} \sim N(\mu_2, \sigma_2^2)$$
# $X$ и $Y$  - независимы, $\sigma_1 = \sigma_2$ 
# $$H_0: \mu_1-\mu_2 = \theta $$ $$H_1: \mu_1-\mu_2 \neq \theta$$
# Центральная статистика имеет вид: $$\frac{\bar{X}-\bar{Y}-\theta}{S_x\sqrt{1/n_1+1/n_2}} \sim t(n_1+n_2-2)$$.
# $$S_x^2=\frac{(s_1^2(n_1-1)+s_2^2(n_2-1))}{n_1+n_2-2}$$

# In[ ]:





# ## Проверка гипотез про равенство дисперсий
# 
# Дано: $$X_1, ..., X_n \sim N(\mu_1, \sigma_1^2)$$
# $$Y_1, ..., Y_m \sim N(\mu_2, \sigma_2^2)$$
# $X$ и $Y$  - независимы, требуется проверить гипотезу $H_0: \sigma_1 = \sigma_2$
# 
# Сначала считаем выборочные дисперсии:
# $$\widehat{\sigma_1}=\frac{1}{n-1}*\sum_{i=1}^{n}(X_i-\bar{X})^2$$
# $$\widehat{\sigma_2}=\frac{1}{m-1}*\sum_{i=1}^{m}(Y_i-\bar{Y})^2$$
# 
# Cтатистика: $$F=\widehat{\sigma_1}/\widehat{\sigma_2} \sim F(n-1, m-1)$$

# In[15]:


Y = np.random.randn(220);
X = np.random.randn(170);
X = 1*X


x_mean = np.mean(X);
S_1 = (sum((X-np.mean(X))**2))/(len(X)-1.0);
n = len(X);
print('x_mean = ', x_mean, 'S1^2 = ', S_1, 'n1 = ', n);

y_mean = np.mean(Y);
S_2 = (sum((Y-np.mean(Y))**2))/(len(Y)-1.0);
m = len(Y);
print('y_mean = ', y_mean, 'S2^2 = ', S_2, 'n2 = ', m);

St = S_2/S_1;

print('St = ', St)

print('_'*40);
print('Krit1 = ', st.f.ppf(0.025, n-1, m-1));
print('Krit2 = ', st.f.ppf(0.975, n-1, m-1));


# # Критерии согласия.

# ## Критерий соласия $\chi^2$ Пирсона

# Критерий Пирсона, или критерий χ2(Хи-квадрат) - применяют для проверки гипотезы о соответствии эмпирического распределения предполагаемому теоретическому распределению F(x) при большом объеме выборки (n ≥ 100). Критерий применим для любых видов функции F(x), даже при неизвестных значениях их параметров, что обычно имеет место при анализе результатов механических испытаний. В этом заключается его универсальность.
# 
# Использование критерия χ2 предусматривает разбиение размаха варьирования выборки на интервалы и определения числа наблюдений (частоты) для каждого из интервалов. Для удобства оценок параметров распределения интервалы выбирают одинаковой длины. Число интервалов зависит от объема выборки.

# In[16]:


Y = np.random.randn(1000);
plt.hist(Y, bins=10, density = 1);


# In[17]:


k = 15
N = len(Y)
print(N)

grid = np.linspace(min(Y), max(Y), k)
print(grid)

#строим статистический ряд
f_obs = np.zeros(k-1)

for i in range(k-1):
    for j in range(N):
        if Y[j] == grid[i] and i == 0:
            f_obs[i] = f_obs[i] + 1.0
        elif (Y[j] > grid[i] and Y[j] <= grid[i+1]):
            f_obs[i] = f_obs[i] + 1.0
            
print(f_obs)
#f_obs =f_obs/N;
#print(f_obs)
print(np.sum(f_obs));

#for i in range(k-1):
#    x_i[i] = grid[i+1] - grid[i];

    
f_exp = np.zeros(k-1);


for i in range(k-1):
    f_exp[i] = N*(st.norm.cdf(grid[i+1]) - st.norm.cdf(grid[i]));#z.cfd(grid[i+1]) - z.cfd(grid[i])
 
print(f_exp)

print('\n',st.chisquare(f_obs, f_exp), '\n', '_'*40)

St = 0.0;

for i in range(k-1): #считаем критерий согласия
    St = St + ((f_obs[i]-f_exp[i])**2)/f_exp[i]

print('Stat = ', St)
print('Krit1 = ', st.chi2.ppf(0.025, k-1))
print('Krit2 = ', st.chi2.ppf(0.975, k-1))


# ## Критерий нормальности Шапиро-Уилки (Shapiro-Wilk)

# In[18]:


#X = np.random.rand(5000);
#X = np.random.randn(50);
st.shapiro(Y)


# ## Статистическая проверка наличия корреляции. Коэффициент корреляции Пирсона

# $$r_{xy} = \frac {\sum_{i=1}^{m} \left( x_i-\bar{x} \right)\left( y_i-\bar{y} \right)}{\sqrt{\sum_{i=1}^{m} \left( x_i-\bar{x} \right)^2 \sum_{i=1}^{m} \left( y_i-\bar{y} \right)^2}} = \frac {cov(x,y)}{\sqrt{s_x^2 s_y^2}}$$
# 
# $|r_{xy} | =1 $ x, y линейно зависимы, 
# 
# $r_{xy}=0 $ x, y линейно независимы

# In[19]:


X = np.random.randn(5000);
Y = np.random.randn(5000);

print(st.pearsonr(X, Y))
plt.plot(X, Y, '*');


Y = 2.0*X + 17

print(st.pearsonr(X, Y))
plt.figure()
plt.plot(X, Y, '*');


# $$H_0: r_{xy} = 0$$
# 
# $$T = \frac{r_{xy}\sqrt{n-2}}{\sqrt{1-r^2_{xy}}} \sim t_{n-2}$$ $T \in [t_\alpha,t_{1-\alpha}]$

# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # Непараметрические критерии

# In[2]:


import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

default_dpi = mpl.rcParamsDefault['figure.dpi']
factor = 1
mpl.rcParams['figure.dpi'] = default_dpi * factor


# ### Двувыборочный критерий Колмогорова-Смирнова (тест на однородность).
# Статистика критерия задается следуюим образом: $$D_{n,\;m}=\sup _{x}|F_{1,\;n}-F_{2,\;m}|$$.
# Теорема Смирнова.
# Пусть $ F_{1,\;n}(x),\;F_{2,\;m}(x)$  — эмпирические функции распределения, построенные по независимым выборкам объёмом  n и m случайной величины $\xi$. Тогда, если $ F(x)\in C^{1}(\mathbb {X} )$, то $$ \forall t>0\colon \lim _{n,\;m\to \infty }P\left({\sqrt {\frac {nm}{n+m}}}D_{n,\;m}\leqslant t\right)=K(t)=\sum _{j=-\infty }^{+\infty }(-1)^{j}e^{-2j^{2}t^{2}}$$ 
# Если статистика $ {\sqrt {\frac {nm}{n+m}}}D_{n,\;m}$  превышает квантиль распределения Колмогорова $ K_{\alpha }$  для заданного уровня значимости $ \alpha $ , то нулевая гипотеза $ H_{0}$  (об однородности выборок) отвергается. Иначе гипотеза принимается на уровне $\alpha $

# In[3]:


#Одновыборочный
X = np.random.randn(55)

print(st.kstest(X, 'norm'))
print('-'*40)

X = 2.0+X*4.0;

print(st.kstest(X, 'norm'))
print('-'*40)

print(st.kstest(X, 'norm', alternative = 'less'))
print('-'*40)

print(st.kstest(X, 'norm', alternative = 'greater'))
print('-'*40)

print(st.kstest(X, 'norm', args=(2.0, 4.0)))
print('-'*40)


# In[4]:


#Двувыборочный
Y = np.random.randn(200);
X = np.random.randn(120);

print(st.ks_2samp(X, Y))
print('-'*40)

X = 2.0*X
print(st.ks_2samp(X, Y))


# ## Ранговые критерии
# ### Различия между независимыми выборками
# #### Критерий Мани-Уитни (U-test)
# 
# Заданы две выборки $X_n, Y_m$
# 
# 1. обе выборки независимые;
# 2. выборки взяты из неизвестных непрерывных распределений F(x) и G(y).
# 
# Нулевая гипотеза: $$H_0:\; \mathbb{P} \{ x<y \} = 1/2$$
# Построить общий вариационный ряд объединённой выборки $x^{(1)} \leq \cdots \leq x^{(m+n)}$ и найти ранги $r(x_i),\; r(y_i)$ всех элементов обеих выборок в общем вариационном ряду.
# Вычислить суммарные ранги обеих выборок и статистику Манна-Уитни  $U$:
# $$R_x = \sum_{i=1}^m r(x_i);\;\;\;\; U_x = mn + \frac12m(m+1) - R_x;$$
# $$R_y = \sum_{i=1}^n r(y_i);\;\;\;\; U_y = mn + \frac12n(n+1) - R_y;$$
# $$U = \min\left\{U_x,U_y\right\}$.$$
# 
# Менее рациональный способ вычисления статистик Манна-Уитни $U_x,\: U_y$:
# $$U_x = \sum_{i=1}^m \sum_{j=1}^n \left[ x_i < y_j\right];$$
# $$U_y = \sum_{i=1}^m \sum_{j=1}^n \left[ x_i > y_j\right].$$
# **Критерий (при уровне значимости $\alpha$):**
# 
# Критическая область асимптотического критерия Манна-Уитни.
# 
# против альтернативы $H_1:\; \mathbb{P} \{ x<y \} \neq 1/2$
# $$U \notin \left[ U_{\alpha/2},\, U_{1-\alpha/2} \right]$$
# 
# $H_1:\; \mathbb{P} \{ x<y \} > 1/2$
# $$U_x > U_{1-\alpha} $$
# $H_1:\; \mathbb{P} \{ x<y \} < 1/2$
# $$U_y > U_{1-\alpha}$$
# где $ U_{\alpha}$  есть $\alpha$-квантиль табличного распределения Уилкоксона-Манна-Уитни с параметрами $m,\,n$.
# 
# Критические значения критерия Манна-Уитни можно найти, например, в справочнике: Кобзарь А. И. Прикладная математическая статистика. — М.: Физматлит, 2006. — 816 с. [455] 
# В Python они затабулированы, для n>20 действует нормальная аппроксимация.

# In[39]:


#st.mannwhitneyu T H H H H H T T T T T H
x = np.array([1, 0, 0, 0, 0, 0]);
y = np.array([1, 1, 1, 1, 1, 0]);
z = np.hstack((x, y));

m = len(x);
n = len(y);

R = st.rankdata(z, method='average')

print(z)
print(R)

R_x = 0;
R_y = 0;

for i in range(m):
    R_x = R_x+R[i]
    R_y = R_y+R[m+i]

print(R_x, R_y)


U_x = (m*n) + ((m*(m+1))/2.0) - R_x;
U_y = (m*n) + ((n*(n+1))/2.0) - R_y;
print('\nПроверка U_x+U_y = n*m\n', U_x+U_y, '=',n*m)

U = min(U_x, U_y);

print('\n', U_x, U_y, U)

print(st.mannwhitneyu(x, y))

x = np.random.randn(10)
y = np.random.randn(10)

print(st.mannwhitneyu(x, y))


# ### Критерий Краскера-Уоллиса
# Проверяется нулевая гипотеза $ H_{0}\colon F_{1}(x)=\ldots =F_{k}(x)$  при альтернативе $ H_{1}\colon F_{1}(x)=F_{2}(x-\Delta _{1})=\ldots =F_{k}(x-\Delta _{k-1})$ 

# In[34]:


x_1 = np.random.randn(1000)
x_2 = np.random.randn(1000)
x_3 = np.random.randn(1000)
x_4 = np.random.randn(1000)

st.kruskal(x_1, x_2, x_3, x_4)


# ## Различия между зависимыми выборками
# ### Критерий Уилкоксона (Критерий знаковых рангов)
# Данные приходят парами
# 
# Гипотеза $H_0$: медиана разностей в парах равна 0
# 
# Альтернативная - $H_1$: медиана разностей в парах не равна 0
# 
# Пусть N — размер выборки (число пар). Обозначим $x_{1,i}$ — элементы 1 выборки и $x_{2,i}$ — элементы 2 выборки.
# 
# 1. Для $i = 1, ..., N$, вычислить $|x_{2,i} - x_{1,i}|$ и $sign(x_{2,i} - x_{1,i})$
# 
# 2. Исключить пары, где $|x_{2,i} - x_{1,i}| = 0.$ Пусть $N_r$ — размер полученной выборки после удаления таких пар
# 
# 3. Упорядочить оставшиеся $N_r$ пар в порядке возрастания модуля разности, $|x_{2,i} - x_{1,i}|$.
# 
# 4. Построить ранги всех пар, $R_i$ обозначает ранг i-й пары.
# 
# 5. Вычислить статистику W $$W = |\sum_{i=1}^{N_r} [sign(x_{2,i} - x_{1,i}) \cdot R_i]|$$, модуль суммы знаковых рангов.

# In[7]:


t_1 = np.random.randn(10)
t_2 = np.random.randn(10)
#t_2 = t_1

st.wilcoxon(t_1,t_2)


# ## Коэффициент корреляции Спирмена
# 
# Заданы две выборки $x = (x_1,\ldots,x_n),\;\; y = (y_1,\ldots,y_n).$
# 
# 
# Коэффициент корреляции Спирмена вычисляется по формуле: $$\rho=1-\frac{6}{n(n-1)(n+1)}\sum_{i=1}^n(R_i-S_i)^2,$$где R_i - ранг наблюдения $x_i$ в ряду $x$, $S_i$ - ранг наблюдения $y_i$ в ряду $y$.
# Коэффициент $\rho$ принимает значения из отрезка $[-1;\;1]$. Равенство $\rho=1$ указывает на строгую прямую линейную зависимость, $\rho=-1$ на обратную.

# In[8]:


print(t_1)
R = st.rankdata(t_1, method='min')
print(R)

print('\n', '-'*40)

print(t_2)
S = st.rankdata(t_2, method='min')
print(S)

d = 0.0;
for i in range(len(t_1)):
    d = d + (R[i]-S[i])**2

n = float(len(t_1))

my_sp = 1.0 - ((6.0*d)/(n*(n-1.0)*(n+1.0)))
print(my_sp)

print(st.spearmanr(t_1, t_2))


# **Если есть связные ранги**

# In[ ]:





# ## Bootstrap

# In[20]:


X = np.random.randn(1500);
X = 2.0 + 4.0*X
plt.hist(X, bins = 10);
print('Shapiro = ',st.shapiro(X), '\n');


# In[40]:


#print(X);
from sklearn.utils import resample

boot = []

print(len(boot))

for i in range(5000):
    imit = resample(X, replace = True);
    boot.append(np.mean(imit))
    
print(len(boot))

plt.hist(boot, bins = 100);
print (st.skew(boot))
print('Shapiro = ',st.shapiro(boot), '\n');

alpha = 0.05
#boot = np.sort(boot);
print('bootstrap: ', np.percentile(boot,50), '\n', [np.percentile(boot,(alpha/2)*100.0), np.percentile(boot,100-(alpha/2)*100)])


S = np.std(X);
delta = (st.t.ppf(1.0-(alpha/2.0), len(X)-1)*S)/np.sqrt(float(len(X)))

#print('\n', np.percentile(boot,100-(alpha/2)*100)-np.percentile(boot,(alpha/2)*100.0))
#print(2*delta)

print('\n\ntheoretical: ', np.mean(X), '\n', [np.mean(X)-delta, delta+np.mean(X)])


# In[ ]:





# In[ ]:





# In[ ]:




