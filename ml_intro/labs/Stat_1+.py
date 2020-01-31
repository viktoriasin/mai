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




