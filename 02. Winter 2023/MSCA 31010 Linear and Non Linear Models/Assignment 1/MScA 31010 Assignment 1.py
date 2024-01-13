#!/usr/bin/env python
# coding: utf-8

# # Assignment 1
# 
# Author: Shweta Sampath Kumar
# Created: 1/15/2023
# MScA 31010 Linear and Non Linear Models

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import math



# ## --------------------------------------------- QUESTION 1 ----------------------------------------------------------------

# In[2]:


econ = pd.read_csv("Economy_2020_to_2022.csv")
print(econ.head())

print("QUESTION 1\n\n")
# ### (a) Generate a matrix of scatter plot (SPLOM) of these seven features: PCEPI, CPIAUCSL, ICSA_Week1, ICSA_Week2, ICSA_Week3, ICSA_Week4, and ICSA_Week5. You mut properly label the axes and add grid lines to all the scatter plots.

# In[3]:


fig = px.scatter_matrix(econ,
                        width = 800,
                        height = 1000,
                        dimensions = ['PCEPI', 'CPIAUCSL', 'ICSA_Week1', 'ICSA_Week2', 'ICSA_Week3', 'ICSA_Week4', 'ICSA_Week5'])
fig.show()

# ### (b) Calculate the Pearson correlations for each pair of the seven features. Display your result up to four decimal places appropriately as a matrix.

# In[4]:


cols = ['PCEPI', 'CPIAUCSL', 'ICSA_Week1', 'ICSA_Week2', 'ICSA_Week3', 'ICSA_Week4', 'ICSA_Week5']
pearsoncorr = round(econ[cols].corr(method='pearson'), 4)
print("(b)")
print("The Pearson correlation for these features are:")
print(pearsoncorr)
print()


# ### (c) Calculate the Spearman rank-order correlations for each pair of the seven features. Display your result up to four decimal places appropriately as a matrix.

# In[5]:


spearman_corr = round(econ[cols].corr('spearman'), 4)
print("(c)")
print("The Spearman correlation for these features are:")
print(spearman_corr)
print()

# ### (d) Calculate the Kendall’s Tau-b correlations for each pair of the seven features. Display your result up to four decimal places appropriately as a matrix.

# In[6]:


kendalls_corr = round(econ[cols].corr('kendall'), 4)
print("(d)")
print("The Kendall's Tau-b correlation for these features are:")
print(kendalls_corr)
print()

# ### (e) Calculate the Distance correlations for each pair of the seven features. Display your result up to four decimal places appropriately as a matrix.

# In[7]:


def empirical_distance(M):
    m = []
    m_mean = []
    for x in M:
        l = []
        for i in M:
            l.append(abs(x-i))
        m.append(l)
        m_mean.append(sum(l)/len(l))

    m = np.matrix(m)
    m_adjusted = []
    total_mean = sum(m_mean)/len(m_mean)
    c = m.shape[1]
    s = 0
    for i in enumerate(m):
        l = []
        for j in range(c):
            x = m.item(i[0], j) - m_mean[i[0]] - m_mean[j] + total_mean
            s = s + (x*x) 
            l.append(x)
        m_adjusted.append(l)
    vn = s/(c*c)
    return vn, np.matrix(m_adjusted)

def distance_correlation(A, B):
    vn_A, s1 = empirical_distance(A)
    vn_B, s2 = empirical_distance(B)
    s = 0
    for i in enumerate(s1):
        for j in range(s1.shape[1]):
            s = s + (s1.item(i[0], j) * s2.item(i[0], j))

    vn_AB = s/(len(A) * len(B))

    R_squared = vn_AB/(math.sqrt(vn_A * vn_B))
    R = math.sqrt(R_squared)
    
    return R

cols = ['PCEPI', 'CPIAUCSL', 'ICSA_Week1', 'ICSA_Week2', 'ICSA_Week3', 'ICSA_Week4', 'ICSA_Week5']
d = []
for x in cols:
    d_row = []
    for y in cols:
        df = pd.concat([econ[x], econ[y]], axis = 1)
        df = df.dropna()
        distance_corr = distance_correlation(df.iloc[:,0], df.iloc[:,1])
        d_row.append(round(distance_corr, 4))
    d.append(d_row)

distancecorr = pd.DataFrame(np.matrix(d), columns = cols, index = cols, dtype = 'float32')
print("(e)") 
print("The Distance correlation for these features are:") 
print(distancecorr)
print()

# ## --------------------------------------------- QUESTION 2 ----------------------------------------------------------------

# ### (a) What is the first derivative of the function 𝑓(𝑥)=𝑥^{2}−𝑎 with respect of 𝑥?

# ### f'(x) = 2x

# ### (b) You will use the Newton-Raphson method to solve the equation 𝑓(𝑥)=𝑥^2−𝑎=0. What is the formula for updating the estimate?

# ### Formula to update the estimate:
# 
# ### x_{n+1} = x_n - f(x_n)/f'(x_n)
# Solving for f(x) = x^2 - a
# f(x) = x^2 - a
# f'(x) = 2x
# Let a = 2, n = 0
# Thus, x_0 = 1
# x_1 = x_0 - f(x_0)/f'(x_0) = 1 - ((1)^2 - 2)/2(1) = 1.5
# x_2 = x_1 - f(x_1)/f'(x_1) = 1.5 - ((1.5)^2 - 2)/2(1.5) = 1.416666666666667
# x_3 = x_2 - f(x_2)/f'(x_2) = 1.416666666666667 - ((1.416666666666667)^2 - 2)/2(1.416666666666667) = 1.41421568627451
# x_4 = x_3 - f(x_3)/f'(x_3) = 1.41421568627451 - ((1.41421568627451)^2 - 2)/2(1.41421568627451) = 1.41421356237469
# x_5 = x_4 - f(x_4)/f'(x_4) = 1.41421356237469 - ((1.41421356237469)^2 - 2)/2(1.41421356237469) = 1.414213562373095
# x_6 = x_5 - f(x_5)/f'(x_5) = 1.414213562373095 - ((1.414213562373095)^2 - 2)/2(1.414213562373095) = 1.414213562373095
# The root for this is 1.414213562373095

# In[8]:
print("QUESTION 2\n\n")

#Using Python to show the above           
def func (x, a):
   y = x * (x) - a
   return (y)

def dfunc(x):
   dy = 2 * x
   return (dy)

def newton_raphson (init_x, a, max_iter, eps_conv, q_history):
   i_iter = 0
   q_continue = True
   reason = 0
   x_curr = init_x

   if (q_history):
      history = []
   while (q_continue):
      f_curr = func(x_curr, a)
      dfunc_curr = dfunc(x_curr)
      if (q_history):
         history.append([i_iter, x_curr, f_curr, dfunc_curr])
      if (f_curr != 0.0):
         if (dfunc_curr != 0.0):
            i_iter = i_iter + 1
            x_next = x_curr - f_curr / dfunc_curr
            if (abs(x_next - x_curr) <= eps_conv):
               q_continue = False
               reason = 1               # Successful convergence
            elif (i_iter >= max_iter):
               q_continue = False
               reason = 2               # Exceeded maximum number of iterations
            else:
               x_curr = x_next
         else:
            q_continue = False
            reason = 3                  # Zero derivative
      else:
         q_continue = False
         reason = 4                     # Zero function value

   if(q_history):
      print(pd.DataFrame(history, columns = ['Iteration', 'Estimate', 'Function', 'Derivative']))
   
   if reason == 1:
      r = "Successful convergance"
   elif reason == 2:
      r = "Exceeded maximum number of iterations"
   elif reason == 3:
      r = "Zero derivative"
   elif reason == 4:
      r = "Zero function value"
   
   return (x_curr, r)  

print("(b)")
x_solution, reason = newton_raphson (init_x = 1, a = 2, max_iter = 100, eps_conv = 1e-13, q_history = True)    
print("\nThe root of this equation is: " + str(x_solution) + "\nReason: " + reason)
print()

# ### (c) Suppose 𝑎=9 and the initial estimate is 𝑥_0=1. The iteration will converge if |𝑥_{𝑘+1}−𝑥_𝑘|≤10^{−13}. Please show the iteration history.

# In[9]:

print("(c)")
x_solution, reason = newton_raphson (init_x = 1, a = 9, max_iter = 100, eps_conv = 1e-13, q_history = True)    
print("\nThe root of this equation is: " + str(x_solution) + "\nReason: " + reason)
print()

# ### (d) Suppose 𝑎=9000 and the initial estimate is 𝑥_0=1. The iteration will converge if |𝑥_{𝑘+1}−𝑥_𝑘|≤10^{−13}. Please show the iteration history.

# In[10]:

print("(d)")
x_solution, reason = newton_raphson (init_x = 1, a = 9000, max_iter = 100, eps_conv = 1e-13, q_history = True)    
print("\nThe root of this equation is: " + str(x_solution) + "\nReason: " + str(reason))
print()

# ### (e) Suppose 𝑎=0.0000009 and the initial estimate is 𝑥_0=1. The iteration will converge if |𝑥_{𝑘+1}−𝑥_𝑘|≤10^{−13}. Please show the iteration history.

# In[11]:

print("(e)")
x_solution, reason = newton_raphson (init_x = 1, a = 0.0000009, max_iter = 100, eps_conv = 1e-13, q_history = True)    
print("\nThe root of this equation is: " + str(x_solution) + "\nReason: " + str(reason))
print()
