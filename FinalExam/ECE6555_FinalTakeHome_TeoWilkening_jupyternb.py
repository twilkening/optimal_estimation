#!/usr/bin/env python
# coding: utf-8

# # ECE6555 HW5
# 
# Author: Teo Wilkening 
# Due Date: 2022-12-16

# ## [1-Q1] Optimal estimator of V from U of the form $\alpha U$

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


u_mu, u_sigma = 0, 1
n = 10000 # number of samples
u = np.random.normal(u_mu, u_sigma,n)
count, bins, ignored = plt.hist(u, 30, density=True)
plt.plot(bins, 1/(u_sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - u_mu)**2 / (2 * u_sigma**2) ),
                linewidth=2, color='r')
plt.title('Q1: Distribution of U')
plt.show()


# ### MSE numerical estimate

# In[3]:


v = np.sqrt(u**2)
Rv  = np.sum(v**2)/n
Rvu = np.sum(u*v)/n
v_mu = np.sum(v)/n
MSE_linear = Rv - Rvu**2 - v_mu**2
print(f"""The Mean Square error of the linear estimate (for v centered) is: {MSE_linear}""")

MSE_uncentered = Rv - Rvu**2
print(f"""The Mean Square error of the linear estimate (for v un-centered) is: {MSE_uncentered}""")


# ## [1-Q2] Optimal estimator of V from U of the form $\alpha  + \beta U$

# In[4]:


MSE_affine = Rv - Rvu**2 - v_mu**2
print(f"""The Mean Square error of the affine estimate (for v centered) is: {MSE_affine}""")


# ## [1-Q3] Optimal estimator of V from U of the form $\alpha + \beta U + \gamma U^2$

# In[5]:


alpha = v_mu
beta = Rvu
phi = np.sum((u**2)*v)/n
gamma = 1/3*(phi - v_mu)
MSE_quadratic = Rv - 2*(beta*Rvu + gamma*phi + v_mu**2) + (v_mu**2 + beta**2 + 3*gamma**2 + 2*v_mu*gamma)
print(f"""The Mean Square error of the quadratic estimate (for v centered) is: {MSE_quadratic}""")


# ## [2-Q6] EKF implementation
# 
# (will be borrowing my code from HW #4)

# ### Plot the measured and truth data for visualization

# In[6]:

tscale, x = np.load("groundtruth.npy") # ground truth at 1ms
tscale_measurement, y = np.load("measurements.npy") # sampled at 20ms
tscale_measurement2, y2 = np.load("measurements2.npy") # sampled at 2ms

fig, ax = plt.subplots(figsize=(5,3), dpi=120)

ax.plot(tscale,x,'b')
ax.grid(True)
ax.plot(tscale_measurement,y,'c+')
ax.legend(['x','y'])
ax.set_ylabel(r'Pendulum angle: $\alpha = x_1$')
ax.set_xlabel(r'Time (seconds)')
fig.suptitle('Angle and Measurements')

plt.show()


# ### Initialize the necessary parameters/variables

# In[7]:


# initialize all of the variables that we're going to need
Delta = 0.020 # 20 ms
sigma_m = 0.3
sigma_p = 0.1
Qk = (sigma_p**2)*np.matrix([[Delta**3/3,Delta**2/2],
                            [Delta**2/2,Delta]])
Rk = np.matrix([sigma_m**2])
g = 9.8 # m/s^2

def f(xk,dt):
    fk_x = np.matrix([[xk[0][0] + xk[1][0]*dt],
                    [xk[1][0] - g*dt*np.sin(xk[0][0])]])
    return fk_x

def F(xhat,dt):
    F_x = np.matrix([[1                       , dt],
                    [-g*dt*np.sin(xhat[0][0]), 1]])
    return F_x

def h(xk,dt=None):
    return np.sin(xk[0][0])

def H(xhat,dt=None):
    return np.matrix([np.cos(xhat[0][0]), 0])

# # display Qk
# print(f"""Qk = {Qk} \n""")
# display(Rk)

# # test function f:
# xhat_kk = np.array([[1],[2]])
# print("f")
# display(f(xhat_kk,Delta))

# # test function F:
# print("xhat_kk")
# display(xhat_kk[0][0])
# print("F:")
# display(F(xhat_kk,Delta))

# # test function h
# xhat_km1 = np.array([[1],[2]])
# print("h:")
# display(h(xhat_km1))

# # test function K
# print("H:")
# display(H(xhat_km1))


# ### Implement the EKF for measurements.npy

# In[8]:


# the initial guesses of x and P
xhat_init = np.array([[y[0]],[0]]) # set alpha = y[0] and d(alpha)/dt = 0
P_init = np.identity(2)
NumSteps = len(y)

# initializing the matrices for computing Kalman filter state evolution over time
xhat_k_pred = np.zeros((NumSteps,2,1))
xhat_k_curr = np.zeros((NumSteps,2,1))
P_k_pred = np.zeros((NumSteps,2,2))
P_k_curr = np.zeros((NumSteps,2,2))
Kfk_curr = np.zeros((NumSteps,2,1))
I = np.identity(2)

# setup the initial states of the prediction steps, xhat 0|-1 and P 0|-1
xhat_k_pred[0,:,:] = xhat_init
P_k_pred[0,:,:] = P_init[:,:]

# start running the Extended Kalman Filter, using the NoisyMeasurements
for t in np.arange(1,NumSteps):
    ## update step, given the measurement
    # K f,i-1
    Hkm1 = H(xhat_k_pred[t-1,:,:])
    Kfk_curr[t-1,:,:] = P_k_pred[t-1,:,:] @ Hkm1.T @ np.linalg.inv(Hkm1 @ P_k_pred[t-1,:,:] @ Hkm1.T + Rk)
    # P i-1|i-1
    P_k_curr[t-1,:,:] = (I - Kfk_curr[t-1,:,:] @ Hkm1) @ P_k_pred[t-1,:,:]
    # x i-1|i-1
    xhat_k_curr[t-1,:,:] = xhat_k_pred[t-1,:,:] + Kfk_curr[t-1,:,:] * (y[t-1] - h(xhat_k_pred[t-1,:,:]) )
    
    ## Predicition Step
    # x i|i-1
    xhat_k_pred[t,:,:] = f(xhat_k_curr[t-1,:,:],Delta)
    # P i|i-1
    P_k_pred[t,:,:] = F(xhat_k_curr[t-1,:,:],Delta) @ P_k_curr[t-1,:,:] @ F(xhat_k_curr[t-1,:,:],Delta).T + Qk  

## and set the last Update step
t = NumSteps - 1
# K f,t
Hkm1 = H(xhat_k_pred[t,:,:])
Kfk_curr[t,:,:] = P_k_pred[t,:,:] @ Hkm1.T @ np.linalg.inv(Hkm1 @ P_k_pred[t,:,:] @ Hkm1.T + Rk)
# P t|t
P_k_curr[t,:,:] = (I - Kfk_curr[t,:,:] @ Hkm1) @ P_k_pred[t,:,:]
# x t|t
xhat_k_curr[t,:,:] = xhat_k_pred[t,:,:] + Kfk_curr[t,:,:] * (y[t] - h(xhat_k_pred[t,:,:]) )

fig, ax = plt.subplots(figsize=(7,5), dpi=120)

ax.plot(tscale,x,'b')
ax.grid(True)
ax.plot(tscale_measurement,y,'c+')
ax.plot(tscale_measurement,xhat_k_curr[:,0,:],'r-')
ax.legend([r'$x_1$',r'y, $\Delta t = 20ms$',r'$\hat{x}_{1,k|k}$'])
ax.set_ylabel(r'Pendulum angle: $\alpha = x_1$')
ax.set_xlabel(r'Time (seconds)')
fig.suptitle(r'EKF for $\Delta t = 20ms$')

plt.show()


# #### RMS for y

# In[9]:


x_20ms = x[::20]
error = y - x_20ms
error_kalman_f = xhat_k_curr[:,0,:].reshape(-1) - x_20ms
rms_error = np.sqrt(np.sum(error**2)/len(y))
rms_kalman_f = np.sqrt(np.sum(error_kalman_f**2)/len(y))

print(f'''RMS of the measurement y error: {rms_error}''')
print(f'''RMS of the EKF estimates: {rms_kalman_f}''')


# ### Implement the EKF for measurements2.npy

# In[10]:


# initialize all of the variables that we're going to need
Delta = 0.002 # 2 ms
sigma_m = 0.3
sigma_p = 0.1
Qk = (sigma_p**2)*np.matrix([[Delta**3/3,Delta**2/2],
                            [Delta**2/2,Delta]])
Rk = np.matrix([sigma_m**2])
g = 9.8 # m/s^2

def f(xk,dt):
    fk_x = np.matrix([[xk[0][0] + xk[1][0]*dt],
                    [xk[1][0] - g*dt*np.sin(xk[0][0])]])
    return fk_x

def F(xhat,dt):
    F_x = np.matrix([[1                       , dt],
                    [-g*dt*np.sin(xhat[0][0]), 1]])
    return F_x

def h(xk,dt=None):
    return np.sin(xk[0][0])

def H(xhat,dt=None):
    return np.matrix([np.cos(xhat[0][0]), 0])

# the initial guesses of x and P
xhat_init = np.array([[y2[0]],[0]]) # set alpha = y2[0] and d(alpha)/dt = 0
P_init = np.identity(2)
NumSteps = len(y2)

# initializing the matrices for computing Kalman filter state evolution over time
xhat_k_pred = np.zeros((NumSteps,2,1))
xhat_k_curr = np.zeros((NumSteps,2,1))
P_k_pred = np.zeros((NumSteps,2,2))
P_k_curr = np.zeros((NumSteps,2,2))
Kfk_curr = np.zeros((NumSteps,2,1))
I = np.identity(2)

# setup the initial states of the prediction steps, xhat 0|-1 and P 0|-1
xhat_k_pred[0,:,:] = xhat_init
P_k_pred[0,:,:] = P_init[:,:]

# start running the Extended Kalman Filter, using the NoisyMeasurements
for t in np.arange(1,NumSteps):
    ## update step, given the measurement
    # K f,i-1
    Hkm1 = H(xhat_k_pred[t-1,:,:])
    Kfk_curr[t-1,:,:] = P_k_pred[t-1,:,:] @ Hkm1.T @ np.linalg.inv(Hkm1 @ P_k_pred[t-1,:,:] @ Hkm1.T + Rk)
    # P i-1|i-1
    P_k_curr[t-1,:,:] = (I - Kfk_curr[t-1,:,:] @ Hkm1) @ P_k_pred[t-1,:,:]
    # x i-1|i-1
    xhat_k_curr[t-1,:,:] = xhat_k_pred[t-1,:,:] + Kfk_curr[t-1,:,:] * (y2[t-1] - h(xhat_k_pred[t-1,:,:]) )
    
    ## Predicition Step
    # x i|i-1
    xhat_k_pred[t,:,:] = f(xhat_k_curr[t-1,:,:],Delta)
    # P i|i-1
    P_k_pred[t,:,:] = F(xhat_k_curr[t-1,:,:],Delta) @ P_k_curr[t-1,:,:] @ F(xhat_k_curr[t-1,:,:],Delta).T + Qk  

## and set the last Update step
t = NumSteps - 1
# K f,t
Hkm1 = H(xhat_k_pred[t,:,:])
Kfk_curr[t,:,:] = P_k_pred[t,:,:] @ Hkm1.T @ np.linalg.inv(Hkm1 @ P_k_pred[t,:,:] @ Hkm1.T + Rk)
# P t|t
P_k_curr[t,:,:] = (I - Kfk_curr[t,:,:] @ Hkm1) @ P_k_pred[t,:,:]
# x t|t
xhat_k_curr[t,:,:] = xhat_k_pred[t,:,:] + Kfk_curr[t,:,:] * (y2[t] - h(xhat_k_pred[t,:,:]) )

fig, ax = plt.subplots(figsize=(7,5), dpi=120)

ax.plot(tscale,x,'b')
ax.grid(True)
# ax.plot(tscale_measurement2,y2,'c+')
ax.plot(tscale_measurement2,xhat_k_curr[:,0,:],'r-')
# ax.legend([r'$x_1$','y2',r'$\hat{x}_{1,k|k}$'])
ax.legend([r'$x_1$',r'$\hat{x}_{1,k|k}$'])
ax.set_ylabel(r'Pendulum angle: $\alpha = x_1$')
ax.set_xlabel(r'Time (seconds)')
fig.suptitle(r'EKF for $\Delta t = 2ms$')

plt.show()


# #### RMS for y2

# In[11]:


x_2ms = x[::2]
error = y2 - x_2ms
error_kalman_f = xhat_k_curr[:,0,:].reshape(-1) - x_2ms
rms_error = np.sqrt(np.sum(error**2)/len(y2))
rms_kalman_f = np.sqrt(np.sum(error_kalman_f**2)/len(y2))

print(f'''RMS of the measurement y2 error: {rms_error}''')
print(f'''RMS of the EKF estimates: {rms_kalman_f}''')

