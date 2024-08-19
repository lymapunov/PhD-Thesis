from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz
import pandas as pd

np.random.seed(123)

def update_ann(eta,s,dt,w,a,mu):
    if np.linalg.norm(w,2) < mu or (np.linalg.norm(w,2) == mu and eta*s*np.dot(w.T,a) < 0.0):
        w = w + eta*s*a*dt
    else:
        w = w + np.dot(np.eye(len(w)) - np.dot(w.reshape(-1,1),w.reshape(1,-1))/np.dot(w.T,w) ,eta*s*a*dt)
    return w

def ativ(x,w,c):
    return np.array([np.exp(-0.5*((x-c[i])/w[i])**2) for i in range(len(c))])

def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({ 
        'data' : y_sine} ,index=t_sine)
    return result

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
def filter_u(SA):
    Rm = 20
    rows, columns = SA.shape
    beg = SA[0,0]
    SA[:,0] = SA[:,0] - beg
    dt = np.mean(SA[1:rows,0] - SA[0:rows-1,0])
    fps = 1.0/dt
    F_approx = np.array([SA[1:rows,0], cumtrapz(SA[:,1], SA[:,0])]).T
    result_to_filter = F_approx
    result = pd.DataFrame({'data' : result_to_filter[:,1]} ,index=result_to_filter[:,0])

    filtered_signal = butter_highpass_filter(result.data,1,fps)
    
    F_approx = np.array([SA[2:rows,0], cumtrapz(filtered_signal[:], SA[1:rows,0])]).T
    result_to_filter = F_approx
    result = pd.DataFrame({'data' : result_to_filter[:,1]} ,index=result_to_filter[:,0])

    filtered_signal = butter_highpass_filter(result.data,1,fps)
    
    return np.append([0.0,0.0],filtered_signal/Rm)

def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)

def FSA(t, P):
    rho_SA = P['rho_SA'] + np.random.normal(scale=P['std'])
    omega_SA = P['omega_SA'] + np.random.normal(scale=P['std'])
    return rho_SA * sin(omega_SA * t)
def FAV(t, P):
    rho_AV = P['rho_AV']
    omega_AV = P['omega_AV']
    return rho_AV * sin(omega_AV * t)
def FHP(t, P):
    rho_HP = P['rho_HP']
    omega_HP = P['omega_HP']
    return rho_HP * sin(omega_HP * t)

def ode(t, Q, Qp, Q_delay, u, P):
    alp_SA = P['alp_SA']
    nu1_SA = P['nu1_SA']
    nu2_SA = P['nu2_SA']
    d_SA = P['d_SA']
    e_SA = P['e_SA']
    k_AV_SA = P['k_AV_SA']
    kt_AV_SA = P['kt_AV_SA']
    k_HP_SA = P['k_HP_SA']
    kt_HP_SA = P['kt_HP_SA']
    
    alp_AV = P['alp_AV']
    nu1_AV = P['nu1_AV']
    nu2_AV = P['nu2_AV']
    d_AV = P['d_AV']
    e_AV = P['e_AV']
    k_SA_AV = P['k_SA_AV']
    kt_SA_AV = P['kt_SA_AV']
    k_HP_AV = P['k_HP_AV']
    kt_HP_AV = P['kt_HP_AV']
    
    alp_HP = P['alp_HP']
    nu1_HP = P['nu1_HP']
    nu2_HP = P['nu2_HP']
    d_HP = P['d_HP']
    e_HP = P['e_HP']
    k_SA_HP = P['k_SA_HP']
    kt_SA_HP = P['kt_SA_HP']
    k_AV_HP = P['k_AV_HP']
    kt_AV_HP = P['kt_AV_HP']
    
    x1 = Q[0]
    x2 = Qp[0]
    x3 = Q[1]
    x4 = Qp[1]
    x5 = Q[2]
    x6 = Qp[2]
    
    x2p = FSA(t,P) - alp_SA * x2 * (x1 - nu1_SA) * (x1 - nu2_SA) - (x1 * (x1 + d_SA) * (x1 + e_SA)) / (d_SA * e_SA) - k_AV_SA * x1 + kt_AV_SA * Q_delay[0] - k_HP_SA * x1 + kt_HP_SA * Q_delay[1] + u[0]
    
    x4p = FAV(t,P) - alp_AV * x4 * (x3 - nu1_AV) * (x3 - nu2_AV) - (x3 * (x3 + d_AV) * (x3 + e_AV)) / (d_AV * e_AV) - k_SA_AV * x3 + kt_SA_AV * Q_delay[2] - k_HP_AV * x3 + kt_HP_AV * Q_delay[3] + u[1]
    
    x6p = FHP(t,P) - alp_HP * x6 * (x5 - nu1_HP) * (x5 - nu2_HP) - (x5 * (x5 + d_HP) * (x5 + e_HP)) / (d_HP * e_HP) - k_SA_HP * x5 + kt_SA_HP * Q_delay[4] - k_AV_HP * x5 + kt_AV_HP * Q_delay[5] + u[2]
    
    Qpp = np.zeros(3)
    Qpp[0] = x2p
    Qpp[1] = x4p
    Qpp[2] = x6p
    
    return Qpp


def rk4_heart(t, tf, Q, Qp, Q_delay, u, P):
    dt = tf - t

    # Initialize k vectors
    k1p = np.zeros(3)
    k2p = np.zeros(3)
    k3p = np.zeros(3)
    k4p = np.zeros(3)
    k1v = np.zeros(3)
    k2v = np.zeros(3)
    k3v = np.zeros(3)
    k4v = np.zeros(3)
    Qpp = np.zeros(3)
    k_delay = np.zeros(6)

    # Step 1
    Qpp = ode(t, Q, Qp, Q_delay, u, P)
    k1p = dt * Qp
    k1v = dt * Qpp
    k_delay = np.array([k1p[1], k1p[2], k1p[0], k1p[2], k1p[0], k1p[1]])

    # Step 2
    Qpp = ode(t + dt / 2.0, Q + k1p / 2.0, Qp + k1v / 2.0, Q_delay + k_delay / 2.0, u, P)
    k2p = dt * (Qp + k1v / 2.0)
    k2v = dt * Qpp
    k_delay = np.array([k2p[1], k2p[2], k2p[0], k2p[2], k2p[0], k2p[1]])

    # Step 3
    Qpp = ode(t + dt / 2.0, Q + k2p / 2.0, Qp + k2v / 2.0, Q_delay + k_delay / 2.0, u, P)
    k3p = dt * (Qp + k2v / 2.0)
    k3v = dt * Qpp
    k_delay = np.array([k3p[1], k3p[2], k3p[0], k3p[2], k3p[0], k3p[1]])

    # Step 4
    Qpp = ode(t + dt, Q + k3p, Qp + k3v, Q_delay + k_delay, u, P)
    k4p = dt * (Qp + k3v)
    k4v = dt * Qpp

    # Update Q and Qp
    Q += (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
    Qp += (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0

    return Q, Qp

t_beg = 0.0 
t_end = 210.0

beta_t = 0.1048
csr = 100.0
cst = 1.0/csr

ssr = 1000.0
sst = 1.0/ssr

t = np.arange(t_beg,t_end+cst,cst)
t_sim = 0.0

ECG_ = np.zeros(len( t ))
x_ = np.zeros(len( t ))
xd_ = np.zeros(len( t ))
u_ = np.zeros(len( t ))

P = {'alp_SA':3.0,'nu1_SA':1.0,'nu2_SA':-1.9,'d_SA':1.9,'e_SA':0.55,
     'alp_AV':3.0,'nu1_AV':0.5,'nu2_AV':-0.5,'d_AV':4.0,'e_AV':0.67,
     'alp_HP':7.0,'nu1_HP':1.65,'nu2_HP':-2.0,'d_HP':7.0,'e_HP':0.67,
     'rho_SA':8.0,'omega_SA':3.3,'rho_AV':0.0,'omega_AV':0.0,'rho_HP':0.0,'omega_HP':0.0,
     'k_SA_AV':3.0,'k_AV_HP':55.0,'kt_SA_AV':3.0,'kt_AV_HP':55.0,'tau_SA_AV':0.8,'tau_AV_HP':0.1,
     'k_AV_SA':0.0,'k_HP_SA':0.0,'k_HP_AV':0.0,'k_SA_HP':0.0,
     'kt_AV_SA':0.0,'kt_HP_SA':0.0,'kt_HP_AV':0.0,'kt_SA_HP':0.0,
     'tau_AV_SA':0.0,'tau_HP_SA':0.0,'tau_HP_AV':0.0,'tau_SA_HP':0.0,
     'beta_t':beta_t,'std':0.0}

u = np.zeros(3)
Q = np.array([-0.1,-0.6,-3.3])
Qp = np.array([0.025,0.1,2/3])
Q_old = np.zeros(3)
Q_delay = np.zeros(6)
cont_delay = np.zeros(6).astype(np.int64)

ECG = 1.0 + 0.06 * Q[0] + 0.1 * Q[1] + 0.3 * Q[2]

xt1_SA_AV_old = Q[0]
xt1_SA_HP_old = Q[0]
xt3_AV_SA_old = Q[1]
xt3_AV_HP_old = Q[1]
xt5_HP_SA_old = Q[2]
xt5_HP_AV_old = Q[2]

delay = np.array([P['tau_AV_SA'], P['tau_HP_SA'], P['tau_SA_AV'], P['tau_HP_AV'], P['tau_SA_HP'], P['tau_AV_HP']])
delay_size = np.array([P['tau_AV_SA']/sst, P['tau_HP_SA']/sst, P['tau_SA_AV']/sst, P['tau_HP_AV']/sst, P['tau_SA_HP']/sst, P['tau_AV_HP']/sst])
delay_size = delay_size.astype(np.int64)

Q1d_SA_HP = np.zeros(delay_size[4])
Q1d_SA_AV = np.zeros(delay_size[2])
Q3d_AV_HP = np.zeros(delay_size[5])
Q3d_AV_SA = np.zeros(delay_size[0])
Q5d_HP_AV = np.zeros(delay_size[3])
Q5d_HP_SA = np.zeros(delay_size[1])

desired = np.loadtxt('data_csr100.txt')
lamb = 2.0
d_est = 0.0

cam = 20.0
width = [cam/2.0, cam/3.0, cam/6.0, cam/6.0, cam/3.0, cam/2.0]
center = [-cam/2, -cam/8.0, -cam/16.0, cam/16.0, cam/8.0, cam/2]

weight_values = np.zeros((6,))
limit_w = 100.0
learning_rate = 10.0

for tt in range(len(t)):
    #SA:0,1,2; ECG:3,4,5
    xd = desired[tt,0]
    xpd = desired[tt,1]
    xppd = desired[tt,2]
    
    if t[tt] >= 100.0:
        e = Q[0] - xd
        ep = Qp[0] - xpd
        
        s = ep + lamb * e
        
        activation_function = ativ(s, width, center)
        
        weight_values = update_ann(learning_rate,s,cst,weight_values,activation_function,limit_w)
        
        d_est = np.dot(weight_values.T,activation_function)
        
        u[0] = xppd - 2.0 * lamb * ep - lamb * lamb * e - d_est;
    
    t_sim = t[tt]
    while t_sim < t[tt] + cst:#must be less, not <=
        if delay_size[0] < 1:
            Q_delay[0] = Q[1]
        else:
            if t_sim < delay[0] and cont_delay[0] < delay_size[0]:
                Q_delay[0] = Q_old[1] - delay[0]*(Q[1] - Q_old[1])/sst
                Q3d_AV_SA[cont_delay[0]] = Q_delay[0]
                cont_delay[0] += 1
            else:
                Q_delay[0] = Q3d_AV_SA[0]
                Q3d_AV_SA = np.roll(Q3d_AV_SA, -1)
                Q3d_AV_SA[-1] = Q[1]
        
        if delay_size[1] < 1:
            Q_delay[1] = Q[2]
        else:
            if t_sim < delay[1] and cont_delay[1] < delay_size[1]:
                Q_delay[1] = Q_old[2] - delay[1]*(Q[2] - Q_old[2])/sst
                Q5d_HP_SA[cont_delay[1]] = Q_delay[1]
                cont_delay[1] += 1
            else:
                Q_delay[1] = Q5d_HP_SA[0]
                Q5d_HP_SA = np.roll(Q5d_HP_SA, -1)
                Q5d_HP_SA[-1] = Q[2]
                
        if delay_size[2] < 1:
            Q_delay[2] = Q[0]
        else:
            if t_sim < delay[2] and cont_delay[2] < delay_size[2]:
                Q_delay[2] = Q_old[0] - delay[2]*(Q[0] - Q_old[0])/sst
                Q1d_SA_AV[cont_delay[2]] = Q_delay[2]
                cont_delay[2] += 1
            else:
                Q_delay[2] = Q1d_SA_AV[0]
                Q1d_SA_AV = np.roll(Q1d_SA_AV, -1)
                Q1d_SA_AV[-1] = Q[0]
                
        if delay_size[3] < 1:
            Q_delay[3] = Q[2]
        else:
            if t_sim < delay[3] and cont_delay[3] < delay_size[3]:
                Q_delay[3] = Q_old[2] - delay[3]*(Q[2] - Q_old[2])/sst
                Q5d_HP_AV[cont_delay[3]] = Q_delay[3]
                cont_delay[3] += 1
            else:
                Q_delay[3] = Q5d_HP_AV[0]
                Q5d_HP_AV = np.roll(Q5d_HP_AV, -1)
                Q5d_HP_AV[-1] = Q[2]
                
        if delay_size[4] < 1:
            Q_delay[4] = Q[0]
        else:
            if t_sim < delay[4] and cont_delay[4] < delay_size[4]:
                Q_delay[4] = Q_old[0] - delay[4]*(Q[0] - Q_old[0])/sst
                Q1d_SA_HP[cont_delay[4]] = Q_delay[4]
                cont_delay[4] += 1
            else:
                Q_delay[4] = Q1d_SA_HP[0]
                Q1d_SA_HP = np.roll(Q1d_SA_HP, -1)
                Q1d_SA_HP[-1] = Q[0]
                
                
        if delay_size[5] < 1:
            Q_delay[5] = Q[1]
        else:
            if t_sim < delay[5] and cont_delay[5] < delay_size[5]:
                Q_delay[5] = Q_old[1] - delay[5]*(Q[1] - Q_old[1])/sst
                Q3d_AV_HP[cont_delay[5]] = Q_delay[5]
                cont_delay[5] += 1
            else:
                Q_delay[5] = Q3d_AV_HP[0]
                Q3d_AV_HP = np.roll(Q3d_AV_HP, -1)
                Q3d_AV_HP[-1] = Q[1]
        
        Q_old = Q
        Q,Qp = rk4_heart(t_sim, t_sim+sst, Q, Qp, Q_delay, u, P)
        t_sim += sst
        
    ECG = 1.0 + 0.06 * Q[0] + 0.1 * Q[1] + 0.3 * Q[2]
    ECG_[tt] = ECG
    x_[tt] = Q[0]
    xd_[tt] = xd
    u_[tt] = u[0]
    
t = t*beta_t
uf_ = filter_u(np.array([t,u_]).T)
data = np.array([t,x_,xd_,ECG_,u_,uf_])
title = 'figs/data/data_rho'+str(P['rho_SA'])+'_omega'+str(P['omega_SA'])+'_lam'+str(lamb)+'_eta'+str(learning_rate)
np.savetxt(title+'.txt', data.T, delimiter=' ', fmt='%.6f', newline='\n')
np.savetxt(title+'_weights.txt', weight_values.reshape(1,-1), delimiter=' ', fmt='%.6f', newline='\n')

plt.figure(1)
plt.plot(t[t>15],xd_[t>15])
plt.plot(t[t>15],x_[t>15])

plt.figure(2)
plt.plot(t[t>15],ECG_[t>15])

plt.figure(3)
plt.plot(t[t>15],uf_[t>15])














