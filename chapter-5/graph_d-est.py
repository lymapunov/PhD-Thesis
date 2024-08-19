from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

def ativ(x,w,c):
    return np.array([np.exp(-0.5*((x-c[i])/w[i])**2) for i in range(len(c))])

rho_SA = 9.6
omega_SA = 2.1
lamb = 2.0
title = 'data/data_rho'+str(rho_SA)+'_omega'+str(omega_SA)+'_lam'+str(lamb)+'_eta'#+str(learning_rate)

cam = 20.0
width = [cam/2.0, cam/3.0, cam/6.0, cam/6.0, cam/3.0, cam/2.0]
center = [-cam/2, -cam/8.0, -cam/16.0, cam/16.0, cam/8.0, cam/2]

fig = plt.figure(1)
fig.set_figheight(4)
fig.set_figwidth(5.5)
ax0 = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
fig.tight_layout(pad=3.2)

learning_rate = [0.1,0.5,1.0,5.0,10.0,20.0,50.0]
weights1 = np.loadtxt(title+str(learning_rate[0])+'_weights.txt').reshape(-1,1)
weights2 = np.loadtxt(title+str(learning_rate[1])+'_weights.txt').reshape(-1,1)
weights3 = np.loadtxt(title+str(learning_rate[2])+'_weights.txt').reshape(-1,1)
weights4 = np.loadtxt(title+str(learning_rate[3])+'_weights.txt').reshape(-1,1)
weights5 = np.loadtxt(title+str(learning_rate[4])+'_weights.txt').reshape(-1,1)
weights6 = np.loadtxt(title+str(learning_rate[5])+'_weights.txt').reshape(-1,1)

x = np.arange(-2*cam,2*cam,0.01)
d_est1 = np.zeros(len(x))
d_est2 = np.zeros(len(x))
d_est3 = np.zeros(len(x))
d_est4 = np.zeros(len(x))
d_est5 = np.zeros(len(x))
d_est6 = np.zeros(len(x))

for i in range(len(x)):
    d_est1[i] = np.dot(weights1.T,ativ(x[i],width,center))
    d_est2[i] = np.dot(weights2.T,ativ(x[i],width,center))
    d_est3[i] = np.dot(weights3.T,ativ(x[i],width,center))
    d_est4[i] = np.dot(weights4.T,ativ(x[i],width,center))
    d_est5[i] = np.dot(weights5.T,ativ(x[i],width,center))
    d_est6[i] = np.dot(weights6.T,ativ(x[i],width,center))
    
ax0.plot(x,d_est1,label=r'$\eta = 0.1$',c='blue')
ax0.plot(x,d_est2,label=r'$\eta = 0.5$',c='orange')
ax0.plot(x,d_est3,label=r'$\eta = 1$',c='green')
ax0.plot(x,d_est4,label=r'$\eta = 5$',c='red')
ax0.plot(x,d_est5,label=r'$\eta = 10$',c='purple')
ax0.plot(x,d_est6,label=r'$\eta = 20$',c='brown')
ax0.set_xlim([-40,40])
ax0.set_ylim([-80,80])
ax0.set_xticks([-40,-20,-0,20,40],[-40,-20,-0,20,40], fontsize=12)
ax0.set_yticks([-80,-40,0,40,80],[-80,-40,0,40,80], fontsize=12)
ax0.set_xlabel(r"$s$ [mV/s]", fontsize=14)
ax0.set_ylabel(r"$\hat{p}$ [$\mu$A]", fontsize=14, labelpad=1)
ax0.legend(loc='lower right')

plt.savefig('heart-fblann-d-est1.pdf', bbox_inches='tight')
    