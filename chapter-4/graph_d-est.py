from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

def ativ(x,w,c):
    return np.array([np.exp(-0.5*((x-c[i])/w[i])**2) for i in range(len(c))])

cam = 60.0
width = [cam/1.0, cam/2.0, cam/3.0, cam/3.0, cam/2.0, cam/1.0]
center = [-cam, -cam/2.0, -cam/3.0, cam/3.0, cam/2.0, cam]

fig = plt.figure(1)
fig.set_figheight(4)
fig.set_figwidth(11)
ax0 = plt.subplot2grid(shape=(1, 2), loc=(0, 0))
ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 1))
fig.tight_layout(pad=3.2)

case = 'A'
weights1 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(1.0)+'_weights.txt').reshape(-1,1)
weights2 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(5.0)+'_weights.txt').reshape(-1,1)
weights3 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(10.0)+'_weights.txt').reshape(-1,1)
weights4 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(20.0)+'_weights.txt').reshape(-1,1)
weights5 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(50.0)+'_weights.txt').reshape(-1,1)
weights6 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(70.0)+'_weights.txt').reshape(-1,1)

x = np.arange(-3*cam,3*cam,0.01)
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
    
ax0.plot(x,d_est1,label=r'$\eta = 1$',c='blue')
ax0.plot(x,d_est2,label=r'$\eta = 5$',c='orange')
ax0.plot(x,d_est3,label=r'$\eta = 10$',c='green')
ax0.plot(x,d_est4,label=r'$\eta = 20$',c='red')
ax0.plot(x,d_est5,label=r'$\eta = 50$',c='purple')
ax0.plot(x,d_est6,label=r'$\eta = 70$',c='brown')
ax0.set_xlim([-80,40])
ax0.set_ylim([-1200,1200])
ax0.set_xticks([-80,-60,-40,-20,-0,20,40],[-80,-60,-40,-20,-0,20,40], fontsize=12)
ax0.set_yticks([-1200,-600,0,600,1200],[-1200,-600,0,600,1200], fontsize=12)
ax0.set_xlabel(r"$s$ [mV]", fontsize=14)
ax0.set_ylabel(r"$\hat{d}$ [$\mu$A]", fontsize=14, labelpad=1)
ax0.set_title("Caso 1", fontsize=14, pad=4.5)
# ax0.legend(loc='lower right')

case = 'B'
weights1 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(1.0)+'_weights.txt').reshape(-1,1)
weights2 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(5.0)+'_weights.txt').reshape(-1,1)
weights3 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(10.0)+'_weights.txt').reshape(-1,1)
weights4 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(20.0)+'_weights.txt').reshape(-1,1)
weights5 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(50.0)+'_weights.txt').reshape(-1,1)
weights6 = np.loadtxt('data/modelo de massas_case'+str(case)+'_learn'+"{:.2e}".format(70.0)+'_weights.txt').reshape(-1,1)

x = np.arange(-3*cam,3*cam,0.01)
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
    
ax1.plot(x,d_est1,label=r'$\eta = 1$',c='blue')
ax1.plot(x,d_est2,label=r'$\eta = 5$',c='orange')
ax1.plot(x,d_est3,label=r'$\eta = 10$',c='green')
ax1.plot(x,d_est4,label=r'$\eta = 20$',c='red')
ax1.plot(x,d_est5,label=r'$\eta = 50$',c='purple')
ax1.plot(x,d_est6,label=r'$\eta = 70$',c='brown')
ax1.set_xlim([-80,40])
ax1.set_ylim([-1200,1200])
ax1.set_xticks([-80,-60,-40,-20,-0,20,40],[-80,-60,-40,-20,-0,20,40], fontsize=12)
ax1.set_yticks([-1200,-600,0,600,1200],[-1200,-600,0,600,1200], fontsize=12)
ax1.set_xlabel(r"$s$ [mV]", fontsize=14)
ax1.set_ylabel(r"$\hat{d}$ [$\mu$A]", fontsize=14, labelpad=1)
ax1.set_title("Caso 2", fontsize=14, pad=4.5)
ax1.legend(loc='lower right')

ax1.legend(loc='lower right', bbox_to_anchor=(0.55, -0.34), ncols=7)

plt.savefig('modelo de massas-fblann-d-est.pdf', bbox_inches='tight')
    