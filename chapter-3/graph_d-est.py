from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

def ativ(x,w,c):
    return np.array([np.exp(-0.5*((x-c[i])/w[i])**2) for i in range(len(c))])

cam = 30.0
width = [cam/2.0, cam/3.0, cam/4.0, cam/4.0, cam/3.0, cam/2.0]
center = [-cam, -cam/2.0, -cam/4.0, cam/4.0, cam/2.0, cam]

fig = plt.figure(1)
fig.set_figheight(7)
fig.set_figwidth(11)
ax0 = plt.subplot2grid(shape=(2, 2), loc=(0, 0))
ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 1))
ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 1))
fig.tight_layout(pad=3.2)

case = 1
weights1 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-06_weights.txt').reshape(-1,1)
weights2 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-05_weights.txt').reshape(-1,1)
weights3 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0001_weights.txt').reshape(-1,1)
weights4 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0002_weights.txt').reshape(-1,1)
weights5 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_weights.txt').reshape(-1,1)
weights6 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.001_weights.txt').reshape(-1,1)
weights7 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.01_weights.txt').reshape(-1,1)

x = np.arange(-3*cam,3*cam,0.01)
d_est1 = np.zeros(len(x))
d_est2 = np.zeros(len(x))
d_est3 = np.zeros(len(x))
d_est4 = np.zeros(len(x))
d_est5 = np.zeros(len(x))
d_est6 = np.zeros(len(x))
d_est7 = np.zeros(len(x))

for i in range(len(x)):
    d_est1[i] = np.dot(weights1.T,ativ(x[i],width,center))
    d_est2[i] = np.dot(weights2.T,ativ(x[i],width,center))
    d_est3[i] = np.dot(weights3.T,ativ(x[i],width,center))
    d_est4[i] = np.dot(weights4.T,ativ(x[i],width,center))
    d_est5[i] = np.dot(weights5.T,ativ(x[i],width,center))
    d_est6[i] = np.dot(weights6.T,ativ(x[i],width,center))
    d_est7[i] = np.dot(weights7.T,ativ(x[i],width,center))
    
ax0.plot(x,d_est1,label=r'$\eta = 1 \times 10^{-6}$',c='blue')
ax0.plot(x,d_est2,label=r'$\eta = 1 \times 10^{-5}$',c='orange')
ax0.plot(x,d_est3,label=r'$\eta = 1 \times 10^{-4}$',c='green')
ax0.plot(x,d_est4,label=r'$\eta = 2 \times 10^{-4}$',c='red')
ax0.plot(x,d_est5,label=r'$\eta = 5 \times 10^{-4}$',c='purple')
ax0.plot(x,d_est6,label=r'$\eta = 1 \times 10^{-3}$',c='brown')
ax0.plot(x,d_est7,label=r'$\eta = 1 \times 10^{-2}$',c='olive')
ax0.set_xlim([-80,40])
ax0.set_ylim([-12,4])
ax0.set_xticks([-80,-60,-40,-20,-0,20,40],[-80,-60,-40,-20,-0,20,40], fontsize=12)
ax0.set_yticks([-12,-8,-4,0,4],[-12,-8,-4,0,4], fontsize=12)
ax0.set_xlabel(r"$s$ [mV]", fontsize=14)
ax0.set_ylabel(r"$\hat{d}$ [mA]", fontsize=14)
ax0.set_title("Caso 1", fontsize=14, pad=4.5)
# ax0.legend(loc='lower right')

case = 2
weights1 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-06_weights.txt').reshape(-1,1)
weights2 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-05_weights.txt').reshape(-1,1)
weights3 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0001_weights.txt').reshape(-1,1)
weights4 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0002_weights.txt').reshape(-1,1)
weights5 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_weights.txt').reshape(-1,1)
weights6 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.001_weights.txt').reshape(-1,1)
weights7 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.01_weights.txt').reshape(-1,1)

x = np.arange(-3*cam,3*cam,0.01)
d_est1 = np.zeros(len(x))
d_est2 = np.zeros(len(x))
d_est3 = np.zeros(len(x))
d_est4 = np.zeros(len(x))
d_est5 = np.zeros(len(x))
d_est6 = np.zeros(len(x))
d_est7 = np.zeros(len(x))

for i in range(len(x)):
    d_est1[i] = np.dot(weights1.T,ativ(x[i],width,center))
    d_est2[i] = np.dot(weights2.T,ativ(x[i],width,center))
    d_est3[i] = np.dot(weights3.T,ativ(x[i],width,center))
    d_est4[i] = np.dot(weights4.T,ativ(x[i],width,center))
    d_est5[i] = np.dot(weights5.T,ativ(x[i],width,center))
    d_est6[i] = np.dot(weights6.T,ativ(x[i],width,center))
    d_est7[i] = np.dot(weights7.T,ativ(x[i],width,center))
    
ax1.plot(x,d_est1,label=r'$\eta = 1 \times 10^{-6}$',c='blue')
ax1.plot(x,d_est2,label=r'$\eta = 1 \times 10^{-5}$',c='orange')
ax1.plot(x,d_est3,label=r'$\eta = 1 \times 10^{-4}$',c='green')
ax1.plot(x,d_est4,label=r'$\eta = 2 \times 10^{-4}$',c='red')
ax1.plot(x,d_est5,label=r'$\eta = 5 \times 10^{-4}$',c='purple')
ax1.plot(x,d_est6,label=r'$\eta = 1 \times 10^{-3}$',c='brown')
ax1.plot(x,d_est7,label=r'$\eta = 1 \times 10^{-2}$',c='olive')
ax1.set_xlim([-80,40])
ax1.set_ylim([-12,4])
ax1.set_xticks([-80,-60,-40,-20,-0,20,40],[-80,-60,-40,-20,-0,20,40], fontsize=12)
ax1.set_yticks([-12,-8,-4,0,4],[-12,-8,-4,0,4], fontsize=12)
ax1.set_xlabel(r"$s$ [mV]", fontsize=14)
ax1.set_ylabel(r"$\hat{d}$ [mA]", fontsize=14)
ax1.set_title("Caso 2", fontsize=14, pad=4.5)
# ax1.legend(loc='lower right')

case = 3
weights1 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-06_weights.txt').reshape(-1,1)
weights2 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-05_weights.txt').reshape(-1,1)
weights3 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0001_weights.txt').reshape(-1,1)
weights4 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0002_weights.txt').reshape(-1,1)
weights5 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_weights.txt').reshape(-1,1)
weights6 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.001_weights.txt').reshape(-1,1)
weights7 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.01_weights.txt').reshape(-1,1)

x = np.arange(-3*cam,3*cam,0.01)
d_est1 = np.zeros(len(x))
d_est2 = np.zeros(len(x))
d_est3 = np.zeros(len(x))
d_est4 = np.zeros(len(x))
d_est5 = np.zeros(len(x))
d_est6 = np.zeros(len(x))
d_est7 = np.zeros(len(x))

for i in range(len(x)):
    d_est1[i] = np.dot(weights1.T,ativ(x[i],width,center))
    d_est2[i] = np.dot(weights2.T,ativ(x[i],width,center))
    d_est3[i] = np.dot(weights3.T,ativ(x[i],width,center))
    d_est4[i] = np.dot(weights4.T,ativ(x[i],width,center))
    d_est5[i] = np.dot(weights5.T,ativ(x[i],width,center))
    d_est6[i] = np.dot(weights6.T,ativ(x[i],width,center))
    d_est7[i] = np.dot(weights7.T,ativ(x[i],width,center))
    
ax2.plot(x,d_est1,label=r'$\eta = 1 \times 10^{-6}$',c='blue')
ax2.plot(x,d_est2,label=r'$\eta = 1 \times 10^{-5}$',c='orange')
ax2.plot(x,d_est3,label=r'$\eta = 1 \times 10^{-4}$',c='green')
ax2.plot(x,d_est4,label=r'$\eta = 2 \times 10^{-4}$',c='red')
ax2.plot(x,d_est5,label=r'$\eta = 5 \times 10^{-4}$',c='purple')
ax2.plot(x,d_est6,label=r'$\eta = 1 \times 10^{-3}$',c='brown')
ax2.plot(x,d_est7,label=r'$\eta = 1 \times 10^{-2}$',c='olive')
ax2.set_xlim([-80,40])
ax2.set_ylim([-12,4])
ax2.set_xticks([-80,-60,-40,-20,-0,20,40],[-80,-60,-40,-20,-0,20,40], fontsize=12)
ax2.set_yticks([-12,-8,-4,0,4],[-12,-8,-4,0,4], fontsize=12)
ax2.set_xlabel(r"$s$ [mV]", fontsize=14)
ax2.set_ylabel(r"$\hat{d}$ [mA]", fontsize=14)
ax2.set_title("Caso 3", fontsize=14, pad=4.5)
# ax2.legend(loc='lower right')

case = 4
weights1 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-06_weights.txt').reshape(-1,1)
weights2 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_1e-05_weights.txt').reshape(-1,1)
weights3 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0001_weights.txt').reshape(-1,1)
weights4 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0002_weights.txt').reshape(-1,1)
weights5 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_weights.txt').reshape(-1,1)
weights6 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.001_weights.txt').reshape(-1,1)
weights7 = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.01_weights.txt').reshape(-1,1)

x = np.arange(-3*cam,3*cam,0.01)
d_est1 = np.zeros(len(x))
d_est2 = np.zeros(len(x))
d_est3 = np.zeros(len(x))
d_est4 = np.zeros(len(x))
d_est5 = np.zeros(len(x))
d_est6 = np.zeros(len(x))
d_est7 = np.zeros(len(x))

for i in range(len(x)):
    d_est1[i] = np.dot(weights1.T,ativ(x[i],width,center))
    d_est2[i] = np.dot(weights2.T,ativ(x[i],width,center))
    d_est3[i] = np.dot(weights3.T,ativ(x[i],width,center))
    d_est4[i] = np.dot(weights4.T,ativ(x[i],width,center))
    d_est5[i] = np.dot(weights5.T,ativ(x[i],width,center))
    d_est6[i] = np.dot(weights6.T,ativ(x[i],width,center))
    d_est7[i] = np.dot(weights7.T,ativ(x[i],width,center))
    
ax3.plot(x,d_est1,label=r'$\eta = 1 \times 10^{-6}$',c='blue')
ax3.plot(x,d_est2,label=r'$\eta = 1 \times 10^{-5}$',c='orange')
ax3.plot(x,d_est3,label=r'$\eta = 1 \times 10^{-4}$',c='green')
ax3.plot(x,d_est4,label=r'$\eta = 2 \times 10^{-4}$',c='red')
ax3.plot(x,d_est5,label=r'$\eta = 5 \times 10^{-4}$',c='purple')
ax3.plot(x,d_est6,label=r'$\eta = 1 \times 10^{-3}$',c='brown')
ax3.plot(x,d_est7,label=r'$\eta = 1 \times 10^{-2}$',c='olive')
ax3.set_xlim([-80,40])
ax3.set_ylim([-12,4])
ax3.set_xticks([-80,-60,-40,-20,-0,20,40],[-80,-60,-40,-20,-0,20,40], fontsize=12)
ax3.set_yticks([-12,-8,-4,0,4],[-12,-8,-4,0,4], fontsize=12)
ax3.set_xlabel(r"$s$ [mV]", fontsize=14)
ax3.set_ylabel(r"$\hat{d} [mA]$", fontsize=14)
ax3.set_title("Caso 4", fontsize=14, pad=4.5)
ax3.legend(loc='lower right', bbox_to_anchor=(0.98, -0.34), ncols=7)

plt.savefig('Amygdala_flbann-d-est.pdf', bbox_inches='tight')
    