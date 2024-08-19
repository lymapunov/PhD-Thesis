from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

############################################

def update_ann(eta,s,dt,w,a,mu):
    if np.linalg.norm(w,2) < mu or (np.linalg.norm(w,2) == mu and eta*s*np.dot(w.T,a) < 0.0):
        w = w + eta*s*a*dt
    else:
        w = w + np.dot(np.eye(len(w)) - np.dot(w.reshape(-1,1),w.reshape(1,-1))/np.dot(w.T,w) ,eta*s*a*dt)
    return w

def ativ(x,c,w):
    return np.array([np.exp(-0.5*((x-c[i])/w[i])**2) for i in range(len(c))])

def edo_Duffing(t, x, xp, u, data):
    delta = data['delta']
    alpha = data['alpha']
    beta = data['beta']
    gamm = data['gamm']
    omega = data['omega']
    return -delta*xp - alpha*x - beta*x**3 + gamm*np.sin(omega*t) + u

def rk4(t, dt, u, x, xp, data):
    k1a = dt*xp
    k1b = dt*edo_Duffing(t, x, xp, u, data)
    
    k2a = dt*(xp + k1b/2)
    k2b = dt*edo_Duffing(t, x + k1a/2, xp + k1b/2, u, data)
    
    k3a = dt*(xp + k2b/2)
    k3b = dt*edo_Duffing(t, x + k2a/2, xp + k2b/2, u, data)
    
    k4a = dt*(xp + k3b)
    k4b = dt*edo_Duffing(t, x + k3a, xp + k3b, u, data)
    
    x = x + (k1a + 2*k2a + 2*k3a + k4a)/6
    xp = xp + (k1b + 2*k2b + 2*k3b + k4b)/6
    
    return x, xp

############################################

t = 0.0
tf = 100.0
dt = 0.01

u = 0.0
x = 0.5
xp = 0.0
lam = 1.0

d_est = 0.0
w = np.zeros((7,))
phi = 0.2
c = [-phi, -phi/2, -phi/4 ,0.0, phi/4, phi/2, phi]
l = [phi/2, phi/3, phi/4, phi/5, phi/4, phi/3, phi/2]
learn = 0.0
mu = 100.0

data = {'delta':0.15,'alpha':-1.0,'beta':1.0,'gamm':0.3,'omega':1.0}

delta = data['delta']
alpha = data['alpha']
beta = data['beta']
gamm = data['gamm']
omega = data['omega']

t_ = []
x_ = []
xd_ = []
u_ = []
d_est_ = []
y_ = []
while t < tf:
    xd = np.sin(0.2*t)
    xpd = 0.2*np.cos(0.2*t)
    xppd = -0.2*0.2*np.sin(0.2*t)
    
    e = x - xd
    ep = xp - xpd
    
    s = ep + lam*e
    
    a = ativ(s,c,l)
    w = update_ann(learn,s,dt,w,a,mu)
    d_est = np.dot(w.T,a)
    
    f = 0.0*(-delta*xp - alpha*x - beta*x**3 + gamm*np.sin(omega*t))
    u = -f + xppd - d_est - lam*ep - lam*s
    
    x,xp = rk4(t, dt, u, x, xp, data)
    xpp = edo_Duffing(t, x, xp, u, data)
    t_.append(t)
    x_.append(x)
    xd_.append(xd)
    u_.append(u)
    d_est_.append(d_est)
    y_.append(xpp - u)
    t += dt
  
error = [x_[i]-xd_[i] for i in range(len(x_))]

e1 = sum(np.array(t_)*abs(np.array(error)))
u1 = sum(np.array(t_)*abs(np.array(u_)))

fig = plt.figure(1)
fig.set_figheight(12)
fig.set_figwidth(14)

ax0 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
ax1 = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 1))
fig.tight_layout(pad=4)

ax0.plot(t_, xd_, linewidth=1, c='black', label='Desejado')
ax0.plot(t_, x_, linewidth=1, c='red', label='Convencional')

ax1.plot(t_, error, linewidth=1, c='red', label='Convencional')

ax2.plot(t_, u_, linewidth=1, c='red', label='Convencional')

t = 0.0
learn = 50.0
t_ = []
x_ = []
xd_ = []
u_ = []
d_est_ = []
y_ = []
while t < tf:
    xd = np.sin(0.2*t)
    xpd = 0.2*np.cos(0.2*t)
    xppd = -0.2*0.2*np.sin(0.2*t)
    
    e = x - xd
    ep = xp - xpd
    
    s = ep + lam*e
    
    a = ativ(s,c,l)
    w = update_ann(learn,s,dt,w,a,mu)
    d_est = np.dot(w.T,a)
    
    f = 0.0*(-delta*xp - alpha*x - beta*x**3 + gamm*np.sin(omega*t))
    u = -f + xppd - d_est - lam*ep - lam*s
    
    x,xp = rk4(t, dt, u, x, xp, data)
    xpp = edo_Duffing(t, x, xp, u, data)
    t_.append(t)
    x_.append(x)
    xd_.append(xd)
    u_.append(u)
    d_est_.append(d_est)
    y_.append(xpp - u)
    t += dt
    
error = [x_[i]-xd_[i] for i in range(len(x_))]

e2 = sum(np.array(t_)*abs(np.array(error)))
u2 = sum(np.array(t_)*abs(np.array(u_)))
    
ax0.plot(t_, x_, linewidth=1, c='blue', label='Inteligente')
ax0.set_xlabel(r'Tempo [s]', fontsize=20)
ax0.set_ylabel(r'$x$', fontsize=20)
ax0.set_xlim([0,100])
ax0.set_xticks([0,20,40,60,80,100],[0,20,40,60,80,100], fontsize=16)
ax0.set_ylim([-1.5,1.5])
ax0.set_yticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],[-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5], fontsize=16)
ax0.legend(fontsize=18, loc='upper right')

ax1.plot(t_, error, linewidth=1, c='blue', label='Inteligente')
ax1.set_xlabel(r'Tempo [s]', fontsize=20)
ax1.set_ylabel(r'$\tilde{x}$', fontsize=20)
ax1.set_xlim([0,100])
ax1.set_xticks([0,20,40,60,80,100],[0,20,40,60,80,100], fontsize=16)
ax1.set_ylim([-0.6,0.9])
ax1.set_yticks([-0.6,-0.3,0.0,0.3,0.6,0.9],[-0.6,-0.3,0.0,0.3,0.6,0.9], fontsize=16)
ax1.legend(fontsize=18, loc='upper right')

ax2.plot(t_, u_, linewidth=1, c='blue', label='Inteligente')
ax2.set_xlabel(r'Tempo [s]', fontsize=20)
ax2.set_ylabel(r'$u$', fontsize=20)
ax2.set_xlim([0,100])
ax2.set_xticks([0,20,40,60,80,100],[0,20,40,60,80,100], fontsize=16)
ax2.set_ylim([-0.8,0.8])
ax2.set_yticks([-0.8,-0.4,0.0,0.4,0.8],[-0.8,-0.4,0.0,0.4,0.8], fontsize=16)
ax2.legend(fontsize=18, loc='upper right')

plt.text(-130,2.64,r'\textbf{(a)}',fontsize=20)
plt.text(-130,0.78,r'\textbf{(b)}',fontsize=20)
plt.text(-14,0.78,r'\textbf{(c)}',fontsize=20)

print((e2-e1)/e1)
print((u2-u1)/u1)

plt.savefig('duffing_withANN_f equal 0.pdf', bbox_inches='tight')
