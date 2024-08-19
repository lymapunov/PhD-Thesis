from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy import signal
import datetime
import os

plt.rcParams['text.usetex'] = True

np.random.seed(123)

def rew_function(px,x,x_max,py,y,y_max):
    #return -0.01*(px*np.exp(abs(x)/x_max) + py*np.exp(y/y_max) - 40)
    return 10.0* np.log(px*abs(x)/x_max + py*abs(y)/y_max)

def filtro(x, xd, cte):
    return (xd - x)/cte

def Izhikevich(v, u, h, a, b, I, stim_app):
    v = v + h * (0.04 * v**2 + 5 * v + 140 - u + I + stim_app)/2.0
    v = v + h * (0.04 * v**2 + 5 * v + 140 - u + I + stim_app)/2.0
    u = u + h * a * (b * v - u)

    return v, u

def stimulaton(t_ini,t_end,dt,amp,freq,duration,dt_pulse_min,stim_type,stim_case):
    freq = round(1000*freq,1)/1000
    t = np.arange(t_ini,t_end,dt)
    if t_end-t_ini > 1000*10:
        print('error!!!! The time interval is not permitted!')
        return 0
    if stim_case == 'Periodic':
        N = len(t)
        t = np.arange(t_ini,t_ini+1000*10,dt)#caso freq=3.5Hz, 35 pontos em 10s->pega os pontos dentro do 1o segundo
        stim = np.zeros(len(t))
        if freq > 0.0:
            stim[(t).astype(int) % int(1.0/freq) == 0] = amp
            stim = stim[:N]
        else:
            stim = stim[:N]
        for i in np.where(stim>1.0e-6)[0]:
            if stim_type == 'mono':
                for j in range(1,int(duration/dt)):
                    stim[i+j] = amp
            else:
                for j in range(0,int((duration/2)/dt)):
                    stim[i+j] = -amp
                for j in range(int((duration/2)/dt),int(duration/dt)):
                    stim[i+j] = amp
        if stim_type == 'mono':
            stim = np.roll(stim,-int((duration/2)/dt))
        else:
            stim = -np.roll(stim,-int((duration/2)/dt))
    elif stim_case == 'NPSLH':
        t = np.arange(t_ini,t_ini+1000*10,dt)
        stim = np.zeros(len(t))
        v = np.sort(np.random.uniform(size=(int(1000*freq*10),)))*1000*10 + t_ini
        for j in range(len(v)):
            if j > 0:
                if v[j]-v[j-1] < dt_pulse_min and dt_pulse_min > 0.0:
                    v[j] = v[j-1] + dt_pulse_min
            stim[(np.abs(t - v[j])).argmin()] = amp
        stim = stim[t<t_end]
        for i in np.where(stim>1.0e-6)[0]:
            if stim_type == 'mono':
                for j in range(1,int(duration/dt)):
                    stim[i+j] = amp
            else:
                for j in range(0,int((duration/2)/dt)):
                    stim[i+j] = -amp
                for j in range(int((duration/2)/dt),int(duration/dt)):
                    stim[i+j] = amp
        if stim_type == 'mono':
            stim = np.roll(stim,-int((duration/2)/dt))
        else:
            stim = -np.roll(stim,-int((duration/2)/dt))
         
    return stim

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def fourier(sig,dt):
    fourier = np.abs(rfft(sig))/(len(sig)/2)
    f_axis = 2*np.pi*rfftfreq(len(sig), d=dt)
    amp_max = max(fourier)
    amp_max_main_range = fourier > 0.95*amp_max
    amp_max_main = np.mean(fourier[amp_max_main_range])
    f_max_main = np.mean(f_axis[amp_max_main_range])
    power_spec = sum(np.maximum(fourier[f_axis<2.0],0.07)**2)
    
    return [f_axis,fourier,amp_max_main,f_max_main,power_spec]

def model(P):
    
    finalTime = P['finalTime'] #simulation time
    Fs = P['Fs'] #sampling frequency (kHz)
    
    dt = 1./Fs # period
    
    t = np.arange(0,finalTime,dt)
    nbSamples = len(t) # number of samples - simulation
    
    duration = P['duration']
    freq = P['freq']
    t_stim = P['t_stim']
    amp = P['amp stimulus']
    
    N_alg = int(P['t_alg']/dt)
    tt_alg = 0
    stim = np.zeros(N_alg)
    
    simulatedLFP = np.zeros(nbSamples)
    stim_ = np.zeros(nbSamples)
    
    state_weight = np.loadtxt('state_weight_Amygdala.txt').reshape(-1,1)
    stim_weight = np.loadtxt('stim_weight_Amygdala.txt').reshape(-1,1)
    
    Na = P['Na']
    Nc = P['Nc']
    NI = P['NI']
    
    a_ = P['a_']
    b_ = P['b_']
    c_ = P['c_']
    
    # ra = np.random.uniform(size=(Na,1))
    # rc = np.random.uniform(size=(Nc,1))
    # rI = np.random.uniform(size=(NI,1))
    a = np.vstack((a_  * np.ones((Na,1)), a_  * np.ones((Nc,1)), a_  * np.ones((NI,1))))
    b = np.vstack((b_  * np.ones((Na,1)), b_  * np.ones((Nc,1)), 0.25* np.ones((NI,1))))
    c = np.vstack((c_  * np.ones((Na,1)), -50 * np.ones((Nc,1)), c_  * np.ones((NI,1))))
    d = np.vstack((8   * np.ones((Na,1)), 2   * np.ones((Nc,1)), 2   * np.ones((NI,1)))) 
    
    S_old = np.hstack((0.5 * np.random.uniform(size=(Na+Nc+NI,Na)), 
                       0.5 * np.random.uniform(size=(Na+Nc+NI,Nc)),
                      -3.5 * np.random.uniform(size=(Na+Nc+NI,NI))))
    S = np.copy(S_old)
    S1f = 0.5
    S2f = 0.5
    S3f = -3.5
    S1fd = P['S1fd']
    S2fd = P['S2fd']
    S3fd = P['S3fd']
    sf_cte = P['sf_cte']
    t_sf = P['t_sf']
    flag_sf = False#True
    
    v = c # Initial values of v
    u = b*v # Initial values of u
    
    t_var = P['t_var']
    N_var = int(t_var/dt)
    data_var = np.zeros(N_var)
    std_var = np.zeros(int(min(1.5*N_var,N_alg)))
    data_var_ = np.zeros(nbSamples)
    std_ = np.zeros(nbSamples)
    stdf = 0.0
    std = 0.0
    rew_ = np.zeros(nbSamples)
    fired_ = np.zeros(nbSamples)
    
    alpha = 0.001
    
    
    firings1 = np.array([]).reshape(-1,2)
    firings2 = np.array([]).reshape(-1,2)
    firings3 = np.array([]).reshape(-1,2)
    firings4 = np.array([]).reshape(-1,2)
    LFP = 0.0
    
    for tt in range(nbSamples):
        
        stim_app = 2/3*stim[tt_alg] * abs(stim_weight+np.random.normal(scale=stim_weight/100)) + 1/3*stim[tt_alg]
        #!!!!tt_alg += 1
            
        if t[tt] > t_sf and flag_sf:
            S1f = S1f + dt*filtro(S1f, S1fd, sf_cte)
            S2f = S2f + dt*filtro(S2f, S2fd, sf_cte)
            S3f = S3f + dt*filtro(S3f, S3fd, sf_cte)
            S[:,0:Na] = S1f*S_old[:,0:Na]/0.5
            S[:,Na:Na+Nc] = S2f*S_old[:,Na:Na+Nc]/0.5
            S[:,Na+Nc:Na+Nc+NI] = S3f*S_old[:,Na+Nc:Na+Nc+NI]/(-3.5)
            flag_sf = flag_sf if (abs(S1f-S1fd) > 1.0e-3 or abs(S2f-S2fd) > 1.0e-3 or abs(S3f-S3fd) > 1.0e-3) and stdf < 100.0 else False
            if not flag_sf:
                t_stim = t[tt]
        
        I = np.vstack((5*np.random.normal(size=(Na,1)), 5.1*np.random.normal(size=(Nc,1)), -1.3*np.random.normal(size=(NI,1))))  # thalamic input
        fired = np.array(np.asarray(v >= 30.0).nonzero()[0].tolist())
        if (tt+1) % 2 == 0:
            if True:
                firings1 = np.concatenate((firings1, np.concatenate((t[tt]*np.ones((len(fired),1)),fired.reshape(-1,1)),axis=1)),axis=0)
            if t[tt] > 1000.0 and t[tt] <= 3000.0:
                firings2 = np.concatenate((firings2, np.concatenate((t[tt]*np.ones((len(fired),1)),fired.reshape(-1,1)),axis=1)),axis=0)
            if t[tt] > 8000.0 and t[tt] <= 10000.0:
                firings3 = np.concatenate((firings3, np.concatenate((t[tt]*np.ones((len(fired),1)),fired.reshape(-1,1)),axis=1)),axis=0)
            # elif finalTime - t[tt] <= 3000.0:
            #     firings4 = np.concatenate((firings4, np.concatenate((t[tt]*np.ones((len(fired),1)),fired.reshape(-1,1)),axis=1)),axis=0)
        
        if len(fired) > 0:
            v[fired] = c[fired]
            u[fired] = u[fired] + d[fired]
            I = I + np.sum(S[:,fired],axis=1).reshape(-1,1)
                
        v, u = Izhikevich(v, u, dt, a, b, I, stim_app)
        
        LFP = float(np.dot((state_weight+np.random.normal(scale=state_weight/100)).T,v))
         
        data_var = np.roll(data_var, -1)
        data_var[-1] = LFP
        std = max(data_var)-min(data_var)#!!!!!np.std(data_var[max(-tt-1,-N_var):])
        std_var = np.roll(std_var, -1)
        if t[tt] < P['t_corte']:
            stdf = 0.0
        elif t[tt] == P['t_corte']:
            stdf = alpha*std
        else:
            stdf = alpha*std + (1-alpha)*stdf
        std_var[-1] = stdf
        std_[tt] = stdf
        data_var_[tt] = max(std_var)-min(std_var)
        simulatedLFP[tt] = LFP
        rew_[tt] = rew_function(1.0,data_var_[tt],50.0,0.1,freq,0.5)
        fired_[tt] = len(fired)/(Na+Nc+NI)*100
        
        # if tt_alg == N_alg:
        #     tt_alg = 0
        #     if t[tt] >= int(t_stim):
        #         ###############
        #         #algoritmo aqui
        #         ###############
        #         if t[tt] > t_stim and t[tt] <= 13000:
        #             freq = 0.2
        #         elif t[tt] > 13000 and t[tt] <= 16000:
        #             freq = 0.131
        #         elif t[tt] > 16000:
        #             freq = 0.1
        #         stim = stimulaton(t[tt],t[tt]+P['t_alg'],dt,amp,freq,duration,P['dt_pulse_min'],P['stim_type'],P['stim_case'])
        #         N_alg = len(stim)
                
        stim_[tt] = stim[tt_alg]
            
    #print(flag_sf,np.max(S[:,0:Na]),np.max(S[:,Na:Na+Nc]),np.min(S[:,Na+Nc:Na+Nc+NI]))
    return t,simulatedLFP,data_var_,std_,firings1,firings2,firings3,firings4,rew_,fired_

#simulation parameters 
t_corte = 200
finalTime = 10000 + t_corte#simulation time [ms]
t_corte = 200
Fs = 1 #sampling frequency (kHz)

Na = 768
Nc = 312
NI = 120

a_ = 0.02
b_ = 0.2
c_ = -65

case = 3
if case == 1:
    S1fd,S2fd,S3fd = 1.0, 0.5, -3.5
elif case == 2:
    S1fd,S2fd,S3fd = 0.5, 1.0, -3.5
elif case == 3:
    S1fd,S2fd,S3fd = 1.0, 1.0, -3.5
elif case == 4:
    S1fd,S2fd,S3fd = 0.5, 0.5, 0.0

sf_cte = 1500.0
t_sf = 3000.0

amp = 200.0
duration = 1.0#tem que ser maior ou igual que 1/Fs
Freq = 0.0 #kHz - maximum at 1/duration
t_stim = 10000-1/Fs

t_var = 250

P = {'finalTime':finalTime,'Fs':Fs,'amp stimulus':amp,'duration':duration,'freq':Freq,
     'Na':Na,'Nc':Nc,'NI':NI,'a_':a_,'b_':b_,'c_':c_,'S1fd':S1fd,'S2fd':S2fd,'S3fd':S3fd,
     'sf_cte':sf_cte,'t_sf':t_sf,'t_stim':t_stim,'t_var':t_var,'stim_type':'mono',
     'stim_case':'Periodic','dt_pulse_min':20.0,'t_alg':t_var,'t_corte':t_corte}

t,simulatedLFP,data_var_,std_,firings1,firings2,firings3,firings4,rew_,fired_ = model(P)

fig = plt.figure(1)
fig.set_figheight(7)
fig.set_figwidth(11)
ax0 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=3, colspan=2)
ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 2))
ax2 = plt.subplot2grid(shape=(3, 3), loc=(1, 2))
ax3 = plt.subplot2grid(shape=(3, 3), loc=(2, 2))
fig.tight_layout(pad=3.2)

ax0.scatter(firings1[firings1[:,0]>t_corte,0]-t_corte, firings1[firings1[:,0]>t_corte,1], s=0.1, c='black')
ax0.set_title("Raster plot", fontsize=14, pad=4.5)
ax0.set_xlabel("Tempo [s]", fontsize=14)
ax0.set_ylabel("Neurônio", fontsize=14)
ax0.set_xlim([0,10000])
ax0.set_ylim([0,1200])
ax0.set_xticks([0,2000,4000,6000,8000,10000],[0,2,4,6,8,10], fontsize=12)
ax0.set_yticks([0,200,400,600,800,1000,1200],[0,200,400,600,800,1000,1200], fontsize=12)
ax1.plot(t[t>t_corte]-t_corte, fired_[t>t_corte], linewidth=1, c='black')
ax1.set_title("Ativação neuronal", fontsize=14, pad=4.5)
ax1.set_xlabel("Tempo [s]", fontsize=14)
ax1.set_ylabel("$\%$ de disparos", fontsize=14)
ax1.set_xlim([0,10000])
ax1.set_ylim([0,100])
ax1.set_xticks([0,2000,4000,6000,8000,10000],[0,2,4,6,8,10], fontsize=12)
ax1.set_yticks([0,25,50,75,100],[0,25,50,75,100], fontsize=12)
ax2.plot(t[t>t_corte]-t_corte, simulatedLFP[t>t_corte], linewidth=1, c='black')
ax2.set_title("Potencial de campo local", fontsize=14, pad=4.5)
ax2.set_xlabel("Tempo [s]", fontsize=14)
ax2.set_ylabel("LFP [mV]", fontsize=14)
ax2.set_xlim([0,10000])
ax2.set_ylim([-100,0])
ax2.set_xticks([0,2000,4000,6000,8000,10000],[0,2,4,6,8,10], fontsize=12)
ax2.set_yticks([-100,-75,-50,-25,0],[-100,-75,-50,-25,0], fontsize=12)
xf = butter_highpass_filter(simulatedLFP[t>2000+t_corte],0.001,Fs)
freq,amp,amp_max,f_amp_max,power_spec = fourier(xf,1)
ax3.plot(freq*1000, amp, linewidth=1, c='black')
ax3.set_title("Transformada rápida de Fourier", fontsize=14, pad=4.5)
ax3.set_xlabel("Frequência [Hz]", fontsize=14)
ax3.set_ylabel("Magnitude", fontsize=14)
ax3.set_xlim([0,300])
ax3.set_ylim([0,3])
ax3.set_xticks([0,100,200,300],[0,100,200,300], fontsize=12)
ax3.set_yticks([0,1,2,3],[0,1,2,3], fontsize=12)

plt.text(-860,11.9,r'\textbf{(a)}',fontsize=14)
plt.text(-70,11.9,r'\textbf{(b)}',fontsize=14)
plt.text(-70,7.45,r'\textbf{(c)}',fontsize=14)
plt.text(-70,3,r'\textbf{(d)}',fontsize=14)

plt.savefig('Amygdala_vGeral_fig1-pt.pdf', bbox_inches='tight')
