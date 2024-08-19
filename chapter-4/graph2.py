from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy import signal

plt.rcParams['text.usetex'] = True

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
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

case = 'A'
t_corte = 1.0
data_ = np.loadtxt('data/modelo de massas_-fblann-case'+str(case)+'_learn7.00e+01_data.txt')

data = data_[data_[:,0]>=t_corte,:]
t = data[:,0]-t_corte
LFP = data[:,[1,2]]
u = data[:,-1]
dt = np.mean(np.diff(t))

t_chan = 10.0
first_peak_2 = t[np.argmax(LFP[:,0]>2)]
first_peak_1 = t[np.argmax(LFP[:,1]>2)]
t_c = 40.0

fig = plt.figure(1)
fig.set_figheight(10.5)
fig.set_figwidth(11)
ax0 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2)
ax1 = plt.subplot2grid(shape=(2, 4), loc=(1, 1), colspan=2)
ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2)
fig.tight_layout(pad=3.4)

ax0.plot(t, LFP[:,0], c='black')
ax0.set_title("Potencial de campo local", fontsize=14, pad=4.5)
ax0.set_xlabel("Tempo [s]", fontsize=14)
ax0.set_ylabel("LFP [mV]", fontsize=14)
ax0.set_xlim([0,90])
ax0.set_ylim([-20,20])
ax0.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax0.set_yticks([-20,-10,0,10,20],[-20,-10,0,10,20], fontsize=12)
l1 = mlines.Line2D([t_chan,t_chan], [-20,20], color='b', linewidth=1.2, zorder=1)
ax0.add_line(l1)
rect1 = patches.Rectangle((t_chan+7, 11.5), 30, 5, linewidth=1.2, edgecolor='b', facecolor=(1, 1, 1, 0.85), zorder=2)
ax0.add_patch(rect1)
rx, ry = rect1.get_xy()
cx = rx + rect1.get_width()/2.0
cy = ry + rect1.get_height()/2.0
l11 = mlines.Line2D([rx,t_chan], [cy,10.5], color='b', linewidth=1.2, zorder=1)
ax0.add_line(l11)
ax0.annotate("\\textbf{Parâmetro "+str(case)+"}\n\\textbf{começa a mudar}", (cx, cy-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)
l2 = mlines.Line2D([first_peak_1,first_peak_1], [-20,20], color='r', linewidth=1.2, zorder=1)
ax0.add_line(l2)
rect2 = patches.Rectangle((first_peak_1+5, -18.5), 30, 5, linewidth=1.2, edgecolor='r', facecolor=(1, 1, 1, 0.85), zorder=2)
ax0.add_patch(rect2)
rx2, ry2 = rect2.get_xy()
cx2 = rx2 + rect2.get_width()/2.0
cy2 = ry2 + rect2.get_height()/2.0
l22 = mlines.Line2D([rx2,first_peak_1], [cy2,-17], color='r', linewidth=1.2, zorder=1)
ax0.add_line(l22)
ax0.annotate("\\textbf{Início da crise}\n\\textbf{em "+str(round(first_peak_1,3))+" s}", (cx2, cy2-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)

ax1.plot(t, u, linewidth=1, c='black')
ax1.set_title("Sinal de controle", fontsize=14, pad=4.5)
ax1.set_xlabel("Tempo [s]", fontsize=14)
ax1.set_ylabel("$J_2$ [$\mu$A]", fontsize=14)
ax1.set_xlim([0,90])
ax1.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax1.set_ylim([-180,0])
ax1.set_yticks([-180,-120,-60,0],[-180,-120,-60,0], fontsize=12)

ax2.plot(t, LFP[:,1], c='black')
ax2.set_title("Potencial de campo local", fontsize=14, pad=4.5)
ax2.set_xlabel("Tempo [s]", fontsize=14)
ax2.set_ylabel("LFP [mV]", fontsize=14)
ax2.set_xlim([0,90])
ax2.set_ylim([-20,20])
ax2.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax2.set_yticks([-20,-10,0,10,20],[-20,-10,0,10,20], fontsize=12)
l1 = mlines.Line2D([t_c,t_c], [-20,20], color='b', linewidth=1.2, zorder=1)
ax2.add_line(l1)
rect1 = patches.Rectangle((t_c+5, 11.5), 25, 5, linewidth=1.2, edgecolor='b', facecolor=(1, 1, 1, 0.85), zorder=2)
ax2.add_patch(rect1)
rx, ry = rect1.get_xy()
cx = rx + rect1.get_width()/2.0
cy = ry + rect1.get_height()/2.0
l11 = mlines.Line2D([rx,t_c], [cy,10.5], color='b', linewidth=1.2, zorder=1)
ax2.add_line(l11)
ax2.annotate("\\textbf{Controlador}\n\\textbf{ativado}", (cx, cy-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)
l2 = mlines.Line2D([first_peak_2,first_peak_2], [-20,20], color='r', linewidth=1.2, zorder=1)
ax2.add_line(l2)
rect2 = patches.Rectangle((first_peak_2+5, -18.5), 30, 5, linewidth=1.2, edgecolor='r', facecolor=(1, 1, 1, 0.85), zorder=2)
ax2.add_patch(rect2)
rx2, ry2 = rect2.get_xy()
cx2 = rx2 + rect2.get_width()/2.0
cy2 = ry2 + rect2.get_height()/2.0
l22 = mlines.Line2D([rx2,first_peak_2], [cy2,-17], color='r', linewidth=1.2, zorder=1)
ax2.add_line(l22)
ax2.annotate("\\textbf{Início da crise}\n\\textbf{em "+str(round(first_peak_2,3))+" s}", (cx2, cy2-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)

plt.text(-122.4,20,r'\textbf{(a.1)}',fontsize=14)
plt.text(-15,20,r'\textbf{(a.2)}',fontsize=14)
plt.text(-65,-27,r'\textbf{(b)}',fontsize=14)

plt.savefig('graph2-fblann-'+str(case)+'-pt.pdf', bbox_inches='tight')

