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

def fourier(sig,dt):
    fourier = np.abs(rfft(sig))/(len(sig)/2)
    f_axis = 2*np.pi*rfftfreq(len(sig), d=dt)
    amp_max = max(fourier)
    amp_max_main_range = fourier > 0.95*amp_max
    amp_max_main = np.mean(fourier[amp_max_main_range])
    f_max_main = np.mean(f_axis[amp_max_main_range])
    power_spec = sum(np.maximum(fourier[f_axis<2.0],0.07)**2)
    
    return [f_axis,fourier,amp_max_main,f_max_main,power_spec]

t_corte = 200
Fs = 1

fig = plt.figure(1)
fig.set_figheight(15)
fig.set_figwidth(11)
ax0 = plt.subplot2grid(shape=(8, 3), loc=(0, 0), rowspan=4, colspan=2)
ax1 = plt.subplot2grid(shape=(8, 3), loc=(0, 2))
ax2 = plt.subplot2grid(shape=(8, 3), loc=(1, 2))
ax3 = plt.subplot2grid(shape=(8, 3), loc=(2, 2))
ax4 = plt.subplot2grid(shape=(8, 3), loc=(3, 2))
ax5 = plt.subplot2grid(shape=(8, 3), loc=(4, 0), rowspan=4, colspan=2)
ax6 = plt.subplot2grid(shape=(8, 3), loc=(4, 2))
ax7 = plt.subplot2grid(shape=(8, 3), loc=(5, 2))
ax8 = plt.subplot2grid(shape=(8, 3), loc=(6, 2))
ax9 = plt.subplot2grid(shape=(8, 3), loc=(7, 2))
fig.tight_layout(pad=3.2)

case = 3
data = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_data.txt')
firings = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_firings1.txt')

t = data[:,0]
simulatedLFP = data[:,1]
data_var_ = data[:,2]
std_ = data[:,3]
freq_ = data[:,4]
stim_fbl_ = data[:,5]
fired_ = data[:,6]
S1f_ = data[:,7]
S2f_ = data[:,8]
S3f_ = data[:,9]

ax0.scatter(firings[firings[:,0]>t_corte+5000,0]-t_corte, firings[firings[:,0]>t_corte+5000,1], s=0.1, c='black')
ax0.set_title("Raster plot", fontsize=14, pad=4.5)
ax0.set_xlabel("Tempo [s]", fontsize=14)
ax0.set_ylabel("Neurônio", fontsize=14)
ax0.set_xlim([5000,25000])
ax0.set_ylim([0,1200])
ax0.set_xticks([5000,10000,15000,20000,25000],[5,10,15,20,25], fontsize=12)
ax0.set_yticks([0,200,400,600,800,1000,1200],[0,200,400,600,800,1000,1200], fontsize=12)
l1 = mlines.Line2D([10000,10000], [0,1200], color='r', linewidth=2.5)
ax0.add_line(l1)
rect1 = patches.Rectangle((10500, 500), 3000, 85, linewidth=1.8, edgecolor='r', facecolor=(1, 1, 1, 0.8), zorder=4)
ax0.add_patch(rect1)
rx, ry = rect1.get_xy()
cx = rx + rect1.get_width()/2.0
cy = ry + rect1.get_height()/2.0
ax0.annotate("\\textbf{Controle}\n\\textbf{ativado}", (cx, cy), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=5)
ax0.arrow(cx,ry,10000-cx, -100, color='r', linewidth=1.5,length_includes_head=True, zorder=3)

ax1.plot(t[t>t_corte]-t_corte, fired_[t>t_corte], linewidth=1, c='black')
ax1.set_title("Ativação neuronal", fontsize=14, pad=4.5)
ax1.set_xlabel("Tempo [s]", fontsize=14)
ax1.set_ylabel("$\%$ de disparos", fontsize=14)
ax1.set_xlim([0,25000])
ax1.set_ylim([0,100])
ax1.set_xticks([0,5000,10000,15000,20000,25000],[0,5,10,15,20,25], fontsize=12)
ax1.set_yticks([0,25,50,75,100],[0,25,50,75,100], fontsize=12)

ax2.plot(t[t>t_corte]-t_corte, simulatedLFP[t>t_corte], linewidth=1, c='black')
ax2.set_title("Potencial de campo local", fontsize=14, pad=4.5)
ax2.set_xlabel("Tempo [s]", fontsize=14)
ax2.set_ylabel("LFP [mV]", fontsize=14)
ax2.set_xlim([0,25000])
ax2.set_ylim([-200,800])
ax2.set_xticks([0,5000,10000,15000,20000,25000],[0,5,10,15,20,25], fontsize=12)
ax2.set_yticks([-200,0,200,400,600,800],[-200,0,200,400,600,800], fontsize=12)

ax3.plot(t[t>t_corte]-t_corte, stim_fbl_[t>t_corte], linewidth=1, c='black')
ax3.set_title("Sinal de controle", fontsize=14, pad=4.5)
ax3.set_xlabel("Tempo [s]", fontsize=14)
ax3.set_ylabel("$J$ [mA]", fontsize=14)
ax3.set_xlim([0,25000])
ax3.set_ylim([-5,15])
ax3.set_xticks([0,5000,10000,15000,20000,25000],[0,5,10,15,20,25], fontsize=12)
ax3.set_yticks([-5,0,5,10,15],[-5,0,5,10,15], fontsize=12)

xf = butter_highpass_filter(simulatedLFP[t>19000+t_corte],0.001,Fs)
freq,amp,amp_max,f_amp_max,power_spec = fourier(xf,1)
ax4.plot(freq*1000, amp, linewidth=1, c='black')
ax4.set_title("Transformada rápida de Fourier", fontsize=14, pad=4.5)
ax4.set_xlabel("Frequência [Hz]", fontsize=14)
ax4.set_ylabel("Magnitude", fontsize=14)
ax4.set_xlim([0,300])
ax4.set_ylim([0,2])
ax4.set_xticks([0,100,200,300],[0,100,200,300], fontsize=12)
ax4.set_yticks([0,0.5,1,1.5,2],[0.0,0.5,1.0,1.5,2.0], fontsize=12)

plt.text(-870,63*0.4,r'\textbf{(a)}',fontsize=14)
plt.text(-70,63*0.4,r'\textbf{(b)}',fontsize=14)
plt.text(-70,54.5*0.4,r'\textbf{(c)}',fontsize=14)
plt.text(-70,46.3*0.4,r'\textbf{(d)}',fontsize=14)
plt.text(-70,38.1*0.4,r'\textbf{(e)}',fontsize=14)

case = 4
data = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_data.txt')
firings = np.loadtxt('data/Amygdala_vGeral_FBLANN_LFP_case'+str(case)+'learn_0.0005_firings1.txt')

t = data[:,0]
simulatedLFP = data[:,1]
data_var_ = data[:,2]
std_ = data[:,3]
freq_ = data[:,4]
stim_fbl_ = data[:,5]
fired_ = data[:,6]
S1f_ = data[:,7]
S2f_ = data[:,8]
S3f_ = data[:,9]

t_synaptic_beg = t[np.logical_or.reduce((abs(S1f_[0]-S1f_)>1.0e-10,abs(S2f_[0]-S2f_)>1.0e-10,abs(S3f_[0]-S3f_)>1.0e-10))][0]-t_corte
t_synaptic_end = t[np.logical_and.reduce((abs(S1f_[-1]-S1f_)<1.0e-10,abs(S2f_[-1]-S2f_)<1.0e-10,abs(S3f_[-1]-S3f_)<1.0e-10))][0]-t_corte

ax5.scatter(firings[firings[:,0]>t_corte+5000,0]-t_corte, firings[firings[:,0]>t_corte+5000,1], s=0.1, c='black')
ax5.set_title("Raster plot", fontsize=14, pad=4.5)
ax5.set_xlabel("Tempo [s]", fontsize=14)
ax5.set_ylabel("Neurônio", fontsize=14)
ax5.set_xlim([5000,25000])
ax5.set_ylim([0,1200])
ax5.set_xticks([5000,10000,15000,20000,25000],[5,10,15,20,25], fontsize=12)
ax5.set_yticks([0,200,400,600,800,1000,1200],[0,200,400,600,800,1000,1200], fontsize=12)
l1 = mlines.Line2D([10000,10000], [0,1200], color='r', linewidth=2.5)
ax5.add_line(l1)
rect1 = patches.Rectangle((10500, 500), 3000, 85, linewidth=1.8, edgecolor='r', facecolor=(1, 1, 1, 0.8), zorder=4)
ax5.add_patch(rect1)
rx, ry = rect1.get_xy()
cx = rx + rect1.get_width()/2.0
cy = ry + rect1.get_height()/2.0
ax5.annotate("\\textbf{Controle}\n\\textbf{ativado}", (cx, cy), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=5)
ax5.arrow(cx,ry,10000-cx, -100, color='r', linewidth=1.5,length_includes_head=True, zorder=3)

ax6.plot(t[t>t_corte]-t_corte, fired_[t>t_corte], linewidth=1, c='black')
ax6.set_title("Ativação neuronal", fontsize=14, pad=4.5)
ax6.set_xlabel("Tempo [s]", fontsize=14)
ax6.set_ylabel("$\%$ de disparos", fontsize=14)
ax6.set_xlim([0,25000])
ax6.set_ylim([0,100])
ax6.set_xticks([0,5000,10000,15000,20000,25000],[0,5,10,15,20,25], fontsize=12)
ax6.set_yticks([0,25,50,75,100],[0,25,50,75,100], fontsize=12)

ax7.plot(t[t>t_corte]-t_corte, simulatedLFP[t>t_corte], linewidth=1, c='black')
ax7.set_title("Potencial de campo local", fontsize=14, pad=4.5)
ax7.set_xlabel("Tempo [s]", fontsize=14)
ax7.set_ylabel("LFP [mV]", fontsize=14)
ax7.set_xlim([0,25000])
ax7.set_ylim([-200,800])
ax7.set_xticks([0,5000,10000,15000,20000,25000],[0,5,10,15,20,25], fontsize=12)
ax7.set_yticks([-200,0,200,400,600,800],[-200,0,200,400,600,800], fontsize=12)

ax8.plot(t[t>t_corte]-t_corte, stim_fbl_[t>t_corte], linewidth=1, c='black')
ax8.set_title("Sinal de controle", fontsize=14, pad=4.5)
ax8.set_xlabel("Tempo [s]", fontsize=14)
ax8.set_ylabel("$J$ [mA]", fontsize=14)
ax8.set_xlim([0,25000])
ax8.set_ylim([-5,15])
ax8.set_xticks([0,5000,10000,15000,20000,25000],[0,5,10,15,20,25], fontsize=12)
ax8.set_yticks([-5,0,5,10,15],[-5,0,5,10,15], fontsize=12)

xf = butter_highpass_filter(simulatedLFP[t>19000+t_corte],0.001,Fs)
freq,amp,amp_max,f_amp_max,power_spec = fourier(xf,1)
ax9.plot(freq*1000, amp, linewidth=1, c='black')
ax9.set_title("Transformada rápida de Fourier", fontsize=14, pad=4.5)
ax9.set_xlabel("Frequência [Hz]", fontsize=14)
ax9.set_ylabel("Magnitude", fontsize=14)
ax9.set_xlim([0,300])
ax9.set_ylim([0,2])
ax9.set_xticks([0,100,200,300],[0,100,200,300], fontsize=12)
ax9.set_yticks([0,0.5,1,1.5,2],[0.0,0.5,1.0,1.5,2.0], fontsize=12)

plt.text(-870,30*0.4,r'\textbf{(f)}',fontsize=14)
plt.text(-70,30*0.4,r'\textbf{(g)}',fontsize=14)
plt.text(-70,21.5*0.4,r'\textbf{(h)}',fontsize=14)
plt.text(-70,13.3*0.4,r'\textbf{(i)}',fontsize=14)
plt.text(-70,5.1*0.4,r'\textbf{(j)}',fontsize=14)

plt.savefig('Amygdala_flbann-fig3cd.pdf', bbox_inches='tight')
    