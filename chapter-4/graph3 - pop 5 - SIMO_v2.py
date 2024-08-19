from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

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
learn = 100
N_pop = 5
directory = 'data/modelo de massas_-fblann-case'+str(case)+'_learn'+"{:.2e}".format(learn)+'_pop'+str(N_pop)+'_SIMO_data.txt'
data_ = np.loadtxt(directory)

data = data_[data_[:,0]>=t_corte,:]
t = data[:,0]-t_corte
LFP = data[:,range(1,N_pop+1)]
u2 = data[:,-1]
u3 = data[:,-2]
dt = np.mean(np.diff(t))

t_chan = 10.0
first_peak_1 = t[np.argmax(LFP[:,0]>2)]
first_peak_2 = t[np.argmax(LFP[:,1]>2)]
first_peak_3 = t[np.argmax(LFP[:,2]>2)]
first_peak_4 = t[np.argmax(LFP[:,3]>2)]
first_peak_5 = t[np.argmax(LFP[:,4]>2)]
t_c = 40.0

# Create a figure
fig = plt.figure(figsize=(11, 15))

# Define a GridSpec with more space on the bottom row for centering
gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, LFP[:,0], c='black')
ax1.set_title("Potencial de campo local - Pop. 1", fontsize=14, pad=4.5)
ax1.set_xlabel("Tempo [s]", fontsize=14)
ax1.set_ylabel("LFP [mV]", fontsize=14)
ax1.set_xlim([0,90])
ax1.set_ylim([-20,20])
ax1.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax1.set_yticks([-20,-10,0,10,20],[-20,-10,0,10,20], fontsize=12)
l1 = mlines.Line2D([t_chan,t_chan], [-20,20], color='b', linewidth=1.2, zorder=1)
ax1.add_line(l1)
rect1 = patches.Rectangle((t_chan+7, 11.5), 30, 5, linewidth=1.2, edgecolor='b', facecolor=(1, 1, 1, 0.85), zorder=2)
ax1.add_patch(rect1)
rx, ry = rect1.get_xy()
cx = rx + rect1.get_width()/2.0
cy = ry + rect1.get_height()/2.0
l11 = mlines.Line2D([rx,t_chan], [cy,10.5], color='b', linewidth=1.2, zorder=1)
ax1.add_line(l11)
ax1.annotate("\\textbf{Parâmetro "+str(case)+"}\n\\textbf{começa a mudar}", (cx, cy-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)
l2 = mlines.Line2D([first_peak_1,first_peak_1], [-20,20], color='r', linewidth=1.2, zorder=1)
ax1.add_line(l2)
rect2 = patches.Rectangle((first_peak_1+5, -18.5), 30, 5, linewidth=1.2, edgecolor='r', facecolor=(1, 1, 1, 0.85), zorder=2)
ax1.add_patch(rect2)
rx2, ry2 = rect2.get_xy()
cx2 = rx2 + rect2.get_width()/2.0
cy2 = ry2 + rect2.get_height()/2.0
l22 = mlines.Line2D([rx2,first_peak_1], [cy2,-17], color='r', linewidth=1.2, zorder=1)
ax1.add_line(l22)
ax1.annotate("\\textbf{Início da crise}\n\\textbf{em "+str(round(first_peak_1,3))+" s}", (cx2, cy2-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, LFP[:,1], c='black')
ax2.set_title("Potencial de campo local - Pop. 2", fontsize=14, pad=4.5)
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

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t, LFP[:,2], c='black')
ax3.set_title("Potencial de campo local - Pop. 3", fontsize=14, pad=4.5)
ax3.set_xlabel("Tempo [s]", fontsize=14)
ax3.set_ylabel("LFP [mV]", fontsize=14)
ax3.set_xlim([0,90])
ax3.set_ylim([-20,20])
ax3.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax3.set_yticks([-20,-10,0,10,20],[-20,-10,0,10,20], fontsize=12)
l1 = mlines.Line2D([t_c,t_c], [-20,20], color='b', linewidth=1.2, zorder=1)
ax3.add_line(l1)
rect1 = patches.Rectangle((t_c+5, 11.5), 25, 5, linewidth=1.2, edgecolor='b', facecolor=(1, 1, 1, 0.85), zorder=2)
ax3.add_patch(rect1)
rx, ry = rect1.get_xy()
cx = rx + rect1.get_width()/2.0
cy = ry + rect1.get_height()/2.0
l11 = mlines.Line2D([rx,t_c], [cy,10.5], color='b', linewidth=1.2, zorder=1)
ax3.add_line(l11)
ax3.annotate("\\textbf{Controlador}\n\\textbf{ativado}", (cx, cy-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)
l2 = mlines.Line2D([first_peak_3,first_peak_3], [-20,20], color='r', linewidth=1.2, zorder=1)
ax3.add_line(l2)
rect2 = patches.Rectangle((first_peak_3+5, -18.5), 30, 5, linewidth=1.2, edgecolor='r', facecolor=(1, 1, 1, 0.85), zorder=2)
ax3.add_patch(rect2)
rx2, ry2 = rect2.get_xy()
cx2 = rx2 + rect2.get_width()/2.0
cy2 = ry2 + rect2.get_height()/2.0
l22 = mlines.Line2D([rx2,first_peak_3], [cy2,-17], color='r', linewidth=1.2, zorder=1)
ax3.add_line(l22)
ax3.annotate("\\textbf{Início da crise}\n\\textbf{em "+str(round(first_peak_2,3))+" s}", (cx2, cy2-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t, LFP[:,3], c='black')
ax4.set_title("Potencial de campo local - Pop. 4", fontsize=14, pad=4.5)
ax4.set_xlabel("Tempo [s]", fontsize=14)
ax4.set_ylabel("LFP [mV]", fontsize=14)
ax4.set_xlim([0,90])
ax4.set_ylim([-20,20])
ax4.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax4.set_yticks([-20,-10,0,10,20],[-20,-10,0,10,20], fontsize=12)
l2 = mlines.Line2D([first_peak_4,first_peak_4], [-20,20], color='r', linewidth=1.2, zorder=1)
ax4.add_line(l2)
rect2 = patches.Rectangle((first_peak_4+5, -18.5), 30, 5, linewidth=1.2, edgecolor='r', facecolor=(1, 1, 1, 0.85), zorder=2)
ax4.add_patch(rect2)
rx2, ry2 = rect2.get_xy()
cx2 = rx2 + rect2.get_width()/2.0
cy2 = ry2 + rect2.get_height()/2.0
l22 = mlines.Line2D([rx2,first_peak_4], [cy2,-17], color='r', linewidth=1.2, zorder=1)
ax4.add_line(l22)
ax4.annotate("\\textbf{Início da crise}\n\\textbf{em "+str(round(first_peak_4,3))+" s}", (cx2, cy2-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)

ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(t, LFP[:,4], c='black')
ax5.set_title("Potencial de campo local - Pop. 5", fontsize=14, pad=4.5)
ax5.set_xlabel("Tempo [s]", fontsize=14)
ax5.set_ylabel("LFP [mV]", fontsize=14)
ax5.set_xlim([0,90])
ax5.set_ylim([-20,20])
ax5.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax5.set_yticks([-20,-10,0,10,20],[-20,-10,0,10,20], fontsize=12)
l2 = mlines.Line2D([first_peak_5,first_peak_5], [-20,20], color='r', linewidth=1.2, zorder=1)
ax5.add_line(l2)
rect2 = patches.Rectangle((first_peak_5+5, -18.5), 30, 5, linewidth=1.2, edgecolor='r', facecolor=(1, 1, 1, 0.85), zorder=2)
ax5.add_patch(rect2)
rx2, ry2 = rect2.get_xy()
cx2 = rx2 + rect2.get_width()/2.0
cy2 = ry2 + rect2.get_height()/2.0
l22 = mlines.Line2D([rx2,first_peak_5], [cy2,-17], color='r', linewidth=1.2, zorder=1)
ax5.add_line(l22)
ax5.annotate("\\textbf{Início da crise}\n\\textbf{em "+str(round(first_peak_5,3))+" s}", (cx2, cy2-0.3), color='black', weight='bold', fontsize=12, ha='center', va='center', zorder=3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(t, u2, linewidth=1, c='black', label='$J_2$')
ax6.plot(t, u3, linewidth=1, c='grey', label='$J_3$')
ax6.set_title("Sinal de controle", fontsize=14, pad=4.5)
ax6.set_xlabel("Tempo [s]", fontsize=14)
ax6.set_ylabel("$J$ [$\mu$A]", fontsize=14)
ax6.set_xlim([0,90])
ax6.set_xticks([0,30,60,90],[0,30,60,90], fontsize=12)
ax6.set_ylim([-400,0])
ax6.set_yticks([-400,-300,-200,-100,0],[-400,-300,-200,-100,0], fontsize=12)
ax6.legend(fontsize=14)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.text(-121,964,r'\textbf{(a.1)}',fontsize=14)
plt.text(-14,964,r'\textbf{(a.2)}',fontsize=14)
plt.text(-121,484,r'\textbf{(a.3)}',fontsize=14)
plt.text(-14,484,r'\textbf{(a.4)}',fontsize=14)
plt.text(-121,5,r'\textbf{(a.5)}',fontsize=14)
plt.text(-13,5,r'\textbf{(b)}',fontsize=14)

plt.savefig('graph3-fblann-'+str(case)+'_pop'+str(N_pop)+'-SIMO-pt.pdf', bbox_inches='tight')

