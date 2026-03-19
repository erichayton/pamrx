import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, lfilter

# Parameters
baud = 80_000
sps = 30  # samples per symbol
fs = baud * sps
pulse_width_ratio=.5
noise_sigma=0.2
channel_bandwidth =.16
num_symbols = int(baud * 50 * 1e-3)
discard_symbols = int(0.1 * num_symbols)
pulse_tw = sps/2*pulse_width_ratio/fs*1e3
pulse_fw = 1/pulse_tw

print ("sample rate[khz", fs/1e3)
print ("pulse width ",end='')
print (f"time[ms] {pulse_tw} freq[khz] {pulse_fw}")
print ("ch_bw[khz]", channel_bandwidth*baud * sps/1e3)
print ("discard symbols ", discard_symbols)
print ("num symbols ",num_symbols)

spp = int(sps*pulse_width_ratio)  # samples per pulse
# PAM4 signal
levels = np.array([-3, -1, 1, 3])

data = levels[np.random.randint(0, len(levels), num_symbols)]
tx = np.zeros(num_symbols * sps)
tx[::sps] = data


# place pulse0 within pulse
pulse0 = 0.5 * (1 - np.cos(2*np.pi/spp*np.arange(spp)))
wings = int((sps - spp) / 2)
pulse = np.zeros(sps)
pulse[wings:wings+len(pulse0)] = pulse0

signal = np.convolve(tx, pulse, mode="same")

# limited channel bandwidth
b, a = butter(5, channel_bandwidth )   
rx = lfilter(b, a, signal)
#rx=signal  #bypass bw limiting filter

# AWGN
rx += np.random.normal(0, noise_sigma, len(rx))

# Matched filter on the recv'd signal
matched = pulse[::-1]
rx_mf = np.convolve(rx, matched, mode="same")


def lms_equalizer(rx_mf, desired,eq_taps=7,mu=.001):

    rx_eq = np.zeros_like(rx_mf)

    eq = np.zeros(eq_taps)
    eq[eq_taps // 2] = 1.0  # identity initialization

    for n in range(eq_taps, len(rx_mf)):
        x = rx_mf[n-eq_taps+1:n+1][::-1]
        #y = np.dot(eq, x)
        y = eq@x
        e = desired[n] - y
        if np.isnan(e) or np.isnan(x).any():  # skip invalid updates
            rx_eq[n] = y
            continue
        eq += mu * e * x
        rx_eq[n] = y
    return rx_eq,eq



def rls_equalizer(rx_mf, desired, eq_taps=7, lambda_=0.999, delta=50.):
    """
    Recursive Least Squares (RLS) equalizer.

    Parameters:
        rx_mf   : array, matched-filtered received signal
        desired : array, target signal (pulse-shaped + matched filter)
        eq_taps : int, number of equalizer taps
        lambda_ : float, forgetting factor (0 < lambda_ <= 1)
        delta   : float, initial inverse correlation scaling

    Returns:
        rx_eq   : array, equalized output
        eq      : array, final equalizer coefficients
    """


    eq = np.zeros(eq_taps)
    eq[eq_taps // 2] = 1.0           # identity initialization
    rx_eq = np.zeros_like(rx_mf)
    P = np.eye(eq_taps) * delta       # inverse correlation matrix

    for n in range(eq_taps, len(rx_mf)):

        # input vector (most recent first)
        x = rx_mf[n-eq_taps+1:n+1][::-1]

        # filter output
        y = eq @ x

        # error between desired and output
        e = desired[n] - y

        # gain vector
        Px = P @ x
        K = Px / (lambda_ + x @ Px )

        # update weights
        eq = eq + K * e

        # update inverse correlation matrix
        P = (P - np.outer(K, Px)) / lambda_

        # save output
        rx_eq[n] = y

    return rx_eq, eq

# prepare desired:  upsample data and pulse-shape to match MF output
tx_upsampled = np.zeros(len(data)*sps)
tx_upsampled[::sps] = data
desired = np.convolve(tx_upsampled, pulse, mode="same")
desired = np.convolve(desired, matched, mode="same")

rx_eq,eq=lms_equalizer(rx_mf,desired)

#RLS with delta set by <power level>
#print (np.var(rx_mf))
#rx_eq,eq=rls_equalizer(rx_mf,desired,delta=np.var(rx_mf)*2)

#RLS with clipping
#rx_mf_norm = rx_mf / np.std(rx_mf)
#desired_norm = desired / np.std(rx_mf)
#rx_eq, eq = rls_equalizer(rx_mf_norm, desired_norm, delta=1.0,eq_taps=9)
#rx_eq = np.clip(rx_eq, -np.max(np.abs(desired))*1.2, np.max(np.abs(desired))*1.2)

# find baud-rate spectral line
def bandpass(low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='bandpass')
    return b, a

sq = rx_eq**2
sq = sq - np.mean(sq)

b, a = bandpass(baud*0.8, baud*1.2, fs)
tone = filtfilt(b, a, sq)

tone = tone / np.std(tone)

###########################
# PLL

phase = 0
freq = 2*np.pi*baud/fs*1.04
integrator = 0

phase_error = np.zeros_like(tone)
freq_est = np.zeros_like(tone)
nco = np.zeros_like(tone)

Kp = 0.02
Ki = 0.0005

phase_track = np.zeros_like(tone)

for n in range(len(tone)):

    I = tone[n] * np.cos(phase)
    Q = tone[n] * np.sin(phase)

    err = -Q

    integrator += Ki * err
    control = Kp * err + integrator

    phase += freq + control
    phase = (phase + np.pi) % (2*np.pi) - np.pi

    phase_track[n]=phase

    phase_error[n] = err
    freq_est[n] = (freq + control) * fs / (2*np.pi)
    nco[n] = np.cos(phase)

#################
#pulse shape

plt.figure(figsize=(4,2))
plt.plot(pulse, 'r.')

########################3
# recv'd vs transmit (last 10000)

plt.figure(figsize=(20,4))
plt.plot(signal[-10000:], 'r-', label="transmit")
plt.plot(rx[-10000:], 'b-', label="recv'd")
plt.plot(rx_mf[-10000:], 'g-', label="matched filter")
plt.plot(rx_eq[-10000:], 'm-', label="equalized")

plt.title("recv'd vs transmit (last 10000)")
plt.legend()



######################
# baud tone view

seg = 2<<11

f1, P1 = welch(rx, fs, nperseg=seg)
f2, P2 = welch(sq, fs, nperseg=seg)
f3, P3 = welch(tone, fs, nperseg=seg)

baud_bin = np.argmin(np.abs(f2 - baud))
tone_power = P2[baud_bin]

noise_bins = np.r_[baud_bin-20:baud_bin-3, baud_bin+3:baud_bin+20]
noise_power = np.mean(P2[noise_bins])

btnr = 10*np.log10(tone_power / noise_power)


plt.figure(figsize=(20,5))
plt.semilogy(f1, P1, label="recv'd signal")
plt.semilogy(f2, P2, label="mfilter&sq...")
plt.semilogy(f3, P3, label="...bandpass")
plt.xlim(0, 4*baud)

plt.axvline(baud, color='r', linestyle='--')
plt.text(baud*1.1, max(P2), f"BTNR = {btnr:.1f} dB")

plt.legend()
plt.title("Baud tone")

#################
# PLL diagnostics

fig, axs = plt.subplots(3, 1, figsize=(20, 8), sharex=True, gridspec_kw={'height_ratios':[1,1,1]})

# Phase Detector Output
axs[0].plot(phase_error, label="Phase Error", color='tab:blue')
axs[0].set_ylabel("Phase Error")
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Frequency Estimate
axs[1].plot(freq_est, label="Frequency Estimate", color='tab:orange')
axs[1].axhline(baud, color='r', linestyle='--', label="Baud Rate")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].legend(loc='upper right')
axs[1].grid(True)

# PLL Locking to Baud Tone
axs[2].plot(tone, label="Recovered Tone", color='tab:green')
axs[2].plot(nco, label="PLL Clock", color='tab:red', linestyle='solid')
axs[2].set_xlabel("Sample")
axs[2].set_ylabel("Amplitude")
axs[2].legend(loc='upper right')
axs[2].grid(True)

# Overall figure title
fig.suptitle("PLL Diagnostics", fontsize=16)

# Adjust layout so titles/labels don't overlap
fig.tight_layout(rect=[0, 0, 1, 0.96])



############################
# Sampling phase detection

phases = np.arange(sps)
metric = np.zeros(sps)

for p in phases:
 
    samples = rx_eq[p::sps][:num_symbols]
    
    #metric[p] = np.var(samples)
    metric[p] = np.mean(np.abs(samples))

    
best_phase = np.argmax(metric)

print("Best sampling phase:", best_phase)

plt.figure()
plt.plot(phases, metric, 'o-')
plt.axvline(best_phase, color='r', linestyle='--')
plt.title("Eye Opening Metric vs Sampling Phase")
plt.xlabel("Sample Offset")
plt.ylabel("metric")


#####################
# Eye Diagram 

# diagram is eye_symbols wide
eye_symbols = 2
eye_samples = eye_symbols * sps

# make a matrix out of stacked symbols, sps wide
usable = (len(rx_eq) // sps) * sps
reshaped = rx_eq[:usable].reshape(-1, sps)

plt.figure(figsize=(20,5))

for i in range(len(reshaped)-eye_symbols):
    segment = reshaped[i:i+eye_symbols].flatten()
    t = np.arange(eye_samples) / sps
    plt.plot(t, segment, color='blue', alpha=0.05)

plt.axvline(best_phase/sps, color='red', linestyle='--', linewidth=2, label="Best Sample")

plt.xlabel("Symbol Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()


#####################################
# Adaptive Symbol Slicing for PAM4

# Sample the matched filter output at the best phase
samples = rx_eq[best_phase::sps][:num_symbols]

# Estimate PAM4 levels using KMeans clustering
from sklearn.cluster import KMeans
init = levels.reshape(-1,1)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=0)

samples_steady = samples[discard_symbols:]

kmeans.fit(samples_steady.reshape(-1,1))

# Extract and sort cluster centers
levels_rx = np.sort(kmeans.cluster_centers_.flatten())

print (levels_rx)
# Compute decision thresholds
thresholds = (levels_rx[:-1] + levels_rx[1:]) / 2

# Slice the symbols based on thresholds
sliced = np.zeros_like(samples)

sliced[samples < thresholds[0]] = levels_rx[0]
sliced[(samples >= thresholds[0]) & (samples < thresholds[1])] = levels_rx[1]
sliced[(samples >= thresholds[1]) & (samples < thresholds[2])] = levels_rx[2]
sliced[samples >= thresholds[2]] = levels_rx[3]


plt.figure()
plt.plot(samples, '.', label="Samples")
plt.plot(sliced, 'r.', label="Sliced")
plt.title("Symbol Slicing")
plt.xlabel("Symbol Index")
plt.ylabel("Amplitude")

plt.grid(True)


#########################
# Constellation

plt.figure(figsize=(6,6))

plt.plot(samples, np.zeros_like(samples), '.', alpha=0.3)

for lvl in levels_rx:
    plt.axvline(lvl, linestyle='--')

plt.title("PAM4 Constellation")
plt.xlabel("In-phase")
plt.ylabel("Quadrature (unused)")
plt.ylim(-0.5,0.5)
plt.grid(True)


#########################
# Symbol Error Rate

# map TX levels to indices
level_map = { -3:0, -1:1, 1:2, 3:3 }
tx_idx = np.array([level_map[v] for v in data[:len(samples)]])

# determine RX symbol indices using thresholds
rx_idx = np.zeros_like(samples, dtype=int)

rx_idx[samples < thresholds[0]] = 0
rx_idx[(samples >= thresholds[0]) & (samples < thresholds[1])] = 1
rx_idx[(samples >= thresholds[1]) & (samples < thresholds[2])] = 2
rx_idx[samples >= thresholds[2]] = 3

# --- alignment via correlation ---
corr = np.correlate(rx_idx, tx_idx, mode='full')
lag = np.argmax(corr) - len(tx_idx) + 1


if lag > 0:
    rx_valid = rx_idx[lag:]
    tx_valid = tx_idx[:len(rx_valid)]
else:
    tx_valid = tx_idx[-lag:]
    rx_valid = rx_idx[:len(tx_valid)]

# discard transient
tx_valid = tx_valid[discard_symbols:]
rx_valid = rx_valid[discard_symbols:]

# compute errors
error_vector = (tx_valid != rx_valid)

ser = np.mean(error_vector)


print(f"{ser=:.2%} errors {np.sum(error_vector)} of {len(error_vector)}")


############################################
# PAM4 Histogram and Decision Thresholds
plt.figure(figsize=(8,5))

plt.hist(samples, bins=100, density=True, alpha=0.7, color='steelblue')

# plot estimated levels
for lvl in levels_rx:
    plt.axvline(lvl, color='green', linestyle='--', linewidth=1)

# plot decision thresholds
for th in thresholds:
    plt.axvline(th, color='red', linestyle='-', linewidth=2)
    
plt.title("PAM4 Sample Histogram")
plt.xlabel("Amplitude")
plt.ylabel("Probability Density")
plt.grid(True)

plt.show()
