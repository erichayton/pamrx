import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, lfilter

# Parameters
baud = 80_000
channel_bandwidth =68_000
sps = 20  # samples per symbol
fs = baud * sps
nyq=fs/2
pulse_width_ratio=.3
spp = int(sps*pulse_width_ratio)  # samples per pulse
noise_sigma=0.2
num_symbols = int(baud * 50 * 1e-3)
eye_symbols = 4

discard_symbols = int(0.1 * num_symbols)

print ("sample rate[khz]", fs/1e3)
print ("ch_bw[khz]", channel_bandwidth/1e3)
print ("discard symbols ", discard_symbols)
print ("num symbols ",num_symbols)
print ("===============")

print ("create PAM4 signal")

levels = np.array([-3, -1, 1, 3])

data = levels[np.random.randint(0, len(levels), num_symbols)]
tx = np.zeros(num_symbols * sps)
tx[::sps] = data

def rcos_shape(sps,spp):
    

    # place pulse0 within pulse
    pulse0 = 0.5 * (1 - np.cos(2*np.pi/spp*np.arange(spp)))
    wings = int((sps - spp) / 2)
    pulse = np.zeros(sps)
    pulse[wings:wings+len(pulse0)] = pulse0
    pulse/=np.max(pulse)
    return pulse

def gaussian_shape(sps,spp):
    
    sigma = spp / 2.355
    pulse = np.exp(-0.5 * ((np.arange(sps) - (sps-1)/2) / sigma)**2)
    pulse /= np.max(pulse)

    return pulse


def bandpass(low, high, nyq, order=4):

    b, a = butter(order, [low/nyq, high/nyq], btype='bandpass')
    return b, a


def lms_equalizer(rx_mf, desired, eq_taps=5, mu=1e-3, sps=20, best_phase=0):
    """
    Symbol-spaced LMS equalizer.

    Parameters:
        rx_mf    : matched-filtered received signal (array)
        desired  : reference signal (array, pulse-shaped + matched filter)
        eq_taps  : number of equalizer taps
        mu       : step size (LMS gain)
        sps      : samples per symbol
        best_phase : symbol sampling offset (0..sps-1)

    Returns:
        rx_eq : equalized output
        eq    : final equalizer coefficients
    """
    rx_eq = np.zeros_like(rx_mf)
    eq = np.zeros(eq_taps)
    eq[eq_taps // 2] = 1.0  # identity init

    for n in range(eq_taps, len(rx_mf)):
        x = rx_mf[n-eq_taps+1:n+1][::-1]
        y = eq @ x

        # --- update only at symbol centers ---
        if (n - best_phase) % sps == 0:
            e = desired[n] - y

            eq += mu * e * x / (np.dot(x, x) + 1e-12)  # NLMS update

        rx_eq[n] = y

    return rx_eq, eq


def rls_equalizer(rx_mf, desired, eq_taps=7, lambda_=0.999, delta=50., sps=20, best_phase=0):
    """
    Symbol-spaced RLS equalizer.

    Parameters:
        rx_mf    : matched-filtered received signal
        desired  : reference signal (pulse-shaped + matched filter)
        eq_taps  : number of equalizer taps
        lambda_  : forgetting factor (0 < lambda_ <= 1)
        delta    : initial inverse correlation scaling
        sps      : samples per symbol
        best_phase : symbol sampling offset (0..sps-1)

    Returns:
        rx_eq : equalized output
        eq    : final equalizer coefficients
    """
    eq = np.zeros(eq_taps)
    eq[eq_taps // 2] = 1.0           # identity init
    rx_eq = np.zeros_like(rx_mf)
    P = np.eye(eq_taps) * delta       # inverse correlation matrix

    for n in range(eq_taps, len(rx_mf)):

        x = rx_mf[n-eq_taps+1:n+1][::-1]
        y = eq @ x

        # --- update only at symbol centers ---
        if (n - best_phase) % sps == 0:
            e = desired[n] - y

            Px = P @ x
            K = Px / (lambda_ + x @ Px)
            eq = eq + K * e
            P = (P - np.outer(K, Px)) / lambda_

        rx_eq[n] = y

    return rx_eq, eq

#pulse= rcos_shape(sps,spp)
pulse= gaussian_shape(sps,spp)

print ("plot the pulse shape")
# ----- TIME AXIS -----
t = np.arange(len(pulse)) / fs 

# ----- FFT -----
Nfft = 2<<12
P = np.fft.fft(pulse, Nfft)
f = np.fft.fftfreq(Nfft, d=1/fs)

# Positive freqs
P = np.abs(P[:Nfft//2])
f = f[:Nfft//2]

# Normalize
P = P / np.max(P)
P_db = 20 * np.log10(P + 1e-12)

# -3 dB bandwidth
peak_idx = np.argmax(P)
half_power = P[peak_idx] / np.sqrt(2)

indices = np.where(P >= half_power)[0]
bw_3db = (f[indices[-1]] - f[indices[0]])


#print(f"-3 dB bandwidth ≈ {bw_3db/1e3:.2f} khz ")

# time-domain -3 dB (FWHM) 
half_power = 0.707  # -3 dB in amplitude

indices = np.where(pulse >= half_power * np.max(pulse))[0]

t_left = t[indices[0]]
t_right = t[indices[-1]]
fwhm = t_right - t_left

#print(f"FWHM ≈ {fwhm*1e6:.2f} us ")


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t*1e6, pulse, label="Pulse")
plt.axhline(half_power * np.max(pulse), color='red', linestyle='--', label='-3 dB level')
plt.axvline(t_left, color='green', linestyle='--')
plt.axvline(t_right, color='green', linestyle='--', label=f'FWHM ≈ {fwhm*1e6:.3f} us')
plt.title("Pulse (Time Domain)")
plt.xlabel("Time (us)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(f/1e3, P_db, label="Spectrum")
plt.axhline(-3, color='red', linestyle='--', label='-3 dB')
plt.axvline(bw_3db/1e3, color='green', linestyle='--', label=f'BW ≈ {bw_3db/1e3:.1f} kHz')
plt.title("Pulse Spectrum")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.legend()

plt.tight_layout()




signal = np.convolve(tx, pulse, mode="same")

# limit channel bandwidth
b, a = butter(5, channel_bandwidth/nyq,btype='low')   
rx = lfilter(b, a, signal)
#rx=signal  #bypass bw limiting filter

# AWGN
rx += np.random.normal(0, noise_sigma, len(rx))


# ok that's it for signal creation and channel impairments
# now, receive it

# Matched filter on the recv'd signal
matched = pulse[::-1]
rx_mf = np.convolve(rx, matched, mode="same")
rx_mf = rx_mf / np.std(rx_mf)

def bestphase(rx, sps=sps, num_symbols=num_symbols):
    """
    determine optimum sampling phase
    """

    phases = np.arange(sps)
    metric = np.zeros(sps)

    for p in phases:
 
        samples = rx[p::sps][:num_symbols]
    
        #metric[p] = np.var(samples)
        metric[p] = np.mean(np.abs(samples))
        #metric[p] = np.mean(samples**2) / (np.var(samples) + 1e-12)  #7 fix
    
    return np.argmax(metric),phases,metric


best_phase,phases,metric = bestphase(rx_mf)

print("best sampling phase:", best_phase)

samples = rx_mf[best_phase::sps][:num_symbols]


# prepare desired:  upsample data and pulse-shape to match MF output
tx_upsampled = np.zeros(len(data)*sps)
tx_upsampled[::sps] = data
desired = np.convolve(tx_upsampled, pulse, mode="same")
desired = np.convolve(desired, matched, mode="same")
desired = desired / np.std(desired)

corr = np.correlate(rx_mf, desired, mode='full')

lag = np.argmax(corr) - (len(desired) - 1)

print("Estimated delay:", lag)

if lag > 0:
    desired = np.pad(desired, (lag, 0))[:len(desired)]
else:
    desired = np.pad(desired, (0, -lag))[-lag:]




rx_eq, eq = rls_equalizer(rx_mf,desired,eq_taps=11 , lambda_=0.999, delta=1.0 / np.var(rx_mf) , sps=sps, best_phase=best_phase)

#rx_eq, eq = lms_equalizer(rx_mf,desired,7,3e-3,sps,best_phase)

# no equalizer
#rx_eq=rx_mf

best_phase,phases,metric = bestphase(rx_eq)

sym_err = (desired - rx_eq)[best_phase::sps]
plt.figure()
plt.plot(sym_err**2)
plt.title("symbol error power")
plt.grid(True)


print ("filter out the baud-rate component")

sq = rx_eq**2
sq = sq - np.mean(sq)

b, a = bandpass(baud*0.8, baud*1.2, nyq)
tone = filtfilt(b, a, sq)

tone = tone / np.std(tone)

print("run the PLL")

def phase_cross(prev, curr, target):
    d1 = (target - prev + np.pi) % (2*np.pi) - np.pi
    d2 = (target - curr + np.pi) % (2*np.pi) - np.pi
    return (d1 > 0) and (d2 <= 0)


phase = 0
prev_phase=phase
freq = 2*np.pi*baud/fs*1.02
integrator = 0

phase_error = np.zeros_like(tone)
freq_est = np.zeros_like(tone)
nco = np.zeros_like(tone)

Kp = 0.02
Ki = 0.0005

phase_track = np.zeros_like(tone)
phase_offset = 2 * np.pi * best_phase / sps
target = (phase_offset + np.pi) % (2*np.pi) - np.pi
strobe = np.zeros_like(tone)

samples=[]
for n in range(len(tone)):

    I = tone[n] * np.cos(phase)
    Q = tone[n] * np.sin(phase)

    err = -Q

    integrator += Ki * err
    control = Kp * err + integrator

    phase += freq + control
    phase = (phase + np.pi) % (2*np.pi) - np.pi


    if phase_cross(prev_phase, phase, target):
        frac = (target - prev_phase) / (phase - prev_phase + 1e-12)
        interp = rx_eq[n-1] + frac * (rx_eq[n] - rx_eq[n-1])
        samples.append(interp)
    
    phase_track[n]=phase
    prev_phase = phase
    
    phase_error[n] = err
    freq_est[n] = (freq + control) * fs / (2*np.pi)
    nco[n] = np.cos(phase)

samples=np.array(samples)

print("plot recv'd vs transmit (last 10000)")

plt.figure(figsize=(20,4))
plt.plot(signal[-10000:], 'r-', label="transmit")
plt.plot(rx[-10000:], 'b-', label="recv'd")
plt.plot(rx_mf[-10000:], 'g-', label="matched filter")
plt.plot(rx_eq[-10000:], 'm-', label="equalized")

plt.title("recv'd vs transmit (last 10000)")
plt.legend()



print ("plot baud tone")

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

print ("plot PLL diagnostics")

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


print ("eye diagram")

# diagram is eye_symbols wide
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


print("calculate slice regions")


###  we're trying to strobe this with PLL now
# Sample at the best phase
#samples = rx_eq[best_phase::sps][:num_symbols]


# Estimate PAM4 levels using KMeans clustering
from sklearn.cluster import KMeans
init = levels.reshape(-1,1)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=0)
#kmeans = KMeans(n_clusters=4, init=init, n_init=1)

samples_steady = samples[discard_symbols:]

kmeans.fit(samples_steady.reshape(-1,1))

# Extract and sort cluster centers
levels_rx = np.sort(kmeans.cluster_centers_.flatten())

#print (levels_rx)
# Compute decision thresholds
thresholds = (levels_rx[:-1] + levels_rx[1:]) / 2

# Slice the symbols based on thresholds
sliced = np.zeros_like(samples)

sliced[samples < thresholds[0]] = levels_rx[0]
sliced[(samples >= thresholds[0]) & (samples < thresholds[1])] = levels_rx[1]
sliced[(samples >= thresholds[1]) & (samples < thresholds[2])] = levels_rx[2]
sliced[samples >= thresholds[2]] = levels_rx[3]





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


print(f"SER={ser:.2%} errors {np.sum(error_vector)} of {len(error_vector)}")


print("plot symbol slices")
fig,axs=plt.subplots(2,2,figsize=(9,9))
axs[0][1].plot(samples, '.', label="Samples")
axs[0][1].plot(sliced, 'r.', label="Sliced")
axs[0][1].set_xlabel("Symbol Index")
axs[0][1].set_ylabel("Amplitude")
axs[0][1].grid(True)
fig.suptitle("Symbol Slicing")

print ("histogram")

axs[1][0].hist(samples, bins=100, density=True, alpha=0.7, color='steelblue')

# plot estimated levels
for lvl in levels_rx:
    axs[1][0].axvline(lvl, color='green', linestyle='--', linewidth=1)

# plot decision thresholds
for th in thresholds:
    axs[1][0].axvline(th, color='red', linestyle='-', linewidth=2)
    
axs[1][0].set_xlabel("Amplitude")
axs[1][0].set_ylabel("Probability Density")
axs[1][0].grid(True)

print ("constellation")
axs[0][0].plot(samples, np.zeros_like(samples), '.', alpha=0.3)

for lvl in levels_rx:
    axs[0][0].axvline(lvl, linestyle='--', color='green',linewidth=1)

axs[0][0].set_xlabel("In-phase")
axs[0][0].set_ylabel("Quadrature (unused)")
axs[0][0].set_ylim(-0.5,0.5)
axs[0][0].grid(True)

# sample point vs metric
axs[1][1].plot(phases, metric, 'o-')
axs[1][1].axvline(best_phase, color='r', linestyle='--')
axs[1][1].set_xlabel("Sample Offset")
axs[1][1].set_ylabel("metric")


plt.show()
