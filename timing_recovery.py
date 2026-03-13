import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, lfilter


# Parameters
baud = 80_000
sps = 20  # samples per symbol
fs = baud * sps
num_symbols = int(baud * 16 * 1e-3)   
spp = int(sps*0.8)  # samples per pulse
noise_sigma=0.2

# PAM4 signal
levels = np.array([-3, -1, 1, 3])

data = levels[np.random.randint(0, len(levels), num_symbols)]

tx = np.zeros(num_symbols * sps)
tx[::sps] = data


# plase pulse0 within pulse
pulse0 = 0.5 * (1 - np.cos(2*np.pi/spp*np.arange(spp)))
wings = int((sps - spp) / 2)
pulse = np.zeros(sps)
pulse[wings:wings+len(pulse0)] = pulse0

signal = np.convolve(tx, pulse, mode="same")

# limited channel bandwidth
b, a = butter(5, 0.0663)   
rx = lfilter(b, a, signal)

# AWGN
rx += np.random.normal(0, noise_sigma, len(rx))


# Matched filter on the recv'd signal
matched = pulse[::-1]
rx_mf = np.convolve(rx, matched, mode="same")


# find baud-rate spectral line
def bandpass(low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='bandpass')
    return b, a

sq = rx_mf**2
sq = sq - np.mean(sq)

b, a = bandpass(baud*0.8, baud*1.2, fs)
tone = filtfilt(b, a, sq)

tone = tone / np.std(tone)

# ------------------------------------------------
# Digital PLL
# ------------------------------------------------
phase = 0
freq = 2*np.pi*baud/fs*1.04
integrator = 0

phase_error = np.zeros(len(tone))
freq_est = np.zeros(len(tone))
nco = np.zeros(len(tone))

Kp = 0.02
Ki = 0.0005

for n in range(len(tone)):

    I = tone[n] * np.cos(phase)
    Q = tone[n] * np.sin(phase)

    err = Q

    integrator += Ki * err
    control = Kp * err + integrator

    phase += freq + control
    phase = (phase + np.pi) % (2*np.pi) - np.pi

    phase_error[n] = err
    freq_est[n] = (freq + control) * fs / (2*np.pi)
    nco[n] = -np.cos(phase)


# recv'd vs transmit
plt.figure()
plt.plot(rx[-10000:], 'b-', label="recv'd")
plt.plot(rx_mf[-10000:], 'g-', label="matched filter output")
plt.plot(signal[-10000:], 'r-', label="transmit")
plt.legend()



# baud tone view
seg = 2<<11

f1, P1 = welch(rx, fs, nperseg=seg)
f2, P2 = welch(sq, fs, nperseg=seg)
f3, P3 = welch(tone, fs, nperseg=seg)

baud_bin = np.argmin(np.abs(f2 - baud))
tone_power = P2[baud_bin]

noise_bins = np.r_[baud_bin-20:baud_bin-4, baud_bin+4:baud_bin+20]
noise_power = np.mean(P2[noise_bins])

btnr = 10*np.log10(tone_power / noise_power)

plt.figure()
plt.plot(pulse, 'r.')

plt.figure(figsize=(10,6))
plt.semilogy(f1, P1, label="recv'd signal")
plt.semilogy(f2, P2, label="mfilter&sq...")
plt.semilogy(f3, P3, label="...bandpass")
plt.xlim(0, 4*baud)

plt.axvline(baud, color='r', linestyle='--')
plt.text(baud*1.1, max(P2), f"BTNR = {btnr:.1f} dB")

plt.legend()
plt.title("Spectral Line Creation")

# ------------------------------------------------
# PLL diagnostics
# ------------------------------------------------
plt.figure()
plt.plot(phase_error)
plt.title("PLL Phase Detector Output")
plt.xlabel("Sample")

plt.figure()
plt.plot(freq_est)
plt.axhline(baud, color='r', linestyle='--')
plt.title("PLL Frequency Estimate")
plt.xlabel("Sample")
plt.ylabel("Frequency (Hz)")

plt.figure()
plt.plot(tone, label="Recovered tone")
plt.plot(nco, label="PLL clock")
plt.legend()
plt.title("PLL Locking to Baud Tone")


# Sampling phase detection
phases = np.arange(sps)
metric = np.zeros(sps)

for p in phases:

    samples = rx_mf[p::sps][:num_symbols]

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



# Eye Diagram 

# diagram is eye_symbols wide
eye_symbols = 2
eye_samples = eye_symbols * sps

# make a matrix out of stacked symbols, sps wide
usable = (len(rx_mf) // sps) * sps
reshaped = rx_mf[:usable].reshape(-1, sps)

plt.figure(figsize=(9,5))

for i in range(len(reshaped)-eye_symbols):
    segment = reshaped[i:i+eye_symbols].flatten()
    t = np.arange(eye_samples) / sps
    plt.plot(t, segment, color='blue', alpha=0.05)

plt.axvline(best_phase/sps, color='red', linestyle='--', linewidth=2, label="Best Sample")

plt.title("Eye Diagram")
plt.xlabel("Symbol Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()



# Adaptive Symbol Slicing for PAM4

# Sample the matched filter output at the best phase
samples = rx_mf[best_phase::sps][:num_symbols]

# Estimate PAM4 levels from samples
sorted_samples = np.sort(samples)
N = len(samples)

lvl1 = np.mean(sorted_samples[:N//4])
lvl2 = np.mean(sorted_samples[N//4:N//2])
lvl3 = np.mean(sorted_samples[N//2:3*N//4])
lvl4 = np.mean(sorted_samples[3*N//4:])
levels_rx = np.array([lvl1, lvl2, lvl3, lvl4])

# Compute decision thresholds (midpoints between levels)
thresholds = (levels_rx[:-1] + levels_rx[1:]) / 2

# Slice the symbols based on thresholds
sliced = np.zeros_like(samples)

sliced[samples < thresholds[0]] = levels_rx[0]
sliced[(samples >= thresholds[0]) & (samples < thresholds[1])] = levels_rx[1]
sliced[(samples >= thresholds[1]) & (samples < thresholds[2])] = levels_rx[2]
sliced[samples >= thresholds[2]] = levels_rx[3]


plt.figure()
plt.plot(samples[:500], '.', label="Samples")
plt.plot(sliced[:500], 'r.', label="Sliced")
plt.title("Symbol Slicing with Adaptive Levels")
plt.xlabel("Symbol Index")
plt.ylabel("Amplitude")

plt.grid(True)



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



# Symbol Error Rate

# map TX levels to indices
level_map = { -3:0, -1:1, 1:2, 3:3 }
tx_idx = np.array([level_map[v] for v in data[:len(samples)]])

# determine RX symbol indices using thresholds
rx_idx = np.zeros(len(samples), dtype=int)

rx_idx[samples < thresholds[0]] = 0
rx_idx[(samples >= thresholds[0]) & (samples < thresholds[1])] = 1
rx_idx[(samples >= thresholds[1]) & (samples < thresholds[2])] = 2
rx_idx[samples >= thresholds[2]] = 3

# compute errors
error_vector = (tx_idx != rx_idx)
ser = np.mean(error_vector)

print("Symbol error rate:", ser, np.sum(error_vector), len(error_vector))



# PAM4 Histogram and Decision Thresholds
plt.figure(figsize=(8,5))

plt.hist(samples, bins=100, density=True, alpha=0.7, color='steelblue')

# plot estimated levels
for lvl in levels_rx:
    plt.axvline(lvl, color='green', linestyle='--', linewidth=2)

# plot decision thresholds
for th in thresholds:
    plt.axvline(th, color='red', linestyle='-', linewidth=2)

plt.title("PAM4 Sample Histogram")
plt.xlabel("Amplitude")
plt.ylabel("Probability Density")
plt.grid(True)

plt.legend(["Estimated Levels", "Decision Thresholds"])


plt.show()
