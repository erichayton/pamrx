import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ------------------------------------------------
# Parameters
# ------------------------------------------------
baud = 80_000
sps = 20  # samples per symbol
fs = baud * sps
num_symbols = int(baud * 8 * 1e-3)   
spp = int(sps*1.0)  # samples per pulse
noise_sigma=0.2

# ------------------------------------------------
# Generate PAM4 signal
# ------------------------------------------------
levels = np.array([-3, -1, 1, 3])

data = levels[np.random.randint(0, len(levels), num_symbols)]

tx = np.zeros(num_symbols * sps)
tx[::sps] = data

pulse0 = 0.5 * (1 - np.cos(2*np.pi/spp*np.arange(spp)))

wings = int((sps - spp) / 2)
pulse = np.zeros(sps)
pulse[wings:wings+len(pulse0)] = pulse0

signal = np.convolve(tx, pulse, mode="same")

rx = signal + np.random.normal(0, noise_sigma, len(signal))

# ------------------------------------------------
# Matched filter
# ------------------------------------------------
matched = pulse[::-1]
rx_mf = np.convolve(rx, matched, mode="same")

# ------------------------------------------------
# Create baud-rate spectral line
# ------------------------------------------------

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

# ------------------------------------------------
# Spectrum view
# ------------------------------------------------
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
plt.plot(rx[-10000:], 'b-', label="data")
plt.plot(rx_mf[-10000:], 'g-', label="matched filter output")
plt.plot(signal[-10000:], 'r-', label="transmit")
plt.legend()

plt.figure()
plt.plot(pulse, 'r.')

plt.figure(figsize=(10,6))
plt.semilogy(f1, P1, label="Original signal")
plt.semilogy(f2, P2, label="After squaring")
plt.semilogy(f3, P3, label="After bandpass")
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

# ------------------------------------------------
# Sampling phase detection
# ------------------------------------------------
phases = np.arange(sps)
metric = np.zeros(sps)

for p in phases:

    samples = rx_mf[p::sps]
    samples = samples[:num_symbols]

    metric[p] = np.var(samples)

best_phase = np.argmax(metric)

print("Best sampling phase:", best_phase)

plt.figure()
plt.plot(phases, metric, 'o-')
plt.axvline(best_phase, color='r', linestyle='--')
plt.title("Eye Opening Metric vs Sampling Phase")
plt.xlabel("Sample Offset")
plt.ylabel("Variance")

# ------------------------------------------------
# Eye Diagram
# ------------------------------------------------
eye_symbols = 2
eye_samples = eye_symbols * sps

plt.figure(figsize=(9,5))

for i in range(num_symbols - eye_symbols):

    start = i * sps
    segment = rx_mf[start:start + eye_samples]

    t = np.arange(eye_samples) / sps
    plt.plot(t, segment, color='blue', alpha=0.05)

plt.axvline(best_phase/sps, color='red', linestyle='--', linewidth=2, label="Best Sample")

plt.title("Eye Diagram")
plt.xlabel("Symbol Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# ------------------------------------------------
# Symbol Sampling
# ------------------------------------------------

samples = rx_mf[best_phase::sps][:num_symbols]


# ------------------------------------------------
# Adaptive Symbol Slicing for PAM4
# ------------------------------------------------

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
plt.legend()
plt.grid(True)


# ------------------------------------------------
# Constellation
# ------------------------------------------------

plt.figure(figsize=(6,6))

plt.plot(samples, np.zeros_like(samples), '.', alpha=0.3)

for lvl in levels_rx:
    plt.axvline(lvl, linestyle='--')

plt.title("PAM4 Constellation")
plt.xlabel("In-phase")
plt.ylabel("Quadrature (unused)")
plt.ylim(-0.5,0.5)
plt.grid(True)

# ------------------------------------------------
# Symbol Error Rate
# ------------------------------------------------



# Transmitted symbols (aligned to recovered samples)
tx_symbols = data[:len(sliced)]

# Scale recovered symbols to match tx amplitude
scale = np.std(sliced) / np.std(tx_symbols)
sliced_scaled = sliced / scale

# Compute minimum spacing between PAM levels
unique_levels = np.sort(np.unique(tx_symbols))
min_spacing = np.min(np.diff(unique_levels))

# Set tolerance as half the minimum spacing
tol = min_spacing / 2

# Compute error vector: 1 = error, 0 = correct
error_vector = (np.abs(sliced_scaled - tx_symbols) > tol).astype(int)

# Symbol error rate
ser = np.mean(error_vector)

print("Minimum PAM spacing:", min_spacing)
print("Tolerance used for error detection:", tol)
print("Symbol error rate:", ser)


plt.figure()
plt.plot(error_vector, '.', markersize=2)
plt.title("Symbol Error Vector")
plt.xlabel("Symbol Index")
plt.ylabel("Error (1=yes, 0=no)")
plt.show()

plt.show()
