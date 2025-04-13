# this is the SERIES/CASCADE

import numpy as np
import scipy.signal as sig
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import soundfile as sf

# gain incl spectral shaping filter
# spectral shaping doesn't change overall gain of resonator (relative measure, shifting up or down)


""" prevent clicks in files
def apply_fade(signal):
    fade_length = min(len(signal) // 2, 4096)  # Ensure it fits within the signal
    window = sig.windows.hann(2 * fade_length)
    signal[:fade_length] *= window[:fade_length]
    signal[-fade_length:] *= window[fade_length:]
    return signal """

# function to create a second-order bandpass filter (formant resonator)
# uses iirpeak method from scipy.signal, ideal for creating RESONATORS with a sharp peak at the desired formant frequency
# bw of 100 hz for each formant
def create_formant_filter(frequency, bandwidth, fs):
    # normalize to the nyquist frequency
    w0 = frequency / (fs / 2)
    bw = bandwidth / (fs / 2)
    
    # Q factor
    Q = w0 / bw
    
    # create second-order bandpass filter
    b, a = sig.iirpeak(w0, Q)  # IIR filter for bandpass
    return b, a

def generate_impulse_train(F0, fs, duration):
    # generates an impulse train
    N = int(fs * duration)
    impulse = np.zeros(N)
    phase = 0.0
    for n in range(N):
        phase += F0 / fs
        if phase >= 1.0:
            impulse[n] = 1.0
            phase -= 1.0
    return impulse

# creating a lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# applying the lowpass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# convert dB values to linear gain
def dB_to_linear(dB_values):
    return 10 ** (np.array(dB_values) / 20)

# compute FFT amplitude spectrum
def compute_spectrum(signal, fs):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    return freqs, spectrum

# normalize formant amplitudes dynamically
def normalize_formant_amplitudes(formant_signals, fs, target_dB_levels):
    target_gains = dB_to_linear(target_dB_levels)
    
    # Compute FFT-based amplitude for each formant
    formant_magnitudes = []
    for sig in formant_signals:
        freqs, spectrum = compute_spectrum(sig, fs)
        formant_magnitudes.append(np.max(spectrum))  # Peak magnitude

    # Convert formant amps to dB
    formant_dB_levels = 20 * np.log10(np.array(formant_magnitudes) + 1e-8)
    
    # Compute gain correction factor
    gain_factors = target_gains / dB_to_linear(formant_dB_levels)
    
    # Apply gain correction
    adjusted_formants = [sig * gain for sig, gain in zip(formant_signals, gain_factors)]
    
    return adjusted_formants

# formant frequencies and bandwidths for different vowels
formants_a = [730, 1100, 2540, 3400, 4000]  # formants for vowel "a"
formants_i = [280, 2250, 2890, 3400, 4000]  # formants for vowel "i"
formants_u = [310, 870, 2250, 3300, 4200]  # formants for vowel "u"

bandwidths = [100, 100, 100, 100, 100]  # Same BW for simplicity

# empirical target formant amplitude levels in dB (adjust as needed)
target_formant_levels = [-5, -10, -20, -20, -20]  # reference spectrum

# choose the vowel sound to synthesize
vowel = 'a'  # Choose from 'a', 'i', 'u'

if vowel == 'a':
    formants = formants_a
elif vowel == 'i':
    formants = formants_i
else:
    formants = formants_u

if __name__ == '__main__':
    # sampling rate and duration
    fs = 16000   # hz
    duration = 1.0 # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    F0 = 100              # fundamental frequency for the waveform
    
    # generate an impulse signal
    impulse = generate_impulse_train(F0, fs, duration)

    # initializes the output signal
    output_signal = impulse

    # apply the fade to our signal
    # output_signal = apply_fade(output_signal)

    # Apply each formant filter in cascade
    formant_signals = []
    for f in formants:
        b, a = create_formant_filter(f, 100, fs)  # get filter coefficients
        output_signal = sig.lfilter(b, a, output_signal)  # apply to signal
        formant_signals.append(output_signal.copy())  # Store intermediate filtered signal

    # normalize formant amplitudes dynamically
    normalized_formants = normalize_formant_amplitudes(formant_signals, fs, target_formant_levels)

    # sum the adjusted formants to create the final vowel sound
    output_signal = np.sum(normalized_formants, axis=0)

    # apply the final lowpass filter
    b_lp, a_lp = butter_lowpass(500, fs)
    low_pass_out = sig.lfilter(b_lp, a_lp, output_signal)

    # normalize the output signals to avoid silence
    output_signal /= np.max(np.abs(output_signal))
    low_pass_out /= np.max(np.abs(low_pass_out))

    # Ensure the signal lengths match
    assert len(t) == len(impulse) == len(low_pass_out), "mismatch in signal lengths!"

    # Save the sound file
    sf.write(f"{vowel}_vowel.wav", low_pass_out, fs)

    # == Plot frequency response of formants before and after amplitude correction ===
    # Get the first formant filter coefficients (before and after amplitude correction)
    b_raw, a_raw = create_formant_filter(formants[0], 100, fs)

    # Frequency response of the original filter
    freqs, h_raw = sig.freqz(b_raw, a_raw, fs=fs)

    # Plot the filter response
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, 20 * np.log10(abs(h_raw) + 1e-8), label="Original Formant Filter", linestyle='dashed')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title("Frequency Response of First Formant Filter")
    plt.legend()
    plt.grid()
    plt.show()

    # choose a representative formant frequency (e.g., the first formant)
    formant_freq = formants[0]  # first formant for the vowel

    # get formant filter coefficients
    b_formant, a_formant = create_formant_filter(formant_freq, 100, fs)

    # get lowpass filter coefficients
    b_lp, a_lp = butter_lowpass(500, fs)

    # compute frequency responses
    freq_formant, h_formant = sig.freqz(b_formant, a_formant, fs=fs)
    freq_lp, h_lp = sig.freqz(b_lp, a_lp, fs=fs)

    # plot the individual transfer functions
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    # formant resonator amplitude response
    ax[0, 0].plot(freq_formant, 20 * np.log10(np.maximum(abs(h_formant), 1e-5)), color='blue')
    ax[0, 0].set_title("formant resonator amp/phase response")
    ax[0, 0].set_ylabel("amplitude [dB]", color='blue')
    ax[0, 0].set_xlim([100, 5000])
    ax[0, 0].set_ylim([-50, 20])
    ax[0, 0].grid(True)

    # formant resonator phase response
    ax[1, 0].plot(freq_formant, np.unwrap(np.angle(h_formant)) * 180 / np.pi, color='green')
    ax[1, 0].set_ylabel("phase [degrees]", color='green')
    ax[1, 0].set_xlabel("frequency [Hz]")
    ax[1, 0].set_xlim([100, 5000])
    ax[1, 0].set_ylim([-360, 360])
    ax[1, 0].grid(True)

    # lowpass filter amplitude response
    ax[0, 1].plot(freq_lp, 20 * np.log10(np.maximum(abs(h_lp), 1e-5)), color='red')
    ax[0, 1].set_title("lowpass filter amp/phase response")
    ax[0, 1].set_ylabel("amplitude [dB]", color='red')
    ax[0, 1].set_xlim([100, 5000])
    ax[0, 1].set_ylim([-50, 20])
    ax[0, 1].grid(True)

    # lowpass filter phase response
    ax[1, 1].plot(freq_lp, np.unwrap(np.angle(h_lp)) * 180 / np.pi, color='purple')
    ax[1, 1].set_ylabel("phase [degrees]", color='purple')
    ax[1, 1].set_xlabel("frequency [Hz]")
    ax[1, 1].set_xlim([100, 5000])
    ax[1, 1].set_ylim([-360, 360])
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # add plots for: [time domain]
    # original signal
    # formant resonator output
    # low pass filter output

    # Create separate subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot the original impulse signal
    axs[0].plot(t, impulse, linestyle="dashed", color="blue", alpha=0.7)
    axs[0].set_title("Original Impulse Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()
    axs[0].set_xlim(0, 0.05)  # zoom in to the first 0.05 seconds
    axs[0].set_ylim(-0.20, 0.20)  # zoom in 

    # Plot the filtered signal
    axs[1].plot(t, low_pass_out, color="red", linewidth=2)
    axs[1].set_title("Low-Pass Filtered Signal")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid()
    axs[1].set_xlim(0, 0.05)  # zoom in to the first 0.05 seconds
    axs[1].set_ylim(-0.15, 0.15)  # zoom in


    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
