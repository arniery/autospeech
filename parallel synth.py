# this is the PARALLEL
# using 3 formants for this one

import numpy as np
import scipy.signal as sig
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import soundfile as sf


""" prevent clicks in files
def apply_fade(signal):
    window = sig.windows.hann(8192)
    fade_length = window.shape[0] // 2
    signal[:fade_length] *= window[:fade_length]
    signal[-fade_length:] *= window[fade_length:]
    return signal """

# function to create a second-order bandpass filter (formant resonator)
# uses iirpeak method from scipy.signal, ideal for creating resonators with a sharp peak at the desired formant frequency
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

# creating a lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def generate_impulse_train(F0, sample_rate, duration):
    """generates an impulse train."""
    N = int(sample_rate * duration)
    signal = np.zeros(N)
    phase = 0.0
    for n in range(N):
        phase += F0 / sample_rate
        if phase >= 1.0:
            signal[n] = 1.0
            phase -= 1.0
    return signal

# Convert dB values to linear gain
def dB_to_linear(dB_values):
    return 10 ** (np.array(dB_values) / 20)

# Compute FFT amplitude spectrum
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
formants_a = [730, 1100, 2540]  # formants for vowel "a"
formants_i = [270, 2290, 2890]  # formants for vowel "i"
formants_u = [325, 870, 2250]  # formants for vowel "u"

bandwidths = [100, 100, 100]  # Same BW for simplicity

# empirical target formant amplitude levels in dB (adjust as needed)
target_formant_levels = [-5, -10, -20]  # reference spectrum

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
    duration = 1 # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    F0 = 100              # fundamental frequency for the waveform
    
    # generates the input excitation
    # could instead replace this with impulse signal from andy's code
    excitation_signal = generate_impulse_train(F0, fs, duration)

    # initializes the output signal
    output_signal = excitation_signal

    # apply the fade to our signal
    # output_signal = apply_fade(output_signal)

    # applies each formant resonator (filter) in parallel
    formant_signals = []
    for f in formants:
        b, a = create_formant_filter(f, 100, fs)
        filtered_signal = sig.lfilter(b, a, excitation_signal)  # filtered signal for each formant
        # reverse polarity in time domain
        formant_signals.append(filtered_signal)  # Store filtered signals
        output_signal += filtered_signal  # sum the outputs in parallel

    # normalize formant amplitudes dynamically
    normalized_formants = normalize_formant_amplitudes(formant_signals, fs, target_formant_levels)

    # sum the adjusted formants to create the final vowel sound
    output_signal = np.sum(normalized_formants, axis=0)

    # apply the final lowpass filter
    # b_lp, a_lp = butter_lowpass(500, fs)
    # low_pass_out = sig.lfilter(b_lp, a_lp, output_signal) 

    # normalizes the output signal to avoid silence
    output_signal = output_signal / np.max(np.abs(output_signal))
    # low_pass_out /= np.max(np.abs(low_pass_out))

    # generate wav files !!!
    # writes to a .wav file or plays it directly
    sf.write(f"{vowel}_vowel.wav", output_signal, fs)

    # plot freq responses
    # choose a representative formant frequency (e.g., the first formant)
    formant_freq = formants[0]  # first formant for the vowel

    # get formant filter coefficients
    b_formant, a_formant = create_formant_filter(formant_freq, 100, fs)

    # compute frequency responses
    freq_formant, h_formant = sig.freqz(b_formant, a_formant, fs=fs)
    # freq_lp, h_lp = sig.freqz(b_lp, a_lp, fs=fs)

    # Create separate subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot the original impulse signal
    axs[0].plot(t, excitation_signal, linestyle="dashed", color="blue", alpha=0.7)
    axs[0].set_title("Original Impulse Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()
    axs[0].set_xlim(0, 0.05)  # zoom in to the first 0.05 seconds
    axs[0].set_ylim(-0.20, 0.20)  # zoom in 

    # Plot the output (formant filtered) signal
    axs[1].plot(t, output_signal, color="red", linewidth=2)
    axs[1].set_title("Formant Filtered Signal")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid()
    axs[1].set_xlim(0, 0.05)  # zoom in to the first 0.05 seconds
    axs[1].set_ylim(-0.15, 0.15)  # zoom in


    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    # formant resonator amplitude response
    """ ax[0, 0].plot(freq_formant, 20 * np.log10(np.maximum(abs(h_formant), 1e-5)), color='blue')
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
    plt.show() """
