# co. jan wilczek @wolfsound
# a formant is just a bandpass

import numpy as np
import scipy.signal as sig
import soundfile as sf

# prevent clicks in files
def apply_fade(signal):
    window = sig.windows.hann(8192)
    fade_length = window.shape[0] // 2
    signal[:fade_length] *= window[:fade_length]
    signal[-fade_length:] *= window[fade_length:]
    return signal

# if bandpass=false, then function will perform bandstop filtering
def bandstop_bandpass(signal, Q, center_frequency, fs, bandpass=False):
    # define filtering function
    filtered = np.zeros_like(signal)
    # define buffers of filter - 2 buffers per side
    x1 = 0 # input sample from prev iteration
    x2 = 0 # from 2 iterations back
    y1 = 0 # output sample ...
    y2 = 0
    # processing loop, computing coef of the filters

    for i in range(signal.shape[0]):
        BW = center_frequency[i] / Q

        tan = np.tan(np.pi * BW / fs)

        # all pass of the 2nd order
        c = (tan - 1) / (tan + 1)
        d = - np.cos(2 * np.pi * center_frequency[i] / fs)

        b = [-c, d * (1 - c), 1] # coef in num of transfer func
        a = [1, d * (1 - c), -c] # coef in denom ...

        x = signal[i] # input sample at this it

        y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2 # diff eq of filter

        # assigning output samples so they remain correct in next iterations
        # order of assignment matters!
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x 

        filtered[i] = y
    
    sign = -1 if bandpass else 1

    output = 0.5 * (signal + sign * filtered)

    return output

# createNote
# def impulse():



def main():
    fs = 44100 # sampling rate
    length_seconds = 6
    length_samples = fs * length_seconds # how many samples mean this 6 seconds of audio
    Q = 3 # relative bandwidth of our filters, fc/BW

    # generate range/array of center freq that we want to go over
    # sweep cannot be linear, has to be exponentially
    # higher the frequency, the larger the gaps btwn freq

    center_frequency = np.geomspace(700, 16000, length_samples)

    # how many should our array contain?
    # generate input signal
    # uniform generator ensures we have a signal in the rate from -1 to 1; not possible with gaussian noise

    noise = np.random.default_rng().uniform(-1, 1, (length_samples))

    bandstop_filtered_noise = bandstop_bandpass(noise, Q, center_frequency, fs)
    bandpass_filtered_noise = bandstop_bandpass(noise, Q, center_frequency, fs, bandpass=True)

    # make signals a little quieter
    amplitude = 0.5
    bandstop_filtered_noise *= amplitude
    bandpass_filtered_noise *= amplitude

    # apply fade in/fade out
    bandstop_filtered_noise = apply_fade(bandstop_filtered_noise)
    bandpass_filtered_noise = apply_fade(bandpass_filtered_noise)

    # write our signals to an audio file
    sf.write('bandstop_filtered_noise.flac', bandstop_filtered_noise, fs)
    sf.write('bandpass_filtered_noise.flac', bandpass_filtered_noise, fs)

if __name__ == '__main__':
    main()