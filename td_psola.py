import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks

def shift_pitch(signal, fs, f_ratio):
    peaks = find_peaks(signal)[0]  
    new_signal = psola(signal, peaks, f_ratio, len(signal))
    return new_signal

def psola(signal, peaks, f_ratio, N):
    new_signal = np.zeros(N)
    new_peaks_ref = np.linspace(0, len(peaks) - 1, int(len(peaks) * f_ratio))  # Convert to integer
    new_peaks = np.zeros(len(new_peaks_ref)).astype(int)
    for i in range(len(new_peaks)):
        weight = new_peaks_ref[i] % 1
        left = np.floor(new_peaks_ref[i]).astype(int)
        right = np.ceil(new_peaks_ref[i]).astype(int)
        new_peaks[i] = int(peaks[left] * (1 - weight) + peaks[right] * weight)
    for j in range(len(new_peaks)):
        i = np.argmin(np.abs(peaks - new_peaks[j]))
        P1 = [new_peaks[j] if j == 0 else new_peaks[j] - new_peaks[j-1],
              N - 1 - new_peaks[j] if j == len(new_peaks) - 1 else new_peaks[j+1] - new_peaks[j]]
        if peaks[i] - P1[0] < 0:
            P1[0] = peaks[i]
        if peaks[i] + P1[1] > N - 1:
            P1[1] = N - 1 - peaks[i]
        window = list(np.linspace(0, 1, P1[0] + 1)[1:]) + list(np.linspace(1, 0, P1[1] + 1)[1:])
        new_signal[new_peaks[j] - P1[0]: new_peaks[j] + P1[1]] += window * signal[peaks[i] - P1[0]: peaks[i] + P1[1]]
    return new_signal

if __name__ == "__main__":
    orig_signal, fs = librosa.load("C:/Users/PocoLoco24/Desktop/finalRiffifyCode/Audio/Original Audio/female_scale.wav", sr=44100)
    N = len(orig_signal)
    f_ratio = 2 ** (-2 / 12)
    new_signal = shift_pitch(orig_signal, fs, f_ratio)
    plt.style.use('ggplot')
    plt.plot(orig_signal[:-1])
    plt.show()
    plt.plot(new_signal[:-1])
    plt.show()

    # Use librosa directly without the output attribute
    librosa.write_wav("female_scale_transposed_{:01.2f}.wav".format(f_ratio), new_signal, fs)

