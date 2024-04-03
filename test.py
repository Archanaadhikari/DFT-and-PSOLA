import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class PitchShifterGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pitch Shifter")

        # Create GUI elements
        self.label = ttk.Label(master, text="Enter Semitone Shift:")
        self.label.grid(row=0, column=0, pady=10)

        self.semitone_entry = ttk.Entry(master)
        self.semitone_entry.grid(row=0, column=1, pady=10)

        self.browse_button = ttk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, pady=10)

        self.selected_song_label = ttk.Label(master, text="Selected Song: None")
        self.selected_song_label.grid(row=1, column=0, columnspan=3, pady=10)

        self.shift_button = ttk.Button(master, text="Shift Pitch", command=self.shift_pitch)
        self.shift_button.grid(row=2, column=0, columnspan=3, pady=10)

        self.file_path = ""

    def browse_file(self):
        # Open a file dialog to select an audio file
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
        # Update the label with the selected song name
        self.selected_song_label.config(text=f"Selected Song: {os.path.basename(self.file_path)}")

    def shift_pitch(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a song before shifting pitch.")
            return

        try:
            semitone_shift = float(self.semitone_entry.get())
            if semitone_shift < -12 or semitone_shift > 12:
                raise ValueError("Semitone shift must be between -12 and 12.")

            # Load audio file dynamically based on the selected file path
            orig_signal, fs = librosa.load(self.file_path, sr=44100)

            new_signal = shift_pitch(orig_signal, fs, semitone_shift)

            # Choose the path to save the pitch-shifted file
            save_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 files", "*.mp3")])

            if save_path:
                plt.style.use('ggplot')
                plt.subplot(2, 1, 1)
                plt.plot(orig_signal[:-1], label="Original Signal", color='blue')
                plt.legend(loc="upper right")
                plt.subplot(2, 1, 2)
                plt.plot(new_signal[:-1], label="Pitch Shifted Signal", color='red')
                plt.legend(loc="upper right")
                plt.show()

                # Write to disk using soundfile with the chosen file path
                sf.write(save_path, new_signal, fs)
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", str(e))

def dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)

def idft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, signal) / N

def shift_pitch(signal, fs, semitone_shift):
    N = len(signal)
    # Calculate pitch shift factor
    f_ratio = 2 ** (semitone_shift / 12)

    # Find peaks and apply PSOLA
    peaks = find_peaks(signal, fs, N)
    new_signal = psola(signal, peaks, f_ratio)
    return new_signal

def find_peaks(signal, fs, N, max_hz=950, min_hz=75, analysis_win_ms=40, max_change=1.005, min_change=0.995):
    min_period = fs // max_hz
    max_period = fs // min_hz

    # compute pitch periodicity
    sequence = int(analysis_win_ms / 1000 * fs)  # analysis sequence length in samples
    periods = compute_periods_per_sequence(signal, sequence, min_period, max_period, N)

    # simple hack to avoid octave error: assume that the pitch should not vary much, restrict range
    mean_period = np.mean(periods)
    max_period = int(mean_period * 1.1)
    min_period = int(mean_period * 0.9)
    periods = compute_periods_per_sequence(signal, sequence, min_period, max_period, N)

    # find the peaks
    peaks = [np.argmax(signal[:int(periods[0]*1.1)])]
    while True:
        prev = peaks[-1]
        idx = prev // sequence  # current autocorrelation analysis window
        if prev + int(periods[idx] * max_change) >= N:
            break
        # find maximum near expected location
        peaks.append(prev + int(periods[idx] * min_change) +
                     np.argmax(signal[prev + int(periods[idx] * min_change): prev + int(periods[idx] * max_change)]))
    return np.array(peaks)

def compute_periods_per_sequence(signal, sequence, min_period, max_period, N):
    offset = 0  # current sample offset
    periods = []  # period length of each analysis sequence

    while offset < N:
        fourier = fft(signal[offset: offset + sequence])
        fourier[0] = 0  # remove DC component
        autoc = ifft(fourier * np.conj(fourier)).real
        autoc_peak = min_period + np.argmax(autoc[min_period: max_period])
        periods.append(autoc_peak)
        offset += sequence
    return periods

def psola(signal, peaks, f_ratio):
    N = len(signal)
    # Interpolate
    new_signal = np.zeros(N)
    new_peaks_ref = np.linspace(0, len(peaks) - 1, int(len(peaks) * f_ratio))
    new_peaks = np.zeros(len(new_peaks_ref)).astype(int)

    for i in range(len(new_peaks)):
        weight = new_peaks_ref[i] % 1
        left = np.floor(new_peaks_ref[i]).astype(int)
        right = np.ceil(new_peaks_ref[i]).astype(int)
        new_peaks[i] = int(peaks[left] * (1 - weight) + peaks[right] * weight)

    for j in range(len(new_peaks)):
        # find the corresponding old peak index
        i = np.argmin(np.abs(peaks - new_peaks[j]))
        # get the distances to adjacent peaks
        P1 = [new_peaks[j] if j == 0 else new_peaks[j] - new_peaks[j-1],
              N - 1 - new_peaks[j] if j == len(new_peaks) - 1 else new_peaks[j+1] - new_peaks[j]]
        # edge case truncation
        if peaks[i] - P1[0] < 0:
            P1[0] = peaks[i]
        if peaks[i] + P1[1] > N - 1:
            P1[1] = N - 1 - peaks[i]
        # linear OLA window
        window = list(np.linspace(0, 1, P1[0] + 1)[1:]) + list(np.linspace(1, 0, P1[1] + 1)[1:])
        # center window from the original signal at the new peak
        new_signal[new_peaks[j] - P1[0]: new_peaks[j] + P1[1]] += window * signal[peaks[i] - P1[0]: peaks[i] + P1[1]]
    return new_signal

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchShifterGUI(root)
    root.mainloop()