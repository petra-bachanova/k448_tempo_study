import librosa
import numpy as np
import pandas as pd


def average_over_1s(acoustic_ft, frames_per_sec):
    n_seconds = int(len(acoustic_ft) / frames_per_sec)
    ft_1s = np.array([
        acoustic_ft[int(i * frames_per_sec) : int((i+1) * frames_per_sec)].mean()
        for i in range(n_seconds)
    ])
    return ft_1s


project_path = "/dartfs-hpc/rc/lab/E/ECoG/k448_tempo_study"
hop_length = 512
sr = 22050
n_fft = 2048
frames_per_sec = sr / hop_length
music = ""

music_dic = {
    "COLDPLAY": "Coldplay_Clocks_132bpm.wav",
    "BACH": "JSBach_PreludeNo2_Cminor_137bpm.wav",
    "WAGNER": "Wagner_LohengrinWWV75-PreludeToActI.wav",
    "K448_MONO": "K448-136bpm-Monotonic.wav",
    "K448_106BPM": "K448-Audio-106bpm.wav",
    "K448_136BPM": "K448-Audio-136bpm.wav",
    "K448_166BPM": "K448-Audio-166bpm.wav",
}

for key, path in music_dic.items():
    wav_path = f"{project_path}/stimuli/wav files/{path}"

    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # RMS energy (proxy for loudness/intensity)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    rms_1s = average_over_1s(acoustic_ft=rms, frames_per_sec=frames_per_sec)

    # Spectral centroid ("brightness" / dominant frequency center)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    centroid_1s = average_over_1s(acoustic_ft=centroid, frames_per_sec=frames_per_sec)

    # Replace first and last 1s bin with local median of neighbouring bins
    centroid_1s[0] = np.median(centroid_1s[1:5])
    centroid_1s[1] = np.median(centroid_1s[1:5])
    centroid_1s[-1] = np.median(centroid_1s[-5:-1])


    # Spectral flux — frame-to-frame spectral change (onset proxy)
    # Computed manually as L2 norm of difference between consecutive magnitude spectra
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0], flux])  # pad first frame with 0
    flux_1s = average_over_1s(acoustic_ft=flux, frames_per_sec=frames_per_sec)

    output_df = pd.DataFrame({"RMS_1s_bins": rms_1s, "Centroid_1s_bins": centroid_1s, "Flux_1s_bins": flux_1s})
    output_df.to_csv(f"{project_path}/derivatives/acoustic_ft_analysis/{key}.csv")
