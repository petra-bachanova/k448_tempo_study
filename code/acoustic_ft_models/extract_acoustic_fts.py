import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf


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

modspec_path = f"{project_path}/derivatives/acoustic_ft_analysis/modspec"

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
    stem = os.path.splitext(path)[0]  # strip ".wav" to match modspec filenames

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
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0], flux])
    flux_1s = average_over_1s(acoustic_ft=flux, frames_per_sec=frames_per_sec)

    acoustic_df = pd.DataFrame({
        "RMS_1s_bins": rms_1s,
        "Centroid_1s_bins": centroid_1s,
        "Flux_1s_bins": flux_1s,
    })

    # --- modulation spectrum, extracted separately (Matlab) by Michal Casey, loaded here ---
    modspec = np.loadtxt(f"{modspec_path}/{stem}_0_modspec.txt")
    modfreqs = np.loadtxt(f"{modspec_path}/{stem}_0_modfreqs.txt")

    modspec_cols = [f"modband_{round(f, 1)}" for f in modfreqs]
    modspec_df = pd.DataFrame(modspec, columns=modspec_cols)

    # compare lenghts between the wav file, acoustic df and modulation 
    # spectrum features to make sure they align
    info = sf.info(f"{wav_path}")
    duration_sec = info.frames / info.samplerate
    print(f"{key}: modspec={modspec.shape}, acoustic_fts={acoustic_df.shape}, wav_duration={duration_sec}")

    # modspec always ends up 1 row short (see modspec_1s_dB.m frame-count
    # off-by-one).Truncate acoustic_df to match, dropping its LAST row so
    # both dataframes stay aligned to the same real seconds of audio

    n_rows = min(acoustic_df.shape[0], modspec_df.shape[0])
    acoustic_df = acoustic_df.iloc[:n_rows].reset_index(drop=True)
    modspec_df = modspec_df.iloc[:n_rows].reset_index(drop=True)

    output_df = pd.concat([acoustic_df, modspec_df], axis=1)
    output_df.to_csv(f"{project_path}/derivatives/acoustic_ft_analysis/{key}.csv", index=False)