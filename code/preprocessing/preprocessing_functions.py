# directory setup, data wrangling
import re 
import logging
import time

# data analysis
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# mne dependencies
import mne
from mne import io, read_proj
from mne.datasets import sample
from mne.channels import read_custom_montage
#from mne.time_frequency import psd_multitaper
#from mne import io, read_proj, read_custom_montage
from mne.time_frequency import psd_array_multitaper

import warnings
warnings.filterwarnings('ignore')


def configure_logfile(path):
    """Configure path and settings for logfile.

    Args:
        path (string): save path for the log file
    """
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_raw_ieeg(path):
    """Load EEG data from file. Log file info."""
    raw = mne.io.read_raw_edf(path, preload=True)
    logging.info(f"Path: {path}")
    logging.info(f"Duration of EEG data: {raw.times[-1]/60:.2f} min")

    return raw


def apply_filters(df):
    """Apply high, low-pass and notch filters. Log their values.

    Args:
        df (mne.io.Raw): raw ieeg data

    Returns:
        mne.io.Raw: filtered ieeg data
    """
    # Rereference data (average rereference)
    df.set_eeg_reference('average', projection=True)

    # Apply notch filter of 60 Hz and harmonics; low and high pass filters
    notch_filter = 60
    low_pass = 1
    high_pass = 250

    notch_frequencies = np.arange(notch_filter, df.info['sfreq'] // 2, notch_filter)
    filtered = df.copy().notch_filter(notch_frequencies, filter_length='auto', phase='zero')
    filtered = filtered.filter(l_freq=low_pass, h_freq=high_pass, l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length='auto', phase='zero')

    # Log info
    logging.info(f"Notch filter: {notch_filter}")
    logging.info(f"Low pass filter: {low_pass}")
    logging.info(f"High pass filter: {high_pass}")

    return filtered


def manually_reject_bads(df, channels_path):
    """Rejects channels of type MISC (such as scalp electrodes, DC channels etc) and manually marked electrodes

    Args:
       df (mne.io.Raw): Input EEG data.

    Returns:
        tuple:
            - clean_df (mne.io.Raw): EEG data with bad channels removed.
            - rejected_picks (list): manually rejected channels.
    """

    # load channels.tsv file
    channels_df = pd.read_csv(channels_path, sep="\t", dtype={"name": "string"})

    # Channels to exclude
    misc_channels = set(channels_df.loc[channels_df["type"] == "MISC", "name"])
    bads_contacts = set(channels_df.loc[channels_df["status"] == "bad", "name"])

    # Keep only channels that are not in misc or bads
    picks = [ch for ch in df.ch_names if ch not in misc_channels and ch not in bads_contacts]
    rejected_picks = [ch for ch in df.ch_names if ch not in picks]
    clean_df = df.copy().pick(picks)

    return clean_df, rejected_picks

def downsample_and_log(df, freq):
    # Downsampling
    logging.info(f"Original sampling rate: {df.info['sfreq']} Hz")
    downsampled = df.copy().resample(freq, npad='auto')
    logging.info(f"Preprocessed sampling rate: {downsampled.info['sfreq']} Hz")

    return downsampled


def automatically_reject_bads(df):
    """
    Downsample EEG data to 200 Hz and automatically reject bad channels.
    Rejection uses an adaptive variance method:
    1. Automatically detects bads by iteratively flagging variance outliers (> 3 SD from the good channel mean in that iteration).
    2. In each subsequent iteration, recalculates distribution without bads to detect more subtle outliers.

    Args:
        df (mne.io.Raw): Input EEG data.

    Returns:
        tuple:
            - clean_df (mne.io.Raw): EEG data with bad channels removed.
            - rejected_picks (list): automatically rejected channels.
    """
  
    def check_bads_adaptive(df, picks, fun=np.var, thresh=3, max_iter=100):
        ch_x = fun(df.get_data(picks=picks), axis=-1)
        my_mask = np.zeros(len(ch_x), dtype=bool)
        for i_iter in range(int(max_iter)):
            ch_x = np.ma.masked_array(ch_x, mask=my_mask)  # array of channel variances
            this_z = np.abs(stats.zscore(ch_x))
            local_bad = this_z > thresh
            if not np.any(local_bad):
                break
            my_mask |= local_bad
            print(f'iteration {i_iter} : total bads: {np.sum(my_mask)}')
        bads = [df.ch_names[i] for i in np.where(my_mask)[0]]
        return bads

    endIndex = next((i for i, name in enumerate(df.ch_names) if re.match(r'C\d{3}', name)), len(df.ch_names))
    rejected_picks = df.ch_names[endIndex:]
    rejected_picks.extend(check_bads_adaptive(df, picks=range(endIndex), thresh=3))
    df.info['bads'] = rejected_picks

    # Pick only good channels
    clean_df = df.copy().pick_types(eeg=True, meg=False, exclude='bads')

    return clean_df, rejected_picks


def log_rejected_channels(raw, manual_bads, auto_bads, df_auto_clean):
    logging.info(f"All channels, count: {len(raw.ch_names)}")
    logging.info(f"Manually rejected channels, count: {len(manual_bads)}")
    logging.info(f"Manually rejected channels, names: {manual_bads}")
    logging.info(f"Automatically rejected channels, count: {len(auto_bads)}")
    logging.info(f"Automatically rejected channels, names: {auto_bads}")
    logging.info(f"Remaining channels, count: {len(df_auto_clean.ch_names)}")
    logging.info(f"Maths check: Raw - rejected (manual + automatic) = {len(raw.ch_names) - len(manual_bads) - len(auto_bads)}")


def plot_PSD(df, title, savepath):
    """
    Plot the power spectral density (between 1 to 40 Hz) of all EEG channels.

    Args:
        df (mne.io.Raw): Input EEG data.
        title (str): Title of the plot.
        savepath (str): File path to save the figure.

    Returns:
        None: The plot is saved to the specified path.
    """
    psd = df.compute_psd(fmin=1, fmax=80, n_fft=2048)
    frequencies = psd.freqs
    psd_values = psd_values = 10 * np.log10(psd.get_data())

    plt.figure(figsize=(8, 4))
    picks = df.ch_names
    colors = plt.cm.viridis(np.linspace(0, 1, len(picks))) # type: ignore
    for i, color in enumerate(colors):
        plt.plot(frequencies, psd_values[i], color=color, linewidth=1)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", dpi=300)
    plt.close()


def log_runtime_info(start_time, end_time):
    """Log preprocessing information."""
    logging.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logging.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logging.info(f"Runtime duration: {end_time - start_time:.2f} s (={(end_time - start_time)/60:.2f} min)")


def save_preprocessed_data(preprocessed_data, path):
    """Save the preprocessed iEEG data."""
    header = ','.join(preprocessed_data.ch_names)
    np.savetxt(fname=path, X=preprocessed_data.get_data().T, delimiter=',', header=header, comments="")
