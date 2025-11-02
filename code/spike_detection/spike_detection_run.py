from bids import BIDSLayout
from pathlib import Path
from time import time
import pandas as pd
import logging
import os
from spike_detection_functions import *


# SETUP
# --------------------------
# sub = 'sub-17'
# ses = 'ses-01'  # use a leading 0 i.e. "01" instead of "1"
sub = os.getenv("SUBJECT")
ses = os.getenv("SESSION")  # use a leading 0 i.e. "01" instead of "1"

project_path = "/dartfs-hpc/rc/lab/E/ECoG/music_study_tempo"
derivatives_path = f"{project_path}/derivatives"
spikedetection_sub_dir = f"{derivatives_path}/spike_detection/{sub}/{ses}"
os.makedirs(spikedetection_sub_dir, exist_ok=True)

# template-matching graphs
template_match_path = f"{derivatives_path}/spike_detection/graphs/template_matching"
os.makedirs(template_match_path, exist_ok=True)
# SPECT graphs
spectrogram_path = f"{derivatives_path}/spike_detection/graphs/SPECTS/IEDS"
os.makedirs(spectrogram_path, exist_ok=True)
# --------------------------


# LOAD EEG & INITIALISE LOGGING
# --------------------------
start_time = time.time()
log_path = f"{spikedetection_sub_dir}/{sub}_{ses}_spike_detection_log.log"
configure_logfile(path=log_path)

processed_sub_dir = f"{derivatives_path}/preprocessing/{sub}/{ses}"
processed_eeg, channels = load_processed_eeg(path=f"{processed_sub_dir}/{sub}_{ses}_preprocessed_ieeg.csv")
# --------------------------


# SPIKE DETECTION
# --------------------------
# Automatically detect spikes by template-matching
template_m_spikes = auto_detect(eegdata=processed_eeg, saveplotpath=template_match_path, subject_and_session=f"{sub}_{ses}", cross_corr_thresh=7)
# Generate and save spectrograms for each template match
spectimgs(eegdata=processed_eeg, spikedf=template_m_spikes, spectdir=spectrogram_path)
# Apply pretrained CNN to classify spikes
cnn_spikes = detect_with_cnn(project_dir=project_path, subject_and_session=f"{sub}_{ses}")
cnn_spikes.to_excel(f'{spikedetection_sub_dir}/{f"{sub}_{ses}"}_spike_detection_BEFORE_DEDUPE.xlsx', index=False)

# Clean and format the spike output
cnn_spikes_clean = clean_and_format_spike_data(
    df=cnn_spikes, subject_and_session=f"{sub}_{ses}", channels=channels, samp_freq=template_m_spikes.fs.values[0])
cnn_spikes_clean.to_excel(f'{spikedetection_sub_dir}/{sub}_{ses}_spike_detection.xlsx', index=False)
logging.info(f"Final number of spikes detected: {len(cnn_spikes_clean)}")

# Clear subject's IED images from the spectrogram directory
clear_spectrogram_dir(path=spectrogram_path, subject_and_session=f"{sub}_{ses}")

# Save the log file
print(f"Runtime duration: {time.time() - start_time:.2f} s (={(time.time() - start_time)/60:.2f} min)")
log_runtime_info(start_time=start_time, end_time=time.time())
# --------------------------
