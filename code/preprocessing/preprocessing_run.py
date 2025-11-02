from preprocessing_functions import *
import time
import os

# SETUP & PATHS
# --------------------------
# sub = 'sub-17'
# ses = 'ses-01'  # use a leading 0 i.e. "01" instead of "1"
sub = os.getenv("SUBJECT")
ses = os.getenv("SESSION")  # use a leading 0 i.e. "01" instead of "1"

project_path = "/dartfs-hpc/rc/lab/E/ECoG/music_study_tempo"
bids_sub_dir = f"{project_path}/rawdata/{sub}/{ses}/ieeg"
deriv_sub_dir = f"{project_path}/derivatives/preprocessing/{sub}/{ses}"
os.makedirs(deriv_sub_dir, exist_ok=True)
# --------------------------


# LOAD EEG & INITIALISE LOGGING
# --------------------------
start_time = time.time()
log_path = f"{deriv_sub_dir}/{sub}_{ses}_preprocessing_log.log"
configure_logfile(path=log_path)

raw_path = f"{bids_sub_dir}/{sub}_{ses}_task-k448tempo_ieeg.edf"
raw = load_raw_ieeg(path=raw_path)
# --------------------------


# PREPROCESSING & QC
# --------------------------
filtered = apply_filters(df=raw)
df_manual_clean, manual_bads = manually_reject_bads(df=filtered, channels_path=f"{bids_sub_dir}/{sub}_{ses}_task-k448tempo_channels.tsv")
df_manual_clean_ds = downsample_and_log(df=df_manual_clean, freq=200)
df_auto_clean, auto_bads = automatically_reject_bads(df=df_manual_clean_ds)

# Log bads, plot a PSD of good channels
log_rejected_channels(raw=raw, manual_bads=manual_bads, auto_bads=auto_bads, df_auto_clean=df_auto_clean)
plot_PSD(df=df_auto_clean, title=f"{sub}_{ses} PSD: Preprocessed", savepath=f"{deriv_sub_dir}/{sub}_{ses}_processed_PSD_plot.png")

# Save & log runtime
save_preprocessed_data(preprocessed_data=df_auto_clean, path=f"{deriv_sub_dir}/{sub}_{ses}_preprocessed_ieeg.csv")
log_runtime_info(start_time=start_time, end_time=time.time())
# --------------------------
