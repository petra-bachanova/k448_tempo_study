from pathlib import Path
import mne
from mne_bids import BIDSPath, write_raw_bids

# List of subjects/sessions
subjects = [
    "sub-17_ses-01", "sub-18_ses-01", "sub-18_ses-02", "sub-19_ses-01",
    "sub-20_ses-01", "sub-21_ses-01", "sub-22_ses-01", "sub-23_ses-01",
    "sub-24_ses-01", "sub-26_ses-01", "sub-27_ses-01", "sub-27_ses-02",
    "sub-27_ses-03", "sub-28_ses-01", "sub-30_ses-01", "sub-31_ses-01",
    "sub-32_ses-01", "sub-33_ses-01"
]

# Paths
raw_backup_root = Path("/dartfs-hpc/rc/lab/E/ECoG/music_study_tempo/rawdata_original_Natus")
raw_root = Path("/dartfs-hpc/rc/lab/E/ECoG/music_study_tempo/rawdata")  # new BIDS path

# Define patterns for non-iEEG channels
not_ieeg_patterns = ['DC', 'TRIG', 'OSAT', 'PR', 'Pleth', 'FP', 'Fp', 'F', 'C', 'P', 'O', 'T', 'EKG']

for subj_ses in subjects:
    subj, ses = subj_ses.split("_")
    
    # Original file
    raw_file = raw_backup_root / f"{subj_ses}.edf"
    
    if not raw_file.exists():
        print(f"File not found: {raw_file}")
        continue
    
    # Load raw EEG
    raw = mne.io.read_raw_edf(raw_file, preload=False)
    
    # Build channel type mapping
    channel_type_updates = {}
    for ch in raw.ch_names:
        if any(ch.startswith(pattern) for pattern in not_ieeg_patterns):
            # Assign non-iEEG channels to 'misc'
            channel_type_updates[ch] = 'misc'
        else:
            # All other channels are ieeg
            channel_type_updates[ch] = 'seeg'
    
    # Apply channel type updates
    raw.set_channel_types(channel_type_updates)
    
    # Create BIDSPath for writing
    bids_path = BIDSPath(
        subject=subj.replace("sub-", ""),
        session=ses.replace("ses-", ""),
        task="k448tempo",
        suffix="ieeg",
        extension=".edf",
        datatype="ieeg",
        root=raw_root
    )
    
    # Write to BIDS iEEG folder
    write_raw_bids(raw, bids_path=bids_path, overwrite=True)
    print(f"Written iEEG data for {subj_ses} to {bids_path.fpath}")
