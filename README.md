# Music3

This repository contains code for Music3, a study investigating the relationship between musical characteristics (e.g. tempo) and interictal epileptiform discharges (IEDs).

Only code is tracked. The full repository structure is as follows:

## Data directories (all in repository root)

- `/rawdata` - raw data in [BIDS](https://bids.neuroimaging.io/) format
  - Subject directories follow the structure `sub-<id>/ses-01/ieeg/` and contain:
    - `*_channels.tsv` - channel metadata
    - `*_ieeg.json` - session-level metadata
    - `*_ieeg.edf` - de-identified EEG data in European Data Format (EDF)
  - Root-level BIDS files: `dataset_description.json`, `participants.tsv`, `participants.json`, `sessions.tsv`, `sessions.json`

- `/_raw_from_Natus` - original EDF exports from the Natus clinical software, including subjects later excluded due to data quality issues. Retained as a convenience for BIDS conversion training.

- `/stimuli` - WAV files used in the study and corresponding musical structure label files

- `/derivatives` — preprocessing and analysis outputs; subdirectories follow BIDS structure 
  (e.g. `sub-<id>/ses-01/*`):
  - `preprocessing/`
  - `spike_detection/`
  - `time_markers/`
  - `calculated_spike_rates/`
  - `psd_analysis/`

## Code

See the `/code` directory, tracked in full on GitHub.

## Reports

`/reports` will contain HTML outputs from linear models fit in R.