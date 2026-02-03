import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import os
import ast


# TODO: Add raster plot data formatting function

def load_detected_spikes(sub_spikes_path, sampling_rate, channels_path, SOZ_analysis):
    spikes_df = pd.read_excel(sub_spikes_path)
    spikes_df['time in seconds'] = spikes_df['spike_start']/sampling_rate
    spikes_df['channels_names'] = spikes_df['channels_names'].apply(ast.literal_eval)

    if SOZ_analysis:
        # Add SOZ (yes/no) nSOZ (yes/no) columns to the detected spikes
        channels_df = pd.read_csv(channels_path, sep="\t", dtype={"name": "string"})
        soz_contacts = list(channels_df[channels_df["soz_contact"] == "yes"]["name"].values)
        spikes_df["spike_in_SOZ"] = spikes_df["channels_names"].apply(lambda x: "yes" if any(item in x for item in soz_contacts) else "no")
        spikes_df["spike_in_nSOZ"] = spikes_df["channels_names"].apply(lambda x: "yes" if set(x) - set(soz_contacts) else "no")

    return spikes_df


def get_times_from_event(time_marker_df, event):
    """
    Get start time, end time and duration of an event from the time marker dataframe.
    """
    event_row = time_marker_df[time_marker_df['event'] == event]
    event_duration = event_row['duration_s'].values[0]
    event_start = event_row['start_time'].values[0]
    event_end = event_row['end_time'].values[0]

    return event_start, event_end, event_duration


def subset_event_spikes(spike_df, event_start, event_end):
    """
    Subset spike dataframe based on start and end times.
    """
    spikes_times = spike_df['time in seconds']
    # Find the index of the first spike greater than the start time
    idx_start = (spikes_times - event_start).loc[spikes_times > event_start].idxmin()
    # Find the index of the last spike less than the end time
    idx_end = (spikes_times - event_end).loc[spikes_times < event_end].idxmax()
    # Subset based on indices
    subset = spike_df.iloc[idx_start:idx_end+1, :].copy()

    return subset


def compute_binned_spike_rates(spike_df, start_time, end_time, bin_len, max_duration):
    """
    Compute spike rates in consecutive bins within a time window.

    Args:
        spike_df (pd.DataFrame): DataFrame containing spikes with 'time in seconds' column.
        start_time (float): Start time of the period.
        end_time (float): End time of the period.
        bin_len (float): Length of each time bin in seconds.
        max_duration (float | None): Optional cutoff duration from start_time.

    Returns:
        dict[str, float]: Dictionary mapping time bins ('0-15s', '15-30s', etc.) to spike rates.
    """
    spike_rates = {}
    spike_counts = {}

    total_duration = end_time - start_time
    if max_duration:
        total_duration = min(total_duration, max_duration)

    for i in range(0, int(total_duration), bin_len):
        bin_start = start_time + i
        bin_end = min(bin_start + bin_len, end_time)

        mask = (spike_df['time in seconds'] >= bin_start) & (spike_df['time in seconds'] < bin_end)
        n_spikes = mask.sum()
        rate = n_spikes / bin_len

        spike_rates[f"{int(i)}-{int(i + bin_len)}s"] = round(rate, 3)
        spike_counts[f"{int(i)}-{int(i + bin_len)}s"] = n_spikes

    return spike_rates, spike_counts


def bootstrap_spike_rates(spike_rates, n_iterations):
    bootstrap_means = []
    for _ in range(n_iterations):
        # Sample with replacement
        bootstrap_sample = np.random.choice(spike_rates, size=len(spike_rates), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)
    return bootstrap_means


def plot_bootstrapped_hist(spike_rates, savepath):
    spike_rates_li = list(spike_rates.values())
    bootstrapped_spike_rates_li = bootstrap_spike_rates(spike_rates=spike_rates_li, n_iterations=10000)
    plt.hist(bootstrapped_spike_rates_li, bins=50)  # density=False would make counts
    plt.ylabel('Count')
    plt.xlabel('Boostrapped baseline spike rates')
    plt.savefig(savepath, dpi=300)
    plt.close()


# Load metadata
project_path = "/dartfs-hpc/rc/lab/E/ECoG/music_study_tempo"
participants_path = f"{project_path}/rawdata/participants.tsv"
participants = pd.read_csv(participants_path, sep="\t")
sessions = pd.read_csv(f"{project_path}/rawdata/sessions.tsv", sep="\t")
sessions = sessions[sessions["exclude"] == 0]
# merge metadata
meta = sessions.merge(participants, on="participant_id", how="left")


for _, row in meta.iterrows():
    sub, ses = row["participant_id"], row["session"]
    sr_sub_path = f"{project_path}/derivatives/calculated_spike_rates/{sub}/{ses}"
    os.makedirs(sr_sub_path, exist_ok=True)
    
    # Define which contact types to analyze
    include_soz = row["include_soz_analysis"] == 1
    # TODO: ONCE CHANNELS.TSV ARE FINISHED FOR SUB-19 AND SUB-20, REMOVE THIS IF LOOP
    if sub in ("sub-19", "sub-20"):
        include_soz = False
    contact_types = ["all", "SOZ", "nSOZ"] if include_soz else ["all"]

    # Load detected spikes & add a column denoting whether each spike happened in SOZ/nSOZ
    spikes_sub_path = f'{project_path}/derivatives/spike_detection/{sub}/{ses}/{sub}_{ses}_spike_detection.xlsx'
    channels_path = f'{project_path}/rawdata/{sub}/{ses}/ieeg/{sub}_{ses}_task-k448tempo_channels.tsv'
    spikes = load_detected_spikes(sub_spikes_path=spikes_sub_path, sampling_rate=200, channels_path=channels_path, SOZ_analysis=include_soz)
    # Read in time markers
    sub_time_marker_path = f'{project_path}/derivatives/time_markers/{sub}/{ses}/{sub}_{ses}_processed_time_markers.csv'
    tm = pd.read_csv(sub_time_marker_path)

    sr_merged = pd.DataFrame()
    sc_merged = pd.DataFrame()
    events = ["pre-exp baseline", 'COLDPLAY', 'BACH', 'K448_MONO', 'K448_106BPM', 'K448_136BPM', 'K448_166BPM', 'WAGNER']
    for event in events:
        # Skip events the participant did not listen to
        if event not in tm["event"].unique():
            continue
        
        # Get event times
        event_start, event_end, duration = get_times_from_event(time_marker_df=tm, event=event)
        spikes_event = subset_event_spikes(spike_df=spikes, event_start=event_start, event_end=event_end)

        # Calculate spike rates per events per contact type (i.e. all, SOZ, nSOZ)
        sr_event = pd.DataFrame()
        sc_event = pd.DataFrame()
        for ct in contact_types:
            spikes_ct = spikes_event if ct == "all" else spikes_event[spikes_event[f"spike_in_{ct}"] == "yes"]

            bin_len = 15 if event == "pre-exp baseline" else 15
            max_dur = None if event == "pre-exp baseline" else 90

            spike_rates, spike_counts = compute_binned_spike_rates(
                spike_df=spikes_ct, start_time=event_start, end_time=event_end, 
                bin_len=bin_len, max_duration=max_dur)

            if event == "pre-exp baseline":
                plot_path = f"{sr_sub_path}/{sub}_{ses}_bootstrbase_hist_{ct}_contacts.png"
                plot_bootstrapped_hist(spike_rates, plot_path)


            # format results for spike rates in an event's time bins (each column contains spike rates across all, SOZ and nSOZ contacts)    
            sr_ct = (pd.DataFrame.from_dict(spike_rates, orient="index", columns=[f"rate_{ct}"])
                        .reset_index().rename(columns={"index": "time_bin"}))
            sr_event = sr_ct if sr_event.empty else pd.merge(sr_event, sr_ct, on="time_bin", how="outer")

            # same for spike counts
            sc_ct = (pd.DataFrame.from_dict(spike_counts, orient="index", columns=[f"count_{ct}"])
                        .reset_index().rename(columns={"index": "time_bin"}))
            sc_event = sc_ct if sc_event.empty else pd.merge(sc_event, sc_ct, on="time_bin", how="outer")

        # Merge events
        sr_event["event"] = event
        sr_merged = pd.concat([sr_merged, sr_event], ignore_index=True)

        sc_event["event"] = event
        sc_merged = pd.concat([sc_merged, sc_event], ignore_index=True)

    # Save per participant/session
    sr_merged.to_csv(f"{sr_sub_path}/{sub}_{ses}_calculated_spike_rates.csv", index=False)
    sc_merged.to_csv(f"{sr_sub_path}/{sub}_{ses}_calculated_spike_counts.csv", index=False)
