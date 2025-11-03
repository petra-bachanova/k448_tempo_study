import pandas as pd
import os
from extract_time_markers import *

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
    time_marker_sub_path = f"{project_path}/derivatives/time_markers/{sub}/{ses}"
    os.makedirs(time_marker_sub_path, exist_ok=True)

    # 1. Extract DC markers from raw.mne.io EEG file
    dc_chan = meta[(meta["participant_id"] == sub) & (meta["session"] == ses)]["dc_channel"].values[0]
    raw_path = f"{project_path}/rawdata/{sub}/{ses}/ieeg/{sub}_{ses}_task-k448tempo_ieeg.edf"
    dc_events_df = extract_DCs_from_edf(dc_chan=f"DC{int(dc_chan)}", subject_raw_path=raw_path)

    # 2. Load video markers
    video_marker_path = f"{time_marker_sub_path}/{sub}_{ses}_video_markers.xlsx"
    video_markers = pd.read_excel(video_marker_path)
    video_markers['start/end'] = [re.sub('START', 'start_dc', str(i)) for i in video_markers['start/end']]
    video_markers['start/end'] = [re.sub('END', 'end_dc', str(i)) for i in video_markers['start/end']]
    # 3. Load music durations
    music_durations = pd.read_csv(f'{project_path}/stimuli/music_stimuli.tsv', sep="\t")

    # 4. Format music, SART and talking events, then concat
    music_ms = format_music_markers(video_markers=video_markers, durations=music_durations)
    sart_ms = format_sart_or_talking_markers(video_markers=video_markers, which_event='SART')
    talking_ms = format_sart_or_talking_markers(video_markers=video_markers, which_event='TALKING')
    events = pd.concat([music_ms, sart_ms, talking_ms]).sort_values(by='start_time').reset_index(drop=True)

    # 5. Calculate gap and pre-exp baselines
    events_adj = adjust_overlapping_events(events=events)
    gap_baselines = calculate_gap_baselines(evs_adj = events_adj)
    events_w_gap_bs = pd.concat([events, gap_baselines]).sort_values(by='start_time').reset_index(drop=True)
    events_w_gap_and_preexp_bs = add_preexp_baseline(df=events_w_gap_bs)
    # 6. Save
    events_w_gap_and_preexp_bs.to_csv(f"{time_marker_sub_path}/{sub}_{ses}_processed_time_markers.csv", index=False)

    # 7. Plot time marker plots
    plot_time_markers(
        savepath = f"{time_marker_sub_path}/{sub}_{ses}_time_markers.png", 
        subject_and_session = f"{sub}_{ses}", 
        video_markers=video_markers, 
        music_durations=music_durations, 
        dc_events_df=dc_events_df, 
        music_ms=music_ms, 
        sart_ms=sart_ms, 
        talking_ms=talking_ms)