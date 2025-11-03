import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
import re

def extract_DCs_from_edf(dc_chan, subject_raw_path):
    """
    Reads in subject-specific DC channel data.
    Calculates time difference between DC events and time difference from clip start of all events.
    Outputs a dataframe with these diffs in columns.
    """
    raw = mne.io.read_raw_edf(subject_raw_path, include=[dc_chan])
    sampling_freq = raw.info['sfreq']
    dc_events = mne.find_events(raw, stim_channel=dc_chan, shortest_event=1, min_duration=0, output='offset')

    # 2. Get DC channel events
    dc_events_df = pd.DataFrame({'sample': dc_events[:,0], 'time_from_clip_start': dc_events[:,0]/sampling_freq})
    dc_events_df['time_diffs'] = dc_events_df['time_from_clip_start'].diff()
    dc_events_df['time_from_first_DC'] = dc_events_df['time_from_clip_start'] - dc_events_df['time_from_clip_start'].iloc[0]
    return dc_events_df


def format_music_markers(video_markers, durations):
    """
    Subsets music-specific markers from subject's video markers. 
    Calculates music end_time based on music duration (since there is no DC channel marking the end of the music stimulus).
    Where the mp3 duration is longer than the actual duration of the music 'block' (this would be due to experiment interruption)
    end_time = end time as marked by the DC channel.
    """
    df = video_markers[video_markers['event'] == 'MUSIC'].sort_values(by='time_from_clip_start').reset_index(drop=True)
    df = df.pivot_table(index=['music type'], columns='start/end', values='time_from_clip_start')
    df = pd.merge(df, durations[['short_name', 'duration_mp3']], left_on='music type', right_on='short_name')

    df['end_time_mp3'] = df['start_dc'] + df['duration_mp3']
    df['end_time'] = df['end_dc']
    df['end_time'] = np.where(
        df['end_time_mp3'] < df['end_dc'], df['end_time_mp3'], df['end_dc'])

    df = df.reset_index().rename(columns={'short_name': 'event', 'start_dc': 'start_time'})
    df = df[['event', 'start_time', 'end_time']]
    
    return df


def format_sart_or_talking_markers(video_markers, which_event):
    """
    Subset SART/talking from subject's video markers, add index of the event (e.g. SART1, SART2, ...)
    """
    df = video_markers[video_markers['event'] == which_event].sort_values(by='time_from_clip_start').reset_index(drop=True)
    # add a column with 'task' number
    task_n = len(df) // 2
    df['task number'] = [f'{which_event} ' + str(i) for i in range(1, task_n + 1) for _ in range(2)]
    df = df.pivot_table(index='task number', columns='start/end', values='time_from_clip_start')
    df = df.reset_index().rename(columns={'task number': 'event', 'start_dc': 'start_time','end_dc': 'end_time'})

    return df


def adjust_overlapping_events(events):
    """
    Because of accounting for talking/distraction, there will be events that partially/fully overlap.
    To correctly calculate gaps (no music, no sart, no talking) for baseline spike rate, we need to account for these.

    This function adjusts for that by:
        1. if two events partially overlap, start time of the second event will be the end time of the first event.
        2. if two events fully overlap, the one that is encompassed within the order is dropped.
    """
    evs_adj = events.copy()
    evs_adj['remove_row'] = False
    for i in range(1, len(evs_adj)):
        # if the following event starts before the current finished
        if evs_adj.loc[i, 'start_time'] < evs_adj.loc[i - 1, 'end_time']:
            # if it finishes later than current, adjust the start time to the end time of the current
            evs_adj.loc[i, 'start_time'] = evs_adj.loc[i - 1, 'end_time'] + 1
            # if the following event also finishes before the current finished
            if evs_adj.loc[i, 'start_time'] > evs_adj.loc[i, 'end_time']:
                evs_adj.loc[i, 'remove_row'] = True
            
    evs_adj = evs_adj[evs_adj['remove_row'] != True].drop(columns='remove_row').reset_index(drop=True)

    return evs_adj


def calculate_gap_baselines(evs_adj):
    """
    Using adjusted events, calculates start and end times of gaps between events.
    Outputs a dataframe with 'gap' events.
    """
    gaps = []
    n = 0
    for i in range(len(evs_adj) - 1):
        current_end = evs_adj.loc[i, 'end_time']
        next_start = evs_adj.loc[i + 1, 'start_time']
        # record any gap longer than 3 seconds
        if next_start - current_end > 3:
            n += 1
            gaps.append({'event': f'GAP {n}', 'start_time': current_end, 'end_time': next_start})
    gaps = pd.DataFrame(gaps)

    return gaps


def add_preexp_baseline(df):
    """
    Calculates start and end times of pre-experiment baseline. 
    This is set to 10 minutes unless the clip is shorter than that.
    """
    start_baseline = df['start_time'].min() - 600
    if start_baseline < 0:
        start_baseline = 0
    df.loc[-1] = ['pre-exp baseline', start_baseline, df['start_time'].min() - 1]
    df = df.sort_index().reset_index(drop=True)
    df['duration_s'] = df['end_time'] - df['start_time']
    df['duration_min'] = [round(i/60, 2) for i in df['duration_s']]

    return df


def plot_time_markers(savepath, subject_and_session, video_markers, music_durations, dc_events_df, music_ms, sart_ms, talking_ms):
    """
    Plot Camilo's time marker horizontal bar chart plot of time events.
    """
    fig, ax = plt.subplots(nrows=2, figsize=(12, 5))

    # PLOT 1: MUSIC
    for _, row in music_ms.iterrows():
        ax[0].axvspan(row['start_time'], row['end_time'], facecolor='blue', alpha=0.7)
        ax[0].text(x=(row['start_time'] + row['end_time']) / 2, y=0.1, 
                   s=row['event'], ha='center', color='white', fontsize=8, fontweight ='bold')
        print(row['event'], row['start_time'], row['end_time'])
    # PLOT 1: SART
    for _, row in sart_ms.iterrows():
        ax[0].axvspan(row['start_time'], row['end_time'], facecolor='#18a558', alpha=0.7)
    # PLOT 1:  TALKING
    for _, row in talking_ms.iterrows():
        ax[0].axvspan(row['start_time'], row['end_time'], facecolor='red', alpha=0.5)
    # PLOT 1: FORMAT
    ax[0].set_xlim([0, np.nanmax(video_markers['time_from_clip_start'] + 300)])
    ax[0].set_yticks([])
    ax[0].set_title('Video markers (without true music finish)')

    # PLOT 2: MUSIC
    for time in dc_events_df['time_from_clip_start']:
        ax[1].axvline(x=time, color='black', linestyle='-', linewidth=0.5)
    for _, row in music_ms.iterrows():
        music_duration = music_durations[music_durations['short_name'] == row['event']]['duration_mp3'].values[0]
        end_time = row['start_time'] + music_duration
        ax[1].axvspan(row['start_time'], end_time, facecolor='blue', alpha=0.7)
        ax[1].text(x=(row['start_time'] + end_time) / 2, y=0.1, s=row['event'], ha='center', color='white', fontsize=8, fontweight ='bold')
        print(row['event'], row['start_time'], end_time)
    # PLOT 2: FORMAT        
    ax[1].set_xlim([0, np.nanmax(video_markers['time_from_clip_start'] + 300)])
    ax[1].set_yticks([])
    ax[1].set_xlabel('Time during recording (s)')
    ax[1].set_title('DC markers with real mp3 duration')

    plt.suptitle(subject_and_session)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()