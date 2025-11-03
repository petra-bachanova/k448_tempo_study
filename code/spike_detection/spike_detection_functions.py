from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import h5py
import re
import shutil
import time
from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from matplotlib.pyplot import specgram
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
import logging
from collections import Counter


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


def load_processed_eeg(path):
    """
    Loads processed EEG data from a CSV file.
    The file is assumed to have channels as columns and samples as rows,
    and is always transposed so that rows correspond to channels and
    columns correspond to timepoints.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: EEG data with rows = channels and columns = timepoints.
        list: List of channel names.
    """
    data = pd.read_csv(path, header=0)
    logging.info(f"Loaded data from {path}")
    data = data.T
    logging.info("Transposed data (rows = channels, cols = timepoints).")

    data = data.apply(pd.to_numeric, errors='coerce')
    channels = data.index.tolist()

    return data, channels


def locate_downsample_freq(sample_freq, min_freq=200, max_freq=340):
    """
    Finds the optimal downsample frequency that minimizes the upsampling factor.

    Args:
        sample_freq (float): Original sampling frequency.
        min_freq (int): Minimum candidate downsample frequency. Default is 200.
        max_freq (int): Maximum candidate downsample frequency. Default is 340.

    Returns:
        int: Optimal downsample frequency.
    """

    min_up_factor = np.inf
    best_candidate_freq = None

    for candidate in range(min_freq, max_freq + 1):
        if sample_freq % candidate == 0:
            return candidate
        # Calculate downsampling ratio and upsampling factor
        down_samp_rate = sample_freq / candidate
        _, up_factor = down_samp_rate.as_integer_ratio()

        if up_factor < min_up_factor:
            min_up_factor = up_factor
            best_candidate_freq = candidate

    logging.info(f"Sampling frequency: {sample_freq}.")
    logging.info(f"Optimal downsample frequency: {best_candidate_freq}.")
    return best_candidate_freq


def plot_detected_peaks(x, mph, mpd, threshold, edge, valley, ax, ind, saveplotpath):
    """
    Creates and saves a plot showing detected peaks or valleys in the data.

    Args:
        x (array-like): Data array with detected peaks or valleys.
        mph (float): Minimum peak height.
        mpd (int): Minimum peak distance (number of samples).
        threshold (float): Minimum difference between peak and its neighbors.
        edge (str): Edge type used for detection ('rising', 'falling', or 'both').
        valley (bool): Whether valleys (True) or peaks (False) are detected.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis for plotting. Default is None.
        ind (array-like): Indices of detected peaks or valleys.
        saveplotpath (str): Path to save the plot image.

    Returns:
        None
    """
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x, 'b', lw=1)
    
    if ind.size:
        label = 'valley' if valley else 'peak'
        label += 's' if ind.size > 1 else ''
        ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                label=f'{ind.size} {label}')
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    
    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    
    finite_x = x[np.isfinite(x)]
    ymin, ymax = finite_x.min(), finite_x.max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    
    mode = 'Valley detection' if valley else 'Peak detection'
    ax.set_title(f"{mode} (mph={mph}, mpd={mpd}, threshold={threshold}, edge='{edge}')")
    plt.tight_layout()
    plt.title(re.sub("^.*?(sub.*?).png$", "\\1", saveplotpath))
    plt.savefig(saveplotpath, dpi=200)
    plt.close()


def detect_peaks(x, saveplotpath, mph=None, mpd=1, threshold=0, edge='rising',
                kpsh=False, valley=False, saveplot=True, ax=None):
    """
    Detects peaks or valleys in data based on amplitude and other criteria.

    Args:
        x (array-like): Data array in which to detect peaks.
        saveplotpath (str): Path to save the plot of detected peaks.
        mph (float, optional): Minimum peak height. Default is None.
        mpd (int, optional): Minimum peak distance (number of samples). Default is 1.
        threshold (float, optional): Minimum difference between peak and its neighbors. Default is 0.
        edge (str, optional): Edge type for peak detection ('rising', 'falling', or 'both'). Default is 'rising'.
        kpsh (bool, optional): Keep smaller peaks with the same height if True. Default is False.
        valley (bool, optional): Detect valleys instead of peaks if True. Default is False.
        saveplot (bool, optional): Whether to save a plot of the detected peaks. Default is True.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis for plotting. Default is None.

    Returns:
        np.ndarray: Indices of detected peaks or valleys.
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if saveplot:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        # _plot(x, mph, mpd, threshold, edge, valley, ax, ind, saveplotpath)
        plot_detected_peaks(x, mph, mpd, threshold, edge, valley, ax, ind, saveplotpath)
    return ind


# FOR OLD DETECTOR THRESH=8, MIN_SPACING=1; NEW: 7 and 0
def template_match(channel, template, down_samp_freq, saveplotpath, thresh, min_spacing=1):
    """
    Detects spikes in a channel by cross-correlating with a template and identifying significant peaks.

    Args:
        channel (array-like): The EEG channel data.
        template (array-like): The template for matching spikes.
        down_samp_freq (int): The downsampled frequency used in processing.
        saveplotpath (str): Path to save the plot of detected peaks.
        thresh (float, optional): Threshold for peak detection. Default for aIED is 7, old detector is 8.
        min_spacing (float, optional): Minimum spacing between detected peaks (in seconds). Default is 0.

    Returns:
        np.ndarray: Array of tuples where each tuple represents the start and end indices of detected spikes.
    """
    
    template_len = len(template)
    # Cross-correlate the input signal with the template
    cross_corr = np.convolve(channel, template, 'valid')
    window = np.ones(down_samp_freq) / float(down_samp_freq)
    # Calculate the mean of the squared cross-correlation
    mean_square = np.convolve(cross_corr ** 2, window, 'valid')
    square_mean = np.convolve(cross_corr, window, 'valid') ** 2
    cross_corr_std = np.sqrt(np.median(mean_square - square_mean))
    
    if cross_corr_std > 0:
        cross_corr_norm = (cross_corr - np.mean(cross_corr)) / cross_corr_std  # Normalize the cross-correlation
        
        # set first and last elements to zero to avoid boundary effects when detecting peaks
        cross_corr_norm[1] = 0
        cross_corr_norm[-1] = 0

        peaks = []
        if np.any(np.abs(cross_corr_norm) > thresh):
            peaks = detect_peaks(np.abs(cross_corr_norm), saveplotpath, mph=thresh, mpd=template_len, saveplot=True)
            peaks += int(np.ceil(template_len / 2.))
            # Ensure peaks are within the valid range of the channel
            valid_peaks = np.logical_and(peaks > template_len, peaks <= len(channel) - template_len)
            peaks = peaks[valid_peaks]

            if len(peaks) > 0:
                # Calculate distances between consecutive peaks
                distant_peaks = np.diff(peaks) > min_spacing * down_samp_freq
                distant_peaks = np.insert(distant_peaks, 0, True)  # always keep the first peak
                peaks = peaks[distant_peaks]

    detections = np.array([(peak - template_len, peak + template_len) for peak in peaks])
    return detections


def detect_template_matches(channel, samp_freq, saveplotpath, thresh, return_eeg=False, temp_func=None, signal_func=None):
    """
    Detects spikes in an EEG channel by downsampling, applying a triangular template, 
    and performing template matching.

    Args:
        channel (array-like): The EEG channel data.
        samp_freq (int): The sampling frequency of the data.
        saveplotpath (str): Path to save the plot of detected spikes.
        return_eeg (bool, optional): If True, returns the matched EEG segments. Default is False.
        temp_func (callable, optional): Function to adjust the template. Default is None.
        signal_func (callable, optional): Function to preprocess the signal. Default is None.

    Returns:
        list: Detected spike indices, optionally with corresponding EEG segments if `return_eeg` is True.
    """
    if samp_freq > 100:
        samp_freq = int(np.round(samp_freq))  # Round samp_freq to the nearest integer if it is large
    down_samp_freq = locate_downsample_freq(samp_freq)
    template = signal.windows.triang(np.round(down_samp_freq * 0.06))
    kernel = np.array([-2, -1, 1, 2]) / float(8)
    template = np.convolve(kernel, np.convolve(template, kernel, 'valid') ,'full')
    if temp_func:
        template = temp_func(template, samp_freq)
    if signal_func:
        channel = signal_func(channel, samp_freq)

    down_samp_rate = samp_freq / float(down_samp_freq)
    down_samp_factor, up_samp_factor = down_samp_rate.as_integer_ratio()
    channel = signal.detrend(channel, type='constant')
    results = template_match(channel, template, down_samp_freq, saveplotpath, thresh)
    up_samp_results = [np.round(spikes * down_samp_factor / float(up_samp_factor)).astype(int) for spikes in results]
    if return_eeg:
        return up_samp_results, [channel[start:end] for start, end in results]
    else:
        return up_samp_results


def auto_detect(eegdata, saveplotpath, subject_and_session, cross_corr_thresh, samp_freq = 200):
    """
    Detects spike events in EEG data for each channel and formats the results.

    Args:
        eegdata (DataFrame): Raw EEG data with each row as a channel.
        subject_and_session (str): Identifier for the subject_and_session being analyzed.
        samp_freq (int, optional): Sampling frequency of the EEG data (default is 200 Hz).

    Returns:
        DataFrame: A long-form DataFrame with spike times, channels, sampling frequency, and subject information.
    """
    ### DETECT SPIKES:
    logging.info("Template matching start.")
    all_detections = []
    channel_names = []
    for i in range(eegdata.shape[0]):
        channel = eegdata.iloc[i,:].astype(float) # run on each row (chan)
        ch_label = eegdata.iloc[i,:].name
        detections = detect_template_matches(channel, samp_freq, thresh=cross_corr_thresh, saveplotpath=f"{saveplotpath}/{ch_label}.png", return_eeg=False, temp_func=None, signal_func=None) 
        all_detections.append(detections)
        channel_names.append(int(float((eegdata.columns[i]))))
    logging.info("Template matching end.")

    ### REFORMAT SPIKES:
    detections = pd.DataFrame(all_detections)
    channels = pd.DataFrame(channel_names)
    spikes = pd.concat([channels,detections], axis = 1)
    newspikes = spikes.transpose()
    newspikes.columns = newspikes.iloc[0]
    newspikes = newspikes.iloc[1:] # remove duplicate channel_name row

    ### AUTO LONG-FORMATTING OF SPIKES
    spikeDf = pd.DataFrame() # empty df to store final spikes and spikeTimes 
    for idx, col in enumerate(newspikes.columns):
        # extract spikes for each column 
        tempSpikes = newspikes.iloc[:,idx].dropna() # column corresponding to channel with all spikes
        tempSpikes2 = tempSpikes.tolist() # convert series to list 
        # extract channel name for each spike (duplicate based on the number of spikes)
        tempName = tempSpikes.name # channel name 
        tempName2 = [tempName] * len(tempSpikes) # repeat col name by the number of spikes in this channel 
        tempDf = pd.DataFrame({'channel': tempName2, 'spikeTime': tempSpikes2})
        # save and append to final df 
        spikeDf = pd.concat([spikeDf, tempDf])
        spikeDf['fs'] = samp_freq
        spikeDf['subject'] = subject_and_session
    return(spikeDf)


def spectimgs(eegdata, spikedf, spectdir):
    """
    Generates and saves spectrograms for each detected spike in the EEG data.

    Args:
        eegdata (DataFrame): The EEG data.
        spikedf (DataFrame): DataFrame containing detected spikes with channel and timing information.
        spectdir (str): Directory where the spectrogram images will be saved.

    Returns:
        spectrograms saved to Graphs/SPECTS/IEDS
    """

    logging.info("Generating spectrograms for CNN.")
    for i in tqdm(range(0,len(spikedf))): 
        samp_freq = int(float(spikedf.fs.values[0]))
        #######################################
        pad = 1 # d:1 number of seconds for window 
        Nfft = 128*(samp_freq/500) # d: 128 
        h = 3
        w = 3
        #######################################
        try:
            subject_and_session = spikedf.subject.values[0]
            chan_name = int(spikedf.channel.values[i]) # zero idxed -1
            spikestart = spikedf.spikeTime.values[i][0] # start spike
            ### select eeg data row 
            ecogclip = eegdata.iloc[chan_name]
            ### filter out line noise
            b_notch, a_notch = signal.iirnotch(60.0, 30.0, samp_freq)
            ecogclip = pd.Series(signal.filtfilt(b_notch, a_notch, ecogclip)) 
        
            ### trim eeg clip based on cushion            
            ### mean imputation if missing indices
            end = int(float((spikestart+int(float(pad*samp_freq)))))
            start = int(float((spikestart-int(float(pad*samp_freq)))))
            if end > max(ecogclip.index):
                temp = list(ecogclip[list(range(spikestart-int(float(pad*samp_freq)), max(ecogclip.index)))])
                cushend = [np.mean(ecogclip)]*(end - max(ecogclip.index))
                temp = np.array(temp + cushend)
            elif start < min(ecogclip.index):
                temp = list(ecogclip[list(range(min(ecogclip.index), spikestart+pad*samp_freq))])
                cushstart = [np.mean(ecogclip)]*(min(ecogclip.index)-start)
                temp = np.array(cushstart + temp)  # , -> + (CC edit)
            else:
                temp = np.array(ecogclip[list(range(spikestart-int(float(pad*samp_freq)), 
                                         spikestart+int(float(pad*samp_freq))))]) 
           
            ### PLOT AND EXPORT:
            plt.figure(figsize=(h,w))
            specgram(temp, NFFT = int(Nfft), Fs = samp_freq, noverlap=int(Nfft/2), detrend = "linear", cmap = "YlOrRd") 
            plt.axis("off")
            plt.xlim(0, pad*2)
            plt.ylim(0,100)
            plt.savefig(f"{spectdir}/{subject_and_session}_{str(spikestart)}_{str(chan_name)}.png", dpi = 300)
            plt.close()
        except Exception as e: 
            print(e)
            print("ERROR with IED portion:", i)
            logging.error(f"ERROR with IED portion: {i}")
            plt.close()
            continue


def detect_with_cnn(project_dir, subject_and_session):
    """
    Processes spectrogram images template matches, uses a pretrained CNN to classify each spectrogram as containing an IED or not.

    Args:
        project_dir (str): Project root directory. Should contain: trained CNN model & spectrogram images.

    Returns:
        pd.DataFrame: A DataFrame containing filenames of the processed spectrograms, along with the 
        predicted class for each (0 for IED, 1 for non-IED).
    
    Steps:
        1. Loads and preprocesses subject_and_session specific spectrogram images from the specified project directory.
        2. Extracts metadata (e.g., subject ID, start time, and channel) from the spectrogram filenames.
        3. Loads a pretrained ResNet-18 model from the project directory.
        4. Applies the CNN to each spectrogram to predict whether it contains an IED.
        5. Compiles the results into a DataFrame, including the spectrogram filenames and their predicted classes.
    
    Note:
        - Ensure that the pretrained model file (`model_aied.pt`) is present in the specified directory.
        - The function assumes that the spectrogram images are stored in the `Graphs/SPECTS` directory within 
        the project directory.
    """

    logging.info("Applying pretrained CNN to each spectrogram.")
    ### A: LOAD ALL DATA, extract clip_id from path
    imgs = f'{project_dir}/derivatives/spike_detection/graphs/SPECTS'

    data_transforms = {
        imgs: transforms.Compose([
            transforms.Resize(224),
            transforms.Pad(1, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths.
        Extends torchvision.datasets.ImageFolder
        """

        def __init__(self, root, transform=None):
            super().__init__(root, transform)
            # Filter out only the images containing the specific string
            self.samples = [(path, target) for path, target in self.samples if subject_and_session in os.path.basename(path)]
            self.imgs = self.samples  # Ensure imgs attribute also gets filtered
            
        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return (tuple_with_path)

    image_datasets = {
        x: ImageFolderWithPaths(os.path.join(project_dir, x), data_transforms[x]) for x in [imgs]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, # use batch=1, shuffle=F
                                                shuffle=False, num_workers=0) for x in [imgs]} 
    class_names = image_datasets[imgs].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # extract image paths
    path_names = []
    for images,labels,paths in dataloaders[imgs]:
        path_names.append(paths)
    # convert list of paths to dataframe col
    df = pd.DataFrame(path_names)
    df.columns = ['clip_ids']
    df[['clip_ids','clip']] = df['clip_ids'].str.split('IEDS/',expand=True)
    df['clip'] = df['clip'].str.rstrip('.png')
    df[['subject', 'session', 'start','chan']] = df['clip'].str.split('_', expand=True)
    df['subject'] = df['subject'] + '_' + df['session']
    df = df.drop(columns=['session'])

    ### B: LOAD PRETRAINED MODEL 
    try:
        model = torch.load(f"{project_dir}/code/spike_detection/model_aied.pt", weights_only=False)
        # model.eval() # model architecture
    except ImportError:
        logging.error("Trained model not found. Check that model is in the right directory.")
        print('TRAINED MODEL NOT FOUND: Check that trained model is in eegdir and name matches: model_aied.pt')

    ### C: RUN MODEL
    y_pred = []
    with torch.no_grad():
        for inputs,labels,paths in dataloaders[imgs]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            _,predicted = torch.max(outputs, 1)
            pred = predicted.numpy()
            lab = labels.numpy()
            y_pred.append(pred)

    logging.info("Finished running CNN.")
    # reformat outputs:
    y_pred_flat = np.concatenate((y_pred),axis=0)
    df['predicted_class'] = y_pred_flat

    return df


def clean_and_format_spike_data(df, subject_and_session, channels, samp_freq, win = 3): 
    """
    Cleans and formats the CNN spike output. This function processes the spike detection data, removes overlapping spikes within a specified time window,
    and formats the data for export. It identifies unique spike events, counts the number of channels involved,
    and maps channel indices to their corresponding names.

    Args:
        df (pd.DataFrame): DataFrame containing the output from the CNN model, with spike detection results.
        subject_and_session (str): Identifier for the subject and session.
        channels (list): List of channel names corresponding to the indices in the spike data.
        samp_freq (int): Sampling frequency of the EEG data.
        win (int, optional): Time window in seconds for spike overlap (default is 3 seconds). Spikes detected within
                             this window are considered as a single event.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing:
                      - 'spike_start': Start time of each unique spike event.
                      - 'channels_idx': List of indices for the channels where spikes were detected.
                      - 'channels_names': Corresponding channel names for the detected spikes.
                      - 'channels_count': Number of channels involved in each spike event.
    """
    ### only keep spikes: predicted_class = 0
    try:
        df = df[df.predicted_class == 0].copy()
    except AttributeError:  # for template_match detector only
        df[['start', 'end']] = pd.DataFrame(df['spikeTime'].tolist(), index=df.index)
        df = df.drop(columns='end')
        df = df.rename(columns={"channel": "chan"})

    df['start'] = df['start'].astype(int) # convert from str to int
    ### sort start times in df:
    df = df.sort_values(by = 'start', ascending = True)
    ### dedupe spikes by col and time
    bins =  np.arange(min(df.start.values), max(df.start.values), samp_freq*win)
    spikebins = np.digitize(df['start'], bins)
    cleandf = df.groupby(spikebins)['start'].describe()
    chanlist = df.groupby(spikebins)['chan'].apply(lambda x: x.values.tolist())
    chanlist = [list(set(x)) for x in chanlist]
    chancounts = [len(l) for l in chanlist]
    meanspikestart = (cleandf['mean']).astype(int)
    subjectid = [subject_and_session]*len(meanspikestart)
    ### reformat into new df
    finaldf = pd.DataFrame({'subject': subjectid, 'spike_start': meanspikestart, 
                            'channels_idx': chanlist, 'channels_count': chancounts})
    ### reject spikes detected in >= 12 channels within time window
    finaldf = finaldf[finaldf.channels_count < 12]
    
    # create a channels dataframe, with indices related to the order in which they were read in from the edf file.
    channels_df = pd.DataFrame({'channels': channels})
    channels_names = []
    for index, row in finaldf.iterrows():
        # grab the list of channel indices from 
        li = row["channels_idx"]
        channels_li = [channels_df.loc[int(i)].channels for i in li]
        channels_names.append(channels_li)

    finaldf["channels_names"] = channels_names
    cols = ["spike_start", "channels_idx", "channels_names", "channels_count"]
    finaldf = finaldf[cols]
    print(finaldf.head())

    # concatenate lists with channel names to identify channels with the most spikes
    channel_counts = []
    for i in finaldf.channels_names:
        channel_counts.extend(i)

    top10chs = pd.DataFrame({'channel_name': Counter(channel_counts).keys(), 'count':Counter(channel_counts).values()})
    top10chs = top10chs.sort_values(by='count', ascending=False).iloc[:10, :]
    top10chs = top10chs.set_index('channel_name')['count'].to_dict()
    logging.info(f"Top 10 channels: {top10chs}")

    return (finaldf)


def clear_spectrogram_dir(path, subject_and_session):
    """
    Clears subject & session specific files from the Graphs/SPECTS/IEDS/ dir.

    Args:
        path (str): The path to the directory to be cleared.
        subject_and_session (str): sub-X_ses-X identifier
    """
    
    # Remove all files and directories within the spectrogram directory
    for filename in os.listdir(path):
        if subject_and_session in filename:
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    logging.info(f"Cleared {subject_and_session} files from spectrogram directory.")


def log_runtime_info(start_time, end_time):
    """Log preprocessing information."""
    logging.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logging.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logging.info(f"Runtime duration: {end_time - start_time:.2f} s (={(end_time - start_time)/60:.2f} min)")
