import skimage as ski
import caiman as cm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def reshape(Y, x_dim, y_dim, z_dim):
    # 1st dimension of Y is time X z_stacks, so reshape according to specified z_stacks
    try:
        Y_reshaped = np.reshape(Y, (int(Y.shape[0]/z_dim), z_dim, x_dim, y_dim))
    except ValueError:
        print(f"Y.shape: {Y.shape}, x_dim: {x_dim}, y_dim: {y_dim}, z_dim: {z_dim}")
        raise ValueError("The dimensions of the video are not compatible with the specified x,y,z dimensions. Please check the dimensions of the video and the x,y,z dimensions specified in experiment_info.py")
    # transpose to (n_frames, x, y, z)
    Y_reshaped2 = np.transpose(Y_reshaped, (0, 2, 3, 1))
    return Y_reshaped2

def background_normalize(Y, background_frames, offset=1000):
    # Y has shape (frames, x,y,z), first `background_frames` frames are background
    Y = Y + offset
    background = np.mean(Y[0:background_frames,:,:,:], axis=0) # get mean of each pixel for 1st 20 frames
    Y_normalized = (Y - background)/(background+0.0000000000000000000001) # subtract and divide by background
    return Y_normalized

def load_videos_into_list(videos, odor_of_interest_indices, p, normalize=False):
    x_dim, y_dim, z_dim = p['x_dim'], p['y_dim'], p['z_dim']
    n_frames_to_analyze = p['n_frames_to_analyze']
    background_frames = p['background_frames']
    
    Y_list = []
    for i in odor_of_interest_indices:
        Y = cm.load(videos[i])
        Y = reshape(Y, x_dim, y_dim, z_dim)
        assert Y.shape[0] >= n_frames_to_analyze, f"Number of frames in video is less than n_frames_to_analyze. Please specify a smaller number of frames to analyze."
        Y = Y[:n_frames_to_analyze] # only analyze first n_frames
        if normalize:
            Y = background_normalize(Y, background_frames)
        Y_list.append(Y)
    return Y_list

def find_binary_mask(Y):
    # temporal smoothing
    Y_smoothed = np.mean(Y, axis=0)
    # spatial smoothing, median filter across pixels
    Y_smoothed = ski.filters.median(Y_smoothed, behavior='ndimage')
    # thresholding
    thresh = ski.filters.threshold_otsu(Y_smoothed)
    binary_mask = Y_smoothed > thresh 
    return binary_mask

def extract_mean_activity_within_binary_mask(Y, binary_mask, p):
    x_dim, y_dim, z_dim = p['x_dim'], p['y_dim'], p['z_dim']
    # vectorize both the binary mask and the video
    binary_volume_R = np.reshape(binary_mask, (x_dim*y_dim*z_dim))
    Y_R = np.reshape(Y, (Y.shape[0], -1))
    Y_R = np.array(Y_R)
    # print(binary_volume_R.shape, Y_R.shape)

    return np.mean(Y_R[:,binary_volume_R], axis=1)

def compute_max_responses(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, n_frames_to_analyze):
    maxs_by_samp = defaultdict(lambda : defaultdict(list))

    for samp in mean_activity_within_segment:
        maxs_per_odor = defaultdict(list)
        # for each odor, extract max value from the corresponding interval
        for i in range(len(odor_of_interest_indices)):
            # get odor name
            odor_of_interest_index = odor_of_interest_indices[i]
            odor = odor_list[odor_of_interest_index]
            odor_name = odor_encodings[odor]

            # since a subset of videos were concatenates, use index i of odor_of_interest_indices to get the corresponding interval
            interval = mean_activity_within_segment[samp][i*n_frames_to_analyze:(i+1)*n_frames_to_analyze]

            maxs_per_odor[odor_name].append(np.max(interval))

        for odor in maxs_per_odor:
            assert len(maxs_per_odor[odor]) <= 2, f" for odor {odor} there were more than 2 trials. This is unexpected."

        maxs_by_samp[samp] = maxs_per_odor  

    return maxs_by_samp



def get_odor_name(index, odor_list, odor_encodings):
    # get the name of an odor at position `index` in odor_list
    odor = odor_list[index]
    return odor_encodings[odor]

def calculate_AOC(activity_traces, odor_of_interest_indices, odor_list, odor_encodings, p, test=False):
    n_frames_to_analyze = p['n_frames_to_analyze']
    background_frames = p['background_frames']
    aocs_by_samp = defaultdict(lambda : defaultdict(list))

    aoc_coords = defaultdict(list)

    for samp in activity_traces:
        aocs_per_odor = defaultdict(list)
        # for each odor, extract max value from the corresponding interval
        for i, index in enumerate(odor_of_interest_indices):
            # get odor name
            odor_name = get_odor_name(index, odor_list, odor_encodings)

            # since a subset of videos were concatenates, use index i of odor_of_interest_indices to get the corresponding interval
            interval = activity_traces[samp][i*n_frames_to_analyze:(i+1)*n_frames_to_analyze]

            # baseline= 1.96*np.std(interval[0:background_frames])
            baseline= np.median(interval[0:background_frames])
            peak_coord = np.argmax(interval)

            # get all indices of interval where activity is above baseline
            above_baseline = np.where(interval > baseline)[0]
            # find sets of indices in above_baseline that are consecutive, if not consecutive, then activity dropped below baseline
            not_consecutive = np.where(np.diff(above_baseline) != 1)[0]
            # an extra 1 is added to account for fact that np.diff returns array that is 1 element shorter than the input array, so add 1 to split at correct indices
            split_indices = np.split(above_baseline, not_consecutive+1)

            # find interal that contains the peak index
            for s in split_indices:
                if peak_coord in s:
                    start = s[0] - 1 # indices are for where activity is above baseline, so start 1 before
                    end = s[-1] + 2 # indices are for where activity is above baseline, so end 1 after, and add extra 1 to include the last index
                    peak_interval = interval[start:end]
                    break

            # calculate area under the curve
            aoc = np.trapz(peak_interval, dx=1)
            aocs_per_odor[odor_name].append(aoc)

            # for testing

            if test and i == 0 and samp == list(activity_traces.keys())[0]:
                plt.plot(interval)
                plt.axhline(baseline, color='black', linestyle='--')
                plt.axvline(peak_coord, color='red')
                plt.axvline(start, color='green')
                plt.axvline(end, color='green')


        for odor in aocs_per_odor:
            assert len(aocs_per_odor[odor]) <= 2, f" for odor {odor} there were more than 2 trials. This is unexpected."

        aocs_by_samp[samp] = aocs_per_odor  

    return aocs_by_samp

def subtract_paraffin_trace(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, n_frames_to_analyze):
    
    mean_activity_within_segment_paraffin_subtracted = {}
    num_odors = len(odor_list)

    # find indices of the odors delivered that correspond to paraffin, should be 2 for 2 trials
    paraffin_indices = []
    for i, index in enumerate(odor_of_interest_indices):
        odor_name = get_odor_name(index, odor_list, odor_encodings)
        if odor_name == 'paraffin':
            paraffin_indices.append(i)
    assert len(paraffin_indices) == 2, f"Expected to find 2 paraffin trials, but found {len(paraffin_indices)}"
    assert paraffin_indices[0] < paraffin_indices[1], f"Expected paraffin trial 1 to come before paraffin trial 2, but found {paraffin_indices}"

    for samp in mean_activity_within_segment:
        new_traces = []
        # collect paraffin traces for this sample, for both trials; in paraffin_traces dict, keys are trials
        paraffin_traces = {}
        for i, index in enumerate(paraffin_indices):
            interval = mean_activity_within_segment[samp][index*n_frames_to_analyze:(index+1)*n_frames_to_analyze]
            paraffin_traces[i] = interval

        # subtract paraffin traces from each odor trace
        for i, index in enumerate(odor_of_interest_indices):
            odor_name = get_odor_name(index, odor_list, odor_encodings)
            interval = mean_activity_within_segment[samp][i*n_frames_to_analyze:(i+1)*n_frames_to_analyze]
            # if this odor was delivered in the first half, subtract paraffin trace from first trial
            if index <= (num_odors/2) - 1:
                interval_subtracted = interval - paraffin_traces[0]
            else:
                interval_subtracted = interval - paraffin_traces[1]
            new_traces.extend(interval_subtracted)

        mean_activity_within_segment_paraffin_subtracted[samp] = np.array(new_traces)

    return mean_activity_within_segment_paraffin_subtracted