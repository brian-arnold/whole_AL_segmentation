import skimage as ski
import caiman as cm
import numpy as np
from collections import defaultdict

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