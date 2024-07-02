
import caiman as cm
import numpy as np


def reshape_vid(V, x_dim, y_dim, z_dim):
    # reshapes a video matrix into 4 dimensions: (n_frames, x, y, z)
    try:
        # 1st dimension of Y is time X z_stacks, so reshape according to specified z_stacks
        V_reshaped = np.reshape(V, (int(V.shape[0]/z_dim), z_dim, x_dim, y_dim))
    except ValueError:
        print(f"V.shape: {V.shape}, x_dim: {x_dim}, y_dim: {y_dim}, z_dim: {z_dim}")
        raise ValueError("The dimensions of the video are not compatible with the specified x,y,z dimensions. Please check the dimensions of the video and the x,y,z dimensions specified in experiment_info.py")
    # transpose to (n_frames, x, y, z)
    V_reshaped2 = np.transpose(V_reshaped, (0, 2, 3, 1))
    return V_reshaped2

def background_normalize(V, background_frames, offset=1000):
    # normalize pixel intensity according to the mean of the first background_frames
    # the default of offset=1000 was taken from Martin's code!
    # V has shape (frames, x,y,z), first `background_frames` frames are background
    V = V + offset
    background = np.mean(V[0:background_frames,:,:,:], axis=0) # get mean of each pixel for 1st 20 frames
    V_normalized = (V - background)/(background+0.0000000000000000000001) # subtract and divide by background
    return V_normalized

def load_videos(vid_fnames, params, normalize=False):
    # takes video file names in vid_fnames and loads them into list of CaImAn movie objects
    x_dim, y_dim, z_dim = params['x_dim'], params['y_dim'], params['z_dim']
    n_frames_to_analyze = params['n_frames_to_analyze']
    background_frames = params['background_frames']
    
    vid_list = []
    for v_fname in vid_fnames:
        V = cm.load(v_fname)
        V = reshape_vid(V, x_dim, y_dim, z_dim)
        assert V.shape[0] >= n_frames_to_analyze, f"Number of frames in video is less than n_frames_to_analyze. Please specify a smaller number of frames to analyze."
        # only analyze first 'n_frames_to_analyze' frames
        V = V[:n_frames_to_analyze] 
        if normalize:
            V = background_normalize(V, background_frames)
        vid_list.append(V)
        
    return cm.concatenate(vid_list)