import os
import pickle
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def find_binary_mask(video, params):
    # uses scikit image to find binary mask from movie object
    x_dim, y_dim, z_dim = params['x_dim'], params['y_dim'], params['z_dim']
    # create single volume by taking the mean across frames
    V_smoothed = np.mean(video, axis=0)
    # spatial smoothing, median filter across pixels
    V_smoothed = ski.filters.median(V_smoothed, behavior='ndimage')
    # thresholding
    thresh = ski.filters.threshold_otsu(V_smoothed)

    binary_mask = V_smoothed > thresh 
    binary_mask_frac = np.sum(binary_mask)/(x_dim*y_dim*z_dim)

    return binary_mask, binary_mask_frac

def save_and_plot_binary_mask(name, binary_mask, params, results_dir):
    
    mask_dir = f'{results_dir}/binary_masks'
    plot_dir = f'{results_dir}/binary_mask_plots'

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # save binary mask
    with open(f'{mask_dir}/{name}.pkl', 'wb') as f:
        pickle.dump(binary_mask, f)

    # plot binary masks
    colors = [(0, 0, 0, 0), (0, 0, 0, 0.1)]  # RGBA tuples, the 1st color's alpha set to 0 to make transparent so white values are ignores, black values are partially transparent; 0's mapped to 1st color, 1's mapped to 2nd color
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1,1, figsize=(3,3))
    for i in range(0, params['z_dim']):
        plt.imshow(binary_mask[:,:,i], cmap=cmap)
    plt.title(name)
    sns.despine()
    plt.savefig(f'{plot_dir}/{name}.png', dpi=300)
    plt.close()

def mean_activity_within_binary_mask(video, binary_mask, params):
    # using pixels contained in binary_mask, measure the mean pixel intensity over time

    x_dim, y_dim, z_dim = params['x_dim'], params['y_dim'], params['z_dim']
    # vectorize both the binary mask and the video
    binary_mask_reshaped = np.reshape(binary_mask, (x_dim*y_dim*z_dim))
    video_reshaped = np.reshape(video, (video.shape[0], -1))
    video_reshaped = np.array(video_reshaped)
    print(binary_mask_reshaped.shape, video_reshaped.shape)

    mean_activity = np.mean(video_reshaped[:,binary_mask_reshaped], axis=1)
    return mean_activity