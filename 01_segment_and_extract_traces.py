#!/usr/bin/env python
# coding: utf-8

# # Using pixel intensity thresholding to segment the entire mosquito antennal lobe (AL)
# 
# This notebook is kept tidy by modularizing the data and code into separate python files that are imported here.
# 
# See `experiment_info.py` to specify important details related to the experiment. Within this file, important information gets stored in the `params` dictionary.
# 
# If you have different experiments, simply store the information in a separate python file and change the name `experiment_info` in the cell below to correspond to the file with the information from this other experiment.
# 
# **NOTES**: 
# - There should be as many .tif videos as there are odors, and it is assumed that these videos are alphanumerically labeled in the same order as they appear in `odor_string`. I.e., if we sort the names of the videos, the first one should correspond to the first odor in `odor_string`.
# - For computational speed, this code should be run wherever the raw data are stored. If you mount the file system where the data are stored (e.g. PNI cluster) and run the code on your local machine, it may go very slow as the data has to transfer over the network. I currently use this notebook for interactive work, but when I'm satisfied with the results for a few samples, I export this notebook as a python script using `jupyter nbconvert --to script 01_segment_and_extract_traces.ipynb` and then execute this script as a job using SLURM.
# 

# ### make directory to store results

# In[2]:


results_dir = 'results'
import os
os.makedirs(results_dir, exist_ok=True)


# In[3]:


from experiment_info import samples, data_dir, puffs, params


# ### Load in `experiment_info.py` along with `functions.py`, which has some custom functions used here. 

# In[5]:


# import custom functions
import functions as fn

print(f'data directory: {data_dir}')
print(f'Number of samples: {len(samples)}')
num_odors = len(puffs)

print(f'Number of odors: {num_odors}')
print(f'x,y,z dimensions:', params['x_dim'], params['y_dim'], params['z_dim'])
print(f'Number of frames to analyze:', params['n_frames_to_analyze'])
print(f'Number of initial frames for df/f normalization:', params['background_frames'])


# ### Import other libraries

# In[6]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import glob
import pandas as pd
from collections import defaultdict
import caiman as cm
import pickle


# ### Get full file paths for all videos, per sample.
# Note that this finds **all** videos per sample. We potentially select a subset of these later on.

# In[7]:


videos = {}
for samp in samples:
    v = glob.glob(f"{data_dir}/{samp}/*.registered.tif")
    v = sorted(v)
    num_vids = len(v)
    assert num_vids == len(puffs), f"I found {num_vids} videos for sample {samp}, but there are {len(puffs)} odors. The number of videos and odors should be equivalent."
    videos[samp] = v


# ### For each sample, concatenate all videos (one per odor) and threshold to segment AL.
# For thresholding, we are currently using `ski.filters.threshold_otsu`

# In[8]:


# data reloaded to ensure cell runs independently
binary_masks = {}
for samp in videos:
    vids = videos[samp]
    Y_list = fn.load_videos_into_list(vids, params, normalize=False)
    Y = cm.concatenate(Y_list)

    binary_mask = fn.find_binary_mask(Y)
    print(f"fraction of volume containing AL segment for sample {samp}: ", np.sum(binary_mask)/(params['x_dim']*params['y_dim']*params['z_dim']))
    binary_masks[samp] = binary_mask


# ### Save binary masks to use in downstream analyses

# In[9]:


with open(f'{results_dir}/binary_masks.pkl', 'wb') as f:
    pickle.dump(binary_masks, f)


# ## Look at 2D projections of 3D binary masks
# Check out all the .png files that get created in the `binary_mask_plots` subdirectory in results_dir.

# In[10]:


os.makedirs(f'{results_dir}/binary_mask_plots', exist_ok=True)
colors = [(0, 0, 0, 0), (0, 0, 0, 0.1)]  # RGBA tuples, the 1st color's alpha set to 0 to make transparent so white values are ignores, black values are partially transparent; 0's mapped to 1st color, 1's mapped to 2nd color
cmap = ListedColormap(colors)

for i,samp in enumerate(samples):
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    for i in range(0,params['z_dim']):
        plt.imshow(binary_masks[samp][:,:,i], cmap=cmap)
    plt.title(samples[0])
    sns.despine()
    plt.savefig(f'{results_dir}/binary_mask_plots/{samp}_binary_mask.png', dpi=300)
    plt.close()


# ### Within each AL segment, compute mean activity over time

# In[13]:


mean_activity_within_segment = {}
for i,samp in enumerate(videos):
    vids = videos[samp]
    Y_list = fn.load_videos_into_list(vids, params, normalize=True) # note normalize = True!
    Y = cm.concatenate(Y_list)
    mean_activity_within_segment[samp] = fn.extract_mean_activity_within_binary_mask(Y, binary_masks[samp], params)
    print(f'finished sample {i+1}')
    
mean_activity_df = pd.DataFrame.from_dict(mean_activity_within_segment)
mean_activity_df.to_csv(f'{results_dir}/mean_activity_within_segment.csv', index=False)


# ### Subtract the paraffin signal from each trace (nonfunctional after code change; not updated because results never used)

# In[10]:


# mean_activity_within_segment_paraffin_subtracted = fn.subtract_paraffin_trace(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, params['n_frames_to_analyze'])
# mean_activity_paraffin_subtracted_df = pd.DataFrame.from_dict(mean_activity_within_segment_paraffin_subtracted)
# mean_activity_paraffin_subtracted_df.to_csv(f'{results_dir}/mean_activity_within_segment_paraffin_subtracted.csv', index=False)


# ## Plot activity traces

# In[12]:


fig, axs = plt.subplots(1, 1, figsize=(16, 4))

for i,samp in enumerate(mean_activity_within_segment):    
    plt.plot(mean_activity_within_segment[samp] + i*0.02, c='black')  # Offset each trace by i*3
    # print sample name on the right
    plt.text(len(mean_activity_within_segment[samp])*1.02, i*0.02, samp, color='black')

# print the names of the odors on the x-axis
odor_names = []
positions = []
for i,puff in enumerate(puffs):
    odor_names.append(puff.odor_name)
    positions.append(i*params['n_frames_to_analyze'] + params['n_frames_to_analyze']/2)
plt.xticks(positions, odor_names, rotation=90)

# draw vertical lines to separate odors
for i in range(len(puffs)):
    plt.axvline((i+1)*params['n_frames_to_analyze'], color="black", linestyle="--", alpha=0.1)

plt.yticks([])
# supress grid lines
plt.grid(False)
sns.despine()

plt.savefig(f'{results_dir}/signal_traces.png', dpi=300)
plt.close()


# ## For each odor, using the mean activity traces, get the maximum intensity during the frames corresponding to that odor.
# 
# While were at it, let's also get the exact frame in which the segment was at it's max activity and store in the `argmaxs_by_samp` dictionary.

# In[21]:


# # reload the functions module to make sure we are using the latest version
import importlib
importlib.reload(fn)

maxs_by_samp, argmaxs_by_samp = fn.compute_max_responses(mean_activity_within_segment, puffs, params['n_frames_to_analyze'])
aucs_by_samp = fn.calculate_AUC(mean_activity_within_segment, puffs, params, test=False)


# ## From the mean activity in each segment, subtract out the activity observed from the paraffin odor (negative control)

# In[16]:


# mean_activity_within_segment_paraffin_subtracted = fn.subtract_paraffin_trace(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, params['n_frames_to_analyze'])


# In[26]:


# maxs_by_samp_paraffin_subtracted, _ = fn.compute_max_responses(mean_activity_within_segment_paraffin_subtracted, odor_of_interest_indices, odor_list, odor_encodings, params['n_frames_to_analyze'])
# aucs_by_samp_paraffin_subtracted = fn.calculate_AUC(mean_activity_within_segment_paraffin_subtracted, odor_of_interest_indices, odor_list, odor_encodings, params)


# ## Convert the data to a DataFrame and save

# In[22]:


peak_max_df = fn.convert_to_df(maxs_by_samp, puffs)
peak_auc_df = fn.convert_to_df(aucs_by_samp, puffs)
peak_max_df.to_csv(f'{results_dir}/peak_max_df.csv', index=False)
peak_auc_df.to_csv(f'{results_dir}/peak_auc_df.csv', index=False)
        
# peak_max_paraffin_df = fn.convert_to_df(maxs_by_samp_paraffin_subtracted, odor_order)
# peak_auc_paraffin_df = fn.convert_to_df(aucs_by_samp_paraffin_subtracted, odor_order)
# peak_max_paraffin_df.to_csv(f'{results_dir}/peak_max_paraffin_df.csv', index=False)
# peak_auc_paraffin_df.to_csv(f'{results_dir}/peak_auc_paraffin_df.csv', index=False)


# ## For each odor, get the frame corresponding to the maximum activity
# 
# -within compute_max_responses, get index of frame of max activity
# 
# -store in dict with odor_name as key, index as value
# 
# -go through each odor_of_interest_indices, get name of odor and load the video for that index
# 
# -extract frame of corresponding to index of max intensity

# In[24]:


for i, puff in enumerate(puffs):
    # i=11 is nonanal4.56
    # i=16 is decanal5.53
    for samp in samples:
        os.makedirs(f'{results_dir}/images_at_max_intensity_per_odor/{samp}', exist_ok=True)
        # only look at the first trial; odors with indices > 36 are repeats from second trial.
        if puff.trial == 1:
            odor_name = puff.odor_name
            frame_at_max = argmaxs_by_samp[samp][odor_name][0]
            Y = cm.load(videos[samp][puff.number])
            Y = fn.reshape(Y, params['x_dim'], params['y_dim'], params['z_dim'])
            Y = fn.background_normalize(Y, params['background_frames'])
            Y = Y[frame_at_max]
            file_name = f'{results_dir}/images_at_max_intensity_per_odor/{samp}/{odor_name}.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(Y, f)
            # projection = np.mean(Y, axis=2)
            # fig, ax = plt.subplots(1,1, figsize=(3,3))
            # plt.imshow(projection, cmap='bone')


