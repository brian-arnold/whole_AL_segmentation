#!/usr/bin/env python
# coding: utf-8

# ## Using pixel intensity thresholding to segment the entire mosquito antennal lobe (AL)
# 
# This notebook is kept tidy by modularizing the data and code into separate python files that are imported here.
# 
# See `experiment_info.py` to specify important details related to the experiment. Within this file, important information gets stored in the `params` dictionary.
# 
# If you have different experiments, simply store the information in a separate python file and change the name `experiment_info` in the cell below to correspond to the file with the information from this other experiment.
# 
# **NOTE**: There should be as many .tif videos as there are odors, and it is assumed that these videos are alphanumerically labeled in the same order as they appear in `odor_string`. I.e., if we sort the names of the videos, the first one should correspond to the first odor in `odor_string`.
# 
# ### Specify odors to analyze
# These odors should correspond to the values in the `odor_encodings` dictionary specified in `experiment_info.py`.
# 
# If you want to analyze ALL odors, leave this list blank `[]`.

# In[1]:


odors_of_interest = []


# ### Load in `experiment_info.py` along with `functions.py`, which has some custom functions used here. 

# In[19]:


# import important experimental variables
from experiment_info import samples, data_dir
from experiment_info import odor_encodings, odor_string
from experiment_info import params
# import custom functions
import functions as fn

print(f'data directory: {data_dir}')
print(f'Number of samples: {len(samples)}')
odor_list = odor_string.split('_')
num_odors = len(odor_list)

# make a dictionary to store the order in which odors are presented
odor_order = {}
for i,odor in enumerate(odor_list):
    odor_name = odor_encodings[odor]
    if odor_name not in odor_order:
        odor_order[odor_name] = i

print(f'Number of odors: {num_odors}')
print(f'x,y,z dimensions:', params['x_dim'], params['y_dim'], params['z_dim'])
print(f'Number of frames to analyze:', params['n_frames_to_analyze'])
print(f'Number of initial frames for df/f normalization:', params['background_frames'])

# sanity checks
import os
assert os.path.isdir(data_dir), f'Data directory does not exist: {data_dir}'
for samp in samples:
    assert os.path.isdir(f"{data_dir}/{samp}"), f"directory {data_dir}/{samp} containing tif files does not exist! Is there one approppriately-named subdirectory per sample?"
for odor in odor_list:
    assert odor in odor_encodings, f"Odor {odor} in odor_list not found in odor_encodings dictionary. Please add it to experiment_info.py"
for o in odors_of_interest:
    assert o in odor_encodings.values(), f"Odor {o} in odors_of_interst not found in odor_encodings dictionary"


# ### Import other libraries

# In[3]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import glob
import pandas as pd
from collections import defaultdict
import caiman as cm
import skimage as ski


# ### Get full file paths for all videos, per sample.
# Note that this finds **all** videos per sample. We potentially select a subset of these later on.

# In[4]:


videos = {}
for samp in samples:
    v = glob.glob(f"{data_dir}/{samp}/*.tif")
    v = sorted(v)
    num_vids = len(v)
    assert num_vids == num_odors, f"I found {num_vids} videos for sample {samp}, but there are {num_odors} odors. The number of videos and odors should be equivalent."
    videos[samp] = v


# ### Find indices of videos corresponding to odors specified in `odors_of_interest`. To do this, we will use the order of odors specified in `odor_string` and assume the ordering of videos is the same.

# In[5]:


odor_of_interest_indices = []
if odors_of_interest:
    for i,odor in enumerate(odor_list):
        if odor_encodings[odor] in odors_of_interest:
            # print(i, odor, odor_encodings[odor])
            odor_of_interest_indices.append(i)
else:
    odor_of_interest_indices = range(len(odor_list))
assert odor_of_interest_indices == sorted(odor_of_interest_indices), "odor_of_interest_indices should be sorted"
print(f"According to your odors of interest, I was able to find this many in the odor_list: {len(odor_of_interest_indices)}") 
# print("for first sample in list, get the name of files with this odor:")
# tmp = [videos[samples[0]][i].split('/')[-1] for i in odor_of_interest_indices]
# for i in tmp:
#     print(i)


# ### For each sample, concatenate all videos and threshold to segment AL.
# For thresholding, we are currently using `ski.filters.threshold_otsu`

# In[6]:


# data reloaded to ensure cell runs independently
binary_masks = {}
for samp in videos:
    Y_list = fn.load_videos_into_list(videos[samp], odor_of_interest_indices, params, normalize=False)
    Y = cm.concatenate(Y_list)

    binary_mask = fn.find_binary_mask(Y)
    print(f"fraction of volume containing AL segment for sample {samp}: ", np.sum(binary_mask)/(params['x_dim']*params['y_dim']*params['z_dim']))
    binary_masks[samp] = binary_mask


# ## Look at 2D projections of 3D binary masks
# Check out all the .png files that get created in the `binary_mask_plots` subdirectory.

# In[7]:


os.makedirs('binary_mask_plots', exist_ok=True)
colors = [(0, 0, 0, 0), (0, 0, 0, 0.1)]  # RGBA tuples, the 1st color's alpha set to 0 to make transparent so white values are ignores, black values are partially transparent; 0's mapped to 1st color, 1's mapped to 2nd color
cmap = ListedColormap(colors)

for i,samp in enumerate(samples):
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    for i in range(0,params['z_dim']):
        plt.imshow(binary_masks[samp][:,:,i], cmap=cmap)
    plt.title(samples[0])
    sns.despine()
    plt.savefig(f'binary_mask_plots/{samp}_binary_mask.png', dpi=300)
    plt.close()


# ### Within each AL segment, compute mean activity over time

# In[1]:


mean_activity_within_segment = {}
for i,samp in enumerate(videos):
    Y_list = fn.load_videos_into_list(videos[samp], odor_of_interest_indices, params, normalize=True) # note normalize = True!
    Y = cm.concatenate(Y_list)
    mean_activity_within_segment[samp] = fn.extract_mean_activity_within_binary_mask(Y, binary_masks[samp], params)
    print(f'finished sample {i+1}')


# ## Plot activity traces

# In[9]:


fig, axs = plt.subplots(1, 1, figsize=(8, 4))

for i,samp in enumerate(mean_activity_within_segment):    
    plt.plot(mean_activity_within_segment[samp] + i*0.02, c='black')  # Offset each trace by i*3
    # print sample name on the right
    plt.text(len(mean_activity_within_segment[samp])*1.02, i*0.02, samp, color='black')

# print the names of the odors on the x-axis
odor_names = []
positions = []
for i,index in enumerate(odor_of_interest_indices):
    odor_names.append(odor_encodings[odor_list[index]])
    positions.append(i*params['n_frames_to_analyze'] + params['n_frames_to_analyze']/2)
plt.xticks(positions, odor_names, rotation=90)

# draw vertical lines to separate odors
for i in range(len(odor_of_interest_indices)):
    plt.axvline((i+1)*params['n_frames_to_analyze'], color="black", linestyle="--", alpha=0.1)

plt.yticks([])
# supress grid lines
plt.grid(False)
sns.despine()


# ## For each odor, using the mean activity traces, get the maximum intensity during the frames corresponding to that odor.

# In[10]:


maxs_by_samp = fn.compute_max_responses(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, params['n_frames_to_analyze'])
aocs_by_samp = fn.calculate_AOC(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, params)


# In[11]:


mean_activity_within_segment_paraffin_subtracted = fn.subtract_paraffin_trace(mean_activity_within_segment, odor_of_interest_indices, odor_list, odor_encodings, params['n_frames_to_analyze'])


# In[12]:


fig, axs = plt.subplots(1, 1, figsize=(8, 4))

for i,samp in enumerate(mean_activity_within_segment_paraffin_subtracted):    
    plt.plot(mean_activity_within_segment_paraffin_subtracted[samp] + i*0.02, c='black')  # Offset each trace by i*3
    # print sample name on the right
    plt.text(len(mean_activity_within_segment_paraffin_subtracted[samp])*1.02, i*0.02, samp, color='black')

# print the names of the odors on the x-axis
odor_names = []
positions = []
for i,index in enumerate(odor_of_interest_indices):
    odor_names.append(odor_encodings[odor_list[index]])
    positions.append(i*params['n_frames_to_analyze'] + params['n_frames_to_analyze']/2)
plt.xticks(positions, odor_names, rotation=90)

# draw vertical lines to separate odors
for i in range(len(odor_of_interest_indices)):
    plt.axvline((i+1)*params['n_frames_to_analyze'], color="black", linestyle="--", alpha=0.1)

plt.yticks([])
# supress grid lines
plt.grid(False)
sns.despine()


# In[13]:


maxs_by_samp_paraffin_subtracted = fn.compute_max_responses(mean_activity_within_segment_paraffin_subtracted, odor_of_interest_indices, odor_list, odor_encodings, params['n_frames_to_analyze'])
aocs_by_samp_paraffin_subtracted = fn.calculate_AOC(mean_activity_within_segment_paraffin_subtracted, odor_of_interest_indices, odor_list, odor_encodings, params)


# ## Convert the data to a DataFrame and save

# In[23]:


def convert_to_df(dict, odor_order):
    df_list = []
    for samp in dict:
        df_tmp = pd.DataFrame.from_dict(dict[samp])
        df_tmp['samp'] = samp
        df_tmp['subpop'] = samp.split('_')[1]
        df_tmp['trial'] = df_tmp.index+1
        df_list.append(df_tmp)
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df = pd.melt(df, id_vars=['samp', 'subpop', 'trial'], var_name='odor', value_name='value')
    df['odor_order'] = df['odor'].map(odor_order)
    return df
    
peak_max_df = convert_to_df(maxs_by_samp, odor_order)
peak_aoc_df = convert_to_df(aocs_by_samp, odor_order)

peak_max_paraffin_df = convert_to_df(maxs_by_samp_paraffin_subtracted, odor_order)
peak_aoc_paraffin_df = convert_to_df(aocs_by_samp_paraffin_subtracted, odor_order)

os.makedirs('results', exist_ok=True)
peak_max_df.to_csv('results/peak_max_df.csv', index=False)
peak_aoc_df.to_csv('results/peak_aoc_df.csv', index=False)
peak_max_paraffin_df.to_csv('results/peak_max_paraffin_df.csv', index=False)
peak_aoc_paraffin_df.to_csv('results/peak_aoc_paraffin_df.csv', index=False)


# In[24]:


peak_max_df


# In[ ]:


sns.scatterplot(x=list(df[df['trial']==1]['peak_value']),
                y=list(df[df['trial']==2]['peak_value']),
                hue=df[df['trial']==1]['subpop'],
                style=df[df['trial']==1]['odor'],
                alpha=0.7)
plt.xlabel('peak_value trial 1')
plt.ylabel('peak_value trial 2')
plt.xlim(0,0.02)
plt.ylim(0,0.02)

# put legend in upper right corner
plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)
# plot the correlation between the two trials and print this in the upper left corner
corr = np.corrcoef(df[df['trial']==1]['peak_value'], df[df['trial']==2]['peak_value'])[0,1]
plt.text(0.8, 0.1, f"correlation: {corr:.2f}", transform=plt.gca().transAxes)
# add x=y line
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black', alpha=0.5)
# add grid lines 
plt.grid(True, alpha=0.2)
sns.despine()

