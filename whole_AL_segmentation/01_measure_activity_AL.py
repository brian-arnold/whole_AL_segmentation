#!/usr/bin/env python
# coding: utf-8

# # Using pixel intensity thresholding to segment the entire mosquito antennal lobe (AL)
# 
# This notebook is kept tidy by modularizing the data and code into separate python files that are imported here. Data is stored in the `Brains` data class specified below, and these data are processed using function in modules in the `utils` directory.
# 
# IMPORTANT: See `experiment_info.py` to specify details related to the experiment. These details are imported below
# 
# If you have different experiments, simply store the information in a separate python file and change the name `experiment_info` in the cell below to correspond to the file with the information from this other experiment.
# 
# **NOTES**: 
# - There should be as many .tif videos as there are odors, and it is assumed that these videos are alphanumerically labeled in the same order as they appear in `odor_string`. I.e., if we sort the names of the videos, the first one should correspond to the first odor in `odor_string`.
# - For computational speed, this code should be run wherever the raw data are stored. If you mount the file system where the data are stored (e.g. PNI cluster) and run the code on your local machine, it may go very slow as the data has to transfer over the network. I currently use this notebook for interactive work, but when I'm satisfied with the results for a few samples, I export this notebook as a python script using `jupyter nbconvert --to script 01_measure_activity_AL.ipynb` and then execute this script as a job using SLURM.
# 

# In[19]:


from dataclasses import dataclass
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
from collections import defaultdict
import numpy as np
import caiman as cm

results_dir = 'results_tmp'
os.makedirs(results_dir, exist_ok=True)


# ## Load custom modules that can be found in the parent directory of this package.

# In[20]:


from experiment_info import samples, data_dir, puffs, params

import utils.activity_traces as activity_traces
import utils.video_IO as video_IO
import utils.binary_mask as binary_mask


# In[21]:


print(f'data directory: {data_dir}')
print(f'Number of samples: {len(samples)}')
num_odors = len(puffs)

print(f'Number of odors: {num_odors}')
print(f'x,y,z dimensions:', params['x_dim'], params['y_dim'], params['z_dim'])
print(f'Number of frames to analyze:', params['n_frames_to_analyze'])
print(f'Number of initial frames for df/f normalization:', params['background_frames'])


# ## Create the `Brain` data class that will store organized information for each sample
# 
# When creating the data class, `slots=True` means no new attributes can be added to the class. Only attributes `name` and `vid_fnames` need to be specified when initializing each instance (see below). Other attributes will be assigned values as the data are processed.
# 

# In[22]:


@dataclass(slots=True)
class Brain:

    name: str   # name of the sample
    vid_fnames: list # the file paths of all the individual videos

    # the following attributes, initialized here, are computed using vid_fnames
    video: cm.base.movies.movie = None # the full video, a concatenation of all individual videos
    
    binary_mask: np.array = None # 3D outline of the antennal lobe, region in video of higher pixel intensities according to otsu thresholding
    binary_mask_frac: float = None # the fraction of pixels contained within binary_mask, compared to the entire 3d volume
    
    mean_activity: np.array = None # the mean pixel intensity wtihin binary mask, over time; this is df_f if videos have been normalized against background activity
    maxs_per_odor: defaultdict(list) = None # a dictionary where keys are names of odors, values are lists of size 2, one for each trial in which the odor was administered
    argmaxs_per_odor: defaultdict(list) = None # a dictionary where keys are names of odors, values are lists of size 2, one for each trial in which the odor was administered
    aucs_per_odor: defaultdict(list) = None # same as maxs_per_odor, except calculating the area under the curve
    
    mean_activity_paraffin_subtracted: np.array = None # same as mean_activity except the signal of the paraffin odor has been subtracted
    maxs_per_odor_paraffin_subtracted: defaultdict(list) = None # same as maxs_per_odor except the signal of the paraffin odor has been subtracted
    argmaxs_per_odor_paraffin_subtracted: defaultdict(list) = None
    aucs_per_odor_paraffin_subtracted: defaultdict(list) = None # same as aucs_per_odor, except calculating the area under the curve


# ## We will store all information within a list called `brains`, that will contain one element per sample

# In[23]:


brains = []

for samp_name in samples:
    # get list of video file names for this sample
    v = glob.glob(f"{data_dir}/{samp_name}/*.registered.tif")
    v = sorted(v)
    num_vids = len(v)
    assert num_vids == len(puffs), f"I found {num_vids} videos for sample {samp_name}, but there are {len(puffs)} odors. The number of videos and odors should be equivalent."

    # initialize a Brain object for this sample
    brain = Brain(samp_name, v)
    brains.append(brain)

for b in brains:
    print(f'{b.name}: {len(b.vid_fnames)} videos')


# ## Here is where most of the work is done.
# For each sample, we pass the attributes to modules in the `utils` directory which has functions that we will reuse in other analyses

# In[24]:


for i,b in enumerate(brains):

    # using filenames of videos, load them into movie objects, creating video instance attribute
    b.video = video_IO.load_videos(b.vid_fnames, params, normalize=False)

    # use otsu thresholding to find binary mask
    b.binary_mask, b.binary_mask_frac = binary_mask.find_binary_mask(b.video, params)

    # save the binary mask object and plot
    binary_mask.save_and_plot_binary_mask(b.name, b.binary_mask, params, results_dir)

    # use the binary mask to compute mean activity over time
    # except here we will re-load the videos, overwriting video attribute, and normalize each video using spontaneous activity during first params['background_frames']
    b.video = video_IO.load_videos(b.vid_fnames, params, normalize=True)
    b.mean_activity = binary_mask.mean_activity_within_binary_mask(b.video, b.binary_mask, params)
    
    # using these mean activities, compute max activity and AUC for each odor in the series
    b.maxs_per_odor, b.argmaxs_per_odor = activity_traces.max_activity_per_odor(b.mean_activity, puffs, params)
    b.aucs_per_odor = activity_traces.activity_auc_per_odor(b.mean_activity, puffs, params)
    
    # repeat the above steps, but subtract the paraffin odor from the mean activity
    b.mean_activity_paraffin_subtracted = activity_traces.subtract_paraffin_response(b.mean_activity, puffs, params)    
    b.maxs_per_odor_paraffin_subtracted, b.argmaxs_per_odor_paraffin_subtracted = activity_traces.max_activity_per_odor(b.mean_activity_paraffin_subtracted, puffs, params)
    b.aucs_per_odor_paraffin_subtracted = activity_traces.activity_auc_per_odor(b.mean_activity_paraffin_subtracted, puffs, params)


# ## Save the mean trace activity as csv file

# In[ ]:


mean_activity = {b.name : b.mean_activity for i,b in enumerate(brains)}
mean_activity_df = pd.DataFrame.from_dict(mean_activity)
mean_activity_df.to_csv(f'{results_dir}/mean_activity_within_mask.csv', index=False)


# ## Convert maxs per odor to a Pandas DataFrame and then save as a CSV file

# In[ ]:


def convert_to_df(brains, puffs, metric):
    df_list = []
    for i,b in enumerate(brains):
        df_tmp = pd.DataFrame.from_dict(getattr(b, metric))
        df_tmp['samp'] = b.name
        df_tmp['subpop'] = b.name.split('_')[1]
        df_tmp['trial'] = df_tmp.index+1
        df_list.append(df_tmp)
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df = pd.melt(df, id_vars=['samp', 'subpop', 'trial'], var_name='odor', value_name='value')

    odor_order = {}
    for puff in puffs:
        if puff.trial == 1:
            odor_order[puff.odor_name] = puff.number

    df['odor_order'] = df['odor'].map(odor_order)
    return df

peak_max_df = convert_to_df(brains, puffs, 'maxs_per_odor')
peak_auc_df = convert_to_df(brains, puffs, 'aucs_per_odor')
peak_max_df.to_csv(f'{results_dir}/peak_max.csv', index=False)
peak_auc_df.to_csv(f'{results_dir}/peak_auc.csv', index=False)

# repeat for paraffin subtracted
peak_max_df = convert_to_df(brains, puffs, 'maxs_per_odor_paraffin_subtracted')
peak_auc_df = convert_to_df(brains, puffs, 'aucs_per_odor_paraffin_subtracted')
peak_max_df.to_csv(f'{results_dir}/peak_max_paraffin_subtracted.csv', index=False)
peak_auc_df.to_csv(f'{results_dir}/peak_auc_paraffin_subtracted.csv', index=False)


# ## Make a pretty plot of all the activity traces

# In[ ]:


def make_mean_activity_plot(brains, params, metric):

    fig, axs = plt.subplots(1, 1, figsize=(16, 4))

    for i,b in enumerate(brains):

        activity = getattr(b, metric)
        plt.plot(activity + i*0.02, c='black')  # Offset each trace by i*3
        # print sample name on the right
        plt.text(len(activity)*1.02, i*0.02, b.name, color='black')

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

    plt.savefig(f'{results_dir}/{metric}.png', dpi=300)
    plt.close()

make_mean_activity_plot(brains, params, metric='mean_activity')
make_mean_activity_plot(brains, params, metric='mean_activity_paraffin_subtracted')


# In[ ]:


# iterate through attributes of brains[0] and print the type
for attr in dir(brains[0]):
    if not attr.startswith('__'):
        print(f'{attr}: {type(getattr(brains[0], attr))}')

