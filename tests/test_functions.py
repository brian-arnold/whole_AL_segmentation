"""
Regression tests for functions.py using the first sample in the experiment_info.py file
"""

import pytest
import sys
import os
import glob
import pickle
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_info import samples, data_dir, puffs, params

import tmp_activity_traces as activity_traces
import tmp_video_IO as video_IO
import tmp_binary_mask as bm


def convert_to_df(maxs, name, puffs):
    # df_list = []

    df = pd.DataFrame.from_dict(maxs)
    df['samp'] = name
    df['subpop'] = name.split('_')[1]
    df['trial'] = df.index+1

    df = df.reset_index(drop=True)
    df = pd.melt(df, id_vars=['samp', 'subpop', 'trial'], var_name='odor', value_name='value')

    odor_order = {}
    for puff in puffs:
        if puff.trial == 1:
            odor_order[puff.odor_name] = puff.number

    df['odor_order'] = df['odor'].map(odor_order)
    return df

#######
# Some functions require unnormalized data, others normalized data, 
# so create fixtures for both
#######

@pytest.fixture
def vids_unnormalized():
    vid_fnames = glob.glob(f"{data_dir}/{samples[0]}/*.registered.tif")
    vid_fnames = sorted(vid_fnames)
    vid_list = video_IO.load_videos(vid_fnames, params, normalize=False)
    return vid_list

@pytest.fixture
def vids_normalized():
    vid_fnames = glob.glob(f"{data_dir}/{samples[0]}/*.registered.tif")
    vid_fnames = sorted(vid_fnames)
    vid_list = video_IO.load_videos(vid_fnames, params, normalize=True)
    return vid_list

#######
# test find_binary_mask function but also return the mask for later testing
#######

@pytest.fixture
def test_binary_mask(vids_unnormalized):
    # the binary mask function performs median filtering, so give it unnormalized data
    binary_mask, binary_mask_frac = bm.find_binary_mask(vids_unnormalized, params)
    with open('tests/data/binary_masks_sample0.pkl', 'rb') as f:
        binary_mask_dict = pickle.load(f)
        binary_mask_expected = binary_mask_dict[samples[0]]
    assert (binary_mask == binary_mask_expected).all()
    return binary_mask

@pytest.fixture
def test_mean_activity(vids_normalized, test_binary_mask):
    mean_activity_within_segment = {}
    mean_activity_within_segment = bm.mean_activity_within_binary_mask(vids_normalized, test_binary_mask, params)
    df = pd.read_csv('tests/data/mean_activity_within_segment_sample0.csv')
    mean_activity_expected = np.array(df.iloc[:,0])
    # assert that the first column of df_expected is the same as mean_activity_within_segment
    assert np.allclose(mean_activity_within_segment, mean_activity_expected)
    return mean_activity_within_segment

def test_max_response(test_mean_activity):
    max_responses, _ = activity_traces.max_activity_per_odor(test_mean_activity, puffs, params)

    max_responses_df = convert_to_df(max_responses, samples[0], puffs)
    observed = max_responses_df.value

    expected_df = pd.read_csv('tests/data/peak_max_df_sample0.csv')
    expected = np.array(expected_df.value)
    # assert that the first column of df_expected is the same as max_responses
    assert np.allclose(observed, expected)
