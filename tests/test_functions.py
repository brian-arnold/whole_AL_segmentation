"""
Regression tests for functions.py using the first sample in the experiment_info.py file
"""

import pytest
import sys
import os
import glob
import caiman as cm
import pickle
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from experiment_info import samples, data_dir, puffs, params
import functions as fn


@pytest.fixture
def vids_unnormalized():
    vids = glob.glob(f"{data_dir}/{samples[0]}/*.registered.tif")
    vids = sorted(vids)
    vid_list = fn.load_videos_into_list(vids, params, normalize=False)
    return vid_list

@pytest.fixture
def vids_normalized():
    vids = glob.glob(f"{data_dir}/{samples[0]}/*.registered.tif")
    vids = sorted(vids)
    vid_list = fn.load_videos_into_list(vids, params, normalize=True)
    return vid_list

@pytest.fixture
def test_binary_mask(vids_unnormalized):
    Y = cm.concatenate(vids_unnormalized)
    binary_mask = fn.find_binary_mask(Y)
    with open('tests/data/binary_masks_sample0.pkl', 'rb') as f:
        binary_mask_dict = pickle.load(f)
        binary_mask_expected = binary_mask_dict[samples[0]]
    assert (binary_mask == binary_mask_expected).all()
    return binary_mask

@pytest.fixture
def test_mean_activity(vids_normalized, test_binary_mask):
    Y = cm.concatenate(vids_normalized)
    mean_activity_within_segment = {}
    mean_activity_within_segment[samples[0]] = fn.extract_mean_activity_within_binary_mask(Y, test_binary_mask, params)
    df = pd.read_csv('tests/data/mean_activity_within_segment_sample0.csv')
    mean_activity_expected = np.array(df.iloc[:,0])
    # assert that the first column of df_expected is the same as mean_activity_within_segment
    assert np.allclose(mean_activity_within_segment[samples[0]], mean_activity_expected)
    return mean_activity_within_segment

def test_max_response(test_mean_activity):
    max_responses, _ = fn.compute_max_responses(test_mean_activity, puffs, params['n_frames_to_analyze'])
    max_responses_df = fn.convert_to_df(max_responses, puffs)
    observed = max_responses_df.value

    expected_df = pd.read_csv('tests/data/peak_max_df_sample0.csv')
    expected = np.array(expected_df.value)
    # assert that the first column of df_expected is the same as max_responses
    assert np.allclose(observed, expected)



