import pytest
import sys
import os
import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment_info import samples, data_dir
from experiment_info import odor_encodings, odor_string
from experiment_info import params


def test_data_dir():
    assert os.path.isdir(data_dir), f'Data directory does not exist: {data_dir}'


def test_samples():
    for samp in samples:
        assert os.path.isdir(f"{data_dir}/{samp}"), f"directory {data_dir}/{samp} containing tif files does not exist! Is there one approppriately-named subdirectory per sample?"


def test_odor_info():
    odor_list = odor_string.split('_')
    for odor in odor_list:
        assert odor in odor_encodings, f"Odor {odor} in odor_list not found in odor_encodings dictionary. Please add it to experiment_info.py"


def test_number_of_videos():
    num_odors = len(odor_string.split('_'))
    for samp in samples:
        v = glob.glob(f"{data_dir}/{samp}/*.registered.tif")
        num_vids = len(v)
        assert num_vids == num_odors, f"I found {num_vids} videos for sample {samp}, but there are {num_odors} odors. The number of videos and odors should be equivalent."
