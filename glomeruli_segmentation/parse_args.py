import sys
import sys
sys.path.insert(0, '../')
from experiment_info import puffs
import argparse

def parse_args():
    description = "Run CaImAn"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--sample_index', type=int, required=True, help="the index of the sample to process, from the list of samples specified in experiment_info.py")
    parser.add_argument('--K', type=int, required=True, help="Number of segments")
    parser.add_argument('--gSig', type=int, nargs=3, required=True, help="gSig is a list of 3 sapce-separated integers specifying the half-width of neurons in x,y,z")
    parser.add_argument('--odor_file', type=str, required=False, default=None, help="a file containing the odors you want to analyze, with one odor on each line")
    parser.add_argument('--out_dir', type=str, required=False, default='./', help="directory to store output")

    return parser.parse_args()

args = parse_args()


# If an odor file is not specified, just use all odors
if args.odor_file == None:
    odors_to_select = []
    puffs_filt = puffs
else:
    with open(args.odor_file, 'r') as f:
        odors_to_select = [line.strip() for line in f.readlines() if line != '\n']
    puffs_filt = [p for p in puffs if p.odor_name in odors_to_select]