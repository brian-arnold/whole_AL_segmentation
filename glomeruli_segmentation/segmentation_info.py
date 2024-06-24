'''
TL;DR -> Change odors_to_select below to alter which odors are considered in the analysis. Leave empty to consider all odors.

This module contains information about the segmentation of the data that is used over multiple notebooks, 
so it's placed here to avoid redundancy.

The primary goal of this module is to only consider select puffs from the puffs list, to e.g. segment and 
analyze the data using only a subset of puffs.
'''

import sys
sys.path.insert(0, '../')
from experiment_info import puffs

# this list contains the lowest 3 concentrations of each aldehyde, 
# followed by the negative control paraffin
# followed by reference odorants
odors_to_select = ['octanal3.525', 
                    'octanal3.09', 
                    'octanal1.595',
                    'nonanal3.445', 
                    'nonanal2.77', 
                    'nonanal1.72',
                    'decanal3.425', 
                    'decanal2.855', 
                    'decanal2.18', 
                    'undecanal3.205', 
                    'undecanal2.765', 
                    'undecanal2.375', 
                    'paraffin',
                    'sulcatone-3',
                    'phenol',
                    'benzaldehyde',
                    'oxoisophorone',
                    '2-ethylhexanol',
                    'hexanoic acid',
                    'camphor',
                    '1-octen-3-ol']

if len(odors_to_select) == 0:
    puffs_filt = puffs
else:
    puffs_filt = [p for p in puffs if p.odor_name in odors_to_select]