from dataclasses import dataclass
from collections import defaultdict

# Names of samples.
# It is assumed there exists a directory (specified below in `data_dir`) which contains a subdirectory for each sample, named according to the strings in this list.
# Within each of these subdirectories there should be .tif files that are also prefixed with the sample name.
samples = ['230913_ORL_GCaMP6f_F1', 
           '230913_ORL_GCaMP6f_F2', 
           '230914_ORL_GCaMP6f_F1', 
           '230914_ORL_GCaMP6f_F2', 
           '230915_ORL_GCaMP6f_F1', 
           '230913_U52_GCaMP6f_F2', 
           '230913_U52_GCaMP6f_F3', 
           '230914_U52_GCaMP6f_F1', 
           '230914_U52_GCaMP6f_F2', 
           '230915_U52_GCaMP6f_F2', 
           '230913_FCV_GCaMP6f_F1', 
           '230914_FCV_GCaMP6f_F1', 
           '230914_FCV_GCaMP6f_F2', 
           '230914_FCV_GCaMP6f_F3', 
           '230915_FCV_GCaMP6f_F1']

# The directory containing the data for each sample.
data_dir = "/jukebox/mcbride/bjarnold/new_analysis/data/Mar_22_2024/1_RegisteredBrains"

# the dimensions of the 3D volumes imaged over time
x_dim, y_dim, z_dim = 128, 128, 24

# The number of frames to anlayze from each odor recording. E.g. if 112, then the first 112 frames are used while the rest are discarded.
# This number should be the minimum number of frames across all recordings, to avoid having different number of frames for different odors.
n_frames_to_analyze = 112 

# the number of initial frames to use for df/f normalization. E.g. if 20, then the mean intensity of the first 20 frames are used to normalize the rest of the frames.
background_frames = 20

# An underscore separated list of encoded odors in the exact order to which they were delivered to the mosquito samples.
# It is assumed that this is the order in which odors are delivered and that videos (.tif files) are named alphanumerically according to this order.
odor_string = "100U_60U_10U_100T_30T_10T_100R_30R_10R_100S_10S_100P_30P_10P_100Q_10Q_100N_30N_10N_100O_10O_100L_50L_10L_100M_10M_100J_100A_100B_100C_100D_100E_100F_100G_100H_100I_100U_60U_10U_100T_30T_10T_100R_30R_10R_100S_10S_100P_30P_10P_100Q_10Q_100N_30N_10N_100O_10O_100L_50L_10L_100M_10M_100J_100A_100B_100C_100D_100E_100F_100G_100H_100I"

# A dictionary to convert the encoded odors to their actual names.
odor_encodings = {'100A': 'sulcatone-2', 
                '100B': 'sulcatone-3', 
                '100C': 'phenol', 
                '100D': 'benzaldehyde', 
                '100E': 'oxoisophorone', 
                '100F': '2-ethylhexanol', 
                '100G': 'hexanoic acid', 
                '100H': 'camphor', 
                '100I': '1-octen-3-ol', 
                '100J': 'paraffin', 
                '100L': 'undecanal4.145', 
                '50L': 'undecanal3.9', 
                '10L': 'undecanal3.205', 
                '100M': 'undecanal2.765', 
                '10M': 'undecanal2.375', 
                '100N': 'decanal4.53', 
                '30N': 'decanal4.015', 
                '10N': 'decanal3.425', 
                '100O': 'decanal2.855', 
                '10O': 'decanal2.18', 
                '100P': 'nonanal4.56', 
                '30P': 'nonanal4.045', 
                '10P': 'nonanal3.445', 
                '100Q': 'nonanal2.77', 
                '10Q': 'nonanal1.72', 
                '100R': 'octanal4.58', 
                '30R': 'octanal4.09', 
                '10R': 'octanal3.525', 
                '100S': 'octanal3.09', 
                '10S': 'octanal1.595',
                '100T': 'heptanal4.51',
                '30T': 'heptanal4.065', 
                '10T': 'heptanal3.44', 
                '100U': 'hexanal4.375', 
                '60U': 'hexanal4.03', 
                '10U': 'hexanal2.825'}


# store information about each odor in a class
@dataclass
class Puff:
    # this class keeps track of information about each odor puff
    odor_name: str
    odor_name_encoded: str
    trial: int # whether this was the first or second time all odors were presented
    number: int # 0-based puff number according to the order in odor_string above, this number is used to grab the corresponding video, e.g. if number=0, then the video is the first video in the list of videos
    
    def __str__(self):
        return f"{self.odor_name}-{self.puff_number}"

puffs = []
trial = defaultdict(int)
for i, odor in enumerate(odor_string.split('_')):
    trial[odor] += 1
    puffs.append(Puff(odor_encodings[odor], odor, trial[odor], i))


params = {}
params['x_dim'] = x_dim
params['y_dim'] = y_dim
params['z_dim'] = z_dim
params['n_frames_to_analyze'] = n_frames_to_analyze
params['background_frames'] = background_frames
