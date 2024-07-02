######
# These functions take in mean activity traces and output summary statistics or modified traces
######

import numpy as np
from collections import defaultdict

def max_activity_per_odor(mean_activity, puffs, params):
    n_frames_to_analyze = params['n_frames_to_analyze']

    maxs_per_odor = defaultdict(list)
    argmaxs_per_odor = defaultdict(list) # this is to find the frame with maximum activity
    
    for i,puff in enumerate(puffs):
        odor_name = puff.odor_name
        # since a subset of videos were concatenates, use index i of odor_of_interest_indices to get the corresponding interval
        interval = mean_activity[i*n_frames_to_analyze : (i+1)*n_frames_to_analyze]

        maxs_per_odor[odor_name].append(np.max(interval))
        argmaxs_per_odor[odor_name].append(np.argmax(interval))

    for odor in maxs_per_odor:
        assert len(maxs_per_odor[odor]) <= 2, f" for odor {odor} there were more than 2 trials. This is unexpected."
        assert len(argmaxs_per_odor[odor]) <= 2, f" for odor {odor} there were more than 2 trials. This is unexpected."

    return maxs_per_odor, argmaxs_per_odor

def activity_auc_per_odor(mean_activity, puffs, params):

    n_frames_to_analyze = params['n_frames_to_analyze']
    background_frames = params['background_frames']

    aucs_per_odor = defaultdict(list)
    # for each odor, extract max value from the corresponding interval
    for i, puff in enumerate(puffs):
        # get odor name
        odor_name = puff.odor_name

        # since a subset of videos were concatenates, use index i of odor_of_interest_indices to get the corresponding interval
        interval = mean_activity[i*n_frames_to_analyze : (i+1)*n_frames_to_analyze]

        # baseline= 1.96*np.std(interval[0:background_frames])
        baseline= np.median(interval[0:background_frames])
        peak_coord = np.argmax(interval)

        # get all indices of interval where activity is above baseline
        above_baseline = np.where(interval > baseline)[0]
        # find sets of indices in above_baseline that are consecutive, if not consecutive, then activity dropped below baseline
        not_consecutive = np.where(np.diff(above_baseline) != 1)[0]
        # an extra 1 is added to account for fact that np.diff returns array that is 1 element shorter than the input array, so add 1 to split at correct indices
        split_indices = np.split(above_baseline, not_consecutive+1)

        # find interal that contains the peak index
        for s in split_indices:
            if peak_coord in s:
                start = s[0]  # indices are for where activity is above baseline, so start 1 before
                end = s[-1] + 1 # indices are for where activity is above baseline, so end 1 after, and add extra 1 to include the last index
                peak_interval = interval[start:end]
                break

        # calculate area under the curve
        auc = np.trapz(peak_interval, dx=1)
        aucs_per_odor[odor_name].append(auc)

        # if test and i == 4 and self.name == list(activity_traces.keys())[0]:
        #     plt.plot(interval)
        #     plt.axhline(baseline, color='black', linestyle='--')
        #     plt.axvline(peak_coord, color='red')
        #     plt.axvline(start, color='green')
        #     plt.axvline(end, color='green')
        #     print(auc)

    for odor in aucs_per_odor:
        assert len(aucs_per_odor[odor]) <= 2, f" for odor {odor} there were more than 2 trials. This is unexpected."

    return aucs_per_odor
    
def subtract_paraffin_response(mean_activity, puffs, params):

    n_frames_to_analyze = params['n_frames_to_analyze']
    num_odors = len(puffs)

    # find indices of the odors delivered that correspond to paraffin, should be 2 for 2 trials
    paraffin_indices = [p.number for p in puffs if p.odor_name == "paraffin"]
    assert len(paraffin_indices) == 2, f"Expected to find 2 paraffin trials, but found {len(paraffin_indices)}"
    assert paraffin_indices[0] < paraffin_indices[1], f"Expected paraffin trial 1 to come before paraffin trial 2, but found {paraffin_indices}"

    new_traces = []
    # collect paraffin traces for this sample, for both trials; in paraffin_traces dict, keys are trials
    paraffin_traces = {}
    for i, index in enumerate(paraffin_indices):
        interval = mean_activity[index*n_frames_to_analyze:(index+1)*n_frames_to_analyze]
        paraffin_traces[i] = interval
    
    # subtract paraffin traces from each odor trace
    for puff in puffs:
        i = puff.number
        interval = mean_activity[i*n_frames_to_analyze:(i+1)*n_frames_to_analyze]
        # if this odor was delivered in the first half, subtract paraffin trace from first trial
        if i <= (num_odors/2) - 1:
            interval_subtracted = interval - paraffin_traces[0]
        else:
            interval_subtracted = interval - paraffin_traces[1]
        new_traces.extend(interval_subtracted)

    mean_activity_paraffin_subtracted = np.array(new_traces)
    return mean_activity_paraffin_subtracted