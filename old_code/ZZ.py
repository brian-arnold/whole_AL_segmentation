def ZZ_calculate_area_peakDrop_once(fluo, real_time, search_idx, baseline_sd):
    """calculate area for calcium df/f trace using the peak-drop method
    fluo_search: 1-d array, df/f curve
    real_time_search: 1-d array, corresponding time points, unit is seconds
    search_idx: where to expect the peaks
    baseline_sd: float, standard deviation of the baseline, used to define boundaries
    return: List, [area, peak_idx, left_bound, right_bound]"""
    
    # define the region of searching 
    fluo_search = fluo[search_idx]
    real_time_search = real_time[search_idx]
    
    # the peak could be excitation or inhibition peak
    # find the point with maximal absolute value
    idx_max = np.argmax(np.abs(fluo_search))
    # convert to original scale
    peak_idx = idx_max + search_idx[0]
    # check if the peak is excitation or inhibition
    peak_value = fluo[peak_idx]
    ## define the entire peak by extending from the highest/lowest point
    # define the peaks by extending from the max point until drops to the noise level (1.96*sd)
    if peak_value >= 0:
        peak_type = 'excitation'
        # find the left boundary
        boundary_value = 1.96*baseline_sd
        for left_bound in range(peak_idx, -1, -1):
            if fluo[left_bound] < boundary_value:
                break
        # find the right boundary
        for right_bound in range(peak_idx, len(fluo), 1):
            if fluo[right_bound] < boundary_value:
                break
    else:
        peak_type = 'inhibition'
        # find the left boundary
        boundary_value = -1.96*baseline_sd
        for left_bound in range(peak_idx, -1, -1):
            if fluo[left_bound] > boundary_value:
                break
        # find the right boundary
        for right_bound in range(peak_idx, len(fluo), 1):
            if fluo[right_bound] > boundary_value:
                break
    # check if the fluroscence drops or not after the highest/lowest point
    # if doesn't drop, it's likely to be a fake peak
    # use linear fit to check if it drops or not
    x = np.arange(0, right_bound-peak_idx)
    yn = fluo[peak_idx:right_bound]
    flag_fake_peak = 0
    if len(x)<=1:
        flag_fake_peak = 1
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,yn)
        if peak_type == 'excitation':
            if slope>=0 or p_value>=0.1:
                flag_fake_peak = 1
        if peak_type == 'inhibition':
            if slope<=0 or p_value>=0.1:
                flag_fake_peak = 1
    ## calculate area by integrating the defined peak
    # return 0 if consider peak as fake
    if flag_fake_peak==1:
        area = 0
    else:
        area = np.trapz(fluo[left_bound:right_bound], real_time[left_bound:right_bound], axis=0)
   
    return([area, peak_idx, left_bound, right_bound])
def ZZ_calculate_area_peakDrop(fluo, real_time, interval=[0,60], markes=False, plot_out=False, return_points=False):
    """given a calcium trace, calculate the area under curve by the peak drop method
    fluo: 1-d array, fluorescence trace, unit is df/f, baseline should be 0
    real_time: 1-d array, real time corresponds to each point in the fluo trace
    interval: List, where to search for the highest/lowest point in a peak, unit is seconds in real time
    markes: Logical, whether it's markes puff, if so check possibility of multiple peaks
    plot_out: Logical, whether to plot the trace and landmark points
    return: Float, total area of identified peaks, unit for area is df/f * second"""
    # first find the major peak; then mask it, run the same algorithm on its left or right to check for possible peaks
    # define the initial searching region
    left = np.where(real_time>=interval[0])
    right = np.where(real_time<=interval[1])
    search_idx = np.intersect1d(left, right)
    # calculate the sd of baseline 
    baseline_range = real_time<0
    baseline_sd = np.std(fluo[baseline_range])
    # search for the 1st peak
    [area1, peak_idx1, left_bound1, right_bound1] = ZZ_calculate_area_peakDrop_once(fluo, real_time, search_idx, baseline_sd)
    if markes:
        # for Markes, mask the 1st peak; 
        # run the same algorithm again for remaing search region on the left and right
        flag_left = 0
        flag_right = 0
        area2 = 0
        area3 = 0
        # left side
        if left_bound1>search_idx[0]:
            # left side is bounded by time zero and left_bound of 1st peak
            flag_left = 1
            fluo_left = fluo[search_idx[0]:left_bound1]
            real_time_left = real_time[search_idx[0]:left_bound1]
            search_idx_left = np.arange(len(fluo_left))
            [area2, peak_idx2, left_bound2, right_bound2] = ZZ_calculate_area_peakDrop_once(fluo_left, real_time_left, search_idx_left, baseline_sd)
            # convert the index into the original scale
            peak_idx2 += search_idx[0]
            left_bound2 += search_idx[0] 
            right_bound2 += search_idx[0]
        # right side
        if right_bound1<search_idx[-1]:
            # right left is bounded by right_bound of 1st peak and last time point
            flag_right = 1
            fluo_right = fluo[right_bound1:]
            real_time_right = real_time[right_bound1:]
            search_idx_right = np.arange(0, search_idx[-1]-right_bound1, 1)
            [area3, peak_idx3, left_bound3, right_bound3] = ZZ_calculate_area_peakDrop_once(fluo_right, real_time_right, search_idx_right, baseline_sd)
            # convert the index into the original scale
            peak_idx3 += right_bound1
            left_bound3 += right_bound1 
            right_bound3 += right_bound1
        if plot_out:
            fig, ax = plt.subplots()
            ax.plot(real_time, fluo)
            #     ax.plot(real_time_search[area_idx], fluo_search[area_idx], color='red')
            ax.plot(real_time[peak_idx1], fluo[peak_idx1], marker='o', color='blue')
            ax.plot(real_time[left_bound1], fluo[left_bound1], marker='o', color='green')
            ax.plot(real_time[right_bound1], fluo[right_bound1], marker='o', color='green')
            if flag_left:
                ax.plot(real_time[peak_idx2], fluo[peak_idx2], marker='+', color='blue')
                ax.plot(real_time[left_bound2], fluo[left_bound2], marker='+', color='green')
                ax.plot(real_time[right_bound2], fluo[right_bound2], marker='+', color='green')
            if flag_right:
                ax.plot(real_time[peak_idx3], fluo[peak_idx3], marker='x', color='blue')
                ax.plot(real_time[left_bound3], fluo[left_bound3], marker='x', color='green')
                ax.plot(real_time[right_bound3], fluo[right_bound3], marker='x', color='green')
#         print(area1, area2, area3)
        return(area1+area2+area3)
    else:
        # for single odorants, just return 1st peak
        if plot_out:
            fig, ax = plt.subplots()
            ax.plot(real_time, fluo)
            #     ax.plot(real_time_search[area_idx], fluo_search[area_idx], color='red')
            ax.plot(real_time[peak_idx1], fluo[peak_idx1], marker='o', color='blue')
            ax.plot(real_time[left_bound1], fluo[left_bound1], marker='o', color='green')
            ax.plot(real_time[right_bound1], fluo[right_bound1], marker='o', color='green')
        if return_points:
            return(area1, [peak_idx1, left_bound1, right_bound1])
        else:
            return(area1)
def ZZ_calculate_area_volume_peakDrop(fn, movie_info, markes=False, channel=0, imaging_freq=3.76, pre_puff=7, search_interval=[0,15], threshold=10, plot_out=False, remove_ends=5):
    """from calcium imaging movie, calculate an area volume by 'peak drop' method
    fn: String, file name of the movie, full path
    movie_info: Dict, xyz size, number of channels and etc, used to reshape the np array
    imaging_freq: Float, volumetric imaging rate, used to calculate real time
    pre_puff: Int, how many seconds before puffing were recorded, used to identify time zero
    search_interval: List, when to expect the highest point in a peak
    drop_level: Float, extend from the highest point until fluorescence drops to this level
    plot_out: Logical, whether to plot the fluorescence trace and identification of landmark points
    remove_ends: Int, how many volumes to ignore when calculate baseline
    return: 3d-array, area volume"""
    # parameters that are relatively fixed
    # filter size to smooth the data
    gaussian_spatial = [4,4,2]
    gaussian_temporal = 3
    ## read in the movie
    vol = ZZ_read_reshape_tiff(fn, movie_info)
    # only use the green channel
    movie = np.squeeze(vol[:,:,channel,:,:])
    ## smoothing to remove noise
    smoothed2 = ZZ_smooth_movies(movie, gaussian_spatial=gaussian_spatial, gaussian_temporal=gaussian_temporal)
    ## define a local function to call the peak drop method
    def ZZ_calculate_area_local(trace):
        # calculate df/f
        # at what time point valve open
        time_zero = int(imaging_freq * pre_puff)
        # calculate baseline fluorescence, excluding the 1st and last 5 volumes
        baseline = np.mean(trace[remove_ends:(time_zero-remove_ends)])
        # ignore voxels those baseline fluorescence is smaller than threshold
        if baseline>=threshold:
            dff = trace/baseline - 1
            # calculate real time for each volume based on volumetric imaging rate
            real_time = np.arange(trace.shape[0])/imaging_freq - pre_puff
            # calculate area by find the highest point, then drop to noise level
            area = ZZ_calculate_area_peakDrop(dff, real_time, interval=search_interval, markes=markes, plot_out=plot_out)
        else:
            area = 0
        return(area)
    ## calculate the area volume 
    area = np.zeros(list(smoothed2.shape[0:3]))
    # loop through each pixel
    for xx in range(smoothed2.shape[0]):
        for yy in range(smoothed2.shape[1]):
            for zz in range(smoothed2.shape[2]):
                trace = smoothed2[xx, yy, zz, :]
                area_temp = ZZ_calculate_area_local(trace)
                area[xx, yy, zz] = area_temp
    # smooth the final area volume
    area_smoothed = ndimage.gaussian_filter(area, gaussian_spatial)
    return(area_smoothed)
def ZZ_calculate_area_volume_fixedInterval(fn, movie_info, channel=0, imaging_freq=3.76, pre_puff=30, interval_baseline_markes=[-25,-5], interval_peak_markes=[0,120], threshold=10):
    """given an odor-evoked movie, calculate an area volume by integrating over a fixed time interval
    fn: String, file name of the movie, full path
    movie_info: Dict, xyz size, number of channels and etc, used to reshape the np array
    imaging_freq: Float, volumetric imaging rate, used to calculate real time
    pre_puff: Int, how many seconds before puffing were recorded, used to identify time zero
    interval_baseline_markes: List, what time interval to calculate baseline fluorescence, unit is seconds
    interval_peak_markes: List, what time interval to integrate for area, unit is seconds
    threshold: Int, ignore voxels whose baseline is smaller than threshold
    return: 3d-array, area volume"""
    # parameters that are relatively fixed
    # filter size to smooth the data
    gaussian_spatial = [4,4,2]
    gaussian_temporal = 3
    ## read in the movie
    vol = ZZ_read_reshape_tiff(fn, movie_info)
    # only use the green channel
    movie = np.squeeze(vol[:,:,channel,:,:])
    ## smoothing to remove noise
    smoothed2 = ZZ_smooth_movies(movie, gaussian_spatial=gaussian_spatial, gaussian_temporal=gaussian_temporal)
    ## calculate df/f 
    # calculate real time
    real_time = np.arange(smoothed2.shape[3])/imaging_freq - pre_puff
    # calculate a baseline volume
    # convert real time into index
    left = np.where(real_time>=interval_baseline_markes[0])
    right = np.where(real_time<=interval_baseline_markes[1])
    baseline_range = np.intersect1d(left, right)
    left = np.where(real_time>=interval_peak_markes[0])
    right = np.where(real_time<=interval_peak_markes[1])
    area_range = np.intersect1d(left, right)
    baseline = np.mean(smoothed2[:,:,:,baseline_range], axis=3)
    # remove non-AL pixels by thresholding
    idx_good = baseline > threshold
    # calculate df/f for each time point
    dff_all = np.zeros(smoothed2.shape)
    # reshape baseline volume for broadcasting
    baseline = baseline[idx_good].reshape(-1,1)
    dff_all[idx_good,:] = smoothed2[idx_good, :] / baseline - 1
    ## calculate area by integrating
    area = np.trapz(dff_all[:,:,:,area_range], axis=3)
    # divide area with imaging frequency, so unit for area is df/f * second
    area = area / imaging_freq
    # smooth the area volume with Gaussian filter 
    area_smoothed = ndimage.gaussian_filter(area, gaussian_spatial)
    return(area_smoothed)