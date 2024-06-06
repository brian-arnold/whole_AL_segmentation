import h5py
import numpy as np
from scipy.stats import pearsonr


def read_metainfo(metafile):
    samples = []
    with open(metafile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        sample= line.strip().split(';')[0]
        samples.append(sample)
    return samples

def load_h5(filename):
    with h5py.File(filename, 'r') as file:
        dataset = file['data'] # Access the dataset containing the 3D matrix
        mat = np.array(dataset)
    return mat

def load_C_X_matrices(base_data_dir, samples, k, power, seg):
    X_mats = []
    C_mats = []
    for s in samples:
        X_file = f'{base_data_dir}/{s}/svd_k_{k}/cluster_power_{power}/segments_{seg}/X.h5'
        C_file = f'{base_data_dir}/{s}/svd_k_{k}/cluster_power_{power}/segments_{seg}/C.h5'
        X_mats.append(load_h5(X_file))
        C_mats.append(load_h5(C_file))
    return X_mats, C_mats

def calculate_pairwise_temporal_corrs(C1, C2):
    # compute the full correlation matrix between two matrices of temporal components
    assert C1.shape == C2.shape
    num_segments = C1.shape[0]
    corr_mat = np.zeros((num_segments, num_segments))

    for i in range(num_segments):
        for j in range(num_segments):  
            correlation, _ = pearsonr(C1[i,:], C2[j,:])
            corr_mat[i, j] = correlation
            corr_mat[j, i] = correlation  # Symmetric

    return corr_mat


def convert_to_3d_coords(X, z_dim, x_dim, y_dim):
    X_3d = np.reshape(X, (z_dim, x_dim, y_dim))
    z, x, y = np.nonzero(X_3d)
    coords_seg = np.column_stack((z, x, y))
    return coords_seg

def calc_IoU(set_a, set_b):
    # Convert array of 3D points to tuples for each set to enable set operations
    tuples_a = set(map(tuple, set_a))
    tuples_b = set(map(tuple, set_b))
    # Calculate intersection and union
    intersection = tuples_a.intersection(tuples_b)
    union = tuples_a.union(tuples_b)
    # Calculate IoU
    iou = len(intersection) / len(union) if len(union) > 0 else 0
    return iou

def calculate_pairwise_IoU(X1, X2, z_dim, x_dim, y_dim):
    assert X1.shape == X2.shape
     # for X matrices, rows are pixels, columns are segments
    num_segments = X1.shape[1]
    IoU_mat = np.zeros((num_segments, num_segments))
    for i in range(num_segments):
        coords_seg_i = convert_to_3d_coords(X1[:,i], z_dim, x_dim, y_dim)
        for j in range(num_segments):  
            coords_seg_j = convert_to_3d_coords(X2[:,j], z_dim, x_dim, y_dim)
            IoU = calc_IoU(coords_seg_i, coords_seg_j)
            IoU_mat[i, j] = IoU
            IoU_mat[j, i] = IoU  # Symmetric
    return IoU_mat




def get_max_coords(X, i, z_dim, x_dim, y_dim):
    X_i = X[:,i]
    X_i = np.reshape(X_i, (z_dim, x_dim, y_dim))
    z_max, x_max, y_max = np.unravel_index(np.argmax(X_i, axis=None), X_i.shape)
    return z_max, x_max, y_max


def calculate_pairwise_dists(X1, X2, x_dim, y_dim, z_dim):
    # compute the full correlation matrix between two matrices of temporal components
    assert X1.shape == X2.shape
    num_segments = X1.shape[1]
    dist_mat = np.zeros((num_segments, num_segments))

    for i in range(num_segments):
        for j in range(num_segments):  
            z_i, x_i, y_i = get_max_coords(X1, i, z_dim, x_dim, y_dim)
            z_j, x_j, y_j = get_max_coords(X2, j, z_dim, x_dim, y_dim) 
            Pi = np.array([z_i, x_i, y_i])
            Pj = np.array([z_j, x_j, y_j])
            distance = np.sqrt(np.sum((Pi - Pj)**2))

            dist_mat[i, j] = distance
            dist_mat[j, i] = distance  # Symmetric

    return dist_mat




def convert_to_3d_coords(X, z_dim, x_dim, y_dim):
    X_3d = np.reshape(X, (z_dim, x_dim, y_dim))
    z, x, y = np.nonzero(X_3d)
    coords_seg = np.column_stack((z, x, y))
    return coords_seg

def calc_intersection_fraction(set_a, set_b):
    # Convert array of 3D points to tuples for each set to enable set operations
    tuples_a = set(map(tuple, set_a))
    tuples_b = set(map(tuple, set_b))
    # Calculate intersection and union
    intersection = tuples_a.intersection(tuples_b)
    # Calculate IoU
    intersection_fraction = len(intersection) / len(tuples_a) if len(tuples_a) > 0 else 0
    return intersection_fraction

def calculate_intersections_per_seg(X1, z_dim, x_dim, y_dim):
     # for X matrices, rows are pixels, columns are segments
    num_segments = X1.shape[1]
    intersection_fractions = np.zeros(num_segments)
    # for each segment in X1, find coordinates in 
    for i in range(num_segments):
        coords_seg_i = convert_to_3d_coords(X1[:,i], z_dim, x_dim, y_dim)
        # combine coordinates of all other segments, intersect with segment i
        coords_seg_rest = np.zeros((0, 3))
        for j in range(num_segments):  
            if i == j:
                continue
            coords_seg_j = convert_to_3d_coords(X1[:,j], z_dim, x_dim, y_dim)
            coords_seg_rest = np.vstack((coords_seg_rest, coords_seg_j))
        intersection_frac = calc_intersection_fraction(coords_seg_i, coords_seg_rest)
        intersection_fractions[i] = intersection_frac

    return intersection_fractions