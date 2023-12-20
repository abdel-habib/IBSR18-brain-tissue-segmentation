import numpy as np
from medpy.metric.binary import hd

def mutual_information(img_reg, img_fixed):
    """Mutual information between two nifti images.
    Args:
        img_reg ('nibabel'): Registered data image.
        img_fixed ('nibabel'): Fixed data image.
    Returns:
        float: Mutual information.
    """
    hist_2d, x_edges, y_edges = np.histogram2d(img_reg.astype(np.double).ravel(),
                                               img_fixed.astype(np.double).ravel(),bins=20)
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def hausdorff_distance(label_volume1, label_volume2, labels=None, voxelspacing=None):
    """
    Calculate the Hausdorff distance between two label volumes.

    Parameters:
    - label_volume1: 3D numpy array representing the first label volume.
    - label_volume2: 3D numpy array representing the second label volume.
    - labels ('dict'): Dictionary mapping tissue types to labels.

    Returns:
    - The Hausdorff distance between the two label volumes.
    """
    
    if labels is None or not isinstance(labels, dict):
        raise ValueError("The 'labels' parameter must be a dictionary mapping tissue types to labels.")

    hausdorff_distances = {}

    for tissue_label in ['WM', 'GM', 'CSF']:
        # Extract coordinates of non-zero elements in each label volume
        mask1 = np.where(label_volume1 == labels[tissue_label], True, False)
        mask2 = np.where(label_volume2 == labels[tissue_label], True, False)

        # Calculate the directed Hausdorff distance from points1 to points2
        hd_distance = hd(mask1, mask2, voxelspacing=voxelspacing)

        # Return the maximum of the two directed Hausdorff distances
        hausdorff_distances[tissue_label] = round(hd_distance, 6)

    return hausdorff_distances