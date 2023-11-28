import numpy as np


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
