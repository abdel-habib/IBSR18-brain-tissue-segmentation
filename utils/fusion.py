import numpy as np


def majority_voting_fusion(labels_dirs, load_nifti_callback):
    '''
    This step is performed after the registration and label propagation to fuse all given labels to a single propabilistic atlas.

    Args:
        labels_dirs ('list'): list of labels directories to be fused togather.

    Returns:
        fused ('np.array'): A numpy array that holds the fused atlas.
    '''

    # place holder for each tissue to create the atlas separately
    _CSF = [] # 1
    _GM  = [] # 2
    _WM  = [] # 3
    
    for dir in labels_dirs:
        # load the volumes
        nifti_volume = load_nifti_callback(dir)[0]

        # select each tissue by its label
        nifti_volume_CSF = nifti_volume == 1
        nifti_volume_GM  = nifti_volume == 2
        nifti_volume_WM  = nifti_volume == 3

        # group labels into their place holders
        _CSF.append(nifti_volume_CSF)
        _GM.append(nifti_volume_GM)
        _WM.append(nifti_volume_WM)

    # get the mean of the three tissues
    mean_csf = np.mean(_CSF, axis=0)
    mean_gm = np.mean(_GM, axis=0)
    mean_wm = np.mean(_WM, axis=0)

    return mean_csf, mean_gm, mean_wm
