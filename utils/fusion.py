import numpy as np
from typing import List, Callable
from EM import NiftiManager, FileManager
from .evaluate import mutual_information
import SimpleITK as sitk

NM = NiftiManager()
FM = FileManager()

def majority_voting_fusion(
        labels_dirs: List[str] = [],
        load_nifti_callback:Callable = None):
    '''
    Fuse all given labels to a propabilistic atlas for each tissue. This step is performed after 
    the registration and label propagation.

    Args:
        labels_dirs ('list'): list of labels directories to be fused togather.

    Returns:
        mean_csf ('np.array'): A numpy array that holds the CSF fused atlas.
        mean_gm ('np.array'): A numpy array that holds the GM fused atlas.
        mean_wm ('np.array'): A numpy array that holds the WM fused atlas.
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


def weighted_voting_fusion(
        labels_dirs: List[str] = [],
        intensity_dirs: List[str] = [],
        target_intensity_path:str = None,
        load_nifti_callback:Callable = None):
    '''
    Fuse all given labels based on their intensity volume similarity to the target volume.
    This step is performed after the registration and label propagation.

    The target_path is required to load the target volume and use it to calculate the 
    similarity. The labels_dirs and intensity_dirs are required to load the labels and
    the intensity volumes respectively for each of the moving images respectively, where
    the label will be used to perform skull stripping on the intensity volume.

    Args:
        labels_dirs ('list'): list of labels directories to be fused togather.
        intensity_dirs ('list'): list of intensity directories to be fused togather.
        target_path ('str'): path to the target volume.
        load_nifti_callback ('function'): a function that loads the nifti volume.

    Returns:
        mean_csf ('np.array'): A numpy array that holds the CSF fused atlas.
        mean_gm ('np.array'): A numpy array that holds the GM fused atlas.
        mean_wm ('np.array'): A numpy array that holds the WM fused atlas.
    '''
    # load the target volume
    target_volume = load_nifti_callback(target_intensity_path)[0]

    # mutual information for each intensity volume with the target volume
    weights = {}
    for intensity, label in zip(intensity_dirs, labels_dirs):
        # load the volumes
        nifti_volume = load_nifti_callback(intensity)[0]
        nifti_label = load_nifti_callback(label)[0]
        nifti_mask  = np.where(nifti_label == 0, 0, 1)

        # skull strip the intensity volume
        nifti_volume = np.multiply(nifti_volume, nifti_mask)

        weights[intensity] = mutual_information(nifti_volume, target_volume)

    # normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # weighted majority voting
    _CSF = [] # 1
    _GM  = [] # 2
    _WM  = [] # 3

    for intensity_path, weight in weights.items():
        intensity_volume = load_nifti_callback(intensity_path)[0]
        label_volume = load_nifti_callback(labels_dirs[intensity_dirs.index(intensity_path)])[0]

        # Skull strip the intensity volume using the label
        intensity_volume = np.multiply(intensity_volume, np.where(label_volume == 0, 0, 1))

        # select each tissue by its label
        nifti_volume_CSF = label_volume == 1
        nifti_volume_GM  = label_volume == 2
        nifti_volume_WM  = label_volume == 3

        # group labels into their place holders
        _CSF.append(weight * nifti_volume_CSF)
        _GM.append(weight * nifti_volume_GM)
        _WM.append(weight * nifti_volume_WM)

    # get the mean of the three tissues
    mean_csf = np.mean(_CSF, axis=0)
    mean_gm = np.mean(_GM, axis=0)
    mean_wm = np.mean(_WM, axis=0)

    return mean_csf, mean_gm, mean_wm

def staple_fusion(        
        labels_dirs: List[str] = [],
        load_nifti_callback:Callable = None):
    '''
    Fuse all given labels based on the STAPLE algorithm. This step is performed after
    the registration and label propagation.

    Args:
        labels_dirs ('list'): list of labels directories to be fused togather.
        load_nifti_callback ('function'): a function that loads the nifti volume.

    Returns:
        mean_csf ('np.array'): A numpy array that holds the CSF fused atlas.
        mean_gm ('np.array'): A numpy array that holds the GM fused atlas.
        mean_wm ('np.array'): A numpy array that holds the WM fused atlas.
    '''

    # load the masks and conver them to SimpleITK format
    masks_stack = []

    for dir in labels_dirs:
        # load the volumes
        nifti_volume = load_nifti_callback(dir)[0]

        masks_stack.append(sitk.GetImageFromArray(nifti_volume.astype(np.int16)))

    # we have 3 labels for the 3 tissues, and STAPLE expects binary image as an input
    num_labels = 3
    consensus_results = []

    # Run the STAPLE algorithm for each label
    # The label index starts at 1, since the zero label is reserved for the background
    for label_index in range(1, num_labels+1):
        # Extract the binary mask for the current label
        binary_masks = [sitk.BinaryThreshold(image, lowerThreshold=label_index, upperThreshold=label_index) for image in masks_stack]

        # Run the STAPLE algorithm for the current label
        staple_filter = sitk.STAPLE(binary_masks, 1.0)

        # Append the result to the list of consensus results
        consensus_results.append(staple_filter)

    # convert back to numpy array
    consensus_results = [ sitk.GetArrayFromImage(STAPLE_seg_sitk) for STAPLE_seg_sitk in consensus_results]

    # return CSF, GM, WM based as they are labelled 1, 2, 3 respectively
    return consensus_results[0], consensus_results[1], consensus_results[2]

