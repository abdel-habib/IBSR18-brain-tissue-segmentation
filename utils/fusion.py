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

    # place holders for the segmented masks
    segmentation_masks = []
    
    for dir in labels_dirs:
        # load the volumes
        nifti_volume = load_nifti_callback(dir)[0]

        # group labels into their place holders
        segmentation_masks.append(nifti_volume)

    # Convert segmentation masks to a 4D NumPy array (N x height x width x channels)
    masks_array = np.stack(segmentation_masks, axis=0)

    # Convert the masks to integers
    masks_array = masks_array.astype(np.int64)

    # Use np.apply_along_axis to apply np.bincount to each pixel along the first axis
    # This returns the bin (label) with the maximum count for each pixel
    segmentation_result = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=masks_array)

    return segmentation_result


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
    _BG  = [] # 0
    _CSF = [] # 1
    _GM  = [] # 2
    _WM  = [] # 3

    for intensity_path, weight in weights.items():
        intensity_volume = load_nifti_callback(intensity_path)[0]
        label_volume = load_nifti_callback(labels_dirs[intensity_dirs.index(intensity_path)])[0]

        # Skull strip the intensity volume using the label
        intensity_volume = np.multiply(intensity_volume, np.where(label_volume == 0, 0, 1))

        # select each tissue by its label
        nifti_volume_BG  = label_volume == 0
        nifti_volume_CSF = label_volume == 1
        nifti_volume_GM  = label_volume == 2
        nifti_volume_WM  = label_volume == 3

        # group labels into their place holders
        _BG.append(weight * nifti_volume_BG)
        _CSF.append(weight * nifti_volume_CSF)
        _GM.append(weight * nifti_volume_GM)
        _WM.append(weight * nifti_volume_WM)

    # get the mean of the three tissues
    mean_bg  = np.mean(_BG, axis=0).flatten()
    mean_csf = np.mean(_CSF, axis=0).flatten()
    mean_gm = np.mean(_GM, axis=0).flatten()
    mean_wm = np.mean(_WM, axis=0).flatten()

    # concatenate the flatenned atlases to form a NxK shaped array of arrays
    concatenated_atlas = np.column_stack((mean_bg, mean_csf, mean_gm, mean_wm))

    # get the argmax for each row to find which cluster does each sample refers to
    atlases_argmax = np.argmax(concatenated_atlas, axis=1) # + 1
    # reshape the argmax to the original shape of the volume
    segmentation = atlases_argmax.reshape(target_volume.shape)[:, :, :, 0]

    return segmentation

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
    for label_index in range(0, num_labels+1):
        # Extract the binary mask for the current label
        binary_masks = [sitk.BinaryThreshold(image, lowerThreshold=label_index, upperThreshold=label_index) for image in masks_stack]

        # Run the STAPLE algorithm for the current label
        reference_segmentation_STAPLE_probabilities  = sitk.STAPLE(binary_masks, 1)

        reference_segmentation_STAPLE  = reference_segmentation_STAPLE_probabilities > 0.60

        # Append the result to the list of consensus results
        consensus_results.append(reference_segmentation_STAPLE)

    # convert back to numpy array
    consensus_results = [ sitk.GetArrayFromImage(STAPLE_seg_sitk) for STAPLE_seg_sitk in consensus_results]
    
    # combined segmentation
    segmentation = np.zeros_like(consensus_results[0])

    segmentation[consensus_results[0] == 1] = 0 # background
    segmentation[consensus_results[1] == 1] = 1 # CSF
    segmentation[consensus_results[2] == 1] = 2 # GM
    segmentation[consensus_results[3] == 1] = 3 # WM

    return segmentation

