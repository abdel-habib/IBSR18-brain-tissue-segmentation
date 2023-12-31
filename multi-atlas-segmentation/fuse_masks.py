import sys
sys.path.append('../')

import argparse
import os
import numpy as np
from glob import glob
from EM import FileManager, ElastixTransformix, NiftiManager, BrainAtlasManager, Evaluate
from utils.fusion import majority_voting_fusion, weighted_voting_fusion, staple_fusion

FM    = FileManager()
ET    = ElastixTransformix()
NM    = NiftiManager()
BM    = BrainAtlasManager()
EVAL  = Evaluate()

if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    parser.add_argument('--reg_params_key', type=str, default='Par0010affine+Par0010bspline', help='registration parameters key generated by create_script.py')
    parser.add_argument('--output_path', type=str, default='output', help='root dir for output scripts')
    parser.add_argument('--experiment_name', type=str, default='NO_PREPROCESSING', help='experiment name')
    parser.add_argument('--fixed_path', type=str, default='../TrainingValidationTestSets/Validation_Set', help='root dir for fixed data') # use the test data for submission 
    parser.add_argument("--generate_report", action='store_true', help='if True, an evaluation report .txt file will be generated. If not, only the transformed labels will be generated for each test sample.')

    # parse the arguments
    args = parser.parse_args()

    # args.experiment_name is useful to distinguish between different experiments (e.g. with or without preprocessing, etc.)
    # reg_params_key is useful to distinguish between different registration parameters (e.g. Par0003, Par0004, etc.)
    args.exp_output = os.path.join(args.output_path, args.experiment_name, args.reg_params_key)

    # get the registered intensity and transformed labels
    registered_intensity = sorted(glob(os.path.join(args.exp_output, "images", "***", "***", "result.1.nii"), recursive=True))
    transformed_labels   = sorted(glob(os.path.join(args.exp_output, "labels", "***", "***", "result.nii"), recursive=True))

    # get the list of fixed folders
    # either images or labels can be used
    fixed_dirs = os.listdir(os.path.join(args.exp_output, 'images')) # ['output_IBSR_11', 'output_IBSR_12', 'output_IBSR_13', 'output_IBSR_14', 'output_IBSR_17']
    print("\nFixed directories: ", fixed_dirs)

    # iterate through all fusion techniques
    for fusion_callback in [majority_voting_fusion, weighted_voting_fusion, staple_fusion]:
        print(f'\nSegmenting using "{fusion_callback.__name__}" fusion technique')

        # iterate through all fixed directories
        for fixed_filename in fixed_dirs:
            fixed_intensity_path = os.path.join(args.exp_output, "images", fixed_filename)
            fixed_label_path     = os.path.join(args.exp_output, "labels", fixed_filename)

            # get the propagated masks that were result of the label propagation to the current fixed
            registered_intensity = [m.replace('\\', '/') for m in glob(os.path.join(fixed_intensity_path, "IBSR_**", "*.nii"), recursive=True)]
            propagated_masks = [m.replace('\\', '/') for m in glob(os.path.join(fixed_label_path, "IBSR_**_seg", "*.nii"), recursive=True)]

            # get the target intensity path
            # we need to load the target intensity and label to get the dice score
            # removing output_ from the filename
            fixed_filename = fixed_filename.replace('output_', '')

            target_intensity_path = os.path.join(args.fixed_path, fixed_filename, f"{fixed_filename}.nii.gz")
            target_intensity_nifti, target_nii_image = NM.load_nifti(target_intensity_path)
            target_intensity_nifti = target_intensity_nifti[:,:,:,0]

            # fuse method
            if fusion_callback.__name__ in ['majority_voting_fusion', 'staple_fusion']:
                segmentation = fusion_callback(
                    labels_dirs = propagated_masks, 
                    load_nifti_callback = NM.load_nifti).astype('float64')
                
            elif fusion_callback.__name__ == 'weighted_voting_fusion':
                segmentation = fusion_callback(
                    labels_dirs = propagated_masks, 
                    intensity_dirs = registered_intensity,
                    target_intensity_path = target_intensity_path,
                    load_nifti_callback = NM.load_nifti).astype('float64')
                
            # save the segmentation result
            args.segmentation_result_output = os.path.join(args.exp_output, "segmentation_results", fusion_callback.__name__)
            FM.create_directory_if_not_exists(args.segmentation_result_output)

            print(f"Saving segmentation result to {args.segmentation_result_output}")
            NM.export_nifti(segmentation, os.path.join(args.segmentation_result_output, f"{fixed_filename}.nii.gz"), nii_image=target_nii_image)

            # check if generate_report is True
            if args.generate_report:
                valid_labels  = sorted(glob(os.path.join(args.fixed_path, "***", "IBSR_*_seg.nii.gz"), recursive=True))
                
                if len(valid_labels) == 0:
                    print(f"No labels found in {args.fixed_path} directory.")
                    sys.exit(1)

                # get the target label path
                target_label_path = os.path.join(args.fixed_path, fixed_filename, f"{fixed_filename}_seg.nii.gz")
                target_label_nifti = NM.load_nifti(target_label_path)[0][:,:,:,0]

                # evaluate dice score
                dice_coefficients = EVAL.evaluate_dice_volumes(segmentation, target_label_nifti, labels={'BG':0, 'CSF':1, 'GM':2, 'WM':3})

                print(f'Testing volume {fixed_filename}:-')
                print(f' - Dice scores {dice_coefficients}')
            