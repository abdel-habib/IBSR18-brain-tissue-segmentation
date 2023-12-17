To create elastix .bat file
`python create_script.py --experiment_name "NO_PREPROCESSING" --type "elastix"`

To modify the interpolator order in all of the transformation files before using elastix
`python prepare_propagation.py --reg_params_key "Par0010affine+Par0010bspline" --output_path "output" --experiment_name "NO_PREPROCESSING"`

To create transformix .bat file
`python create_script.py --experiment_name "NO_PREPROCESSING" --type "transformix"`

To save the segmentation and evaluate
`python fuse_masks.py --reg_params_key "Par0010affine+Par0010bspline" --output_path "output" --experiment_name "NO_PREPROCESSING" --fixed_path "../TrainingValidationTestSets/Validation_Set" --generate_report`

The evaluation is only possible when the test labels are given, the segmentation results are saved (under implementation)