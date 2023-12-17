# <h1 align="center">IBSR Brain Tissue Segmentation</h1>

Multi-Atlas Segmentation 
============
To segment using multi-atlas approach, follow the please instructions steps carefully. The dataset has to be in a folder called `TrainingValidationTestSets` in the root directory of the project.

The following steps are used to automate the registration, segmentation, and evaluation process. Firstly, change the directory to the `multi-atlas-segmentation`.
```
cd multi-atlas-segmentation
```
The registeration and transformation strategies used is to register all volumes to the test space, not taking into account the modality as the number of volumes is not enough to create an accurate segmentation for each modality. Thus, to register the volumes, and propagate the labels, run the following commands to create an executable `.bat` files. Make sure to run the commands in order.

To create elastix .bat file
```
python create_script.py --experiment_name "NO_PREPROCESSING" --type "elastix"
```

After creating the excutable file, inside it you will find how to call the file. A working example based on the previous instruction is shown below. Make sure to run this command inside the `multi-atlas-segmentation` as mentioned previously.
```
call output\NO_PREPROCESSING\Par0010affine+Par0010bspline\elastix.bat
```

Next step is to modify the interpolator order in all of the transformation files before using transformix. This is important to make sure that the propagated labels contain integer values, and not floats.
```
python prepare_propagation.py --reg_params_key "Par0010affine+Par0010bspline" --output_path "output" --experiment_name "NO_PREPROCESSING"
```

Now as the registration is done, we propagate all of the training labels to the same test space by creating an executable `.bat` file for transformix, using the command below. 
```
python create_script.py --experiment_name "NO_PREPROCESSING" --type "transformix"
```

Same as the previous step of elastix, please run the transformix file. The instructions to call the file is written inside the `.bat` file, however, below is an example to run based on the arguments used in the previous command.
```
call output\NO_PREPROCESSING\Par0010affine+Par0010bspline\transformix.bat
```

At this stage, all the labels are propagated to the test space. We can run the following fusion command to fuse the labes using differnt fusion techniques and export the segmentation resuls (exporting the segmentation results is under development).

```
python fuse_masks.py --reg_params_key "Par0010affine+Par0010bspline" --output_path "output" --experiment_name "NO_PREPROCESSING" --fixed_path "../TrainingValidationTestSets/Validation_Set" --generate_report
```

The evaluation is only possible when the test labels are given, the segmentation results are saved (under development).

Deep Learning Segmentation using nnUNet 
============
