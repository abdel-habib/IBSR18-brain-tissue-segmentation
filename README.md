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
The code refering on this part can be founs in the [unnet_segmentation.ipynb](dl_part/unnet_segmentation.ipynb)

In the nnUNet part the dataset was rename it in oder to follow the instruction of the nnUNet [github page](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md "Data Set Convertion NNunet").

We create this function to apply the changes from the UNNet Documentation.

```python
convert_data_structure('../TrainingValidationTestSets/','nnUNet_raw/nnUNet_raw_data/')
```

<table>
    <tr>
        <th>Original</th>
        <th>NNunet</th>
    </tr>
    <tr>
        <td>
            <pre>
TrainingValidationTestSets
├── description.json
├── Test_Set
│   ├── IBSR_02
│   │   └── IBSR_02.nii.gz
│   ├── IBSR_10
│   │   └── IBSR_10.nii.gz
│   └── IBSR_15
│       └── IBSR_15.nii.gz
├── Training_Set
│   ├── IBSR_01
│   │   ├── IBSR_01.nii.gz
│   │   └── IBSR_01_seg.nii.gz
│   ├── IBSR_03
│   │   ├── IBSR_03.nii.gz
│   │   └── IBSR_03_seg.nii.gz
│   ├── IBSR_04
│   │   ├── IBSR_04.nii.gz
│   │   └── IBSR_04_seg.nii.gz
│   ├── IBSR_05
│   │   ├── IBSR_05.nii.gz
│   │   └── IBSR_05_seg.nii.gz
│   ├── IBSR_06
│   │   ├── IBSR_06.nii.gz
│   │   └── IBSR_06_seg.nii.gz
│   ├── IBSR_07
│   │   ├── IBSR_07.nii.gz
│   │   └── IBSR_07_seg.nii.gz
│   ├── IBSR_08
│   │   ├── IBSR_08.nii.gz
│   │   └── IBSR_08_seg.nii.gz
│   ├── IBSR_09
│   │   ├── IBSR_09.nii.gz
│   │   └── IBSR_09_seg.nii.gz
│   ├── IBSR_16
│   │   ├── IBSR_16.nii.gz
│   │   └── IBSR_16_seg.nii.gz
│   └── IBSR_18
│       ├── IBSR_18.nii.gz
│       └── IBSR_18_seg.nii.gz
└── Validation_Set
    ├── IBSR_11
    │   ├── IBSR_11.nii.gz
    │   └── IBSR_11_seg.nii.gz
    ├── IBSR_12
    │   ├── IBSR_12.nii.gz
    │   └── IBSR_12_seg.nii.gz
    ├── IBSR_13
    │   ├── IBSR_13.nii.gz
    │   └── IBSR_13_seg.nii.gz
    ├── IBSR_14
    │   ├── IBSR_14.nii.gz
    │   └── IBSR_14_seg.nii.gz
    └── IBSR_17
        ├── IBSR_17.nii.gz
        └── IBSR_17_seg.nii.gz
            </pre>
        </td>
        <td>
            <pre>
dl_part/nnUNet_raw/nnUNet_raw_data
└──Task975_BrainSegmentation
    ├── imagesTr
    │   ├── IBSR_01_0000.nii.gz
    │   ├── IBSR_03_0000.nii.gz
    │   ├── IBSR_04_0000.nii.gz
    │   ├── IBSR_05_0000.nii.gz
    │   ├── IBSR_06_0000.nii.gz
    │   ├── IBSR_07_0000.nii.gz
    │   ├── IBSR_08_0000.nii.gz
    │   ├── IBSR_09_0000.nii.gz
    │   ├── IBSR_11_0000.nii.gz
    │   ├── IBSR_12_0000.nii.gz
    │   ├── IBSR_13_0000.nii.gz
    │   ├── IBSR_14_0000.nii.gz
    │   ├── IBSR_16_0000.nii.gz
    │   ├── IBSR_17_0000.nii.gz
    │   └── IBSR_18_0000.nii.gz
    ├── imagesTs
    │   ├── IBSR_02_0000.nii.gz
    │   ├── IBSR_10_0000.nii.gz
    │   └── IBSR_15_0000.nii.gz
    └── labelsTr
        ├── IBSR_01.nii.gz
        ├── IBSR_03.nii.gz
        ├── IBSR_04.nii.gz
        ├── IBSR_05.nii.gz
        ├── IBSR_06.nii.gz
        ├── IBSR_07.nii.gz
        ├── IBSR_08.nii.gz
        ├── IBSR_09.nii.gz
        ├── IBSR_11.nii.gz
        ├── IBSR_12.nii.gz
        ├── IBSR_13.nii.gz
        ├── IBSR_14.nii.gz
        ├── IBSR_16.nii.gz
        ├── IBSR_17.nii.gz
        └── IBSR_18.nii.gz
            </pre>
        </td>
    </tr>
</table>



Create the .json file describing the dataset structure with the next command:

```python
create_dataset_json('nnUNet_raw/nnUNet_raw_data/Task975_BrainSegmentation/', "nnUNet_raw/nnUNet_raw_data/Task975_BrainSegmentation/dataset.json")
```


In the notebook we used this code to check if pur custom implementation of the nnUNet exists in the configuration files of the nnUNet. 

> Note only run this after clonning the nnUNet repository

```python
file_path = "nnUNet/nnunet/training/network_training/nnUNetTrainerV2_Fast.py"
create_special_trainer(file_path)
```

To clone the repository and used the V1 of the nnUNet, since the V2 lack the inference part we decide to used the version with inference implementation.

```
! git clone https://github.com/MIC-DKFZ/nnUNet.git
! cd nnUNet; git checkout nnunetv1
```
To install all the dependencies of the nnUNet:

```
! cd nnUNet; pip install -e .
```

To check the gpu and controller you can run `nvidia-smi` or our custom python script `check_gpu()`. The output should look like this of both commands:

```

Fri Dec 15 20:10:59 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1050 Ti     Off | 00000000:01:00.0 Off |                  N/A |
| N/A   46C    P8              N/A / ERR! |      6MiB /  4096MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2239      G   /usr/lib/xorg/Xorg                            4MiB |
|    0   N/A  N/A      2515      G   ...libexec/gnome-remote-desktop-daemon        1MiB |
+---------------------------------------------------------------------------------------+

_____Python, Pytorch, Cuda info____
__Python VERSION: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
__pyTorch VERSION: 2.1.1+cu121
__CUDA RUNTIME API VERSION
__CUDNN VERSION: 8902
_____nvidia-smi GPU details____
index, name, driver_version, memory.total [MiB], memory.used [MiB], memory.free [MiB]
0, NVIDIA GeForce GTX 1050 Ti, 535.129.03, 4096 MiB, 6 MiB, 4034 MiB
_____Device assignments____
Number CUDA Devices: 1
Current cuda device:  0  **May not correspond to nvidia-smi ID above, check visibility parameter
Device name:  NVIDIA GeForce GTX 1050 Ti


```

The next three soft link to this enviroment folder should exist. According to the official documentation of nnUNet:

```
os.environ["nnUNet_raw_data_base"] = str(nnUNet_raw)
os.environ["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
os.environ["RESULTS_FOLDER"] = str(results_folder)
```

After all the setting and dowloading all the necessary stuff to run the nnUNet. You can run the trainning.

The first step is to run the validation and model parameters setting.

 `-t` parameter is the number of the task assing on the rawfolder were you save your data. For more information check the [official Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md "Data Set Convertion NNunet")

```
nnUNet_plan_and_preprocess -t 975 --verify_dataset_integrity
```

The next step is to train the model after the data verification. For this we used a modify class of the normal trainer from the nnUNet configuration files. This is made since the model only has the capability to run until 1000 epochs per fold. This way we only run for 30 epochs. 

```python
import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_Fast(nnUNetTrainerV2):

  def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
            unpack_data=True, deterministic=True, fp16=False):
    super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                      deterministic, fp16)
    self.max_num_epochs = 30
    self.initial_lr = 1e-2
    self.deep_supervision_scales = None
    self.ds_loss_weights = None

    self.pin_memory = True
```

This line of code is used to train the nnUNet model:

```
! nnUNet_train 2d nnUNetTrainerV2_Fast Task975_BrainSegmentation 3 --npz -c 
```

- To train - `nnUNet_train NETWORK NETWORK_TRAINER TASK_NAME_OR_ID FOLD --npz`
- To resume - `nnUNet_train NETWORK NETWORK_TRAINER TASK_NAME_OR_ID FOLD --npz -c `(just add -c to the training command)
- `NETWORK` - 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
- `nnUNetTrainerV2_Fast` - our custom trainer
- `--npz` - if set then nnUNet will export npz files of predicted segmentations in the validation as well
- Everything will be stored in the results folder

After the trainning the predictions of each of the folds are save in the next route:

```
dl_part/nnUNet_results/nnUNet/2d/Task975_BrainSegmentation/nnUNetTrainerV2_Fast__nnUNetPlansv2.1/fold_3/
├── debug.json
├── model_best.model
├── model_best.model.pkl
├── model_final_checkpoint.model
├── model_final_checkpoint.model.pkl
├── network_architecture
├── postprocessing.json
├── progress.png
├── training_log_2023_12_14_20_39_29.txt
├── training_log_2023_12_14_21_34_59.txt
├── training_log_2023_12_15_20_12_06.txt
├── validation_raw
│   ├── IBSR_05.nii.gz
│   ├── IBSR_05.npz
│   ├── IBSR_05.pkl
│   ├── IBSR_10.nii.gz
│   ├── IBSR_10.npz
│   ├── IBSR_10.pkl
│   ├── IBSR_14.nii.gz
│   ├── IBSR_14.npz
│   ├── IBSR_14.pkl
│   ├── summary.json
│   └── validation_args.json
└── validation_raw_postprocessed
    ├── IBSR_05.nii.gz
    ├── IBSR_10.nii.gz
    ├── IBSR_14.nii.gz
    └── summary.json
```

In order to predict the testfiles correctly used the next command line: 

```
! nnUNet_predict -i nnUNet_raw/nnUNet_raw_data/Task975_BrainSegmentation/imagesTs -o task975_Results -t 975 -tr nnUNetTrainerV2_Fast -m 2d --num_threads_preprocessing 1

```

- To predict - `nnUNet_predict -i TEST_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID FOLD -tr NETWORK_TRAINER -m NETWORK --num_threads_preprocessing 1`

- `NETWORK` - 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
- `nnUNetTrainerV2_Fast` - our custom trainer