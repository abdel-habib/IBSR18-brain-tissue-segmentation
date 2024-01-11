import numpy as np
from medpy.metric.binary import hd, ravd
import torch
import sys
import os
import shutil
from subprocess import call
import matplotlib.pyplot as plt
import nibabel as nib
import json

def haussdorf(gt: np.ndarray, pred: np.ndarray, voxelspacing: tuple):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
        voxelspacing (tuple): voxel_spacing
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0]).astype(int)
    hd_values = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        #try:
        hd_values[i-1] = hd(bin_pred, bin_gt, voxelspacing=voxelspacing)
        #except:
            #hd_values[i-1] = np.nan
    return hd_values.tolist()

def avd(gt: np.ndarray, pred: np.ndarray, voxelspacing: tuple):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
        voxelspacing (tuple): voxel_spacing
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0]).astype(int)
    avd = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        vol_pred = np.count_nonzero(bin_pred)
        vol_gt = np.count_nonzero(bin_gt)
        unit_volume = voxelspacing[0] * voxelspacing[1] * voxelspacing[2]
        avd[i-1] = np.abs(vol_pred - vol_gt) * unit_volume
    return avd.tolist()


def rel_abs_vol_dif(gt: np.ndarray, pred: np.ndarray):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0]).astype(int)
    ravd_values = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        try:
            ravd_values[i-1] = ravd(bin_gt, bin_pred)
        except:
            ravd_values[i-1] = np.nan
    return ravd_values.tolist()

def dice_score(gt: np.ndarray, pred: np.ndarray):
    """Compute dice across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0])
    dice = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        dice[i-1] = np.sum(bin_pred[bin_gt == 1]) * 2.0 / (np.sum(bin_pred) + np.sum(bin_gt))
    return dice.tolist()

def create_special_trainer(file_path):
    if os.path.exists(file_path):
        print(f"The file nnUNetTrainerV2_Fast exists, do nothing.")
    else:
        shutil.copy('nnUNetTrainerV2_Fast.py', "nnUNet/nnunet/training/network_training/")
        print(f"The file nnUNetTrainerV2_Fast exists does not exist. Making a copy on trainning dir")

def check_gpu():
    print('_____Python, Pytorch, Cuda info____')
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA RUNTIME API VERSION')
    #os.system('nvcc --version')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('_____nvidia-smi GPU details____')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('_____Device assignments____')
    print('Number CUDA Devices:', torch.cuda.device_count())
    print ('Current cuda device: ', torch.cuda.current_device(), ' **May not correspond to nvidia-smi ID above, check visibility parameter')
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

def graph_brain(result_dir_path = 'task975_Results', test_image_path = 'nnUNet_raw/nnUNet_raw_data/Task975_BrainSegmentation/imagesTs/'):
    result_dir = os.path.join(result_dir_path)
    test_img_name = os.listdir(test_image_path)[np.random.randint(0,3)]
    test_img = np.array(nib.load(os.path.join(test_image_path,test_img_name)).dataobj)[:,:,100:150:5]

    predicted_img_name = test_img_name[:test_img_name.find('_0000.nii.gz')]+'.nii.gz'
    predicted_label = np.array(nib.load(os.path.join(result_dir,predicted_img_name)).dataobj)[:,:,100:150:5]

    print('Test Image Shape: ',test_img.shape)
    print("Predicted Image Shape:",predicted_label.shape)


    max_rows = 2
    max_cols = test_img.shape[2]

    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(30,8))
    for idx in range(max_cols):
        axes[0, idx].axis("off") 
        axes[0, idx].set_title('Test Image'+str(idx+1))
        axes[0 ,idx].imshow(test_img[:,:,idx].squeeze(), cmap="gray")
    for idx in range(max_cols):    
        axes[1, idx].axis("off")
        axes[1, idx].set_title('Test Label'+str(idx+1))
        axes[1, idx].imshow(predicted_label[:,:,idx].squeeze())
        
    plt.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()

def convert_data_structure(original_path, destination_path):
    # Check if 'Task975_BrainSegmentation' exists in the destination path
    task_path = os.path.join(destination_path, 'Task975_BrainSegmentation')
    if os.path.exists(task_path):
        print('New data structure already exists')
        return
    
    # Create new directory structure
    os.makedirs(os.path.join(task_path, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'imagesTs'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'labelsTr'), exist_ok=True)

    # Function to handle the file copying and renaming
    def handle_files(source_folder, is_test=False):
        for folder_name in sorted(os.listdir(source_folder)):
            folder_path = os.path.join(source_folder, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.nii.gz'):
                        source_file = os.path.join(folder_path, file_name)
                        if '_seg' in file_name:
                            # For segmentation files (labels)
                            dest_file = os.path.join(task_path, 'labelsTr', folder_name + '.nii.gz')
                        else:
                            # For image files
                            suffix = '_0000.nii.gz'
                            dest_file = os.path.join(task_path, 'imagesTs' if is_test else 'imagesTr', folder_name + suffix)
                        shutil.copy2(source_file, dest_file)

    # Process each set
    handle_files(os.path.join(original_path, 'Training_Set'))
    handle_files(os.path.join(original_path, 'Validation_Set'))
    handle_files(os.path.join(original_path, 'Test_Set'), is_test=True)

def create_dataset_json(parent_dir, output_file):
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. No action taken.")
        return

    # Define the structure of the JSON file
    dataset_json = {
        "modality": {"0": "T1"},
        "labels": {
            "0": "background",
            "1": "CFS",
            "2": "GM",
            "3": "WM"
        },
        "numTraining": 0,
        "numTest": 0,
        "training": [],
        "test": []
    }

    # Paths for training and test data
    training_images_path = os.path.join(parent_dir, "imagesTr")
    training_labels_path = os.path.join(parent_dir, "labelsTr")
    test_images_path = os.path.join(parent_dir, "imagesTs")

    # Scan for training images and labels
    if os.path.exists(training_images_path) and os.path.exists(training_labels_path):
        training_images = sorted([f for f in os.listdir(training_images_path) if f.endswith('.nii.gz')])
        training_labels = sorted([f for f in os.listdir(training_labels_path) if f.endswith('.nii.gz')])
        for img in training_labels:
            dataset_json["training"].append({
                    "image": os.path.join("./imagesTr", img),
                    "label": os.path.join("./labelsTr", img)
                })
        
        dataset_json["numTraining"] = len(dataset_json["training"])

    # Scan for test images
    if os.path.exists(test_images_path):
        test_images = sorted([f for f in os.listdir(test_images_path) if f.endswith('.nii.gz')])
        for img in test_images:
            dataset_json["test"].append(os.path.join("./imagesTs", img))
        
        dataset_json["numTest"] = len(dataset_json["test"])

    # Write to JSON file
    with open(output_file, 'w') as outfile:
        json.dump(dataset_json, outfile, indent=4)

    print(f"Dataset JSON created at {output_file}")