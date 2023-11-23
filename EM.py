# =============================================================================
# FILE: EM.py
# DESCRIPTION: A Python file containing classes for image segmentation and 
# evaluation using Expectation-Maximization Algorithm.
# AUTHOR: [Abdelrahman Usama Habib]
# DATE: [11/19/2023]
# =============================================================================

# -----------------------------------------------------------------------------
# TABLE OF CONTENTS
# -----------------------------------------------------------------------------
# 1. Import Statements
# 2. FileManager Class
# 3. NiftiManager Class
# 4. Evaluate Class
# 5. ElastixTransformix Class
# 6. BrainAtlasManager Class
# 7. Plot Class
# 8. EM Class
# -----------------------------------------------------------------------------

# 1. Import Statements
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm

import subprocess
from glob import glob
import math
from tqdm import tqdm
import pprint
import pandas as pd

from loguru import logger
import pandas as pd

# 2. FileManager Class
class FileManager:
    '''
    A class for managing file-related operations, such as checking file existence, creating directories,
    pretty-printing objects, and replacing text in files.

    Methods:
        - __init__(self) -> None: Initialize a PrettyPrinter for clear object printing.
        - check_file_existence(self, file, description): Check if a file exists; raise a ValueError if not.
        - create_directory_if_not_exists(self, path): Create a directory if it does not exist.
        - pprint_objects(self, *arg): Print large and indented objects clearly.
        - replace_text_in_file(self, file_path, search_text, replacement_text): Replace text in a text file.

    Attributes:
        - pp (pprint.PrettyPrinter): PrettyPrinter object for clear object printing.
    '''
    def __init__(self) -> None:
        # Initialize a PrettyPrinter for clear object printing
        self.pp = pprint.PrettyPrinter(indent=4)

    def check_file_existence(self, file, description):
        '''
        Check if a file exists; raise a ValueError if not.

        Args:
            file ('str'): File path.
            description ('str'): Description of the file for the error message.
        '''
        if file is None:
            raise ValueError(f"Please check if the {description} file passed exists in the specified directory")
        
    def create_directory_if_not_exists(self, path):
        '''
        Create a directory if it does not exist.

        Args:
            path ('str'): Directory path.
        '''
        if not os.path.exists(path):
            os.makedirs(path)

    def pprint_objects(self, *arg):
        '''
        Print large and indented objects clearly.

        Args:
            *arg: Variable number of arguments to print.
        '''
        self.pp.pprint(arg)

    def replace_text_in_file(self, file_path, search_text, replacement_text):
        '''
        Replace text in a text file.

        Args:
            file_path ('str'): Path to the text file.
            search_text ('str'): Text to search for in the file.
            replacement_text ('str'): Text to replace the searched text with.
        '''
        try:
            # Read the file
            with open(file_path, 'r') as file:
                content = file.read()

            # Replace the search_text with replacement_text
            modified_content = content.replace(search_text, replacement_text)

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.write(modified_content)

            # print(f"Text replaced in {file_path} and saved.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        # except Exception as e:
        #     print(f"An error occurred: {e}")

# 3. NiftiManager Class
class NiftiManager:
    """
    Manager class for handling NIfTI files, including loading, visualization, and export.

    Methods:
        - __init__(self) -> None: Initializes the NiftiManager.
        - load_nifti(self, file_path) -> Tuple[np.array, nibabel.Nifti1Image]:
            Load the NIfTI image and access the image data as a Numpy array.

        - show_nifti(self, file_data, title, slice=25) -> None:
            Display a single slice from the NIfTI volume.

        - show_label_seg_nifti(self, label, seg, subject_id, slice=25) -> None:
            Display both segmentation and ground truth labels for a specific slice.

        - show_mean_volumes(self, mean_csf, mean_wm, mean_gm, slices=[128], export=False, filename=None) -> None:
            Display mean volumes for CSF, WM, and GM for specified slices.

        - show_combined_mean_volumes(self, mean_csf, mean_wm, mean_gm, slice_to_display=128, export=False, filename=None) -> None:
            Display combined averaged volumes for CSF, WM, and GM at a specific slice.

        - min_max_normalization(self, image, max_value) -> np.array:
            Perform min-max normalization on an image.

        - export_nifti(self, volume, export_path, nii_image=None) -> None:
            Export NIfTI volume to a given path.

    Attributes:
        None
    """
    def __init__(self) -> None:
        pass

    def load_nifti(self, file_path):
        '''
        Load the NIfTI image and access the image data as a Numpy array.

        Args:
            file_path ('str'): Path to the NIfTI file.

        Returns:
            data_array ('np.array'): Numpy array representing the image data.
            nii_image: Loaded NIfTI image object.
        '''
        nii_image = nib.load(file_path)
        data_array = nii_image.get_fdata()

        return data_array, nii_image

    def show_nifti(self, file_data, title, slice=25):
        '''
        Display a single slice from the NIfTI volume.

        Args:
            file_data ('np.array'): Numpy array representing the image data.
            title ('str'): Title for the plot.
            slice ('int'): Slice index to display.
        '''        
        plt.imshow(file_data[:, :, slice], cmap='gray')
        plt.title(title)
        # plt.colorbar()
        plt.axis('off')
        plt.show()

    def show_label_seg_nifti(self, label, seg, subject_id, slice=25):
        '''
        Display both segmentation and ground truth labels for a specific slice.

        Args:
            label ('np.array'): Ground truth label image.
            seg ('np.array'): Segmentation label image.
            subject_id ('str'): Identifier for the subject or image.
            slice ('int'): Slice index to display.

        Returns:
            None
        '''
        plt.figure(figsize=(20, 7))
        
        plt.subplot(1, 2, 1)
        plt.imshow(label[:, :, slice], cmap='gray') 
        plt.title(f'Label Image (Subject ID={subject_id})')
        # plt.colorbar()
        plt.axis('off')


        plt.subplot(1, 2, 2)
        plt.imshow(seg[:, :, slice], cmap='gray') 
        plt.title(f'Segmentation Image (Subject ID={subject_id})')
        # plt.colorbar()
        plt.axis('off')

        plt.show()

    def show_mean_volumes(self, mean_csf, mean_wm, mean_gm, slices=[128], export=False, filename=None):
        '''
        Display mean volumes for CSF, WM, and GM for specified slices.

        Args:
            mean_csf ('np.array'): Mean volume for CSF.
            mean_wm ('np.array'): Mean volume for WM.
            mean_gm ('np.array'): Mean volume for GM.
            slices ('list'): List of slice indices to display.
            export ('bool'): Whether to export the plot to a file.
            filename ('str'): Filename for the exported plot.

        Returns:
            None
        '''        
        num_slices = len(slices)
        
        plt.figure(figsize=(20, 7 * num_slices))

        for i, slice in enumerate(slices):
            plt.subplot(num_slices, 3, i * 3 + 1)
            plt.imshow(mean_csf[:, :, slice], cmap='gray')
            plt.title(f'Average CSF Volume - Slice {slice}')
            # plt.colorbar()
            plt.axis('off')

            plt.subplot(num_slices, 3, i * 3 + 2)
            plt.imshow(mean_wm[:, :, slice], cmap='gray')
            plt.title(f'Average WM Volume - Slice {slice}')
            # plt.colorbar()
            plt.axis('off')

            plt.subplot(num_slices, 3, i * 3 + 3)
            plt.imshow(mean_gm[:, :, slice], cmap='gray')
            plt.title(f'Average GM Volume - Slice {slice}')
            # plt.colorbar()
            plt.axis('off')

        if export and filename:
            plt.savefig(filename)
            
        plt.show()

    def show_combined_mean_volumes(self, mean_csf, mean_wm, mean_gm, slice_to_display=128, export=False, filename=None):
        '''
        Display combined averaged volumes for CSF, WM, and GM at a specific slice.

        Args:
            mean_csf ('np.array'): Mean volume for CSF.
            mean_wm ('np.array'): Mean volume for WM.
            mean_gm ('np.array'): Mean volume for GM.
            slice_to_display ('int'): Slice index to display.
            export ('bool'): Whether to export the plot to a file.
            filename ('str'): Filename for the exported plot.

        Returns:
            None
        '''
        # Stack the mean volumes along the fourth axis to create a single 4D array
        combined_mean_volumes = np.stack((mean_csf, mean_wm, mean_gm), axis=3)
    
        # Choose the channel you want to display (0 for CSF, 1 for WM, 2 for GM)
        # channel_to_display = 0  # Adjust as needed
    
        # Display the selected channel
        plt.imshow(combined_mean_volumes[:, :, :, :][:, :, slice_to_display]) # [:, :, :, channel_to_display]
        plt.axis('off')  # Turn off axis labels
        plt.title(f'Combined Averaged Volumes at Slice {slice_to_display}')  # Add a title

        if export and filename:
            plt.savefig(filename)
            
        plt.show()

    def min_max_normalization(self, image, max_value):
        '''
        Perform min-max normalization on an image.

        Args:
            image ('np.array'): Input image to normalize.
            max_value ('float'): Maximum value for normalization.

        Returns:
            normalized_image ('np.array'): Min-max normalized image.
        '''
        # Ensure the image is a NumPy array for efficient calculations
        image = np.array(image)
        
        # Calculate the minimum and maximum pixel values
        min_value = np.min(image)
        max_actual = np.max(image)
        
        # Perform min-max normalization
        normalized_image = (image - min_value) / (max_actual - min_value) * max_value
        
        return normalized_image

    def export_nifti(self, volume, export_path, nii_image=None):
        '''
        Export NIfTI volume to a given path.

        Args:
            volume ('np.array'): Numpy array representing the volume.
            export_path ('str'): Path to export the NIfTI file.
            nii_image ('nibabel'): Loaded NIfTI image object.

        Returns:
            None
        '''
        # Create a NIfTI image from the NumPy array
        # np.eye(4): Identity affine transformation matrix, it essentially assumes that the images are in the same orientation and position 
        # as the original images
        affine = nii_image.affine if nii_image else np.eye(4)
        img = nib.Nifti1Image(volume, affine)

        # Save the NIfTI image
        nib.save(img, str(export_path))

# 4. Evaluate Class
class Evaluate:
    """
    Class for evaluating segmentation performance using Dice coefficients.

    Methods:
        - __init__(self) -> None: Initializes the Evaluate class.

        - calc_dice_coefficient(self, mask1, mask2) -> float:
            Calculate the Dice coefficient between two binary masks.

        - evaluate_dice_volumes(self, volume1, volume2, labels=None) -> dict:
            Evaluate Dice coefficients for different tissue types.

    Attributes:
        None
    """
    def __init__(self) -> None:
        pass

    def calc_dice_coefficient(self, mask1, mask2):
        '''
        Calculate the Dice coefficient between two binary masks.

        Args:
            mask1 ('np.array'): Binary mask.
            mask2 ('np.array'): Binary mask.

        Returns:
            dice ('float'): Dice coefficient.
        '''
        # Ensure the masks have the same shape
        if mask1.shape != mask2.shape:
            raise ValueError("Input masks must have the same shape.")

        # Compute the intersection and union of the masks
        intersection = np.sum(mask1 * mask2)
        union = np.sum(mask1) + np.sum(mask2)

        # Calculate the Dice coefficient
        dice = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero

        return dice
    
    def evaluate_dice_volumes(self, volume1, volume2, labels=None):
        '''
        Evaluate Dice coefficients for different tissue types.

        Args:
            volume1 ('np.array'): Segmentation volume.
            volume2 ('np.array'): Ground truth segmentation volume.
            labels ('dict'): Dictionary mapping tissue types to labels.

        Returns:
            dice_coefficients ('dict'): Dictionary of Dice coefficients for each tissue type.
        '''
        # Ensure the masks have the same shape
        if volume1.shape != volume2.shape:
            raise ValueError("Input masks must have the same shape.")
        
        if labels is None:
            raise ValueError("Missing labels argument.")
        
        dice_coefficients = {}

        for tissue_label in ['WM', 'GM', 'CSF']:
            mask1 = volume1 == labels[tissue_label]
            mask2 = volume2 == labels[tissue_label]

            dice_coefficient = self.calc_dice_coefficient(mask1, mask2)
            dice_coefficients[tissue_label] = round(dice_coefficient, 6)

            # print(f"{tissue_label} DICE: {dice_coefficient}")

        return dice_coefficients

# 5. ElastixTransformix Class           
class ElastixTransformix:
    """
    Class for performing image registration using elastix and label propagation using transformix.

    Methods:
        - __init__(self) -> None: Initializes the ElastixTransformix class.

        - execute_cmd(self, command) -> str:
            Execute a command and check for success.

        - register_elastix(self, fixed_path, moving_path, reg_params, create_dir_callback, execute_cmd_callback, fMask=None) -> None:
            Perform image registration using elastix.

        - label_propagation_transformix(self, fixed_path, moving_path, input_label, transform_path, replace_text_in_file_callback, create_dir_callback, execute_cmd_callback) -> None:
            Apply label propagation using transformix.

    Attributes:
        None
    """
    def __init__(self) -> None:
        pass

    def excute_cmd(self, command):
        '''
        Execute a command and check for success.

        Args:
            command ('str'): Command to execute.
        
        Returns:
            result ('str'): Output of the command if successful.
        '''
        # excute the command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        # Check the return code to see if the command was successful
        if result.returncode == 0:
            # print("Command executed successfully.")
            # print("Output:")
            return result.stdout
        else:
            print(f"Command failed with an error: {command}")
            print(result.stderr)
            return result.stderr

    # Perform registration and label propagation
    def register_elastix(self,
                        fixed_path, 
                        moving_path, 
                        reg_params, 
                        create_dir_callback, 
                        excute_cmd_callback,
                        fMask = None):
        '''
        Perform image registration using elastix.

        Args:
            fixed_path ('str'): Path to the fixed image.
            moving_path ('str'): Path to the moving image.
            reg_params ('str'): Registration parameters for elastix.
            create_dir_callback ('function'): Callback function to create directories.
            excute_cmd_callback ('function'): Callback function to execute commands.
            fMask ('str'): Optional path to a mask file.

        Returns:
            None
        '''
        # Get the names of the fixed and moving images for the output directory, names without the file extensions
        reg_fixed_name  = fixed_path.replace("\\", "/").split("/")[-1].split(".")[0] # \\
        reg_moving_name = moving_path.replace("\\", "/").split("/")[-1].split(".")[0]

        # create output dir
        output_dir = f'output/images/output_{reg_fixed_name}/{reg_moving_name}'
        create_dir_callback(output_dir)

        # create elastix command line
        command_line = f'elastix -f "{fixed_path}" -m "{moving_path}" {reg_params} -out "{output_dir}"' if not fMask else \
                        f'elastix -f "{fixed_path}" -m "{moving_path}" -fMask {fMask} {reg_params} -out "{output_dir}"'

        # call elastix command
        excute_cmd_callback(command_line)

    def label_propagation_transformix(
        self,
        fixed_path, 
        moving_path, 
        input_label, 
        transform_path, 
        replace_text_in_file_callback, 
        create_dir_callback, 
        excute_cmd_callback):
        '''
        Apply label propagation using transformix.

        Args:
            fixed_path ('str'): Path to the fixed image.
            moving_path ('str'): Path to the moving image.
            input_label ('str'): Path to the input label image.
            transform_path ('str'): Path to the transformation parameters.
            replace_text_in_file_callback ('function'): Callback function to replace text in a file.
            create_dir_callback ('function'): Callback function to create directories.
            excute_cmd_callback ('function'): Callback function to execute commands.

        Returns:
            None
        '''
        replace_text_in_file_callback(
            transform_path, 
            search_text = '(FinalBSplineInterpolationOrder 3)', 
            replacement_text =  '(FinalBSplineInterpolationOrder 0)')

        # Get the names of the fixed and moving images for the output directory, names without the file extensions
        reg_fixed_name  = fixed_path.replace("\\", "/").split("/")[-1].split(".")[0] 
        reg_moving_name = os.path.join(moving_path.replace("\\", "/").split("/")[0], moving_path.replace("\\", "/").split("/")[-1].split(".")[0])
            
        # create an output directory for the labels
        output_dir = f'output/labels/output_{reg_fixed_name}/{reg_moving_name}' # rem _float64

        # creates the output directory
        create_dir_callback(output_dir)
        
        # create transformix command line
        command_line = f'transformix -in "{input_label}" -tp "{transform_path}"  -out "{output_dir}"'
        
        # run transformix on all combinations
        excute_cmd_callback(command_line)
    
# 6. BrainAtlasManager Class
class BrainAtlasManager:
    """
    Class for managing brain atlases and performing segmentation using tissue models and label propagation.

    Methods:
        - __init__(self) -> None: Initializes the BrainAtlasManager class.

        - segment_using_tissue_models(self, image, label, tissue_map_csv) -> Tuple[np.array, np.array]:
            Segmentation using intensity information and tissue models.

        - segment_using_tissue_atlas(self, image, label, *atlases) -> Tuple[np.array, np.array]:
            Segmentation using position information and atlases.

        - segment_using_tissue_models_and_atlas(self, image, label, tissue_map_csv, *atlases) -> Tuple[np.array, np.array]:
            Segmentation using both intensity and position information.

    Attributes:
        None
    """
    def __init__(self) -> None:
        pass

    def segment_using_tissue_models(self, image, label, tissue_map_csv):
        '''
        Task (1.1) Tissue models: segmentation using just intensity information.

        Args:
            image ('np.array'):
                A normalized [0, 255] and skull stripped intensity volume for the brain in the form of a numpy array. 
                This is the required volume to be segmented.

            label ('np.array'):
                A nifti volume for the intensity image. Pixels labeled as 0 will be treated as background.

            tissue_map_csv ('Path'): 
                A csf file path that contains the tissue maps probabilities. The file should contain three columns, 
                first column for CSF, then WM, then GM.

        Returns:
            The segmented volume in the same shape of the passed intensity volume. The output is given in labels for 
            the segmentation, where label 0 is for background, 1 for CSF, 2 for WM, and 3 for GM. Those labels changes
            based on the tissue map columns orders. The final atlas probability has a shape of (N, K), where N is the 
            number of samples, and K is the number of clusters.

            segmentation_result ('np.array'):
                The segmentation label image in the form of numpy array.

            tissue_map_array ('np.array'):
                An array that represents the final atlas probabilities. 
        '''
        # read the tissues moodels
        tissue_map_df = pd.read_csv(tissue_map_csv, header=None)
        tissue_map_array = tissue_map_df.values

        # binary mask
        binary_mask = np.where(label == 0, 0, 1) 

        # map background pixels above a threshold to WM (label 2)
        threshold = 100
        bg_mask = np.arange(len(tissue_map_array)) > threshold
        tissue_map_array[bg_mask, 1] = 2

        # flatten the image and select only tissues within the mask
        registered_volume_test = image[binary_mask == 1].flatten()

        # using registered_volume_test as an index to extract specific rows from tissue_map_array
        tissue_map_array = tissue_map_array[registered_volume_test, :]

        # obtain the argmax to know to which cluster each row (histogram bin - 0:255) falls into
        tissue_map_array_argmax = np.argmax(tissue_map_array, axis=1) + 1

        # Reshape the atlases_argmax array to match the shape of the original image
        reshaped_atlases_argmax = tissue_map_array_argmax.reshape(image[binary_mask == 1].shape)

        # Create an empty segmentation result with the same shape as the original image
        segmentation_result = np.zeros_like(image)

        # Set the background (ignored) pixels to label 0
        segmentation_result[binary_mask == 0] = 0

        # set the segmented values where indexes falls to be true
        segmentation_result[binary_mask == 1] = reshaped_atlases_argmax

        return segmentation_result, tissue_map_array

    
    def segment_using_tissue_atlas(self, image, label, *atlases):
        '''
        Task (1.2) Label propagation: segmentation using just position information using atlases

        Args:
            image ('np.array'):
                A normalized [0, 255] and skull stripped intensity volume for the brain in the form of a numpy array. 
                This is the required volume to be segmented. 
            
            label ('np.array'):
                A nifti volume for the intensity image. Pixels labeled as 0 will be treated as background.

            atlases ('np.arrays'): 
                atlases nifti data files for CSF, WM, and GM as in order.        
        
        Returns:
            The segmented volume in the same shape of the passed intensity volume. The output is given in labels for 
            the segmentation, where label 0 is for background, 1 for CSF, 2 for WM, and 3 for GM. Those labels changes
            based on the tissue map columns orders. The final atlas probability has a shape of (N, K), where N is the 
            number of samples, and K is the number of clusters.

            segmentation_result ('np.array'):
                The segmentation label image in the form of numpy array.

            concatenated_atlas ('np.array'):
                An array that represents the final atlas probabilities. 
        '''
        # binary mask
        binary_mask = np.where(label == 0, 0, 1) 

        # get the atlases
        atlas_csf = atlases[0][binary_mask == 1].flatten()
        atlas_wm  = atlases[1][binary_mask == 1].flatten()
        atlas_gm  = atlases[2][binary_mask == 1].flatten()

        # concatenate the flatenned atlases to form a NxK shaped array of arrays
        concatenated_atlas = np.column_stack((atlas_csf, atlas_wm, atlas_gm))
        
        # get the argmax for each row to find which cluster does each sample refers to
        atlases_argmax = np.argmax(concatenated_atlas, axis=1) + 1

        # Create an empty segmentation result with the same shape as the original image
        segmented_image = np.zeros_like(image)

        # Reshape the atlases_argmax array to match the shape of the original image
        reshaped_atlases_argmax = atlases_argmax.reshape(image[binary_mask == 1].shape)

        # Set the background (ignored) pixels to label 0
        segmented_image[binary_mask == 0] = 0

        # set the segmented values where indexes falls to be true
        segmented_image[binary_mask == 1] = reshaped_atlases_argmax

        return segmented_image, concatenated_atlas
    
    def segment_using_tissue_models_and_atlas(self, image, label, tissue_map_csv, *atlases):
        '''(1.3) Tissue models & label propagation: multiplying both results: segmentation using intensity & position information

        Args:
            image ('np.array'):
                A normalized [0, 255] and skull stripped intensity volume for the brain in the form of a numpy array. 
                This is the required volume to be segmented.

            label ('np.array'):
                A nifti volume for the intensity image. Pixels labeled as 0 will be treated as background.

            tissue_map_csv ('Path'): 
                A csf file path that contains the tissue maps probabilities. The file should contain three columns, 
                first column for CSF, then WM, then GM.
            
            atlases ('np.arrays'): 
                atlases nifti data files for CSF, WM, and GM as in order.

        Returns:
            The segmented volume in the same shape of the passed intensity volume. The output is given in labels for 
            the segmentation, where label 0 is for background, 1 for CSF, 2 for WM, and 3 for GM. Those labels changes
            based on the tissue map columns orders. The final atlas probability has a shape of (N, K), where N is the 
            number of samples, and K is the number of clusters.

            segmentation_result ('np.array'):
                The segmentation label image in the form of numpy array.

            posteriors ('np.array'):
                An array that represents the final atlas probabilities. 

        '''
        # read the tissues moodels
        tissue_map_df = pd.read_csv(tissue_map_csv, header=None)
        tissue_map_array = tissue_map_df.values

        # map background pixels above a threshold to WM (label 2)
        threshold = 100
        bg_mask = np.arange(len(tissue_map_array)) > threshold
        tissue_map_array[bg_mask, 1] = 2

        # binary mask
        binary_mask = np.where(label == 0, 0, 1) 

        # get the atlases
        atlas_csf = atlases[0][binary_mask == 1].flatten()
        atlas_wm  = atlases[1][binary_mask == 1].flatten()
        atlas_gm  = atlases[2][binary_mask == 1].flatten()

        # concatenate the flatenned atlases to form a NxK shaped array of arrays
        concatenated_atlas = np.column_stack((atlas_csf, atlas_wm, atlas_gm))
        
        # Perform Bayesian segmentation
        registered_volume_test = image[binary_mask == 1].flatten()

        # using registered_volume_test as an index to extract specific rows from tissue_map_array
        tissue_map_array = tissue_map_array[registered_volume_test, :]

        # multiply the probabilities
        posteriors = tissue_map_array * concatenated_atlas

        # ger the argmax for each sample to know for which cluster does it belongs, +1 to avoid 0 value
        posteriors_argmax = np.argmax(posteriors, axis=1) + 1

        # Create an empty segmentation result with the same shape as the original image
        segmented_image = np.zeros_like(image)

        # Reshape the atlases_argmax array to match the shape of the original image
        reshaped_atlases_argmax = posteriors_argmax.reshape(image[binary_mask == 1].shape)

        # Set the background (ignored) pixels to label 0
        segmented_image[binary_mask == 0] = 0

        # set the segmented values where indexes falls to be true
        segmented_image[binary_mask == 1] = reshaped_atlases_argmax

        return segmented_image, posteriors

# 7. Plot Class
class Plot:
    """
    Class for generating and displaying box plots.

    Methods:
        - __init__(self) -> None: Initializes the Plot class.

        - plot_boxplot_per_tissue(self, WM_values, GM_values, CSF_values, config) -> None:
            Plots a box plot for each tissue type based on the provided data.

        - plot_boxplot_per_patient(self, values, subjects, config) -> None:
            Plots a box plot for each subject separately.

    Attributes:
        None
    """
    def __init__(self) -> None:
        pass

    def plot_boxplot_per_tissue(self, WM_values, GM_values, CSF_values, config):
        """
        Plots a box plot for each tissue type based on the provided data.

        Args:
            WM_values ('list'): List of white matter values for each patient.
            GM_values ('list'): List of gray matter values for each patient.
            CSF_values ('list'): List of cerebrospinal fluid values for each patient.
            config ('str'): Configuration information to include in the plot title.

        Returns:
            None. The function generates and displays the box plot.
        """

        # Calculate quartiles and IQR for each tissue type
        WM_q1, WM_q2, WM_q3 = np.percentile(WM_values, [25, 50, 75])
        WM_iqr = WM_q3 - WM_q1
        
        GM_q1, GM_q2, GM_q3 = np.percentile(GM_values, [25, 50, 75])
        GM_iqr = GM_q3 - GM_q1
        
        CSF_q1, CSF_q2, CSF_q3 = np.percentile(CSF_values, [25, 50, 75])
        CSF_iqr = CSF_q3 - CSF_q1

        # Combine the data for plotting
        data = [WM_values, GM_values, CSF_values]
        
        # Create a box plot
        fig, ax = plt.subplots()
        ax.boxplot(data, labels=['WM', 'GM', 'CSF'])
        
        # Set labels and title
        ax.set_ylabel('Values')
        ax.set_title(f'({config}) configurations')
        
        # Show the plot
        plt.show()


    def plot_boxplot_per_patient(self, values, subjects, config):
        '''
        Plots a boxplot for each subject separately.

        Args:
            values ('list'): A 1D list Contains the values to be plotted.
            subjects ('list'): A 1D list, same length as 'values' for the x-axis of the plot.

        Returns:
            None. The function generates and displays the box plot.
        '''
        
        # Convert WM_values to a list of lists
        data = [[value] for value in values]
        
        # Plot box plots for each patient
        fig, axs = plt.subplots(1, 1, figsize=(10, 12), sharex=True)
        
        axs.boxplot(data, labels=subjects)
        axs.set_title('WM Values ')
        axs.set_ylabel('WM Values')
        axs.set_title(f'Box Plot for Each Patient using ({config}) Configurations')
        plt.xlabel('Patient ID')
        plt.show()

# 8. EM Class
class EM:
    """
    Implementation of the Expectation-Maximization (EM) algorithm for image segmentation.

    Methods:
        - __init__(K=3, params_init_type='random', modality='multi', verbose=True): Initializes the EM algorithm.
        - initialize_for_fit(labels_gt_file, t1_path, t2_path, tissue_model_csv_dir, include_atlas, *atlases): Initializes variables for fitting.
        - skull_stripping(image, label): Performs skull stripping and returns the volume with labeled tissues only.
        - get_tissue_data(labels_gt_file, t1_path, t2_path): Removes black background from skull-stripped volume.
        - initialize_parameters(data, tissue_model_csv_dir, *atlases): Initializes model parameters.
        - multivariate_gaussian_probability(x, mean_k, cov_k, regularization=1e-4): Computes multivariate Gaussian probability.
        - expectation(): Expectation step of the EM algorithm.
        - maximization(w_ik, tissue_data): Maximization step of the EM algorithm.
        - log_likelihood(alpha, clusters_means, clusters_covar, multivariate_gaussian_probability_callback): Computes log-likelihood.
        - generate_segmentation(posteriors, gt_binary): Generates segmentation based on posterior probabilities.
        - correct_pred_labels(segmentation_result, gt_binary): Corrects predicted labels based on prior knowledge.
        - fit(n_iterations, labels_gt_file, t1_path, t2_path=None, correct_labels=True, tissue_model_csv_dir=None, atlas_csf=None, atlas_wm=None, atlas_gm=None, include_atlas=False): Fits the EM algorithm and segments the volume.

    Attributes:
        - K ('int'): Number of clusters/components.
        - params_init_type ('str'): Type of initialization for parameters ('kmeans', 'random', 'tissue_models', 'atlas', 'tissue_models_atlas').
        - modality ('str'): Modality of the input data ('multi' or 'single').
        - verbose ('bool'): Whether to print verbose output.
        - labels_gt_file ('str'): Ground truth labels file path.
        - t1_path ('str'): Path to the T1-weighted image.
        - t2_path ('str'): Path to the T2-weighted image.
        - sum_tolerance ('float'): Tolerance for sum conditions.
        - convergence_tolerance ('int'): Tolerance for convergence check.
        - seed ('int'): Seed for random initialization.
        - NM ('NiftiManager'): Nifti file manager class.
        - FM ('FileManager'): File manager class.
        - BrainAtlas ('BrainAtlasManager'): Brain atlas manager.
        - tissue_data ('np.array'): 2D array of voxel intensities for selected tissues.
        - gt_binary ('np.array'): Binary mask indicating selected tissues.
        - img_shape ('tuple'): Shape of the T1-weighted image volume.
        - n_samples ('int'): Number of samples.
        - n_features ('int'): Number of features.
        - clusters_means ('np.array'): Cluster means.
        - clusters_covar ('np.array'): Cluster covariance matrices.
        - alpha_k ('np.array'): Prior probabilities.
        - posteriors ('np.array'): Normalized posterior probabilities.
        - pred_labels ('np.array'): Predicted labels.
        - loglikelihood ('list'): List to store log-likelihood values.
        - atlas_prob ('np.array'): Atlas probabilities.
        - include_atlas ('bool'): Whether to include atlas information in the initialization.
    """
    def __init__(self, K=3, params_init_type='random', modality='multi', verbose=True):
        '''
        Initialize the Expectation-Maximization (EM) algorithm for image segmentation.

        Args:
            K ('int'): Number of clusters/components.
            params_init_type ('str'): Type of initialization for parameters ('kmeans', 'random', 'tissue_models', 'atlas', 'tissue_models_atlas').
            modality ('str'): Modality of the input data ('multi' or 'single').
            verbose (bool): Whether to print verbose output.

        Returns:
            None
        '''
        self.K                  = K
        self.params_init_type   = params_init_type
        self.modality           = modality
        self.verbose            = verbose

        self.labels_gt_file, self.t1_path, self.t2_path = None, None, None
        self.labels_nifti, self.t1_volume = None, None

        self.sum_tolerance          = 0.15
        self.convergence_tolerance  = 200
        self.seed                   = 42

        # Setting a seed
        np.random.seed(self.seed)

        # Helper classes
        self.NM         = NiftiManager()
        self.FM         = FileManager()
        self.BrainAtlas = BrainAtlasManager()

        self.tissue_data, self.gt_binary, self.img_shape = None, None, None      # (N, d) for tissue data

        self.n_samples      = None      # N samples
        self.n_features     = None      # d = number of features (dimension), 
                                        # based on the number of modalities we pass

        # create parameters objects
        self.clusters_means = None      # (K, d)
        self.clusters_covar = None      # (K, d, d)
        self.alpha_k        = None      # prior probabilities, (K,)

        self.posteriors     = None      # (N, K)
        self.pred_labels    = None      # (N,)
        self.loglikelihood  = [-np.inf]

        # atlas parameters
        self.atlas_prob     = None      # (N, K)
        self.include_atlas  = None

    def initialize_for_fit(self, labels_gt_file, t1_path, t2_path, tissue_model_csv_dir, include_atlas, *atlases):
        '''
        Initialize variables only when fitting the algorithm.

        Args:
            labels_gt_file ('path'): Ground truth labels file path.
            t1_path ('str'): Path to the T1-weighted image.
            t2_path ('str'): Path to the T2-weighted image.
            tissue_model_csv_dir ('str'): Directory containing tissue model CSV files.
            include_atlas ('bool'): Whether to include atlas information in the initialization.
            *atlases: Variable number of atlas objects.

        Returns:
            None
        '''                
        # get the atlases
        atlas_csf = atlases[0]
        atlas_wm  = atlases[1]
        atlas_gm  = atlases[2]

        # initializing skull stripping variables
        self.labels_gt_file, self.t1_path, self.t2_path \
            = labels_gt_file, t1_path, t2_path
        
        # Removing the background for the data
        self.tissue_data, self.gt_binary, self.img_shape \
                            = self.get_tissue_data(
                                self.labels_gt_file,
                                t1_path=self.t1_path,
                                t2_path=self.t2_path
                            )    # (N, d) for tissue data
        
        self.n_samples      = self.tissue_data.shape[0] # N samples
        self.n_features     = self.tissue_data.shape[1] # number of features 2 or 1 (dimension), based on the number of modalities we pass

        self.clusters_means = np.zeros((self.K, self.n_features))                       # (K, d)
        self.clusters_covar = np.zeros(((self.K, self.n_features, self.n_features)))    # (K, d, d)
        self.alpha_k        = np.ones(self.K)                                           # prior probabilities, (K,)

        self.posteriors     = np.zeros((self.n_samples, self.K), dtype=np.float64)      # (N, K)
        self.pred_labels    = np.zeros((self.n_samples,))                               # (N,)

        self.atlas_prob     = np.zeros((self.n_samples, self.K), dtype=np.float64) # atlas probabilities, (N, K)
        self.include_atlas  = include_atlas

        if self.modality not in ['single', 'multi']:
            raise ValueError('Wronge modality type passed. Only supports "single" or "multi" options.')
        
        if tissue_model_csv_dir is None and self.params_init_type == 'tissue_models':
            raise ValueError('Missing tissue_model_csv_dir argument.')
        
        if (atlas_csf is None or atlas_wm  is None or atlas_gm is None) and self.params_init_type == 'atlas':
            raise ValueError('Missing atlases argument.')
        
        if ((atlas_csf is None or atlas_wm  is None or atlas_gm is None) or (tissue_model_csv_dir is None)) and self.params_init_type == 'tissue_models_atlas': 
            raise ValueError('Missing one of the initialization arguments, either tissue_model_csv_dir or atlases.')
        
        if include_atlas and (include_atlas not in ["posteriori"]):
            raise ValueError('Error with include_atlas value. Only "posteriori" method is supported.')

        # assign model parameters their initial values
        self.initialize_parameters(self.tissue_data, tissue_model_csv_dir, atlas_csf, atlas_wm, atlas_gm)

    def skull_stripping(self, image, label):
        '''Performs only skull stripping and returns the volume with the label tissues only.
        
        Args:
            image ('np.array'):
                An intensity volume for the brain in the form of a numpy array.
            
            label ('np.array'):
                The labels volume associated to the intensity volume passed as an image.

        Returns:
            The skull stripped volume in the same shape of the passed intensity volume. The output will still contain
            background labelled as 0 as a result of the multiplication. The output volume is a numpy array.
        '''
        # convert the labels to binary form, all tissues to 1, else is 0
        labels_mask   = np.where(label == 0, 0, 1)

        # multiply the image to get only the tissues
        return np.multiply(image, labels_mask)
    
    def get_tissue_data(self, labels_gt_file, t1_path, t2_path):
        '''
        Remove the black background from the skull-stripped volume and return a 1D array of voxel intensities.

        Args:
            labels_gt_file ('str'): Ground truth labels file path.
            t1_path ('str'): Path to the T1-weighted image.
            t2_path ('str'): Path to the T2-weighted image.

        Returns:
            tissue_data ('np.array'): A 2D array of voxel intensities for selected tissues.
            labels_mask ('np.array'): Binary mask indicating the selected tissues.
            img_shape ('tuple'): Shape of the T1-weighted image volume.
        '''        
        # Check if files passed
        self.FM.check_file_existence(labels_gt_file, "labels")
        self.FM.check_file_existence(t1_path, "T1")

        # load the nifti files & creating a binary mask from the gt labels
        self.labels_nifti, _ = self.NM.load_nifti(labels_gt_file)
        labels_mask   = np.where(self.labels_nifti == 0, 0, 1)

        # loading the volume, performing skull stripping and normalization 
        self.t1_volume  = self.NM.min_max_normalization(
            self.NM.load_nifti(t1_path)[0], 255).astype('uint8')
        
        t1_selected_tissue = self.t1_volume[labels_mask == 1].flatten()

        # The true mask labels count must equal to the number of voxels we segmented
        # np.count_nonzero(labels_mask) returns the sum of pixel values that are True, the count should be equal to the number
        # of pixels in the selected tissue array
        assert np.count_nonzero(labels_mask) == t1_selected_tissue.shape[0], 'Error while removing T1 black background.'

        # put both tissues into the d-dimensional data vector [[feature_1, feature_2]]
        if self.modality == 'multi':
            self.FM.check_file_existence(t2_path, "T2")

            # loading the volumes and performing skull stripping 
            t2_volume  = self.NM.min_max_normalization(
                self.NM.load_nifti(t2_path)[0], 255).astype('uint8')
            
            t2_selected_tissue = t2_volume[labels_mask == 1].flatten()

            assert np.count_nonzero(labels_mask) == t2_selected_tissue.shape[0], 'Error while removing T2_FLAIR black background.'

            tissue_data =  np.array([t1_selected_tissue, t2_selected_tissue]).T     # multi-modality
        else:
            tissue_data =  np.array([t1_selected_tissue]).T                       # single modality

        return tissue_data, labels_mask, self.t1_volume.shape
    
    def initialize_parameters(self, data, tissue_model_csv_dir, *atlases):
        '''Initializes the model parameters and the weights at the beginning of EM algorithm. It returns the initialized parameters.

        Args:
            data ('numpy.ndarray'): The intensity image (tissue data) in its original shape.
            - tissue_map_csv_dir ('str'): path to the tissue model csv file.

        Returns:
            None

        '''
        if self.params_init_type not in ['kmeans', 'random', 'tissue_models', 'atlas', 'tissue_models_atlas']:
            raise ValueError(f"Invalid initialization type {self.params_init_type}. Both 'random' and 'kmeans' initializations are available.")

        if self.verbose: logger.info(f"Initializing model parameters using '{self.params_init_type}'.")

        if self.params_init_type == 'kmeans':
            kmeans              = KMeans(n_clusters=self.K, random_state=self.seed, n_init='auto', init='k-means++').fit(data)
            cluster_labels      = kmeans.labels_                # labels : ndarray of shape (456532,)
            centroids           = kmeans.cluster_centers_       # (3, 2)
            self.alpha_k        = np.array([np.sum([cluster_labels == i]) / len(cluster_labels) for i in range(self.K)]) # ratio for each cluster

        elif self.params_init_type == 'random':  # 'random' initialization
            random_centroids    = np.random.randint(np.min(data), np.max(data), size=(self.K, self.n_features)) # shape (3,2)
            random_label        = np.random.randint(low=0, high=self.K, size=self.n_samples) # (456532,)
            self.alpha_k        = np.ones(self.K, dtype=np.float64) / self.K 
        
        elif self.params_init_type in ['tissue_models', 'atlas', 'tissue_models_atlas']:  # 'tissue_models' or 'atlas' initialization

            # the problem here is that the data is no longer in its original shape, it is (Nxd), and we can't reshape it as it is skull stripped
            # we have to re-form the image, or pass it here in a way that data is the skull stripped image segment_using_tissue_models receives 
            # the images normalized, we normalized in an earlier step
            data_volume         = self.skull_stripping(image=self.t1_volume, label=self.labels_nifti)
            
            # self.NM.show_nifti(self.t1_volume, title="self.t1_volume init segmentation", slice=128)

            # get the segmentation labels
            segmentation = None 
            if self.params_init_type == 'tissue_models': # tissue models
                segmentation, self.atlas_prob = self.BrainAtlas.segment_using_tissue_models(image=data_volume, label=self.labels_nifti,tissue_map_csv=tissue_model_csv_dir)
            elif self.params_init_type == 'atlas': # atlas
                segmentation, self.atlas_prob = self.BrainAtlas.segment_using_tissue_atlas(data_volume, self.labels_nifti, *atlases)
            else: # using both atlas and tissue models
                segmentation, self.atlas_prob = self.BrainAtlas.segment_using_tissue_models_and_atlas(data_volume, self.labels_nifti, tissue_model_csv_dir, *atlases)

            # self.NM.show_nifti(segmentation, title="intermediate parameter init segmentation", slice=137)

            # we have to substract 1 as inside the function `segmentation_tissue_model`, we add 1 to the segmentation argmax predictions
            # we also have to remove the 0 background 
            segmentation_labels = segmentation.flatten()

            cluster_labels      = segmentation_labels[segmentation_labels!=0] - 1
            self.alpha_k        = np.array([np.sum([cluster_labels == i]) / len(cluster_labels) for i in range(self.K)]) # ratio for each cluster

            # compute the new means using the segmentation results and the data
            # adding 1 to K and starting from 1 is important, this is because label 0 is background, and we add 1 to the segmentation argmax predictions inside `segmentation_tissue_model`
            clusters_masks  = [segmentation_labels == k for k in range(1, self.K+1)]
            centroids       = np.array([np.mean(data_volume.flatten()[cluster_mask]) for cluster_mask in clusters_masks])[:, np.newaxis]

        cluster_data            = [data[cluster_labels == i] for i in range(self.K)] if self.params_init_type in ['kmeans', 'tissue_models', 'atlas', 'tissue_models_atlas']  \
                                    else [data[random_label == i] for i in range(self.K)]
        
        # update model parameters (mean and covar)
        self.clusters_means     = centroids if self.params_init_type in ['kmeans', 'tissue_models', 'atlas', 'tissue_models_atlas'] else random_centroids
        self.clusters_covar     = np.array([np.cov(cluster_data[i], rowvar=False) for i in range(self.K)]) # (K, d, d)

        # validating alpha condition
        assert np.isclose(np.sum(self.alpha_k), 1.0, atol=self.sum_tolerance), 'Error in self.alpha_k calculation in "initialize_parameters". Sum of all self.alpha_k elements has to be equal to 1.'

    def multivariate_gaussian_probability(self, x, mean_k, cov_k, regularization=1e-4):
        '''
        Compute the multivariate and single variate gaussian probability density function (PDF) for a given data data.
        The function can handle single or multi-modality (dimensions) and computes the probability on all of the 
        data without a complex iteratitve matrix multiplication.

        Args:
            x ('numpy.ndarray'): The data points.
            mean_k ('numpy.ndarray'): The mean vector for cluster K.
            cov_k ('numpy.ndarray'): The covariance matrix for cluster K.

        Returns:
            float: The probability density at the given data point.
        '''

        dim = self.n_features
        x_min_mean = x - mean_k.T # Nxd
        
        if dim == 1 and cov_k.shape == (): # single modality

            # to handle nan cov_k and inversion in certain cases (mainly when randomly initializing)     
            # we add a small regularisation term to enable the inverse and not to have nan in the final
            # matrix           
            cov_k +=  regularization
            
            # the covariance matrix is a scalar value, thus the inverse is 1 / scalar value
            inv_cov_k = 1 / cov_k

            # to not change the multiplication formula below, we convert it to a (1,1) matrix
            inv_cov_k = np.array([[inv_cov_k.copy()]])
            
            # the determinant is only used for square matrices, for a scalar value, det(a) = a
            determinant = cov_k

        else: # multi-modality

            # to handle nan cov_k and inversion in certain cases (mainly when randomly initializing)     
            # we add a small regularisation term to enable the inverse and not to have nan in the final
            # matrix           
            cov_k += np.eye(cov_k.shape[0]) * regularization

            try:
                inv_cov_k = np.linalg.inv(cov_k)
            except np.linalg.LinAlgError:
                inv_cov_k = np.linalg.pinv(cov_k) # Handle singularity by using the pseudo-inverse

            determinant = np.linalg.det(cov_k)

        exponent = -0.5 * np.sum((x_min_mean @ inv_cov_k) * x_min_mean, axis=1)
        denominator = (2 * np.pi) ** (dim / 2) * np.sqrt(determinant)

        return (1 / denominator) * np.exp(exponent)

    def expectation(self):
        '''
        Expectation step of the EM algorithm.

        The function initializes the probability placeholder on every iteration, then computes the cluster multivariate gaussian probability for every cluster.
        The final normalized posterior probabilities are normalized to ensure the sum of every voxel probabilities for the three clusters is equal to 1.

        Returns:
            posteriors ('np.array'): Normalized posterior probabilities.
        '''
        # initialize membership weights probabilities
        # has to be reset to empty placeholder in every iteration to avoid accumulating the values, the assert below will validate
        posteriors     = np.zeros((self.n_samples, self.K), dtype=np.float64) # posterior probabilities, (456532, 3)

        # calculating the normalised posterior probability for every k cluster using multivariate_gaussian_probability
        for k in range(self.K):

            cluster_prob = self.multivariate_gaussian_probability(
                 x=self.tissue_data, 
                 mean_k=self.clusters_means[k], 
                 cov_k=self.clusters_covar[k]) 
            
            # cluster_prob = multivariate_normal.pdf(
            #    x=self.tissue_data, 
            #    mean=self.clusters_means[k], 
            #    cov=self.clusters_covar[k],
            #    allow_singular=True)
                                    
            # updates every k cluster column 
            posteriors[:,k] = cluster_prob * self.alpha_k[k] 

        # normalize the posteriors "membership weights" row by row separately by dividing by the total sum of each row
        posteriors /= np.sum(posteriors, axis=1)[:, np.newaxis]

        # the sum of the 3 clusters probabilities should be equal to 1
        assert np.isclose(np.sum(posteriors[0,]), 1.0, atol=self.sum_tolerance), 'Error with calculating the posterior probabilities "membership weights" for each voxel.'
        
        return posteriors
    
    def maximization(self, w_ik, tissue_data):
        '''
        Maximization M-Step of EM algorithm.

        The function updates the model parameters (mean and covariance matrix) as well as updates the weights (alphas) for every cluster.

        Args:
            w_ik ('np.array'): Membership weights.
            tissue_data ('np.array'): Tissue data array.

        Returns:
            alpha_k ('np.array'): Updated alpha priors.
            mu_k ('np.array'): Updated cluster means.
            covariance_matrix ('np.array'): Updated covariance matrices.
        '''

        # Computing the new means and covariance matrix
        covariance_matrix = np.zeros(((self.K, self.n_features, self.n_features)))
        mu_k = np.zeros((self.K, self.n_features))
        alpha_k = np.ones(self.K)

        for k in range(self.K):
            # sum of weights for every k
            N_k = np.sum(w_ik[:, k])

            # Mean 
            mu_k[k] = np.array([np.sum(w_ik[:, k] * tissue_data[:, i]) / N_k for i in range(self.n_features)])
            
            # covariance 
            x_min_mean = tissue_data-mu_k[k]
            weighted_diff = w_ik[:, k][:, np.newaxis] * x_min_mean
            covariance_matrix[k] = np.dot(weighted_diff.T, x_min_mean) / N_k

            # alpha priors
            alpha_k[k] = N_k / self.n_samples

        # validating alpha condition
        # assert np.isclose(np.sum(alpha_k), 1, atol=self.sum_tolerance), f'Error in self.alpha_k calculation in "maximization". Sum of all self.alpha_k elements has to be equal to 1. np.sum(alpha_k)={np.sum(alpha_k)}'

        return alpha_k, mu_k, covariance_matrix
    
    def log_likelihood(self, alpha, clusters_means, clusters_covar, multivariate_gaussian_probability_callback):
        '''
        Compute the log-likelihood of the EM algorithm.

        Args:
            alpha ('np.array'): Prior probabilities.
            clusters_means ('np.array'): Cluster means.
            clusters_covar ('np.array'): Cluster covariance matrices.
            multivariate_gaussian_probability_callback ('function'): Callback function for multivariate gaussian probability computation.

        Returns:
            float: Log-likelihood value.
        '''
        
        return np.sum(
                    np.log(
                        np.sum(
                            alpha[k] * multivariate_gaussian_probability_callback(
                            x=self.tissue_data, 
                            mean_k=clusters_means[k], 
                            cov_k=clusters_covar[k]) for k in range(self.K))
                        )
                    )

    def generate_segmentation(self, posteriors, gt_binary):
        '''
        Generate segmentation based on posterior probabilities.

        Args:
            posteriors ('np.array'): Normalized posterior probabilities.
            gt_binary ('np.array'): Binary ground truth volume.

        Returns:
            np.array: Segmentation result.
        '''
        predictions = np.argmax(posteriors, axis=1) + 1
        gt = gt_binary
        gt[gt == 1] = predictions
        
        return gt.reshape(self.img_shape)
    
    def correct_pred_labels(self, segmentation_result, gt_binary):
        '''
        Correct the predicted labels based on prior knowledge.
        
        Args:
            segmentation_result ('np.array'): segmentation volume resulted from the algorithm
            gt_binary ('np.array'): binarized ans flattened volume for the segmented volume, the label/ground truth.

        Returns:
            corrected_segmentation (np.array): segmentation volume with the corrected label for each cluster.
        '''

        if self.verbose: logger.info("Finished segmentation. Correcting prediction labels.")

        means = np.mean(self.clusters_means, axis=1)

        # this is based on prior knowledge, the RHS are the corrected labels
        # assuming that CSF=1 (lowest mean), GM=2, and WM=3(highest mean)
        
        # labels for lab 1, where ing gt: CSF=1, GM=2, and CSF=3
        # highest_mean = np.argmax(means) + 1
        # lowest_mean  = np.argmin(means) + 1
        # middle_mean  = len(means) - highest_mean - lowest_mean + 3
        # labels = {
        #     np.argmax(means) + 1: 3, 
        #     len(means) - np.argmax(means) - np.argmin(means) + 1: 2, 
        #     np.argmin(means) + 1: 1}
        
        # labels for lab 3, where ing gt: CSF=1, WM=2, and GM=3
        # the max is wm k=2, we correct it to gm
        labels = {
            np.argmax(means) + 1: 2,  # highest mean is csf in the new dataset of lab 3
            len(means) - np.argmax(means) - np.argmin(means) + 1: 3,  # middle mean is wm, label 2
            np.argmin(means) + 1: 1}
                
        # Modify the labels based on means
        flattened_result = segmentation_result.flatten()
        for mean_index, label_corrected in labels.items():
            gt_binary[flattened_result == mean_index] = label_corrected

        corrected_segmentation = gt_binary.reshape(self.img_shape)

        return corrected_segmentation

    def fit(self, 
            n_iterations, 
            labels_gt_file, 
            t1_path, 
            t2_path = None ,
            correct_labels=True, 
            tissue_model_csv_dir=None, 
            atlas_csf = None, 
            atlas_wm = None, 
            atlas_gm= None,
            include_atlas = False
            ):
        '''
        Main function that fits the EM algorithm and segments the given volume.

        Args:
            n_iterations ('int'): Number of iterations for the EM algorithm.
            labels_gt_file ('str'): Ground truth labels file path.
            t1_path ('str'): Path to the T1-weighted image.
            t2_path ('str', optional): Path to the T2-weighted image.
            correct_labels ('bool'): Whether to correct the predicted labels.
            tissue_model_csv_dir ('str'): Directory containing tissue model CSV files.
            atlas_csf ('nibabel'): Loaded NIfTI for CSF.
            atlas_wm ('nibabel'): Loaded NIfTI for WM.
            atlas_gm ('nibabel'): Loaded NIfTI for GM.
            include_atlas ('bool'): Whether to include atlas information in the initialization.

        Returns:
            np.array: Segmentation result.
        '''

        if self.verbose: logger.info(f"Starting the algorithm. {n_iterations} iterations were initialized.")

        # Initialize parameters for fitting
        self.initialize_for_fit(labels_gt_file, t1_path, t2_path, tissue_model_csv_dir, include_atlas, atlas_csf, atlas_wm, atlas_gm)
        
        current_idx         = 0

        # Starting with an M-step as we have information from the atlas that we can initialize from
        if include_atlas and include_atlas == "posteriori" and self.params_init_type in ['tissue_models', 'atlas', 'tissue_models_atlas']:
            self.alpha_k, self.clusters_means, self.clusters_covar = self.maximization(self.atlas_prob, self.tissue_data)

        while (current_idx <= n_iterations):            
            # E-Step
            self.posteriors = self.expectation()
                        
            # Log-likelihood convergance check
            current_likelihood = self.log_likelihood(self.alpha_k, self.clusters_means, self.clusters_covar, self.multivariate_gaussian_probability)  

            if (np.abs(current_likelihood - self.loglikelihood[-1]) < self.convergence_tolerance):
                break

            self.loglikelihood.append(current_likelihood)
            
            # M Step
            self.alpha_k, self.clusters_means, self.clusters_covar = self.maximization(self.posteriors, self.tissue_data)

            current_idx += 1

        if include_atlas and include_atlas == "posteriori" and self.params_init_type not in ['kmeans', 'random']:
            if self.verbose: logger.info(f"Including atlas probabilities into EM result using {include_atlas} method.")
            self.posteriors *= self.atlas_prob
        
        if self.verbose: logger.info(f"Iterations performed: {current_idx-1}. Generating segmentation results.")
        
        # creating a segmentation result with the predictions
        segmentation_result = self.generate_segmentation(
            posteriors=self.posteriors, 
            gt_binary=self.gt_binary.flatten())
        
        return self.correct_pred_labels(segmentation_result, self.gt_binary.flatten()) if correct_labels else segmentation_result
