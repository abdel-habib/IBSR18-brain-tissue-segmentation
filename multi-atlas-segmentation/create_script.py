import sys
sys.path.append('../')

import argparse
import os
from glob import glob
from EM import FileManager, ElastixTransformix

FM    = FileManager()
ET    = ElastixTransformix()

if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    parser.add_argument('--fixed_path', type=str, default='../TrainingValidationTestSets/Validation_Set', help='root dir for fixed data') # use the test data for submission
    parser.add_argument('--moving_path', type=str, default='../TrainingValidationTestSets/Training_Set', help='root dir for moving data')
    parser.add_argument('--parameters_path', type=str, default='./registration-parameters', help='root dir for elastix parameters. The script will use all the parameters in this directory. A single .txt file can also be used.')
    parser.add_argument('--output_path', type=str, default='output', help='root dir for output scripts')
    parser.add_argument('--experiment_name', type=str, default='elastix_01', help='experiment name')
    parser.add_argument('--type', type=str, default='elastix', help='type of the bat file to be generated. Either "elastix" or "transformix"')

    # parse the arguments
    args = parser.parse_args()

    # check if parameters_path is .txt file
    if os.path.isfile(args.parameters_path):
        parameters      = args.parameters_path

        # cteate params key folder name
        reg_params      = '-p "{}"'.format(parameters).replace('\\', '/')
        reg_params_key  = parameters.split('/')[-1].replace('.txt', '')
        transform_idx   = 0

    elif os.path.isdir(args.parameters_path):
        # get the parameters from the parameters_path
        parameters = os.listdir(args.parameters_path)

        if len(parameters) == 0:
            print(f"No parameters found in {args.parameters_path} directory.")
            sys.exit(1)
    
        # cteate params key folder name
        reg_params      = ' '.join(['-p "{}"'.format(os.path.join(args.parameters_path, param)) for param in parameters]).replace('\\', '/')    
        reg_params_key  = '+'.join(['{}'.format(param.replace('.txt', '')) for param in parameters])
        transform_idx   = len(parameters) - 1

    # create experiment output
    # args.experiment_name is useful to distinguish between different experiments (e.g. with or without preprocessing, etc.)
    # reg_params_key is useful to distinguish between different registration parameters (e.g. Par0003, Par0004, etc.)
    args.exp_output = os.path.join(args.output_path, args.experiment_name, reg_params_key)
    FM.create_directory_if_not_exists(args.exp_output)

    # get the fixed volumes
    fixed_volumes = sorted(glob(os.path.join(args.fixed_path, "***", "IBSR_*.nii.gz"), recursive=True))
    fixed_volumes = [path for path in fixed_volumes if 'seg' not in path]

    # get the moving volumes and labels
    moving_volumes = sorted(glob(os.path.join(args.moving_path, "***", "IBSR_*.nii.gz"), recursive=True))
    moving_volumes = [path for path in moving_volumes if 'seg' not in path]

    moving_labels = sorted(glob(os.path.join(args.moving_path, "***", "IBSR_*_seg.nii.gz"), recursive=True))

    # creating the .bat file
    with open(os.path.join(args.exp_output, f'{args.type}.bat'), 'w') as file:
        file.write("@echo on\n")
        file.write(f"echo To execute this file, use: call {os.path.join(args.exp_output, f'{args.type}.bat')} \n\n")
        # registration / transformation loop
        for fixed_volume in fixed_volumes: 
            reg_fixed_name  = fixed_volume.replace('\\', '/').split("/")[-1].split(".")[0]

            file.write(f"Fixed {reg_fixed_name}\n")

            for moving_volume, moving_label in zip(moving_volumes, moving_labels):
                reg_moving_name = moving_volume.replace("\\", "/").split("/")[-1].split(".")[0]
                reg_moving_label_name = moving_label.replace("\\", "/").split("/")[-1].split(".")[0]

                transform_path = f'{args.exp_output}/images/output_{reg_fixed_name}/{reg_moving_name}/TransformParameters.{transform_idx}.txt'
                elastix_output_dir = f'{args.exp_output}/images/output_{reg_fixed_name}/{reg_moving_name}'
                transformix_output_dir = f'{args.exp_output}/labels/output_{reg_fixed_name}/{reg_moving_label_name}'

                FM.create_directory_if_not_exists(elastix_output_dir)
                FM.create_directory_if_not_exists(transformix_output_dir)

                elastix_command_line = f'..\\elastix_windows32_v4.2\\elastix -f "{fixed_volume}" -m "{moving_volume}" {reg_params} -out "{elastix_output_dir}"' #-mMask "{moving_label}"
                trasformix_command_line = f'..\\elastix_windows32_v4.2\\transformix -in "{moving_label}" -tp "{transform_path}"  -out "{transformix_output_dir}"'

                if args.type == 'elastix':
                    file.write(f"{elastix_command_line}\n\n")
                elif args.type == 'transformix':
                    file.write(f"{trasformix_command_line}\n\n")


