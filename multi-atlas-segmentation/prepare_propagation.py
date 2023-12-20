import sys
sys.path.append('../')

import argparse
import os
from glob import glob
from EM import FileManager, ElastixTransformix, NiftiManager, BrainAtlasManager, Evaluate

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

    # parse the arguments
    args = parser.parse_args()

    # args.experiment_name is useful to distinguish between different experiments (e.g. with or without preprocessing, etc.)
    # reg_params_key is useful to distinguish between different registration parameters (e.g. Par0003, Par0004, etc.)
    args.exp_output = os.path.join(args.output_path, args.experiment_name, args.reg_params_key)

    # get the number of transformation params files based on the reg_params_key
    args.num_transform_params = len(args.reg_params_key.split('+')) - 1

    # get the transformation params files from each registered intensity folder
    args.transform_params_output = sorted(glob(os.path.join(args.exp_output, "images", "***", "***", f"TransformParameters.{args.num_transform_params}.txt"), recursive=True))

    print("Number of Transform params files: ", len(args.transform_params_output))

    # modify the interpolator order in each of the transformation params files
    for transform_param in args.transform_params_output:
        print(f"Modifying interpolator order in {transform_param}")
        FM.replace_text_in_file(
            transform_param, 
            search_text = '(FinalBSplineInterpolationOrder 3)', 
            replacement_text =  '(FinalBSplineInterpolationOrder 0)')