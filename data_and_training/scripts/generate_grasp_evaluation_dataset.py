'''
Dataset Generation Pipeline:
    -- Randomly spawn a scene.
    -- Filter grasps based on collision.
    -- Generate point cloud for each filtered grasp and save corresponding label.

label: positive      --  1
       negative      --  0
       hard negative -- -1

CSV rows:
pcd tensor path  |  grasp label  |  object category  |  object mesh path

A dictionary of object info will also be saved as pickle file for scene
reconstruction and grasp pose visualization after network training
The obj_info dict is structured as:
obj_info.keys() = ["table", "obj_2", "obj_3"....]
obj_info["obj_2"].keys() = ['mesh', 'pose', 'grasps', 'positive_grasps', 'negative_grasps', 
                            'hard_negative_grasps', 'valid_positive', 'valid_negative', 
                            'valid_hard_negative']
obj_info["obj_2"]["mesh"] is a trimesh object                            
'''

import yaml
import argparse
import random
import numpy as np
from grasp_dataset_generator import GraspDatasetGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_param_file", type=str,
                        default="config/GraspEvaNet_Dataset_SR_Gripper.yaml")
    parser.add_argument("--mesh_dict_file", type=str,
                        default="config/object_mesh_dict.yaml")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output_dir", type=str, 
                        default="grasp_pcd_dataset")
    parser.add_argument("--mesh_list_index", type=int, default=-1)
    
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    data_generator = GraspDatasetGenerator(args)
    num_meshes = len(data_generator.mesh_list)
    if args.mesh_list_index < 0 or args.mesh_list_index >= num_meshes:
        print("Mesh index out of range, randomly pick one...")
        idx = np.random.choice(num_meshes)
    else:
        idx = args.mesh_list_index

    data_generator.run(idx)