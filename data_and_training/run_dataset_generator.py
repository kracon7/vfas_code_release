import os
import argparse
import numpy as np
import random

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
    parser.add_argument("--scene_start_index", type=int, default=-1,
                        help="indes to start in object_mesh_dict.yaml")
    parser.add_argument("--num_scenes", type=int, default=10)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Randomly sample and place object
    if args.scene_start_index < 0:
        for i in range(args.num_scenes):
            seed = args.seed + i
            cmd = 'python scripts/generate_grasp_evaluation_dataset.py '\
                    '--env_param_file %s --mesh_dict_file %s --seed %d --output_dir %s'%(
                        args.env_param_file, 
                        args.mesh_dict_file, 
                        seed, 
                        args.output_dir
                    )

            print('\n\n\n\n================= Seed %d ====================\n'%seed, cmd)
            os.system(cmd)
            
    # Deterministic selection from white list
    else:
        for i in range(args.scene_start_index, args.scene_start_index + args.num_scenes, 1):
            cmd = 'python scripts/generate_grasp_evaluation_dataset.py '\
                    '--env_param_file %s --mesh_dict_file %s --output_dir %s --mesh_list_index %d'%(
                        args.env_param_file, 
                        args.mesh_dict_file,
                        args.output_dir,
                        i
                    )
            print('\n\n\n\n================= Deterministic %d ====================\n'%i, cmd)
            os.system(cmd)