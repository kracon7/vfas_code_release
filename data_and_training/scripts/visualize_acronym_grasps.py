'''
Visualize positive, negative and hard negative grasp poses from ACRONYM dataset
'''

import numpy as np
import argparse
import pyrender
from render_environment import RenderEnvironment, create_gripper_marker

np.random.seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_param_file", type=str,
                        default="config/GraspEvaNet_Dataset_SR_Gripper.yaml")
    parser.add_argument("--mesh_dict_file", type=str,
                        default="config/object_mesh_dict.yaml")
    parser.add_argument("--grasp_label", type=str, default="positive_grasps",
                        help='choose from positive_grasps, negative_grasps or hard_negative_grasps')
    parser.add_argument("--cat_idx", type=int, default=0,
                        help="index for object category in object_mesh_dict.yaml")
    parser.add_argument("--mesh_idx", type=int, default=0,
                        help="index for mesh in the selected object category")

    args = parser.parse_args()

    env = RenderEnvironment(args)
    env.scene.remove_object('gripper')
    mesh_dict = env.mesh_dict
    obj_cat = list(mesh_dict.keys())

    cid, mid, gid = args.cat_idx, args.mesh_idx, args.grasp_idx
    env.scene.arrange_scene(obj_cat[cid], mesh_dict[obj_cat[cid]][mid])

    color_mapping = {'positive_grasps': [0,250,0],
                     'negative_grasps': [80, 10, 10],
                     'hard_negative_grasps': [255, 0, 0]}

    grasps = env.scene.objs['obj'][args.grasp_label]
    num_grasps = len(grasps)

    color =color_mapping[args.grasp_label]
    gripper_marker = create_gripper_marker(color=color, tube_radius=0.001)
        
    for i in range(100):
        
        o_t_g = grasps[np.random.randint(num_grasps)]

        env.scene._renderer.add_object('_%d'%i, gripper_marker, o_t_g)

    pyrender.Viewer(env.scene._renderer._scene, use_raymond_lighting=True)
