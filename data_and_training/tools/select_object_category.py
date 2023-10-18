import os
import argparse
import h5py
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy

np.set_printoptions(precision=1, suppress=True)

def draw_frame(origin, scipy_q, scale=1):
    # open3d quaternion format qw qx qy qz
    o3d_quat = np.array([scipy_q[3], scipy_q[0], scipy_q[1], scipy_q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    return frame_rot

def draw_poses(poses, mesh_path, ncol=8):
    mesh_list = []
    mesh_list.append(draw_frame([0,0,0], [0,0,0,1], scale=0.5))
    for i in range(poses.shape[0]):
        # Add offset to object pose
        w_t_g = poses[i].copy()
        w_t_g[:3,3] += (i%ncol) * np.array([.4, 0, 0])
        w_t_g[:3,3] += (i//ncol) * np.array([0, .4, 0])
        
        gripper_mesh = o3d.io.read_triangle_mesh(mesh_path).transform(w_t_g)
        mesh_list.append(copy.deepcopy(gripper_mesh))

    o3d.visualization.draw_geometries(mesh_list)

def main(args):
    ncol = 5

    object_info = h5py.File(args.info, 'r')
    obj_cats = object_info['categories'].keys()

    white_list = []

    x = ''
    while x != 's':

        cat = input("Enter object category: ")
        if cat == 's':
            break
        if not cat in list(obj_cats):
            print("Category %s does not exist")
            continue
        
        obj_keys = object_info['categories'][cat][()]
        print("Visualize category %s with %d meshes"%(cat, len(obj_keys)))

        mesh_list = [draw_frame([0,0,0], [0,0,0,1], scale=0.2)]
        for i, obj_key in enumerate(obj_keys):
            obj = object_info['meshes'][obj_key]
            mesh_path = os.path.join('dataset', obj['path'][()].decode())
            w_t_g = np.eye(4)
            w_t_g[:3,3] += (i % ncol) * np.array([.4, 0, 0])
            w_t_g[:3,3] += (i // ncol) * np.array([0, .4, 0])
            obj_mesh = o3d.io.read_triangle_mesh(mesh_path).transform(w_t_g)

            mesh_list.append(copy.deepcopy(obj_mesh))

        o3d.visualization.draw_geometries(mesh_list)

        x = input("Do you want to pick %s?\n press 'y' to confirm, press 's' to stop"%(cat))
        if x == 'y' and cat not in white_list:
            white_list.append(cat)

    object_info.close()

    content = {'white_list': white_list}, {'black_list':[]}
    with open(args.output, 'w') as file:
        documents = yaml.dump(content, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--info", type=str, default='dataset/object_info.hdf5')
    parser.add_argument("--output", type=str, default='config/object_instances_whitlist.yaml')

    args = parser.parse_args()
    main(args)

