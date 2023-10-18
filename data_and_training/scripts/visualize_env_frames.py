'''
Test for:
(1) GPU parallel physics
(1) GPU parallel rendering in Isaac Gym
(2) point cloud generation from wrist camera
(3) segmentation id from rendering
'''
import os
import argparse
import random
import numpy as np
import yaml
import trimesh
import pyrender
import open3d as o3d
import h5py
from scipy.spatial.transform import Rotation

import utils

np.random.seed(1111)
np.set_printoptions(precision=4, suppress=True)


def draw_frame(scale=0.2):
    origin = [0, 0, 0]
    cx = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [scale, 0, 0]],
    )
    cx.visual.face_colors = [255,0,0]
    cy = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [0, scale, 0]],
    )
    cy.visual.face_colors = [0,255,0]
    cz = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [0, 0, scale]],
    )
    cz.visual.face_colors = [0,0,255]
    tmp = trimesh.util.concatenate([cx, cy, cz])
    return tmp


def draw_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    # Left finger
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.03, 0, -0.04],
            [0.03, 0, 0],
        ],
    )
    # Right finger
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-0.03, 0, -0.04],
            [-0.03, 0, 0],
        ],
    )

    cb1 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-0.03, 0, -0.04], [0.03, 0, -0.04]],
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, -0.04], [0, 0, -0.055]]
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp

def add_mesh(scene, name, mesh, pose=None):
    if pose is None:
        pose = np.eye(4, dtype=np.float32)

    node = pyrender.Node(
        name=name,
        mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
        matrix=pose,
    )
    scene.add_node(node)


def visualize_pcd(env, env_id):
    xyzrgbas = env.get_wrist_cam_pcd(downsample=True, to_frame='world')
    xyzrgbas = xyzrgbas[env_id]
    points = xyzrgbas.cpu().numpy()[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_color = xyzrgbas.cpu().numpy()[:, 3:6] / 255
    pcd.colors = o3d.utility.Vector3dVector(o3d_color)

    # Origin frame
    frame_base = draw_frame(np.zeros(3), np.array([0, 0, 0, 1]), scale=0.3)

    # End Effector frame (Mesh frame)
    ee_pose = env._get_actor_root_state(env_id, env.env_gripper_map[env_id])
    w_t_ee = utils.tensor_pose_to_matrix(ee_pose[:7]).cpu().numpy()
    quat_ee = Rotation.from_matrix(w_t_ee[:3,:3]).as_quat()
    frame_ee = draw_frame(w_t_ee[:3,3], quat_ee, scale=0.1)

    # Grasp frame
    ee_t_g = np.linalg.inv(env.g_t_ee)
    w_t_g = w_t_ee @ ee_t_g
    quat_grasp = Rotation.from_matrix(w_t_g[:3,:3]).as_quat()
    frame_grasp = draw_frame(w_t_g[:3,3], quat_grasp, scale=0.1)

    # Camera frame
    w_t_c = utils.gym_transform_to_array(env._gym.get_camera_transform(env._sim, env._envs[env_id], 0))
    quat_cam = Rotation.from_matrix(w_t_c[:3,:3]).as_quat()
    frame_cam = draw_frame(w_t_c[:3,3], quat_cam, scale=0.1)

    # Cam view frame
    w_t_v = env.get_w_t_v(env_id, env.env_wrist_cam_map[env_id])
    quat_view = Rotation.from_matrix(w_t_v[:3,:3]).as_quat()
    frame_view = draw_frame(w_t_v[:3,3], quat_view, scale=0.1)

    gripper_mesh = o3d.io.read_triangle_mesh(
                "resources/SR_Gripper_Collision_Open.stl").transform(w_t_ee)
    o3d.visualization.draw_geometries([frame_base, frame_ee, frame_grasp, frame_cam, frame_view, gripper_mesh, pcd])


def main(args):

    with open(args.env_param_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.Loader)

    scene = pyrender.Scene()
    v = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)

    print("Visualizing frames on the gripper")

    origin_frame = draw_frame(scale=0.1)
    add_mesh(scene, 'origin_frame', origin_frame)

    # Gripper mesh offset
    grasp_t_ee = np.eye(4)
    grasp_t_ee[:3, :3] = np.array(params['grasp_t_ee']['rotation']).reshape(3,3)
    grasp_t_ee[:3, 3] = np.array(params['grasp_t_ee']['translation'])

    # eye to mesh (gripper base) transformation
    ee_t_cam = np.eye(4)
    ee_t_cam[:3,:3] = Rotation.from_euler(params['ee_t_cam']['euler_seq'],
                                          params['ee_t_cam']['euler_angles'],
                                          degrees=True).as_matrix()
    ee_t_cam[:3,3] = np.array(params['ee_t_cam']['translation'])

    ee_frame = draw_frame(scale=0.15)
    add_mesh(scene, 'ee_frame', ee_frame, grasp_t_ee)

    grasp_t_cam = grasp_t_ee @ ee_t_cam
    cam_frame = draw_frame(scale=0.1)
    add_mesh(scene, 'cam_frame', cam_frame, grasp_t_cam)

    gripper_mesh = trimesh.load(params['gripper_mesh'])
    color = (np.asarray((0.8, 0.8, 0.8)))
    gripper_mesh.visual.face_colors = np.tile(
        np.reshape(color, [1, 3]), [gripper_mesh.faces.shape[0], 1]
    )

    add_mesh(scene, 'gripper', gripper_mesh, grasp_t_ee)

    x = input('Press any key to view configuration with a grasp pose...')
    v.close_external()

    # ========================================================================
    scene = pyrender.Scene()
    v = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)

    origin_frame = draw_frame(scale=0.1)
    add_mesh(scene, 'origin_frame', origin_frame)

    obj_info = h5py.File(open(params['object_info'], "rb"))
    dataset_path = params['dataset_root']

    with open(args.mesh_dict_file, 'r') as f:
        mesh_dict = yaml.load(f, Loader=yaml.Loader)

    catetory = list(mesh_dict.keys())[args.cat_idx]
    mesh_id = mesh_dict[catetory][args.mesh_idx]
    grasp_label = 'positive_grasps'
    grasp_index = args.grasp_idx

    object_t_grasp = obj_info['meshes'][mesh_id][grasp_label][grasp_index]

    mesh_path = os.path.join(
            dataset_path, obj_info['meshes'][mesh_id]["path"].asstr()[()]
        )
    obj_mesh = trimesh.load(mesh_path, force="mesh")
    add_mesh(scene, 'obj', obj_mesh)

    grasp_frame = draw_frame(scale=0.08)
    add_mesh(scene, 'grasp_frame', grasp_frame, object_t_grasp)

    object_t_ee = object_t_grasp @ grasp_t_ee
    add_mesh(scene, 'gripper', gripper_mesh, object_t_ee)

    object_t_cam = object_t_grasp @ grasp_t_cam
    add_mesh(scene, 'cam_frame', cam_frame, object_t_cam)

    x = input('Press any key to close viewer...')
    v.close_external()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_param_file", type=str,
                        default="config/GraspEvaNet_Dataset_SR_Gripper.yaml")
    parser.add_argument("--mesh_dict_file", type=str,
                        default="config/object_mesh_dict.yaml")
    parser.add_argument("--cat_idx", type=int, default=0,
                        help="index for object category in object_mesh_dict.yaml")
    parser.add_argument("--mesh_idx", type=int, default=0,
                        help="index for mesh in the selected object category")
    parser.add_argument("--grasp_idx", type=int, default=300,
                        help="index for the corresponding grasp pose")
    
    args = parser.parse_args()
    main(args)
