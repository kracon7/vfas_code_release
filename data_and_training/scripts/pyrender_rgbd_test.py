import copy
import argparse
import numpy as np
from render_environment import RenderEnvironment
import open3d as o3d
import matplotlib.pyplot as plt

def draw_frame(origin, q, scale=1):
    # open3d quaternion format qw qx qy qz
    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    return frame_rot

def main(args):
    env = RenderEnvironment(args)
    mesh_dict = env.mesh_dict
    obj_cat = list(mesh_dict.keys())

    cid, mid, gid = args.cat_idx, args.mesh_idx, args.grasp_idx
    env.scene.arrange_scene(obj_cat[cid], mesh_dict[obj_cat[cid]][mid])
    env.set_one_grasp(grasp_idx=gid)

    color, depth = env.get_wrist_cam_rgbd()

    _, axes = plt.subplots(2, 1)
    axes[0].imshow(color, vmin=0, vmax=255)
    axes[1].imshow(depth)
    plt.show()

    pcd = env.get_wrist_cam_pcd(crop=False).numpy()
    points = pcd[0,:,:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd[0,:,3:6].astype('float')/255)
    # Origin frame
    frame_base = draw_frame(np.zeros(3), np.array([0, 0, 0, 1]), scale=0.05)
    o3d.visualization.draw_geometries([frame_base, o3d_pcd])


    pcd = env.get_wrist_cam_pcd(crop=True).numpy()
    points = pcd[0,:,:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd[0,:,3:6].astype('float')/255)
    # Origin frame
    frame_base = draw_frame(np.zeros(3), np.array([0, 0, 0, 1]), scale=0.05)
    o3d.visualization.draw_geometries([frame_base, o3d_pcd])


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
    parser.add_argument("--grasp_idx", type=int, default=200,
                        help="index for the corresponding grasp pose")

    args = parser.parse_args()
    main(args)
