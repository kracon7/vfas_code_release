import os
import torch
import copy
import numpy as np
import pandas as pd
import open3d as o3d

CMAP = [[60, 10, 10],
        [10, 60, 10],
        [10, 10, 60],
        [60, 60, 10],
        [60, 10, 60],
        [10, 60, 60],
        [60, 60, 60],
        [110, 60, 60],
        [60, 110, 60],
        [60, 60, 110]]

def draw_frame(origin, q, scale=1):
    # open3d quaternion format qw qx qy qz
    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    return frame_rot

def visualize_pcd(pcd):
    points = pcd[:, :3]
    seg_id = pcd[:, 3].astype('int')
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_pcd.normals = o3d.utility.Vector3dVector(pcd[:,4:])
    o3d_color = np.array([CMAP[i] for i in seg_id]).astype('float') / 255
    o3d_pcd.colors = o3d.utility.Vector3dVector(o3d_color)

    # Origin frame
    frame_base = draw_frame(np.zeros(3), np.array([0, 0, 0, 1]), scale=0.1)
    o3d.visualization.draw_geometries([frame_base, o3d_pcd])


if __name__ == "__main__":
    scene_dir = 'grasp_pcd_dataset/scene_000008'
    df = pd.read_csv(os.path.join(scene_dir, 'dataset.csv'), header=None)  

    # N = 10
    # idx = np.random.choice(len(df), size=N, replace=False)

    idx = [500*i for i in range(11)]

    for i in idx:
        data_pack = np.load(df.iloc[i, 0])
        pcd = data_pack['pcd'][0]
        print("Item %d, label: %d"%(i, data_pack['label'][0]))
        
        visualize_pcd(pcd)