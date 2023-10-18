import os
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import numpy as np
from scipy.spatial.transform import Rotation as sciR
import trimesh.transformations as tra
from typing import List

def tensor_pose_to_matrix(pose):
    '''
    convert torch tensor of (x, y, z, qx, qy, qz, qw)
    to 4 x 4 matrix format
    pytorch3d.transforms use (w, x, y, z) for quaternion
    '''
    quat = torch.tensor([pose[6], pose[3], pose[4], pose[5]])
    mat = torch.eye(4).type_as(pose)
    mat[:3, :3] = quaternion_to_matrix(quat)
    mat[:3, 3] = pose[:3]
    return mat

def matrix_to_tensor_pose(T):
    '''
    convert torch tensor transformation matrix 4 x 4
    to (x, y, z, qx, qy, qz, qw)
    '''
    T = torch.tensor(T)
    quat = matrix_to_quaternion(T[:3,:3])
    pose = torch.tensor([T[0, 3], 
                         T[1, 3], 
                         T[2, 3], 
                         quat[1], 
                         quat[2], 
                         quat[3], 
                         quat[0]])
    return pose


def get_T(
        x: float = 0.0, 
        y: float = 0.0, 
        z: float = 0.0, 
        seq: str = None,
        angles: List[float] = [],
    ) -> np.array:
    """
    Returns a 4x4 homogeneous transformation matrix.
    :param x: translation in x coordinate
    :param y: translation in y coordinate
    :param z: translation in z coordinate
    :param seq: string of euler axes to define rotation matrix
    :param angles: list of angles of rotation for the axes defined in seq
    :returns: 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3,3] = np.array([x,y,z])
    if seq is not None:
        T[:3,:3] = sciR.from_euler(seq, angles, degrees=True).as_matrix()
    return T

def transform_points(pts, T):
    '''
    Apply SE(3) rigid body transformation to an array of points
    '''
    pts = torch.cat((pts, 
                    torch.ones((pts.shape[0], 1), device=pts.device)),
                    dim=1)
    pts = pts @ T.T
    pts = pts[:,:3]
    return pts

def write_urdf(
    obj_name,
    obj_path,
    output_folder,
):
    content = open("resources/urdf.template").read()
    content = content.replace("NAME", obj_name)
    content = content.replace("MEAN_X", "0.0")
    content = content.replace("MEAN_Y", "0.0")
    content = content.replace("MEAN_Z", "0.0")
    content = content.replace("SCALE", "1.0")
    content = content.replace("COLLISION_OBJ", obj_path)
    content = content.replace("GEOMETRY_OBJ", obj_path)
    urdf_path = os.path.abspath(
        os.path.join(output_folder, obj_name + ".urdf")
    )
    if not os.path.exists(output_folder):
        print("path does not exist!")
        print(f"Output folder: {output_folder}")
        os.makedirs(output_folder)
    open(urdf_path, "w").write(content)
    return urdf_path
