import torch
import copy
import numpy as np
import open3d as o3d
import math
import itertools

VOXEL_SIZE = 0.01
Z_MIN = -0.10    # Crop lower bound

def draw_frame(origin, q, scale=1):
    # open3d quaternion format qw qx qy qz
    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    return frame_rot

def pcd_to_spheres(pcd):
    spheres = []
    pts = np.asarray(pcd.points)
    for p in pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=VOXEL_SIZE).translate(p)
        s.paint_uniform_color(np.random.rand(3))
        spheres.append(copy.deepcopy(s))
    return spheres

frame_base = draw_frame(np.zeros(3), np.array([0, 0, 0, 1]), scale=0.1)
gripper_mesh = o3d.io.read_triangle_mesh(
                "resources/SR_Gripper_Collision_Open.stl")

# Translate to the grasp pose frame in SceneGrasp convention
gripper_mesh.translate(np.array([0,0,-0.21]))

pcd = gripper_mesh.sample_points_uniformly(number_of_points=1500)

# Create bounding box:
bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [Z_MIN, math.inf]]  # set the bounds
bounding_box_points = list(itertools.product(*bounds))  # create limit points
bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
    o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
# Crop the point cloud using the bounding box:
pcd_cropped = pcd.crop(bounding_box)

# Downsample
# pcd_cropped = pcd_cropped.voxel_down_sample(VOXEL_SIZE)
pcd_cropped = pcd_cropped.farthest_point_down_sample(250)

spheres = pcd_to_spheres(pcd_cropped)
print("Number of spheres: %d"%(len(spheres)))
o3d.visualization.draw_geometries(spheres + [gripper_mesh, frame_base])

pts_tensor = torch.from_numpy(np.asarray(pcd_cropped.points))
torch.save(pts_tensor, 'resources/SR_Gripper_Collision_Open.pt')