import yaml
import os
import numpy as np
import h5py
import trimesh
import trimesh.transformations as tra
import torch
import pytorch3d.ops as p3o
from scipy.spatial.transform import Rotation
import pyrender
from pyrender.constants import RenderFlags

OBJ_LABEL = 1
GRIPPER_LABEL = 2
# RGB channel for each instance, last channel is the integer label
SEG_CMAP = {'gripper': [120, 20, GRIPPER_LABEL],
            'obj': [20, 120, OBJ_LABEL]}


class CameraIntrinsics:
    cx = 320
    cy = 240
    fx = 426
    fy = 426
    width = 640
    height = 480
    

class RenderEnvironment:
    
    def __init__(self, args):
        self.args = args
        with open(args.env_param_file, 'r') as f:
            self.params = yaml.load(f, Loader=yaml.Loader)

        with open(args.mesh_dict_file, 'r') as f:
            self.mesh_dict = yaml.load(f, Loader=yaml.Loader)
    
        # Use box passthrough filter in the grasp frame
        if 'pmin' in self.params.keys():
            self.pmin = self.params['pmin']
            self.pmax = self.params['pmax']
        else:
            self.pmin = None
            self.pmax = None

        # Gripper mesh offset
        self.grasp_t_ee = np.eye(4)
        self.grasp_t_ee[:3, :3] = np.array(self.params['grasp_t_ee']['rotation']).reshape(3,3)
        self.grasp_t_ee[:3, 3] = np.array(self.params['grasp_t_ee']['translation'])

        # eye to mesh (gripper base) transformation
        self.ee_T_cam = np.eye(4)
        self.ee_T_cam[:3,:3] = Rotation.from_euler(self.params['ee_t_cam']['euler_seq'],
                                                   self.params['ee_t_cam']['euler_angles'],
                                                   degrees=True).as_matrix()
        self.ee_T_cam[:3,3] = np.array(self.params['ee_t_cam']['translation'])

        # Load gripper point cloud for concatenation later
        self.gripper_points = torch.load(self.params['gripper_points'])

        self.kq = self.params['kq']

        self.scene = Scene(self.args)
        gripper_mesh_path = self.params['gripper_mesh']
        gripper_mesh = trimesh.load(gripper_mesh_path, force="mesh")
        self.scene.add_object('gripper', gripper_mesh, pose=self.grasp_t_ee)

        # Camera projection parameters
        fx = self.scene.camera_intrinsics.fx
        fy = self.scene.camera_intrinsics.fy
        width = self.scene.camera_intrinsics.width
        height = self.scene.camera_intrinsics.height
        camera_u = np.arange(0, width).reshape(1, -1)
        camera_v = np.arange(0, height).reshape(-1, 1)
        self.wrist_cam_projection = {"fxy": [fx, fy],
                                     "wh": [width, height],
                                     "camera_u": camera_u,
                                     "camera_v": camera_v}
        
        self.opencv_t_opengl = np.array([[1,  0,  0, 0],
                                         [0, -1,  0, 0],
                                         [0,  0, -1, 0],
                                         [0,  0,  0, 1]])
        
    def visualize_grasps(self):
        obj_name = 'obj_2'
        positive_grasps = self.scene_manager.objs[obj_name]['valid_positive']
        num_grasps = len(positive_grasps)
        w_t_o = self.scene_manager.get_object_pose(obj_name)

        for i in range(20):
            o_t_g = positive_grasps[np.random.randint(num_grasps)]
            w_t_g = w_t_o @ o_t_g
            self.scene_manager.add_grasp(obj_name+'_%d'%i, w_t_g, [100, 0, 0])

        hard_negative_grasps = self.scene_manager.objs[obj_name]['valid_hard_negative']
        num_grasps = len(hard_negative_grasps)
        w_t_o = self.scene_manager.get_object_pose(obj_name)

        for i in range(20):
            o_t_g = hard_negative_grasps[np.random.randint(num_grasps)]
            w_t_g = w_t_o @ o_t_g
            self.scene_manager.add_grasp(obj_name+'_%d'%i, w_t_g, [0, 100, 0])

    
    def set_one_grasp(self, grasp_label="positive_grasps", grasp_idx=-1, o_t_g_custom=None):
        '''
        Set actor pose for the gripper. 
        The actor pose frame is the same as gripper mesh frame and end effector frame.
        o_t_g stands for object_T_grasp, same as the grasp poses in ACRONYM dataset.

        There are two ways to set the pose:
        1) grasp_label and grasp_idx. 
            grasp label can be "positive_grasps", "negative_grasps", "hard_negative_grasps"
            grasp_idx pick the 4x4 grasp pose from obj_info[grasp_label][grasp_idx]
        2) grasp_label and o_t_g_custom.
            set grasp_label == 'custom'
            pose will be computed based on o_t_g_custom
        '''

        obj_info = self.scene.objs['obj']
        
        # Set grasp pose
        if grasp_label in obj_info.keys():
            if grasp_idx < 0:
                grasp_idx = np.random.choice(len(obj_info[grasp_label][()]))
            o_t_g = obj_info[grasp_label][grasp_idx]

        elif grasp_label == 'custom':
            o_t_g = o_t_g_custom

        else:
            raise Exception("Unrecgonized grasp label!")
        
        o_t_ee = o_t_g @ self.grasp_t_ee
        w_t_o = obj_info['pose']
        gripper_pose = w_t_o @ o_t_ee
        self.scene.set_object_pose('gripper', gripper_pose)

        camera_pose = gripper_pose @ self.ee_T_cam @ self.opencv_t_opengl
        self.scene.camera_pose = camera_pose
                
    def get_wrist_cam_rgbd(self):
        color, depth = self.scene._renderer.render_rgbd()
        return (color, depth)
    
    def get_wrist_cam_pcd(self, downsample=True, crop=True):
        '''
        Return:
            xyzrgbs -- torch.tensor, (1, N, 7)
                If downsample is True, each bach has uniform size,
                otherwise they cannot be packed into a tensor.
        '''
        # Get color and depth
        color, depth = self.get_wrist_cam_rgbd()
        seg_img = self.scene._renderer.render_seg()
        seg = seg_img[:,:,2]

        fx, fy = self.wrist_cam_projection["fxy"]
        width, height = self.wrist_cam_projection["wh"]
        camera_u = self.wrist_cam_projection["camera_u"]
        camera_v = self.wrist_cam_projection["camera_v"]
        
        # 2D - 3D projection
        Z = depth
        X = (camera_u - width/2) / fx * Z
        Y = (camera_v - height/2) / fy * Z

        valid = seg == OBJ_LABEL
        Z = Z[valid].reshape(-1, 1)
        X = X[valid].reshape(-1, 1)
        Y = Y[valid].reshape(-1, 1)
        
        # Transform points into grasp frame
        XYZ1 = np.concatenate([X, Y, Z, np.ones_like(X)], axis=1)
        XYZ1 = XYZ1 @ (self.grasp_t_ee @ self.ee_T_cam).T

        S = seg[valid].reshape(-1, 1)
        RGB = color[valid].reshape(-1, 3)
        xyzrgbs = np.concatenate([XYZ1[:,:3], RGB, S], axis=1)
        
        # Transform to grasp frame and run passthrough filter
        if crop and self.pmin is not None:
            X, Y, Z = xyzrgbs[:,0], xyzrgbs[:,1], xyzrgbs[:,2]
            valid = (X > self.pmin[0]) & (X < self.pmax[0]) \
                    & (Y > self.pmin[1]) & (Y < self.pmax[1]) \
                    & (Z > self.pmin[2]) & (Z < self.pmax[2])
            # If no point is inside the crop, return None for empty grasp
            if np.sum(valid) == 0:
                return None
            else:
                xyzrgbs = xyzrgbs[valid]

        if downsample:
            xyzrgbs = self.pcd_downsample(xyzrgbs, self.kq)

        # Concatenate gripper points
        gripper_pcd = torch.zeros((1, self.gripper_points.shape[0], 7))
        gripper_pcd[:,:,:3] = self.gripper_points.unsqueeze(0)
        gripper_pcd[:,:,6] = SEG_CMAP['gripper'][2]
        xyzrgbs = torch.cat([xyzrgbs, gripper_pcd], dim=1)

        return xyzrgbs
    
    def pcd_downsample(self, pcd_in, kq):
        pcd = torch.tensor(pcd_in).unsqueeze(0)
        # Downsample and select index
        _, idx = p3o.sample_farthest_points(pcd[:,:,:3], K=kq)
        pcd_out = pcd[0, idx[0]].unsqueeze(0)
        assert pcd_out.shape[0] == 1 and pcd_out.shape[1] == kq and pcd_out.shape[2] == 7
        return pcd_out
        

class Scene:
    def __init__(self, args):
        with open(args.env_param_file, 'r') as f:
            self.params = yaml.load(f, Loader=yaml.Loader)

        self._dataset_path = self.params['dataset_root']
        obj_info = h5py.File(open(self.params['object_info'], "rb"))
        self.mesh_info = obj_info["meshes"]
        self.categories = obj_info["categories"]

        self._renderer = SceneRenderer()
        
        # # Add frame of reference for debug purpose
        # frame_mesh = add_origin_frame()
        # self._renderer.add_object('origin_frame', frame_mesh)
        
        self.camera_intrinsics = CameraIntrinsics()
        self._renderer.create_camera(self.camera_intrinsics, 0.03, 0.5)

        self.objs = {}

    def arrange_scene(self, cat, mesh_id):
        '''
        '''
        mesh_path = os.path.join(
            self._dataset_path, self.mesh_info[mesh_id]["path"].asstr()[()]
        )
        obj_mesh = trimesh.load(mesh_path, force="mesh")
        obj_mesh.metadata["key"] = mesh_id
        obj_mesh.metadata["path"] = mesh_path
        obj_mesh.metadata["cat"] = cat
        obj_info = self.mesh_info[mesh_id]
        
        if 'obj' in self.objs:
            self.remove_object('obj')
        self.add_object('obj', obj_mesh, obj_info)


    def add_object(self, name, mesh, info={}, pose=None):

        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        if name == 'obj':
            color = (np.asarray((220, 40, 40)))
        else:
            color = (np.asarray((200, 200, 200)))

        mesh.visual.face_colors = np.tile(
            np.reshape(color, [1, 3]), [mesh.faces.shape[0], 1]
        )
        self.objs[name] = {"mesh": mesh, "pose": pose}

        if "positive_grasps" in info:
            self.objs[name]["positive_grasps"] = info["positive_grasps"][()]
        if "negative_grasps" in info:
            self.objs[name]["negative_grasps"] = info["negative_grasps"][()]
        if "hard_negative_grasps" in info:
            self.objs[name]["hard_negative_grasps"] = info["hard_negative_grasps"][()]
 
        if self._renderer is not None:
            self._renderer.add_object(name, mesh, pose)

    def remove_object(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self._renderer.remove_object(name)
        del self.objs[name]

    def set_object_pose(self, name, pose):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self.objs[name]["pose"] = pose
        self._renderer.set_object_pose(name, pose)
    
    @property
    def camera_pose(self):
        return self._renderer.camera_pose

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        self._renderer.camera_pose = cam_pose


class SceneRenderer:
    def __init__(self):

        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._camera_intr = None
        self._camera_node = None
        self._light_node = None
        self._renderer = None
        self._seg_node_map = {}

    def create_camera(self, intr: CameraIntrinsics, znear: float, zfar: float):
        cam = pyrender.IntrinsicsCamera(
            intr.fx, intr.fy, intr.cx, intr.cy, znear, zfar
        )
        self._camera_intr = intr
        self._camera_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        self._scene.add_node(self._camera_node)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.array([[1,  0,  0, 0],
                                [0,  1,  0, 0],
                                [0,  0,  1, 3],
                                [0,  0,  0, 1]])
        self._light_node = pyrender.Node(light=light, matrix=light_pose)
        self._scene.add_node(self._light_node)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=intr.width,
            viewport_height=intr.height,
            point_size=1.0,
        )

    @property
    def camera_pose(self):
        if self._camera_node is None:
            return None
        return self._camera_node.matrix

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        if self._camera_node is None:
            raise ValueError("No camera in scene!")
        self._scene.set_pose(self._camera_node, cam_pose)
        self._scene.set_pose(self._light_node, cam_pose)

    def render_rgbd(self, depth_only=False):
        if depth_only:
            color = None
            depth = self._renderer.render(self._scene, 
                                          flags=RenderFlags.DEPTH_ONLY)
        else:
            color, depth = self._renderer.render(self._scene)

        return color, depth
    
    def render_seg(self):
        '''
        Segmentation image seg is a 3-channel RGB image.
        The RGB value of each instance is defined by self._seg_node_map
        '''
        seg, _ = self._renderer.render(self._scene, 
                                             flags=RenderFlags.SEG,
                                             seg_node_map=self._seg_node_map)
        return seg

    def render_points(self):
        _, depth = self.render_rgbd(depth_only=True)
        point_norm_cloud = depth.point_normal_cloud(self._camera_intr)

        pts = point_norm_cloud.points.data.T.reshape(
            depth.height, depth.width, 3
        )
        norms = point_norm_cloud.normals.data.T.reshape(
            depth.height, depth.width, 3
        )
        cp = self.get_camera_pose()
        cp[:, 1:3] *= -1

        pt_mask = np.logical_and(
            np.linalg.norm(pts, axis=-1) != 0.0,
            np.linalg.norm(norms, axis=-1) != 0.0,
        )
        pts = tra.transform_points(pts[pt_mask], cp)
        return pts.astype(np.float32)

    def add_object(self, name, mesh, pose=None):
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
            matrix=pose,
        )
        self._node_dict[name] = node
        self._scene.add_node(node)

        # Define instance color from SEG_CMAP for segmentation rendering
        if name in SEG_CMAP:
            self._seg_node_map[node] = SEG_CMAP[name]

    def set_object_pose(self, name, pose):
        self._scene.set_pose(self._node_dict[name], pose)

    def remove_object(self, name):
        self._scene.remove_node(self._node_dict[name])
        del self._node_dict[name]

    def reset(self):
        for name in self._node_dict:
            self._scene.remove_node(self._node_dict[name])
        self._node_dict = {}



def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
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


def add_origin_frame(origin=[0,0,0], scale=0.2):
    cx = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [origin[0]+scale, origin[1], origin[2]]],
    )
    cx.visual.face_colors = [255,0,0]
    cy = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [origin[0], origin[1]+scale, origin[2]]],
    )
    cy.visual.face_colors = [0,255,0]
    cz = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [origin[0], origin[1], origin[2]+scale]],
    )
    cz.visual.face_colors = [0,0,255]
    tmp = trimesh.util.concatenate([cx, cy, cz])
    return tmp