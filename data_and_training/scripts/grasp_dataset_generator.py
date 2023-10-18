'''
An example of the dataset file structure:
grasp_pcd_dataset
    |
    |
    | -- scene_000005
        |
        | -- dataset.csv
        |
        | -- obj_info.pkl
        |
        | -- robot_env.png
        |
        | -- data
            |
            | structure numpy array
              PCD, grasp pose and label

obj_info.pkl 
Inherite the dictionary from scene_manager.objs.
Also added volume estimation for 'obj_2'
'''

import os
import csv
import pickle
import numpy as np
from tqdm import tqdm
from render_environment import RenderEnvironment
import open3d as o3d
import torch
import utils


def estimate_pcd_normals(pcd):
    """
    This function will compute the normals of the input PCD
    Args:
        pcd -- (torch.tensor) shape N x P x 7
                PointFields: xyzrgbas
    Return:
        out_pcd -- (torch.tensor) shape N x P x 10
                PointFields: xyzrgbasuvw, where uvw is the surface normal
    """
    np_pcd = pcd.cpu().numpy()
    
    # Only pick the object point cloud
    scene_mask = np_pcd[:,:,6] == 1

    np_points = np_pcd[:,:,:3]
    batch_size, num_points = np_pcd.shape[0], np_pcd.shape[1]
    
    np_normals = np.zeros((batch_size, num_points, 3))
    for i in range(batch_size):
        o3d_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np_points[i, scene_mask[i]]))
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        o3d_pcd = o3d_pcd.normalize_normals()
        scene_normals = np.asarray(o3d_pcd.normals)
        np_normals[i, scene_mask[i], :] = scene_normals
    
    out_pcd = torch.cat([pcd, torch.tensor(np_normals)], dim=2)
    return out_pcd

class GraspDatasetGenerator:
    def __init__(self, args) -> None:
        self.args = args
        self.output_dir = os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__))),
                                        args.output_dir)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.grasp_labels = {"positive_grasps": 1,
                             "negative_grasps": 0,
                             "hard_negative_grasps": -1}
        
        self.env = RenderEnvironment(args)

        self.mesh_list = []
        for cat, mesh_ids in self.env.mesh_dict.items():
            for mesh_id in mesh_ids:
                self.mesh_list.append([cat, mesh_id])
 
    def run(self, idx):
        '''
        1. Set up a new scene
        2. Render and save PCD dataset

        Args:
            idx -- index in self.mesh_list
        '''
        cat, mesh_id = self.mesh_list[idx]
        self.env.scene.arrange_scene(cat, mesh_id)
        
        # List all the scene directories in the dataset dir root
        flist = os.listdir(self.output_dir)
        flist = [item for item in flist if os.path.isdir(
                        os.path.join(self.output_dir, item))]
        # Create new scene dir without name conflict
        scene_dir = os.path.join(self.output_dir, 
                                'scene_%06d'%(len(flist)))
        os.mkdir(scene_dir)

        # Save object info for scene reconstruction and testing
        obj_info = self.env.scene.objs
        obj_info_path = os.path.join(scene_dir, 'obj_info.pkl')
        with open(obj_info_path, 'wb') as fp:
            pickle.dump(obj_info, fp)
            print('object info saved successfully to file')

        # Generate data for one scene
        csv_fname = os.path.join(scene_dir, 'dataset.csv')
        csvfile = open(csv_fname, 'w+')
        csvwriter = csv.writer(csvfile)

        # Data type to store pcd + label + grasp
        num_pts = self.env.params['kq'] + self.env.gripper_points.shape[0]
        dt = np.dtype([('pcd', 'f8', (num_pts, 7)),
                       ('label', 'int32'),
                       ('grasp', 'f8', (4,4)),
                       ('delta_t', 'f8'),
                       ('delta_r', 'f8')])


        data_dir = os.path.join(scene_dir, 'data')
        os.mkdir(data_dir)

        for label in self.grasp_labels.keys():
            num_grasps = len(obj_info['obj'][label])
            for grasp_idx in tqdm(range(num_grasps)):
                # Place gripper and camera according to grasp pose
                self.env.set_one_grasp(label, grasp_idx)
                xyzrgbs = self.env.get_wrist_cam_pcd()

                if xyzrgbs is None:
                    # print("%s, grasp_idx: %d, empty grasp after cropping!")
                    continue

                xyzrgbsuvw = estimate_pcd_normals(xyzrgbs)
                xyzsuvw = xyzrgbsuvw[:,:,[0,1,2,6,7,8,9]]

                grasp_pose = obj_info['obj'][label][grasp_idx]

                data_pack = np.zeros(1, dtype=dt)
                data_pack['pcd'] = xyzsuvw.numpy().copy()
                data_pack['label'] = self.grasp_labels[label]
                data_pack['grasp'] = grasp_pose.copy()

                data_pack_path = os.path.join(data_dir, '%s_%d.npy'%(label, grasp_idx))
                np.save(data_pack_path, data_pack)

                # Write to CSV file
                row = [data_pack_path, self.grasp_labels[label], obj_info_path]
                csvwriter.writerow(row)

        csvfile.close()