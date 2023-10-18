import os
import math
import copy
from typing import List
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')





class DatasetManager():
    def __init__(self, csv_col_names = None):
        if csv_col_names:
            self.csv_column_names = csv_col_names
        else:
            self.csv_column_names = [
                "Data_path", "Grasp_label", "Scene_info_path"
            ]
        self.train_df = None
        self.test_df = None


    def enforce_data_balance(self,
                             input_df,
                             df_name = '',
                             pos_ratio=0.3,
                             neg_ratio=0.3,
                             hard_neg_ratio=0.4,
                             ):
        sum_ratios = pos_ratio+neg_ratio+hard_neg_ratio
        assert np.isclose(sum_ratios, 1.0), f"Grasp balance ratios does not add up to 1.0 (got={sum_ratios})"
        ratios = {'pos': pos_ratio, 'neg': neg_ratio, 'hard_neg': hard_neg_ratio}
        num_datapoints = len(input_df)
        print(f"Now enforcing data balance on {df_name} with these ratios: {ratios}")
        #We find the limiting factor to enforce the ratios wanted for the datasets
        pos_df = input_df[input_df["Grasp_label"] == 1]
        neg_df = input_df[input_df["Grasp_label"] == 0]
        hard_neg_df = input_df[input_df["Grasp_label"] == -1]
        num_pos = len(pos_df)
        num_neg = len(neg_df)
        num_hard_neg = len(hard_neg_df)
        init_num_labels = {'pos': num_pos, 'neg': num_neg, 'hard_neg': num_hard_neg}
        limit_labels = {key: (x/num_datapoints)/ratios[key] for key, x in init_num_labels.items()}
        #The max value of this dict shows which label is further away from the ratios we want
        #and hence it is the limiting factor
        limiting_label = min(limit_labels, key=limit_labels.get)
        final_num_datapoints = init_num_labels[limiting_label] / ratios[limiting_label] 
        num_labels = {key: int(final_num_datapoints*ratios[key]) for key in init_num_labels}
        out_df = pd.concat((
            pos_df.sample(n=num_labels['pos']),
            neg_df.sample(n=num_labels['neg']),
            hard_neg_df.sample(n=num_labels['hard_neg'])),
            ignore_index=True,
        )
        if df_name:
            print(f"{df_name} with initial length {num_datapoints} was trimmed to length \
                  {int(final_num_datapoints)} with labels: {num_labels}")
        return out_df

    def get_train_test_dataframes(self, 
                                  dataset_root_path,
                                  test_split = 0.1,
                                  total_num_objs = -1,
                                  remove_duplicates = True,
                                  ):
        print(f"Loading data from {dataset_root_path}...")
        all_files = [os.path.join(dataset_root_path, d, 'dataset.csv') 
                     for d in os.listdir(dataset_root_path)]
        #objs_by_file = self._get_dataset_obj_file_dict(all_files, remove_duplicates)
        #all_objs = np.array(list(objs_by_file.keys()))
        all_objs = np.array(list(all_files))
        print(f"Total number of objects loaded: {len(all_objs)}")
        if total_num_objs > 0:
            all_objs = np.random.choice(all_objs, total_num_objs, replace=False)
            print(f"Randomly sampled {len(all_objs)} objects to build our dataset")
        else:
            np.random.shuffle(all_objs)
        if test_split>0.0:
            num_test_objs = math.ceil(len(all_objs)*test_split)
        else:
            num_test_objs = 0
        test_objs = all_objs[:num_test_objs]
        train_objs = all_objs[num_test_objs:]
        # Now build the train and test dataframes 
        #training_files = [path for obj_id in train_objs for path in objs_by_file[obj_id]] #flattened list
        self.train_df = pd.concat((pd.read_csv(f, names=self.csv_column_names) for f in 
                                   tqdm(train_objs, desc="Loading training csv files")), ignore_index=True)
        if len(test_objs)>0:
            #test_files = [path for obj_id in test_objs for path in objs_by_file[obj_id]]        
            self.test_df = pd.concat((pd.read_csv(f, names=self.csv_column_names) for f in 
                                    tqdm(test_objs, desc="Loading test csv files")), ignore_index=True)
            return self.train_df, self.test_df
        else:
            return self.train_df, pd.DataFrame()

    def save_dataframe(self, df, output_dir, fname):
        if df is not None:
            print(f"Saving dataframe to {output_dir}")
            df.to_pickle(os.path.join(output_dir, fname))

    def load_dataframe(self, path):
        return pd.read_pickle(path)


    def plot_df_category_histogram(self, df, savefig=True, outdir='', fname=''):
        histogram = {}
        for path in np.unique(df["Scene_info_path"]):
            scene_info = pickle.load(open(path,'rb'))
            obj_cat = scene_info['obj_2']['mesh'].metadata['cat']
            if obj_cat not in histogram:
                histogram[obj_cat] = 1
            else:
                histogram[obj_cat] += 1
        plt.bar(histogram.keys(), histogram.values())
        plt.title("Object category distribution")
        plt.ylabel('Num objs')
        plt.xticks(rotation=90, ha="right")
        if savefig:
            assert outdir!='' and fname!='', "Must set an output directory and file name to save histogram"
            plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches = "tight")

    def _get_dataset_obj_file_dict(self, all_files_list: List[str], remove_duplicates=True):
        """
        Builds a dictionary where the obj mesh name (obj_id + scale) is the
        key, and the value is a list with the path to all the csv files
        for that object.
        """
        self.objs_by_file = {}
        for file in tqdm(all_files_list, desc="Building Object-File association dictionary"):
            df = pd.read_csv(file, names=self.csv_column_names)
            for path in np.unique(df["Scene_info_path"]):
                scene_info = pickle.load(open(path,'rb'))
                obj_id = scene_info['obj_2']['mesh'].metadata['file_name']
                if obj_id not in self.objs_by_file:
                    self.objs_by_file[obj_id] = []
                if remove_duplicates:
                    #Only add once per obj_id
                    if len(self.objs_by_file[obj_id])==0:
                        self.objs_by_file[obj_id].append(file)
                else:
                    self.objs_by_file[obj_id].append(file)
        return self.objs_by_file

    def test_add_pcd_noise(self, dataframe, idxs, noise):
        for idx in idxs:
            data_pack = np.load(dataframe.iloc[idx, 0])
            label = data_pack['label'][0] 
            if label==-1:
                label = 0
            if label==0:
                pcd_color = np.array([0.8, 0.0, 0.0]).astype(float)
            else:
                pcd_color = np.array([0.0, 0.0, 0.8]).astype(float)
            pcd = data_pack['pcd'][0] 
            all_pcds = copy.deepcopy(pcd)
            noisy_pcd = add_z_noise_to_pcd(pcd, noise)
            noisy_pcd[:,:3] += np.tile([-0.25,0,0],(noisy_pcd[:,:3].shape[0],1))
            all_pcds = np.concatenate((all_pcds, noisy_pcd),axis=0)
            visualize_pcd(all_pcds, pcd_color)

    def test_normal_based_occlusion(self, dataframe, idxs, angle_threshold, drop_probability):
        for idx in idxs:
            data_pack = np.load(dataframe.iloc[idx, 0])
            label = data_pack['label'][0] 
            if label==-1:
                label = 0
            if label==0:
                pcd_color = np.array([0.8, 0.0, 0.0]).astype(float)
            else:
                pcd_color = np.array([0.0, 0.0, 0.8]).astype(float)
            pcd = data_pack['pcd'][0]  
            all_pcds = pcd
            displacements = [
                [-0.25,0,0],    #85 deg
                [-0.50,0,0],    #80 deg
                [-0.75,0,0],    #75 deg
                [-1.00,0,0],    #70 deg
            ]
            #start_angles=[85, 80, 75, 70]
            for i, displacement in enumerate(displacements):
                occluded_pcd = occlude_pcd_normals_based(pcd, angle_threshold, drop_probability)
                #occluded_pcd = occlude_pcd_normals_based(pcd, start_angles[i], drop_probability)
                #Translate this occludded PCD to the side to visualize together
                occluded_pcd[:,:3] += np.tile(displacement,(occluded_pcd[:,:3].shape[0],1))
                all_pcds = np.concatenate((all_pcds,occluded_pcd),axis=0)
            print(f"Showing grasp with idx={idx}")
            visualize_pcd(all_pcds, pcd_color)

    def see_pcd_normals(self, dataframe, idxs):
        for idx in idxs:
            data_pack = np.load(dataframe.iloc[idx, 0])
            label = data_pack['label'][0] 
            if label==-1:
                label = 0
            if label==0:
                pcd_color = np.array([0.8, 0.0, 0.0]).astype(float)
            else:
                pcd_color = np.array([0.0, 0.0, 0.8]).astype(float)
            pcd = data_pack['pcd'][0]
            print(f"PCD shape is: {pcd.shape}")
            visualize_pcd(pcd[:,:4], normals=pcd[:,-3:])
    

class GraspEvaluatorData(Dataset):
    def __init__(self, opt, dataframe):
        self.opt = opt
        self.dataframe = dataframe
        self.size = len(dataframe)

    def __getitem__(self, idx):
        data_pack = np.load(self.dataframe.iloc[idx, 0])
        label = data_pack['label'][0] 
        if label==-1:
            label = 0
        pcd = data_pack['pcd'][0] 
        delta_t = data_pack['delta_t'][0]
        delta_r = data_pack['delta_r'][0]

        if self.opt.occlusion_angle_threshold>0:
            pcd = occlude_pcd_normals_based(pcd,
                                            self.opt.occlusion_angle_threshold,
                                            self.opt.occlusion_drop_rate)
        else:
            pcd = pcd[:,:4]
        if self.opt.xy_depth_noise>0.0:
            pcd = add_xy_noise_to_pcd(pcd, self.opt.xy_depth_noise)
        if self.opt.z_depth_noise>0.0:
            pcd = add_z_noise_to_pcd(pcd, self.opt.z_depth_noise)
        
        return {'pcd': torch.tensor(pcd, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32),
                'tr': torch.tensor([delta_t, delta_r], dtype=torch.float32)}

    def __len__(self):
        return len(self.dataframe)
    


def occlude_pcd_normals_based(pcd, angle_threshold, drop_probability):
    """
    This function will compute the normals of the input PCD 
    and drop points with 'drop_probability' if the angle between
    the normal and the camera ray is higher than angle_threshold 
    in degrees (angle threshold should be a value <=90 degrees)
    """
    cam_normal = R.from_euler('X',-10, degrees=True).as_matrix()[:,2]
    gripper_mask = pcd[:,3]==1
    scene_mask = pcd[:,3]==0
    normals = pcd[:,-3:]
    #Compute dot product between camera ray unit vector and unit normals 
    cos_values = np.einsum('ij, ij->i',
                        np.tile(cam_normal,(pcd.shape[0],1)),
                        normals)
    angles = np.rad2deg(np.arccos(abs(cos_values)))
    #Use boolean mask to apply the drop probability on points which
    #conform to our normal angle threshold criteria
    odds = np.random.choice(
        a=[False,True],
        size=angles.shape,
        p=[drop_probability, 1-drop_probability])
    angle_mask = angles<=angle_threshold
    drop_mask = np.logical_or(angle_mask, odds)
    keep_scene_points_mask = np.logical_and(drop_mask, scene_mask)   
    new_scene_pcd = pcd[keep_scene_points_mask]
    new_pcd = np.concatenate((pcd[gripper_mask],new_scene_pcd),axis=0)
    #Pad the new PCD with zeros to maintain its original shape
    zero_rows = pcd.shape[0] - new_pcd.shape[0]
    new_pcd_padded = np.concatenate((new_pcd, np.zeros((zero_rows,new_pcd.shape[1]))))
    return new_pcd_padded[:,:4]

def add_noise_to_pcd(pcd, noise_strength):
    scene_mask = pcd[:,3]==0
    pcd[scene_mask,:3] += np.random.normal(0.0, noise_strength, size=(pcd[scene_mask,:3].shape))
    return pcd

def add_xy_noise_to_pcd(pcd, noise_strength):
    scene_mask = pcd[:,3]==0
    pcd[scene_mask,:2] += np.random.normal(0.0, noise_strength, size=(pcd[scene_mask,:2].shape))
    return pcd

def add_z_noise_to_pcd(pcd, noise_strength):
    scene_mask = pcd[:,3]==0
    pcd[scene_mask, 2] += np.random.normal(0.0, noise_strength, size=(pcd[scene_mask,2].shape))
    return pcd
    
def visualize_pcd(pcd, scene_color=None, normals=None, csys=[]):
    gripper_color = np.array([0.0, 0.8, 0.0]).astype(float)
    if scene_color is None:
        scene_color = np.array([0.0, 0.0, 0.8]).astype(float)

    #First we separate the scene PCD from gripper PCD using the 4th column value
    gripper_mask = pcd[:,3]==1
    gripper_pcd = pcd[gripper_mask,:3]
    scene_pcd = pcd[~gripper_mask,:3]
    
    o3d_gripper_pcd = o3d.geometry.PointCloud()
    o3d_gripper_pcd.points = o3d.utility.Vector3dVector(gripper_pcd)
    o3d_gripper_pcd.paint_uniform_color(gripper_color)

    o3d_scene_pcd = o3d.geometry.PointCloud()
    o3d_scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd)
    o3d_scene_pcd.paint_uniform_color(scene_color)
    if normals is not None:
        o3d_scene_pcd.normals = o3d.utility.Vector3dVector(normals[~gripper_mask])
    else:
        o3d_scene_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) #5cm arrows
    geometries = [origin, o3d_gripper_pcd, o3d_scene_pcd]
    for csys in csys:
        geometries.append(copy.deepcopy(origin).transform(csys))
    o3d.visualization.draw_geometries(geometries,
                                      zoom=0.55,
                                      front=[ 0.0925, -0.4115, -0.9066 ],
                                      lookat=[ -0.0569, -0.0284, 0.0446 ],
                                      up=[ 0.0334, -0.9088, 0.4158 ])
    