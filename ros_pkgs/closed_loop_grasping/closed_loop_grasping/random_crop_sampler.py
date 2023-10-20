import os
import time
from ament_index_python.packages import get_package_share_directory
import numpy as np
import torch
import pytorch3d.ops as p3o
import pytorch3d.transforms as p3t
from .utils import pointcloud2_to_array


SCENE_LABEL = 0
GRIPPER_LABEL = 1

class RandomCropSampler:

    def __init__(self, params):
        self.params = params

        # number of samples to draw around each grasp candidate
        self.num_draws = self.params['num_draws']
        self.device = self.params['device']

        # Limits for uniform distribution for sampling translation and rotations
        # The limits for x y z is given by self.t_lim and
        # self.r_lim defines the limits for euler angles x, y, z
        self.lim_t = torch.tensor(self.params['lim_t']).to(self.device)
        self.lim_r = torch.tensor(np.deg2rad(self.params['lim_r'])).to(self.device)

        # Define pass through filter
        self.pass_min = torch.tensor(self.params['pmin'])
        self.pass_max = torch.tensor(self.params['pmax'])

        # Grid sampler parameters
        self.grid_xlim = self.params['grid_xlim']
        self.grid_ylim = self.params['grid_ylim']
        self.grid_zlim = self.params['grid_zlim']
        self.grid_density = self.params['grid_density']

        # Max number of points to query
        self.kq = self.params['kq']
        self.radius = self.params['radius']
        self.nc = self.params['nc']
        self.gripper_pcd = torch.load(os.path.join(
                get_package_share_directory('closed_loop_grasping'),
                'resource',
                self.params['gripper_points']
        )).to(self.device).float()


    def sample_crops(self,
                     pcd_msg,
                     robot_T_seed_grasps,
                     num_draws,
                     method='random',
                     t_limits = None, 
                     r_limits_deg = None,
                     add_noise_batches=False,
                     num_noise_batches=4, 
                     xy_noise = 0.002,
                     z_noise = 0.000,
                    ):
        '''
        Process ROS PointCloud2 message with random cropping (Input PCD in robot base frame)
        1. Convert it to PyTorch tensor
        2. Sample grasp poses based on current grasp pose candidates
        3. Crop the point cloud tensor and down-sample
        4. Collision check and concatenate the gripper point cloud
        
        T shape is [Batch, 4, 4]
        batch_cloud shape is [Batch, P, 4]  (P points in the crops)
        Seeds_idxs are the indeces in the first dimention of T corresponding to the seed candidates
        Samples_idxs are the indeces in the first dimention of T corresponding to the perturbed grasp samples
        '''
        
        # Convert PointCloud2 message to Pytorch tensor. Input PCD is in camera frame
        pcd_array = pointcloud2_to_array(pcd_msg)
        pcd_tensor = torch.from_numpy(pcd_array).to(self.device).float()

        # Sample perturbed transformation based on the grasp pose candidates
        #T here is a batch of grasp_T_robot
        if method == "random":
            T = self._sample_transformations(robot_T_seed_grasps, num_draws, t_limits, r_limits_deg)
        elif method == "grid":
            T = self._grid_sample_transformations(robot_T_seed_grasps, with_rot=False)
        else:
            raise Exception("Unrecognized method for grasp pose sampling!")

        #Batch cloud is in gripper frame
        batch_cloud = self._crop_cloud(T, pcd_tensor)

        # ***********************************
        # Skip collision check for now due to speed issue and incompatibility with 
        # the Jacobian computation for grasps
        # ***********************************
        # # Filter to get collision-free grasp poses
        # T, batch_cloud, idx0, idx1 = self._collision_filter(T, batch_cloud)
        
        mask_empty = self._empty_grasp_check(batch_cloud)
        
        if add_noise_batches:
            batch_cloud = self._multiply_crops_with_noise(
                batch_cloud,
                num_noise_batches,
                xy_noise,
                z_noise)
 
        batch_cloud = self._concatenate_gripper_cloud(batch_cloud)

        return T, batch_cloud, mask_empty
    
    def get_crop_params(self):
        return self.params
    

    def _multiply_crops_with_noise(self, batch_crops, num_clones=4, xy_noise=0.001, z_noise=0.002):
        """
        batch_crops has shape [num_samples+1, 1250, 3]
        """
        augmented_crops = batch_crops.repeat(num_clones, 1, 1)       
        non_zero_mask = augmented_crops.bool().int()
        augmented_crops[:,:,:3] += torch.normal(
            torch.zeros(augmented_crops[:,:,:3].shape, device=self.device, dtype=torch.float32),
            xy_noise)
        # augmented_crops[:,:,2] += torch.normal(
        #     torch.zeros(augmented_crops[:,:,2].shape, device=self.device, dtype=torch.float32),
        #     z_noise)
        #Apply noise only to points that are not (0,0,0)
        return augmented_crops * non_zero_mask 

    def _sample_transformations(self, robot_T_seed_grasps, num_draws, t_limits=None, r_limits_deg=None):
        '''
        Sample k random translation and rotation matrix around each
        grasp candidate. k = self.num_draws
        Inputs:
            robot_T_seed_grasps -- (torch.tensor) shape (self.nc, 4, 4)
            t_limits -- (torch.tensor) shape (3) representing the xyz limits [m] for uniform sampling
            r_limits -- (torch.tensor) shape (3) representing the euler angles limits [deg] for uniform sampling
                     
        Return:
            T -- (torch.tensor) shape self.nc x (self.num_draws + 1) x 4 x 4
                Each candiate is concatenated with the randomly drawn grasp poses
                T represents a batch of grasp_T_robot, where the first element
                in the second dimension is the input seed, and the rest are the samples for it
        '''
        assert robot_T_seed_grasps.shape[0] == self.nc, f"Input seed dim0 should be equal to {self.nc}, but it is equal to {robot_T_seed_grasps.shape[0]}"
        nc = self.nc
        ns = num_draws * nc
        if t_limits is None:
            t_limits = self.lim_t
        if r_limits_deg is None:
            r_limits_rad = self.lim_r
        else:
            r_limits_rad = torch.tensor(np.deg2rad(r_limits_deg), device=self.device, dtype=torch.float32)
        
        # rel_T shape: nc x num_draws+1 x 4 x 4
        # The first one is identity matrix
        rel_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(
                        nc, num_draws+1, 1, 1)
        
        rel_T[:,1:,:3,3] = torch.stack([
                    2 * t_limits[0] * (torch.rand(ns, device=self.device, dtype=torch.float32) - 0.5),
                    2 * t_limits[1] * (torch.rand(ns, device=self.device, dtype=torch.float32) - 0.5),
                    2 * t_limits[2] * (torch.rand(ns, device=self.device, dtype=torch.float32) - 0.5)], 
                dim=-1).reshape(nc, num_draws, 3)
    
        # Sample euler angles
        eulers = torch.stack([
                    2 * r_limits_rad[0] * (torch.rand(ns, device=self.device, dtype=torch.float32) - 0.5),
                    2 * r_limits_rad[1] * (torch.rand(ns, device=self.device, dtype=torch.float32) - 0.5),
                    2 * r_limits_rad[2] * (torch.rand(ns, device=self.device, dtype=torch.float32) - 0.5)], 
                dim=-1).view(nc, num_draws, 3)
        rel_T[:,1:,:3,:3] = p3t.euler_angles_to_matrix(eulers, 'XYZ')

        #We invert the grasp pose candidate such that the transform becomes wrist_cam_T_grasp to 
        #later transform the scene PCD to grasp frame easily
        T = torch.stack([torch.inverse(torch.bmm(robot_T_seed_grasps[i,:,:].repeat(num_draws+1,1,1), rel_T[i]))
                    for i in range(nc)]).view(nc * (num_draws+1), 4, 4)
        T.requires_grad = True
        return T
    
    def _grid_sample_transformations(self, robot_T_seed_grasps, with_rot=False):
        assert robot_T_seed_grasps.shape[0] == self.nc, f"Input seed dim0 should be equal to {self.nc}"
        nx, ny, nz = self.grid_density
        xx = torch.linspace(self.grid_xlim[0], self.grid_xlim[1], nx, device=self.device, dtype=torch.float32)
        yy = torch.linspace(self.grid_ylim[0], self.grid_ylim[1], ny, device=self.device, dtype=torch.float32)
        zz = torch.linspace(self.grid_zlim[0], self.grid_zlim[1], nz, device=self.device, dtype=torch.float32)
        # grid shape: (nx, ny, nz, 3)
        grid = torch.stack(torch.meshgrid(xx, yy, zz, indexing='ij'), dim=-1)
        if with_rot:
            # relative transformation shape: (self.nc, 3, nx, ny, nz, 4, 4)
            rel_T = torch.eye(4, device=self.device, dtype=torch.float32).view(1,1,1,1,1,4,4).repeat(
                        self.nc, nx, ny, nz, 3, 1, 1)
            translation = torch.tensor(grid).view(1, nx, ny, nz, 1, 3).repeat(
                self.nc, 1, 1, 1, 3, 1)
            # Sample euler angles
            eulers = torch.tensor([[0,0,-np.pi/6],[0,0,0],[0,0,np.pi/6]])
            rotation = p3t.euler_angles_to_matrix(eulers, 'XYZ').view(1,1,1,1,3,3,3).repeat(
                self.nc, nx, ny, nz, 1, 1, 1)
            rel_T[:,:,:,:,:,:3,3] = translation
            rel_T[:,:,:,:,:,:3,:3] = rotation
        else:
            rel_T = torch.eye(4, device=self.device, dtype=torch.float32).view(1,1,1,1,4,4).repeat(
                self.nc, nx, ny, nz, 1, 1)
            translation = grid.clone().detach().view(1, nx, ny, nz, 3).repeat(
                self.nc, 1, 1, 1, 1)
            rel_T[:,:,:,:,:3,3] = translation

        rel_T = rel_T.view(self.nc, -1, 4, 4)

        # Add the seed grasp at the front
        rel_T = torch.concatenate([torch.eye(4, device=self.device, dtype=torch.float32).view(1,1,4,4).repeat(self.nc, 1,1,1), 
                                   rel_T], dim=1)

        # Number of grid samples for each candidate
        ng = rel_T.shape[1]  # nx * ny * nz + 1
        T = torch.stack([torch.inverse(torch.bmm(robot_T_seed_grasps[i,:,:].repeat(ng,1,1), rel_T[i])) 
                    for i in range(self.nc)]).view(self.nc * ng, 4, 4)
        return T
    

    def _crop_cloud(self, T_in, scene_cloud):
        '''
        Transform the cloud and run passthrough filter
        Args:
            T_in -- (torch.tensor) grasp_T_robot 3D tensor of shape (self.nc x (self.num_draws + 1)) x 4 x 4
            scene_cloud -- (torch.tensor) PCD in robot frame shape n x 3
        Returns:
            Cropped PCDs in grasp frame
        '''
        N = T_in.shape[0]
        r, p = T_in[:,:3,:3], T_in[:,:3,3].unsqueeze(-1)
        # shape: self.num_samples x 3 x n
        batch_cloud_T = scene_cloud.T.repeat(N, 1, 1)
        #Here we transform the input PC to be in grasp coordinates to apply 
        transformed_batch_T = torch.bmm(r, batch_cloud_T) + p
        # shape: self.num_samples x num_points x 3
        transformed_batch = torch.permute(transformed_batch_T, 
                                          (0, 2, 1))

        valid = (transformed_batch[:,:,0] > self.pass_min[0]) \
              & (transformed_batch[:,:,1] > self.pass_min[1]) \
              & (transformed_batch[:,:,2] > self.pass_min[2]) \
              & (transformed_batch[:,:,0] < self.pass_max[0]) \
              & (transformed_batch[:,:,1] < self.pass_max[1]) \
              & (transformed_batch[:,:,2] < self.pass_max[2])

        filtered_cloud = [transformed_batch[i, valid[i]] for i in range(N)]
        #Pad point cloud for batch process in downsampling
        padded_cloud = torch.nn.utils.rnn.pad_sequence(filtered_cloud, batch_first=True)
        # Catch when all crops are empty
        if padded_cloud.shape[1] == 0:
            padded_cloud = torch.zeros(padded_cloud.shape[0],1,padded_cloud.shape[2],
                                       device=self.device, dtype=torch.float32)

        # padded_cloud = transformed_batch * valid.unsqueeze(-1)

        # Downsample and select index
        _, idx = p3o.sample_farthest_points(padded_cloud[:,:,:3], K=self.kq)
        out_cloud = torch.stack([padded_cloud[i, idx] for i, idx in enumerate(idx)])
        out_cloud = out_cloud.view(N, self.kq, 3)

        return out_cloud
    

    def _concatenate_gripper_cloud(self, batch_cloud):
        '''
        Concatenate gripper collision point cloud to each scene cloud
        Args:
            batch_cloud -- (torch.tensor) shape (N, P, 3) in grasp frame
        Return:
            batch_cloud -- (torch.tensor) shape (N, P, 4) in grasp frame
        '''
        N, P = batch_cloud.shape[0], batch_cloud.shape[1]
        batch_cloud = torch.concatenate([batch_cloud, 
                                         SCENE_LABEL * torch.ones((N, P, 1),device=self.device)], dim=2)
        batch_gripper= self.gripper_pcd.repeat(N, 1, 1)
        ng = self.gripper_pcd.shape[0]
        batch_gripper = torch.concatenate([batch_gripper,
                                           GRIPPER_LABEL * torch.ones((N, ng, 1), device=self.device)], dim=2)

        batch_cloud = torch.cat([batch_cloud, batch_gripper], dim=1)
        return batch_cloud
    

    def _empty_grasp_check(self, batch_cloud):
        '''
        Use bounding box between two fingers to check if it is an empty grasp
        Args:
            batch_cloud -- (torch.tensor) shape (N, P, 3)
                            batch point cloud in the grasp frame
        Return:
            empty -- (torch.tensor) shape (N, ) torch.bool 
                            True if grasp is non-empty, False otherwise
        '''
        N = batch_cloud.shape[0]
        pass_min = [-0.05, -0.0225, -0.04]
        pass_max = [0.05, 0.0225, 0.01]
        inbound = (batch_cloud[:,:,0] > pass_min[0]) \
                & (batch_cloud[:,:,1] > pass_min[1]) \
                & (batch_cloud[:,:,2] > pass_min[2]) \
                & (batch_cloud[:,:,0] < pass_max[0]) \
                & (batch_cloud[:,:,1] < pass_max[1]) \
                & (batch_cloud[:,:,2] < pass_max[2])

        # Check for number of inbound points
        num_inbound = torch.sum(inbound, dim=1)
        non_empty = num_inbound >= 20

        inbound_points = batch_cloud * inbound.unsqueeze(-1)
        dist_sum = torch.sum(torch.norm(inbound_points, dim=2), dim=1)
        not_degenerate = dist_sum >= 1e-3
        empty = ~(non_empty & not_degenerate)

        return empty