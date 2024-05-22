import torch
from rclpy.node import Node
import rclpy
import numpy as np
from collections import deque, namedtuple
from vfas_grasp_msgs.srv import MFQuery
import pytorch3d.ops as p3o

SeedGrasp = namedtuple('SeedGrasp', ['pose', 'score'])

class GraspSelector(Node):
    def __init__(self, scoring_mode, history_size=3, device='cuda') -> None:
        super().__init__('grasp_selector')
        print(f"GraspSelector initialized with scoring mode: {scoring_mode}")
        self.scoring_mode = scoring_mode
        self.device = device
        self.history = deque(maxlen=history_size)
        self.mfield_client = self.create_client(MFQuery, '/MFQuery')
        self.mf_request = MFQuery.Request()
        self.sampling_scale_factor = 1.0
        #Sampling region increases 30% in size for each iteration where
        #the best grasp score is below a threshold
        self.sampling_scale_step = 1.3  
        self.max_sampling_scale_factor = 3.0

        while not self.mfield_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Motion Field query service not available, waiting again...')


    def add_last_best_to_history(self, grasp_T_robot, grasp_score):
        """
        This function adds the latest selected seed_grasp_pose (4x4)
        and its corresponding score to the history buffer
        """
        self.history.append(SeedGrasp(grasp_T_robot.view(4,4), grasp_score))
        

    def get_sampling_parameters(self,
                                robot_T_seed_grasp,
                                last_iteration_time,
                                enable_mf=True,
                                enable_region_scaling=True):
        """
        This function receives the latest best grasp pose and score,
        adds them to a history buffer and decides on what the next
        sampling region scale should be, as well as - potentially -
        modify the best grasp with an offset which is decided from
        our 3D motion field measurements

        Inputs:
            robot_T_seed_grasp -- (torch.tensor) shape (1,4,4)
            last_score -- (torch.tensor) shape (1,1)
            enable_mf -- (Bool) use motion field measurement if set True
            enable_region_scaling -- (Bool) 
                increase sampling region if haven't seen good samples for a while
        Returns:
            sampling_region_scale -- (float)
            new_seed -- (torch.tensor) shape (1,4,4)
        """
        if enable_mf:
            offset = self._get_mf_offset(last_iteration_time)
        else:
            offset = torch.eye(4, device=self.device)
        new_seed = offset @ robot_T_seed_grasp.squeeze(0)
        
        if enable_region_scaling:
            sampling_region_scale = self._get_sampling_region_scale()
        else:
            sampling_region_scale = 1.0
        
        return sampling_region_scale, new_seed.view(1,4,4)
       
    def get_best_grasp(self, candidates_scores, grasps_T_robot, seed_score, update_seed):
        """
        This function will take in the current seed grasp, the candidates 
        (grasps_T_robot) and their qualities (as per evaluator network)
        and return the best grasp according to our scoring function 
        Returns the new seed as grasp_T_robot
        """
        best_candidate_idx = torch.argmax(candidates_scores, dim=0)
        best_grasp_T_robot = grasps_T_robot[best_candidate_idx]
        best_grasp_score = candidates_scores[best_candidate_idx]
        
        prob = min(best_grasp_score.item() / (seed_score.item() + 1e-8), 1)
        if update_seed and np.random.rand() < prob:
            robot_T_seed_grasp = torch.inverse(best_grasp_T_robot)
            seed_score = best_grasp_score.clone()
        else:
            robot_T_seed_grasp = torch.inverse(grasps_T_robot[0])

        return best_grasp_T_robot.view(1,4,4), \
               best_grasp_score.view(1,1), \
               robot_T_seed_grasp.view(1,4,4), \
               seed_score.view(1,1)

    def score_grasp_candidates(self, seed_grasp, candidate_grasps, g_quality_stats, mode=None, debug=False):
        """
        candidate_grasps has shape [Batch, 4, 4] in wrist_cam frame
        seed_grasp has shape [1, 4, 4] in wrist_cam frame
        grasp_qualities has shape [Batch, 1]
        """
        allowed_modes = ['q_only', 'trans_rot_penalty', 'q_noise', 'q_noise_with_tr']
        if mode is None:
            mode=self.scoring_mode
        assert mode in allowed_modes, f"Mode: {mode} unrecognized. Allowed modes are: {allowed_modes}"

        if mode=='trans_rot_penalty' or mode=='q_noise_with_tr':
            k1 = 2.0    #Penalty weight for translation distance from seed candidate
            k2 = 0.1    #Penalty weight for rotation distance from seed candidate
            k3 = 0.6    #Penalty weight for grasp quality uncertainty
            #Compute translations between seed and all candidates
            t_dist = torch.linalg.norm(
                (candidate_grasps - seed_grasp.repeat((candidate_grasps.shape[0],1,1)))[:,:3,3],
                dim=1).unsqueeze(dim=1)     #Shape [Batch,1]
            #Compute angles (axis-angle) between seed and all candidates
            #Based of here: https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
            T1_rot_T2 = torch.matmul(candidate_grasps[:,:3,:3] , torch.transpose(seed_grasp[:,:3,:3],1,2))  #Shape [Batch,3,3]
            #Computes the batch trace operation: https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
            batch_trace = T1_rot_T2.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(dim=1) 
            angle_dist = torch.arccos(torch.clamp((batch_trace-1)/2.0, min=-1.0, max=1.0))
    
            if mode=='trans_rot_penalty':
                score = g_quality_stats['mean'].detach().view(-1,1) - \
                    k1*t_dist - k2*angle_dist
            else:
                score = g_quality_stats['mean'].detach().view(-1,1) - \
                k1*t_dist - k2*angle_dist - k3*g_quality_stats['minmax'].detach().view(-1,1)

        elif mode=='q_only':
            score = g_quality_stats['mean'].detach().view(-1,1)

        elif mode=='q_noise':
            k1 = 0.8    #Penalty weight for grasp quality uncertainty
            score = g_quality_stats['mean'].detach().view(-1,1) - k1*g_quality_stats['minmax'].detach().view(-1,1)
    
        return torch.clamp(score, min=0.0, max=1.0)


    def _get_sampling_region_scale(self, threshold=0.7):
        """
        This function looks at history of (grasp,score) and decides 
        how to scale the sampling region.
        For now, if last score is below a threshold, we want to
        increase the sampling region by a fixed factor up to a maximum.
        If its the history is empty, we also increase the region
        size by a fixed factor to hotstart the algorithm
        """
        if self.history[-1].score < threshold:
            self.sampling_scale_factor *= self.sampling_scale_step
            region_scale_factor = np.clip(self.sampling_scale_factor,1.0,self.max_sampling_scale_factor)
            return region_scale_factor
        else:
            # self.get_logger().info("Found a good candidate, set sampling region scale back to 1")
            self.sampling_scale_factor = 1.0
            return 1.0

    def _get_mf_offset(self, last_iteration_time, radius=0.04, threshold=2):
        """
        Looks at the last motion field data, and the last best grasp
        to find the relevant motion vectors close to that grasp
        and output a final offset for the last best grasp
        Radius determines the sphere size around the seed where we 
        look for velocity vectors in the motion field
        """

        # Query the latest motion field data, reshape into a (n x 6) tensor
        self.mf_query_future = self.mfield_client.call_async(self.mf_request)
        rclpy.spin_until_future_complete(self, self.mf_query_future)
        mfield_msg = self.mf_query_future.result().mf_msg
        if len(mfield_msg.data)==0:
            #print("No Motion Field data yet!")
            return torch.eye(4, device=self.device)
        mfield_data = torch.tensor(mfield_msg.data, 
                                   dtype=torch.float32, 
                                   device=self.device).reshape(1, -1, mfield_msg.step)
        # Ball query from the latest_best_grasp
        robot_T_grasp = torch.inverse(self.history[-1].pose).view(4,4)
        ball_center = robot_T_grasp[:3,3].view(1,1,3)
        _, idx, _ = p3o.ball_query(ball_center, mfield_data[:,:,:3], K=20, 
                                   radius=radius)
        idx = idx.reshape(-1)
        idx = idx[idx>=0]
        if len(idx) < threshold:
            offset = torch.eye(4, dtype=torch.float32, device=self.device)
        else:
            # Average mfield vectors to compute the offset
            # self.get_logger().info(f"{len(idx)} Motion Field vectors were found inside query ball")
            vel_vectors = mfield_data[:,idx,3:6].reshape(-1,3)
            mean_vel = torch.mean(vel_vectors, dim=0)
            #print(f"Mean Velocity: {mean_vel}")
            offset = torch.eye(4, dtype=torch.float32, device=self.device)
            offset[:3,3] = last_iteration_time * mean_vel
            #print(f"Offset applied was: {offset[:3,3]} || Iteration time: {last_iteration_time}")
        return offset

