import numpy as np
import os, time
import torch
import yaml
import statistics
from threading import Condition
from enum import Enum
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import PointCloud2, JointState
from geometry_msgs.msg import Transform
from vfas_grasp_msgs.msg import TargetMsg
from closed_loop_grasping.random_crop_sampler import RandomCropSampler
from closed_loop_grasping.grasp_selector import GraspSelector
from closed_loop_grasping.grasp_target_smoother import SE3OneEuroFilter
from closed_loop_grasping.grasp_visualizer import RvizGraspVisualizer

from .utils import (distance_between_transforms, 
                    get_T, 
                    ros_transform_to_matrix,
                    matrix_to_ros_transform)
from .pose_calculator import PoseCalculator
from .models.networks import GraspEvaluatorNetwork

SERVO_TIMEOUT = 60
PREGRASP_TIMEOUT = 40

class State(Enum):
    """FSM States to be used in servoing to grasp method"""
    INIT = 1
    SERVO_TO_PREGRASP = 2
    SERVO_TO_GRASP = 3
    DONE = 4
    TIMEOUT = 5


class ClosedLoopGrasper(Node):

    def __init__(self):
        super().__init__('closed_loop_grasper')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.grasp_visualizer = RvizGraspVisualizer()

        param_file = os.path.join(
            get_package_share_directory('closed_loop_grasping'),
            'config', 'clg_params.yaml'
            )
        with open(param_file, 'r') as f:
            self.params = yaml.load(f, Loader=yaml.Loader)
        self.device = self.params['device']
        self.candidate_sampler = RandomCropSampler(self.params)
        self.grasp_selector = GraspSelector(self.params['scoring_mode'])
        urdf_file_path = os.path.join(get_package_share_directory('closed_loop_grasping'),
                                    'resource/', 
                                    self.params['urdf_file_name'])
        wrist_T_camera_path = os.path.join(get_package_share_directory('closed_loop_grasping'),
                                        'resource/', 
                                        self.params['camera_extrinsics_file_name'])
        self.pose_calculator = PoseCalculator(
            urdf_file_path,
            wrist_T_camera_path,
        )
        
        model_path = os.path.join(
                get_package_share_directory('closed_loop_grasping'),
                'resource',
                self.params['model_file']
            )
        self.eval_network = self._load_grasp_eval_network(model_path)


        self.rstate_sub = self.create_subscription(
            JointState,
            self.params['joint_states_topic_name'],
            self.joint_state_callback,
            qos_profile=qos_profile,
        )

        self.clg_trigger_sub = self.create_subscription(Transform,
                                                   self.params['clg_trigger_topic_name'],
                                                   self.clg_trigger_callback,
                                                   qos_profile = qos_profile)
        # clg_status is False by default and this module will not be activated
        self.clg_status = False
        self.world_T_grasp_init = np.eye(4)
        self.world_T_pregrasp_init = np.eye(4)

        self.pcd_sub = self.create_subscription(PointCloud2,
                                                self.params['pcd_topic_name'],
                                                self.pcd_callback,
                                                qos_profile = qos_profile)
        self.pcd_msg = None

        self.condition_object = Condition()

        self.target_publisher = self.create_publisher(TargetMsg,
                                                      self.params['target_msg_topic_name'],
                                                      qos_profile=rclpy.qos.QoSProfile(depth=10))
    
    def joint_state_callback(
        self, 
        jstate_msg: JointState
    ):
        self.last_joint_state = jstate_msg

    def clg_trigger_callback(self, msg: Transform):
        self.clg_status = True

        self.world_T_grasp_init = ros_transform_to_matrix(msg)
        self.world_T_pregrasp_init = self.world_T_grasp_init @ get_T(z=-self.pregrasp_distance)

        # Attemp to move to initial pregrasp pose
        target_msg = TargetMsg()
        target_msg.home = False
        target_msg.pose = matrix_to_ros_transform(self.world_T_pregrasp_init)
        self.target_publisher.publish(target_msg)

        success = self.servo_to_grasp()
        
        # Send robot to HOME config when servo fails
        if not success:
            target_msg = TargetMsg()
            target_msg.home = True
            target_msg.pose = matrix_to_ros_transform(np.eye(4))
            self.target_publisher.publish(target_msg)

        self.clg_status = False

    def pcd_callback(self, pcd_msg):
        if self.clg_status:
            # Save pcd message
            self.pcd_msg = pcd_msg
            # Notify all
            self.condition_object.acquire()
            self.condition_object.notify_all()
            self.condition_object.release()

    def servo_to_grasp(self,
            dist_threshold = 0.025,     #Tolerance in meters for reaching grasp pose
            angle_threshold = 5,        #Tolerance in degrees for reaching grasp pose
            init_score_threshold = 0.75, #Minimum score to start start servoing
            pregrasp_distance = 0.08,   #Offset in -Z from goal pose
            pregrasp_radius = 0.005,    #Cone diameter at pregrasp distance to define tractor beam
            heading_tolerance = 5,      #Need to be within this many degrees to achieve pregrasp position
        ): 
              
        #Set initial conditions for our algorithm
        self.init_score_threshold = init_score_threshold
        self.pregrasp_distance = pregrasp_distance
        self.dist_threshold = dist_threshold
        self.angle_threshold = angle_threshold
        self.pregrasp_radius = pregrasp_radius
        self.heading_tolerance = heading_tolerance
        self.last_iteration_time = 0.050    #50ms is our typical loop time
        self.state = State.INIT

        self.robot_T_seed_grasp = torch.tensor(self.world_T_grasp_init, device=self.device, 
                                               dtype=torch.float32).unsqueeze(dim=0)
        self.seed_score = torch.zeros((1,1), device=self.device, dtype=torch.float32)
        self.best_grasp_T_robot = self.robot_T_seed_grasp.clone()
        self.best_grasp_score = self.seed_score.clone()
        #Remember the global candidate
        self.global_seed = self.robot_T_seed_grasp.clone()

        loop_time = []
        torch.cuda.synchronize()
        t_init = time.time()
        #Initialize grasp smoother with initial seed
        if self.params['use_grasp_smoother']:
            self.grasp_smoother = SE3OneEuroFilter(t_init, 
                                                self.robot_T_seed_grasp.squeeze(), 
                                                device=self.device,
                                                min_cutoff_r = 0.002, 
                                                min_cutoff_t = 0.004,
                                                beta_r = 5,
                                                beta_t = 10,)

        while True:
            self.servo_init_time = time.time()
            torch.cuda.synchronize()
            #Run state machine
            self.service_state_machine()
            #Collect timing information
            torch.cuda.synchronize()
            self.last_iteration_time = time.time()-self.servo_init_time
            loop_time.append(self.last_iteration_time)
            if len(loop_time)==50:
                self.get_logger().info(f"Avg loop time over 50 iterations was {statistics.mean(loop_time):.4f} seconds")
                loop_time = []
            if self.state == State.DONE:
                return True
            elif self.state == State.TIMEOUT:
                return False


    def service_state_machine(self):
        if self.state == State.INIT:
            # Check if the robot has reached self.world_T_pregrasp_init
            current_pose = self.get_current_pose()
            t_dist, angle_dist = distance_between_transforms(current_pose, 
                                                             self.world_T_pregrasp_init)
            is_valid_init = np.linalg.norm(t_dist) < self.dist_threshold \
                            and np.rad2deg(angle_dist) < self.angle_threshold
            if not is_valid_init:
                return
            
            self.get_logger().info("State: INIT")
            self.run_core_algorithm(
                enable_mf = False,
                enable_region_scaling = True,
                update_seed = False,
                use_grasp_smoother = self.params['use_grasp_smoother'],
                rviz_mode='all',
            )
            # Only start servoing once we find a good grasp
            if self.best_grasp_score > self.init_score_threshold:
                self.valid_pregrasp_counter = 0
                #Set seed to be that best grasp we found
                self.robot_T_seed_grasp = torch.inverse(self.best_grasp_T_robot).clone()
                self.seed_score = self.best_grasp_score.clone()
                self.state = State.SERVO_TO_PREGRASP
                self.get_logger().debug("###### Initialization complete!!! ######")
                self.pregrasp_init_time = time.time()  
                
        elif self.state == State.SERVO_TO_PREGRASP:
            self.get_logger().info("State: SERVO_TO_PREGRASP")
            self.run_core_algorithm(
                enable_mf = True,
                enable_region_scaling = True,
                update_seed = True,
                use_grasp_smoother = self.params['use_grasp_smoother'],
                rviz_mode='all',
            )

            # Attemp to move to new pregrasp pose
            goal_pose = self.robot_T_seed_grasp.squeeze().detach().cpu().numpy() @ get_T(z=-self.pregrasp_distance)
            
            target_msg = TargetMsg()
            target_msg.home = False
            target_msg.pose = matrix_to_ros_transform(goal_pose)
            self.target_publisher.publish(target_msg)

            # Check if we have reached pre-grasp
            current_pose = self.get_current_pose()
            t_dist, angle_dist = distance_between_transforms(current_pose, goal_pose)
            is_valid_pregrasp = np.linalg.norm(t_dist) < self.dist_threshold \
                                and np.rad2deg(angle_dist) < self.angle_threshold \
                                and self.seed_score.item() > 0.85
            self.get_logger().debug("pre-grasp ct: %d, dist t: %.4f, r: %.3f, q: %.2f"%(self.valid_pregrasp_counter, 
                                                                        np.linalg.norm(t_dist), 
                                                                        np.rad2deg(angle_dist),
                                                                        self.seed_score.item()))
            if is_valid_pregrasp:
                self.valid_pregrasp_counter += 1
                if self.valid_pregrasp_counter >= 5 and \
                    (time.time()-self.pregrasp_init_time) > 5.0:
                    self.get_logger().info(" ###### GOT TO PREGRASP!!! ######")
                    self.state = State.SERVO_TO_GRASP
            else:
                #Reset counter if we fall outside of pregrasp tolerance 
                self.valid_pregrasp_counter = 0
            
            if time.time() - self.pregrasp_init_time > PREGRASP_TIMEOUT:
                self.get_logger().info("######  Pregrasp timeout !!! ######")
                self.state = State.TIMEOUT

            if time.time() - self.servo_init_time > SERVO_TIMEOUT:
                self.get_logger().info("######  Servo timeout !!! ######")
                self.state = State.TIMEOUT
            
        elif self.state == State.SERVO_TO_GRASP:
            self.get_logger().info("State: SERVO_TO_GRASP")
            #Servo robot to goal pose
            goal_pose = self.robot_T_seed_grasp.squeeze().detach().cpu().numpy()
            target_msg = TargetMsg()
            target_msg.home = False
            target_msg.pose = matrix_to_ros_transform(goal_pose)
            self.target_publisher.publish(target_msg)

            #Check if we have reached the goal
            current_pose = self.get_current_pose()
            t_dist, angle_dist = distance_between_transforms(current_pose, goal_pose)
            if np.linalg.norm(t_dist)<self.dist_threshold and np.rad2deg(angle_dist)<self.angle_threshold:
                self.get_logger().info("###### We have reached the grasp pose! ######")
                self.state = State.DONE
                
        elif self.state == State.DONE:
            self.get_logger().info("State: DONE")
        
        elif self.state == State.TIMEOUT:
            self.get_logger().info("State: TIMEOUT")


    def run_core_algorithm(self,
                           enable_mf,
                           enable_region_scaling,
                           update_seed,
                           use_grasp_smoother=True,
                           sample_method='random',
                           rviz_mode='none'):
        # Wait for the latest scene PCD, sample crops around seed grasp and evaluate with network
        self.condition_object.acquire()
        self.condition_object.wait()

        segmented_pcd_msg = self.pcd_msg
        self.grasp_selector.add_last_best_to_history(self.best_grasp_T_robot, self.best_grasp_score)
        sampling_scaling_factor, self.robot_T_seed_grasp = self.grasp_selector.get_sampling_parameters(
                    robot_T_seed_grasp=self.robot_T_seed_grasp,
                    last_iteration_time=self.last_iteration_time,
                    enable_mf=enable_mf,
                    enable_region_scaling=enable_region_scaling)
        
        self.grasps_T_robot, self.batch_crops, mask_empty, seed_crop = self.candidate_sampler.sample_crops(
            segmented_pcd_msg,
            self.robot_T_seed_grasp,
            num_draws = int(self.params['num_draws']*sampling_scaling_factor),
            t_limits = np.array(self.params['lim_t'])*sampling_scaling_factor,
            r_limits_deg = np.array(self.params['lim_r'])*sampling_scaling_factor,
            method = sample_method,
            add_noise_batches = self.params['add_noise_batches'],
            num_noise_batches = self.params['num_noise_batches'], 
        )
        self.seed_crop = seed_crop
        with torch.no_grad():
            quality_logits = self.eval_network.forward(
                self.batch_crops[:,:,:3].contiguous(),                   #Pure PCD points
                torch.transpose(self.batch_crops, 1,2).contiguous(),     #PCD+label ('feature')
            )
        g_quality = torch.sigmoid(quality_logits)
        self.g_quality_stats = self.group_inference_results(g_quality, self.params['num_noise_batches'])
        
        # Run our scoring function
        candidates_scores = self.grasp_selector.score_grasp_candidates(
                self.robot_T_seed_grasp,
                torch.inverse(self.grasps_T_robot).detach(),
                self.g_quality_stats,
                mode=self.params['scoring_mode'],
        )

        # Assign zero score to empty grasps
        candidates_scores[mask_empty] = 0.005

        # Select the next seed grasp based on our scoring function
        self.best_grasp_T_robot, self.best_grasp_score, self.robot_T_seed_grasp, self.seed_score\
                    = self.grasp_selector.get_best_grasp(candidates_scores,
                                                         self.grasps_T_robot.detach(),
                                                         self.seed_score,
                                                         update_seed = update_seed)
            
        #Use grasp smoother on this new seed grasp
        if use_grasp_smoother:
            self.robot_T_seed_grasp = self.grasp_smoother(
                time.time(), self.robot_T_seed_grasp.squeeze().clone().detach()).unsqueeze(dim=0)

        if rviz_mode == 'best':
            # Visualize results in Rviz
            self.grasp_visualizer.visualize_grasps_by_score(
                self.robot_T_seed_grasp.clone().detach().cpu().numpy(),
                self.seed_score.clone().detach().cpu().numpy(),
                score_range_filter= [0.0, 1.0],
                white_for_empty=True
            )
        elif rviz_mode == 'all':
            best_score = self.best_grasp_score.clone().detach().cpu().item()
            self.grasp_visualizer.visualize_grasps_by_score(
                    torch.inverse(self.grasps_T_robot).detach().cpu().numpy(), 
                    candidates_scores.clone().detach().cpu().numpy(),
                    #Only show best score as solid, everything else transparent
                    score_range_filter= [best_score-1e-6, best_score+1e-6],
                    white_for_empty=False)         

        self.condition_object.release()

    def get_current_pose(self):
        current_ja = np.array(self.last_joint_state.position[:7])
        current_pose = self.pose_calculator.compute_robot_T_grasp(current_ja)
        return current_pose

    def group_inference_results(self, grouped_results, num_groups):
        result={}
        if not self.params['add_noise_batches']:
            assert num_groups==0, "Make sure num_noise_batches=0 when add_noise_batches is False"
        else:
            assert num_groups>=2,  "Make sure num_noise_batches>=2 when add_noise_batches is True"
        if  num_groups==0:
            result['mean'] = grouped_results
            result['std'] = torch.zeros_like(grouped_results, device=self.device)
            result['minmax'] = torch.zeros_like(grouped_results, device=self.device)
        else:
            grouped_results = grouped_results.view(num_groups,-1)
            result['mean'] = grouped_results.mean(dim=0)
            result['std'] = grouped_results.std(dim=0)
            result['minmax'] = torch.max(grouped_results, dim=0).values - torch.min(grouped_results, dim=0).values
        return result


    def quality_grad_approach_test(self, robot_T_grasp):
        """
        This test will start at a pregrasp position of -10cm from the
        robot_T_grasp and will slowly move towards and past the 
        seed grasp while recording the network output (logits and sigmoid)
        NOTE: You must set num_draws: 0 in clg_params.yaml for this test
        """
        offsets = np.linspace(-0.04, 0.07, 25)
        logit_values = {}
        sigmoid_values = {}
        interpolated_grasps = torch.tensor([robot_T_grasp @ get_T(z=offset) for offset in offsets],
                                        device=self.device,
                                        dtype=torch.float32)
        for idx, robot_T_candidate_grasp in enumerate(interpolated_grasps):
            for i in range(20):
                segmented_pcd_msg = self.pcd_taker.take()
                grasps_T_robot, batch_crops, _ = self.candidate_sampler.sample_crops(
                    segmented_pcd_msg,
                    robot_T_candidate_grasp.unsqueeze(dim=0),
                    num_draws = self.params['num_draws'],
                    t_limits = self.params['lim_t'],
                    r_limits_deg = self.params['lim_r'],
                    method = 'random',
                    add_noise_batches = False,    #For this test, we don't want to inject noise
                    num_noise_batches = 0,  
                )
                with torch.no_grad():
                    quality_logits = self.eval_network.forward(
                        batch_crops[:,:,:3].contiguous(),                   #Pure PCD points
                        torch.transpose(batch_crops, 1,2).contiguous(),     #PCD+label ('feature')
                    )
                g_quality = torch.sigmoid(quality_logits)
                self.grasp_visualizer.visualize_grasps_by_score(
                    np.linalg.inv(grasps_T_robot.detach().clone().cpu().numpy()),
                    g_quality.clone().detach().cpu().numpy(),
                    score_range_filter= [0.0, 1.0])
                if offsets[idx] not in logit_values:
                    logit_values[offsets[idx]] = []
                logit_values[offsets[idx]].append(quality_logits.squeeze().detach().cpu().numpy().item())
                if offsets[idx] not in sigmoid_values:
                    sigmoid_values[offsets[idx]] = []
                sigmoid_values[offsets[idx]].append(g_quality.squeeze().detach().cpu().numpy().item())
        return logit_values, sigmoid_values
                    

    def _load_grasp_eval_network(self, model_path):
        net = GraspEvaluatorNetwork(device=self.device)
        print('Loading GraspEvaluator model from %s' % model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            net = torch.nn.DataParallel(net)
        net.eval()
        net.to(self.device)

        # Run random batches to load model to multi-GPU
        with torch.no_grad():
            batch_pcd = torch.rand((8, 1000, 4), dtype=torch.float32, device=self.device)
            _ = net.forward(
                batch_pcd[:,:,:3].contiguous(),             
                torch.transpose(batch_pcd, 1,2).contiguous(),
            )
        self.get_logger().info("====== Finished loading Grasp Evaluator Network =====")
        return net
    
        

def main(args=None):
    rclpy.init(args=args)
    closed_loop_grasper = ClosedLoopGrasper()
    rclpy.spin(closed_loop_grasper)
    closed_loop_grasper.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
