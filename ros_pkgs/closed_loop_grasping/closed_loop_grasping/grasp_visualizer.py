import os
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from rclpy.node import Node
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory

from .utils import transform_to_pos_quat


def get_mesh_marker(
        marker_id: int,
        frame_id: str,
        pose: np.ndarray,
        mesh_path: str,
        rgb: List[float] = [200.0, 200.0, 200.0],
        alpha=1.0
    ):
        assert pose.shape == (4,4), "Pose is not a (4,4) transformation"
        marker = Marker()
        marker.type = marker.MESH_RESOURCE
        marker.header.frame_id = frame_id
        marker.mesh_resource = mesh_path
        marker.id = marker_id
        pos, quat = transform_to_pos_quat(pose)
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])
        marker.scale.x = 1.
        marker.scale.y = 1.
        marker.scale.z = 1.
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = alpha
        return marker

class RvizGraspVisualizer(Node):
    def __init__(self) -> None:
        super().__init__('RvizGraspVisualizer')
        qos_profile = QoSProfile(depth=10)
        self.grasp_publisher = self.create_publisher(
            MarkerArray, f"/grasp_markers", qos_profile
        )
        cmap = plt.get_cmap('jet_r')
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        self.scalarMap = cm.ScalarMappable(norm, cmap)

    def get_rgba_from_grasp_score(self, score: float, white_for_empty=False) -> np.ndarray:
        if score == 0.005 and white_for_empty:
            return np.array([255.0, 255.0, 255.0])
        else:
            return np.squeeze(self.scalarMap.to_rgba(score))

    def visualize_grasps(self, 
                         grasp_poses: List[np.ndarray],
                         indexes_to_color: List[int] = [-1],
                         seed_grasp: np.ndarray = None,
                         frame_id: str = 'panda_link0',
                         highlight_color = [1.0, 0.0, 0.0],
                         rest_color = [128/255.0, 128/255.0, 128/255.0],
                        ):
        add_marker_array = MarkerArray()
        delete_marker_array = MarkerArray()
        grasp_mesh_path = 'file://' + os.path.join(
            get_package_share_directory("closed_loop_grasping"),
            "resource/Simple_gripper.stl")

        for grasp_idx in range(len(grasp_poses)):
            if grasp_idx in indexes_to_color:
                gripper_color = highlight_color
            else:
                gripper_color = rest_color
            gripper_marker = get_mesh_marker(grasp_idx, frame_id, grasp_poses[grasp_idx], grasp_mesh_path, gripper_color)
            add_marker_array.markers.append(gripper_marker)

        if indexes_to_color == [] and seed_grasp is not None:
            #If seed grasp was in collision, indexes_to_color will be empty, show candidate in orange color
            gripper_marker = get_mesh_marker(1000, frame_id, seed_grasp, grasp_mesh_path, [255/255.0, 102/255.0, 0.0])
            add_marker_array.markers.append(gripper_marker)

        deletion_marker = Marker()
        deletion_marker.header.frame_id = frame_id
        deletion_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(deletion_marker)
        
        self.grasp_publisher.publish(delete_marker_array)
        self.grasp_publisher.publish(add_marker_array)

    def visualize_grasps_by_score(self, 
                         grasp_poses: np.ndarray,
                         grasp_scores: np.ndarray,
                         frame_id = 'panda_link0',
                         score_range_filter= [0.5, 1.0],
                         out_of_range_alpha = 0.2,
                         white_for_empty=False,     #Empty grasps will be colored white
                        ):
        assert grasp_poses.shape[0] == grasp_scores.shape[0], f"Grasp_poses has shape {grasp_poses.shape} and grasp_scores has shape {grasp_scores.shape}"
        add_marker_array = MarkerArray()
        delete_marker_array = MarkerArray()
        grasp_mesh_path = 'file://' + os.path.join(
            get_package_share_directory("closed_loop_grasping"),
            "resource/Simple_gripper.stl")
        for idx, (grasp_pose, score) in enumerate(zip(grasp_poses, grasp_scores)):
            gripper_color = self.get_rgba_from_grasp_score(score, white_for_empty)[0:3]
            if score>= score_range_filter[0] and score<= score_range_filter[1]:
                alpha = 1.0
            else:
                alpha = out_of_range_alpha
            gripper_marker = get_mesh_marker(idx, frame_id, grasp_pose, grasp_mesh_path, gripper_color, alpha)
            add_marker_array.markers.append(gripper_marker)
        
        deletion_marker = Marker()
        deletion_marker.header.frame_id = frame_id
        deletion_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(deletion_marker)
        
        self.grasp_publisher.publish(delete_marker_array)
        self.grasp_publisher.publish(add_marker_array)