from typing import List
from rclpy.node import Node
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np

class MotionFieldVisualizer(Node):
    def __init__(self) -> None:
        super().__init__('MotionFieldVisualizer')
        qos_profile = QoSProfile(depth=10)
        self.mfield_publisher = self.create_publisher(
            MarkerArray, f"/mfield_markers", qos_profile
        )

    def visualize_mfield(self, 
                         mfield_locs: np.ndarray,
                         mfield_vecs: np.ndarray,
                         frame_id: str = 'rgb_camera',
                        ):
        add_marker_array = MarkerArray()
        delete_marker_array = MarkerArray()

        num_pts = mfield_locs.shape[0]

        for idx in range(num_pts):
            loc = mfield_locs[idx]
            vec = mfield_vecs[idx]
            arrow_marker = get_arrow_marker(idx, frame_id, loc, vec)
            add_marker_array.markers.append(arrow_marker)

        deletion_marker = Marker()
        deletion_marker.header.frame_id = frame_id
        deletion_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(deletion_marker)
        
        self.mfield_publisher.publish(delete_marker_array)
        self.mfield_publisher.publish(add_marker_array)

    
def get_arrow_marker(
        marker_id: int,
        frame_id: str,
        loc: np.ndarray,
        vec: np.ndarray,
        rgb: List[float] = [0.8, 0.2, 0.2],
        alpha=1.0
    ):
        marker = Marker()
        marker.type = marker.ARROW
        marker.header.frame_id = frame_id
        marker.id = marker_id
        tail = Point(x=loc[0], 
                     y=loc[1], 
                     z=loc[2])
        tip = Point(x=loc[0]+vec[0], 
                    y=loc[1]+vec[1], 
                    z=loc[2]+vec[2])
        marker.points = [ tail, tip ]
        marker.scale.x = 0.002
        marker.scale.y = 0.003
        marker.scale.z = 0.02
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = alpha
        return marker
