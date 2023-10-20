import os
import rclpy
from rclpy.node import Node
from typing import Tuple
import PIL
import yaml
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo, JointState
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory

from .pose_calculator import PoseCalculator

class WristCamPCDFilter(Node):

    """
    Subscribes to the robot wrist camera and robot state. It publishes 
    a pointcloud in the wrist camera frame with gripper fingers filtered. 
    """

    def __init__(self, voxel_size):
        super().__init__(f"wrist_cam_pcd_filter")
        self.camera_k = None
        self.voxel_downsample_size = voxel_size

        param_file = os.path.join(
            get_package_share_directory('closed_loop_grasping'),
            'config', 'clg_params.yaml'
            )
        with open(param_file, 'r') as f:
            self.params = yaml.load(f, Loader=yaml.Loader)

        # Load finger mask
        mask_file = os.path.join(
                        get_package_share_directory('closed_loop_grasping'),
                        'resource',
                        'finger_mask.png'
                    )
        img = np.asarray(PIL.Image.open(mask_file)).copy()
        self.mask = img[:,:,0] > 0
        
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

        # Initialize all subscritions, publishers and time synchronizer
        qos_profile = rclpy.qos.QoSProfile(depth=10)

        self.rstate_sub = self.create_subscription(
            JointState,
            self.params['joint_states_topic_name'],
            self.joint_state_callback,
            qos_profile=qos_profile,
        )
        self.camera_k_subscription = self.create_subscription(
            CameraInfo,
            self.params['camera_info_topic_name'],
            self.camera_k_callback,
            qos_profile=qos_profile,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.params['depth_topic_name'],
            self.camera_depth_callback,
            qos_profile=qos_profile,
        )
        self.pcd_publisher = self.create_publisher(
            PointCloud2,
            self.params['pcd_topic_name'],
            qos_profile=qos_profile
        )


    def joint_state_callback(
        self, 
        jstate_msg: JointState
    ):
        self.last_joint_state = jstate_msg

    def camera_depth_callback(
        self, 
        depth_m: Image
    ):
        # Wait until we have a camera k
        if self.camera_k is None:
            return
        depth = np.frombuffer(depth_m.data, dtype=np.uint16)
        depth = depth.reshape(depth_m.height, depth_m.width, 1)
        cam_intrinsic = self.camera_k_and_shape_to_intrinsic(depth.shape, self.camera_k)
        # Set depth value in finger region to be 0
        depth[self.mask] = 0
        # create an open3d image and use it to create the PCD
        depth_o3dimg = o3d.geometry.Image(depth)
        open3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth = depth_o3dimg, 
            intrinsic = cam_intrinsic, 
            depth_scale = 1000.0,
            depth_trunc = 2.0,
        )

        if self.voxel_downsample_size>0.0:
            open3d_pcd = open3d_pcd.voxel_down_sample(self.voxel_downsample_size)

        current_ja = np.array(self.last_joint_state.position[:7])
        b_T_cam = self.pose_calculator.compute_robot_T_camera(current_ja)
        open3d_pcd = open3d_pcd.transform(b_T_cam)

        np_pcd = np.asarray(open3d_pcd.points)          #Convert to raw numpy array
        ros_pcd = self.np_pcd_to_ros(pcd=np_pcd, frame_id=self.frame_id)
        self.pcd_publisher.publish(ros_pcd)

    def np_pcd_to_ros(
        self, 
        pcd: np.ndarray, 
        frame_id: str
    ) -> np.ndarray:
        columns = 'xyz'
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        data = pcd.astype(dtype).tobytes()
        fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate(columns)]
        header = Header(frame_id=frame_id)
        return PointCloud2(
            header=header,
            height=1,
            width=pcd.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * len(columns)),
            row_step=(itemsize * len(columns) * pcd.shape[0]),
            data=data
        )
    

    def camera_k_and_shape_to_intrinsic(
        self,
        shape: Tuple[int, int],
        camera_k: np.ndarray,
    ) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Converts a shape and camera k to open-3d intrinsics object.

        :param shape: Two-value tuple of x and y shape.
        :param camera_k: The K matrix.
        :returns: An open-3d intrinsics object.
        """
        return o3d.camera.PinholeCameraIntrinsic(
            shape[0],
            shape[1],
            camera_k[0, 0],
            camera_k[1, 1],
            camera_k[0, 2],
            camera_k[1, 2],
        )

    def camera_k_callback(self, m):
        self.camera_k = m.k.reshape((3, 3))
        self.destroy_subscription(self.camera_k_subscription)
        self.camera_k_subscription = None
        self.get_logger().info(f'Camera K received:\n{self.camera_k}')


def main(args=None):
    rclpy.init(args=args)
    wrist_cam_pcd_filter = WristCamPCDFilter(voxel_size=0.003)
    rclpy.spin(wrist_cam_pcd_filter)
    wrist_cam_pcd_filter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
