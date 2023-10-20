from typing import List
import numpy as np
import json
from urdfpy import URDF


class PoseCalculator:

    """
    Using the configured urdf file, can calculate the camera
    pose and grasp pose from joint angles.
    """

    def __init__(self,
        urdf_file_path: str,
        wrist_T_camera_path: str,
    ):

        """Initializes a PoseCalculator.

        :param urdf_file_path: The path to the urdf file for just the robot
            (probably 'panda.urdf.xml')
        :param wrist_T_camera_path: The path to the json file storing the
            transform between the wrist link and the camera. Must be a
            homogeneous transform. This file is probably 'cam_extrinsics.json'.
        :raises FileNotFoundError: If either file is not found.
        """

        try:
            self.robot = URDF.load(urdf_file_path)
            self.joint_names = list(self.root.joint_map)
        except Exception as exc:
            raise FileNotFoundError(f"could not load urdf file at {urdf_file_path}")

        with open(wrist_T_camera_path, "r") as json_file:
            wrist_T_camera_data = json.load(json_file)

        self.wrist_T_camera = np.reshape(wrist_T_camera_data, (4,4))


    def compute_robot_T_camera(self,
        joint_angles: List[float]
    ) -> np.ndarray: # (4,4)

        """Computes the camera pose in robot base frame from the joint angles.
        
        :param joint_angles: List of joint angles.
        :returns: Camera pose in robot base frame, homogeneous transformation matrix.
        """

        # ====================================================
        # =========   Example for Franka Panda Arm    ========
        # ====================================================
        # joint_config = { self.joint_names[i]: joint_angles[i] for i in range(7) }
        # fk = self.robot.link_fk(cfg = joint_config)
        # # Transformation from link8 to robot base frame
        # robot_T_link8 = fk[self.robot.links[8]]
        # # Transformation from wrist camera to robot base frame
        # robot_T_camera = robot_T_link8 @ self.wrist_T_camera

        # return robot_T_camera 
        # ====================================================
        raise NotImplementedError

    def compute_robot_T_grasp(self,
        joint_angles: List[float]
    ) -> np.ndarray: # (4,4)
        """Computes the grasp pose in robot base frame from the joint angles.
        
        :param joint_angles: List of joint angles.
        :returns: grasp pose in robot base frame, homogeneous transformation matrix.

        Please Refer to self.compute_robot_T_camera() for implementation details
        """
        raise NotImplementedError