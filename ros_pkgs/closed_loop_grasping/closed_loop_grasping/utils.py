import sys, termios, tty
import time
from typing import Tuple, List
from scipy.spatial.transform import Rotation as sciR
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Transform
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


def np_pcd_to_ros_pcd(pcd: np.ndarray, frame_id: str) -> np.ndarray:
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


def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
        
    return np_dtype_list


def pointcloud2_to_array(cloud_msg):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Reshapes the returned array to have shape (height, width), even if the height is 1.

    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    return np.stack([cloud_arr['x'], cloud_arr['y'], cloud_arr['z']]).T


def transform_to_position(T: np.ndarray) -> np.ndarray:
    """
    Extracts the position from a 4x4 homogeneous transformation matrix.
    :param T: Homogeneous 4x4 transformation matrix.
    :returns: Position (3,)
    """
    return T[:3, 3] / T[3, 3]


def transform_to_quaternion(T: np.ndarray) -> np.ndarray:
    """
    Extracts the quaternion from a 4x4 homogeneous transformation matrix.
    :param T: Homogeneous 4x4 transformation matrix.
    :returns: Quaternion (4,)
    """
    return sciR.from_matrix(T[:3, :3]).as_quat()


def transform_to_pos_quat(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the position and quaternion from a 4x4 homogeneous transformation matrix.
    :param T: Homogeneous 4x4 transformation matrix.
    :returns: Tuple of position and quaternion: (3,) and (4,)
    """
    pos = transform_to_position(T)
    quat = transform_to_quaternion(T)
    return pos, quat

def ros_transform_to_matrix(msg: Transform):
    '''
    Convert ROS Geometry_msg/Transform type to matrix
    '''
    pos = np.array([msg.translation.x,
                    msg.translation.y,
                    msg.translation.z])
    quat = np.array([msg.rotation.x,
                     msg.rotation.y,
                     msg.rotation.z,
                     msg.rotation.w])
    T = np.eye(4)
    T[:3,:3] = sciR.from_quat(quat).as_matrix()
    T[:3,3] = pos
    return T

def matrix_to_ros_transform(T: np.ndarray):
    pos = T[:3,3]
    rot = T[:3,:3]
    quat = sciR.from_matrix(rot).as_quat()
    msg = Transform()
    msg.translation.x = pos[0]
    msg.translation.y = pos[1]
    msg.translation.z = pos[2]
    msg.rotation.x = quat[0]
    msg.rotation.y = quat[1]
    msg.rotation.z = quat[2]
    msg.rotation.w = quat[3]
    return msg


def get_T(
        x: float = 0.0, 
        y: float = 0.0, 
        z: float = 0.0, 
        seq: str = None,
        angles: List[float] = [],
    ) -> np.array:
    """
    Returns a 4x4 homogeneous transformation matrix.
    :param x: translation in x coordinate
    :param y: translation in y coordinate
    :param z: translation in z coordinate
    :param seq: string of euler axes to define rotation matrix
    :param angles: list of angles of rotation for the axes defined in seq
    :returns: 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3,3] = np.array([x,y,z])
    if seq is not None:
        T[:3,:3] = sciR.from_euler(seq, angles, degrees=True).as_matrix()
    return T


def distance_between_transforms(T1, T2):
    t_dist = (T1[:3,3] - T2[:3,3])
    T1_rot_T2 = np.transpose(T1[:3,:3]) @ T2[:3,:3]
    angle_dist = np.arccos(np.clip((np.trace(T1_rot_T2) - 1) / 2, a_min=-1.0, a_max=1.0))
    return t_dist, angle_dist

def wait_for_user_input(prompt: str = None):
    if prompt:
        print(prompt)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class Taker(Node):
    def __init__(self, topic, message_type):
        super().__init__(topic.replace('/','_') + '_tmp')
        self.m = None
        # Setup qos profile to be best effort to avoid infinite wait
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            message_type, topic, self.callback, self.qos_profile
        )

    def callback(self, m):
        self.m = m

    def take(self):
        self.m = None
        while self.m is None:
            rclpy.spin_once(self)
        return self.m

    def test_time(self, n=10):
        print(f"{self.get_name()} testing Taker time")
        for i in range(n):
            t0 = time.time()
            _ = self.take()
            t1 = time.time()
            print(f"    {i} performed self.take in {t1-t0:.3f} seconds")