import os
import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from vfas_grasp_msgs.msg import MotionField
from vfas_grasp_msgs.srv import MFQuery

class MfieldSubscriber(Node):

    def __init__(self):
        super().__init__('mfield_subscriber')
        
        param_file = os.path.join(
            get_package_share_directory('closed_loop_grasping'),
            'config', 'clg_params.yaml'
            )
        with open(param_file, 'r') as f:
            clg_params = yaml.load(f, Loader=yaml.Loader)

        self.subscription = self.create_subscription(
            MotionField,
            clg_params['mfield_topic_name'],
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.last_msg = None
        self.pc_srv = self.create_service(MFQuery, '/MFQuery', self.mf_query_callback)

    def listener_callback(self, mfield_msg):
        self.last_msg = mfield_msg

    def mf_query_callback(self, request, response):
        if self.last_msg is not None:
            response.mf_msg.header = self.last_msg.header
            response.mf_msg.step = self.last_msg.step
            response.mf_msg.data = self.last_msg.data 
        return response

def main(args=None):
    rclpy.init(args=args)

    mfield_subscriber = MfieldSubscriber()

    rclpy.spin(mfield_subscriber)

    mfield_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()