import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # parameters
    rosbag_path_parameter_name = 'rosbag_path'
    rosbag_path = LaunchConfiguration(rosbag_path_parameter_name)
    playback_rate_parameter_name = 'rate'
    playback_rate = LaunchConfiguration(playback_rate_parameter_name)


    # launch arguments
    launch_args = [
        DeclareLaunchArgument(
            rosbag_path_parameter_name,
            default_value='',
            description='Path to the rosbag to be played',
        ),
        DeclareLaunchArgument(
            playback_rate_parameter_name,
            default_value='1.0',
            description='Rate at which data is played',
        ),
    ]
    
    # Nodes to be launched
    # RViz
    rviz_config = os.path.join(get_package_share_directory('closed_loop_grasping'), 'rviz','clg_rosbag_test.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config, '--ros-args', '--log-level', 'fatal'],
    )
    return LaunchDescription(
        launch_args + [
            ExecuteProcess(
                cmd=['ros2', 'bag', 'play', rosbag_path, '--loop', '-r', playback_rate],
                output='log'
            ),
            rviz_node,
        ]
    )
