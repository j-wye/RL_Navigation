import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'rl_navigation'

    lidar_node = Node(
        package=pkg_name,
        executable='lidar_preprocessing',
        name='sensor_preprocessing',
        output='screen',
        # parameters=[
        #     {'angle_bins': 180},
        #     {'z_bins': 10},
        # ]
    )
    
    train_node = Node(
        package=pkg_name,
        executable='train',
        name='isaac_env',
        output='screen',
    )
    
    path_visualization_node = Node(
        package=pkg_name,
        executable='path_visualization',
        name='path_visualization',
        output='screen',
    )

    return LaunchDescription([
        lidar_node,
        train_node,
        path_visualization_node,
    ])