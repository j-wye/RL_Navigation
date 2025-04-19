import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    home = os.environ['HOME']
    pkg_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    isaac_python = os.path.join(home, 'isaacsim2', 'python.sh')
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', f'{pkg_path}/scripts/lidar_preprocessing.py'],
            output='screen',
            name='lidar_preprocessing'
        ),
        ExecuteProcess(
            cmd=[isaac_python, f'{pkg_path}/scripts/train.py'],
            output='screen',
            name='train'
        ),
        ExecuteProcess(
            cmd=['python3', f'{pkg_path}/scripts/path_visualize.py'],
            output='screen',
            name='path_visualize'
        ),
    ])
