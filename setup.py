from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'rl_navigation'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*')),
        (os.path.join('share', package_name, 'usd'), glob('usd/*')),
    ],
    zip_safe=False,
    maintainer='j-wye',
    maintainer_email='jongyoonpark0621@gmail.com',
    description='RL based Navigation training with Isaac Sim',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'lidar_preprocessing = rl_navigation.scripts.lidar_preprocessing:main',
            'train = rl_navigation.scripts.train:main',
            'path_visualize = rl_navigation.scripts.path_visualize:main',
        ],
    },
)