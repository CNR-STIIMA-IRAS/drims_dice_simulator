from setuptools import setup
from glob import glob
import os
from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = 'drims2_dice_simulator'

generate_parameter_module(
    "dice_spawner_parameters",
    "config/dice_spawner_parameters.yaml"
)

setup(
    name=package_name,
    version='0.0.1',
    packages=['drims2_dice_simulator'],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cesare Tonola',
    maintainer_email='cesare.tonola@cnr.it',
    description='DRIMS2 Dice Simulator',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dice_spawner = drims2_dice_simulator.dice_spawner:main',
        ],
    },
)
