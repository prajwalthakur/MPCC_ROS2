from setuptools import setup
import os
from glob import glob

package_name = 'mpcc_sim'
submodules = "mpcc_sim/helper"
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')), 
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')
         )
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='prajwal Thakur',
    maintainer_email='prajwalthakur98@gmail.com',
    description=' mpcc-sim file for pyhton3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'mpcc = mpcc_sim.mpcc_main:main',
        'data_logger=mpcc_sim.data_logger:main'
        ],
    },
)


