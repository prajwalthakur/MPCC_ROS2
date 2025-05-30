import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # config = os.path.join(
    #     get_package_share_directory('mpcc_sim'),
    #     'config',
    #      'params.yaml'
        
    # )
    mpcc_node = launch_ros.actions.Node(
        package='mpcc_sim',
        executable='mpcc',
        name = 'mpcc_node',
        namespace = 'ego_vehicle',
        output='screen',
    )


    # data_logger_node = launch_ros.actions.Node(
    #     package='mpcc_sim',
    #     executable='data_logger',
    #     arguments=[],
    #     output='screen',
    # )

    
    return launch.LaunchDescription([
        mpcc_node
    ])



