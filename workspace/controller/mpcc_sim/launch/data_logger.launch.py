import launch
import launch_ros.actions
def generate_launch_description():
    data_logger_node = launch_ros.actions.Node(
        package='mpcc_sim',
        executable='data_logger',
        arguments=[],
        output='screen',
    )

    
    return launch.LaunchDescription([
       data_logger_node
    ])



