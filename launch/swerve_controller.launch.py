# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions.launch_configuration import LaunchConfiguration

def generate_launch_description():
    arg_robot_name = DeclareLaunchArgument('robot_name', default_value='sobit_home')
    arg_enable_gz  = DeclareLaunchArgument('enable_gz', default_value='False')
    arg_config     = DeclareLaunchArgument('config', 
        default_value=os.path.join(get_package_share_directory('swerve_steering_controller'), 'config', "swerve.yaml"
    ))

    return LaunchDescription([
        arg_robot_name,
        arg_enable_gz,
        arg_config,
        OpaqueFunction(function = launch_gz),
    ])


def launch_gz(context, *args, **kwargs):
    robot_name = LaunchConfiguration('robot_name').perform(context)
    enable_gz  = LaunchConfiguration('enable_gz').perform(context)
    config = LaunchConfiguration('config').perform(context)

    # convert enable_gz string to boolean for use_sim_time
    use_sim_time = str(enable_gz).lower() in ['true', '1', 'yes']

    return [
        Node(
            package="swerve_steering_controller",
            executable="swerve_controller",
            name="swerve_controller",
            namespace=robot_name,
            parameters=[
                {'use_sim_time': use_sim_time},
                config
            ],
            output="both",
        )
    ]
