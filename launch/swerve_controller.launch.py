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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions.launch_configuration import LaunchConfiguration

def generate_launch_description():
    arg_robot_name = DeclareLaunchArgument('robot_name', default_value='sobit_home')
    arg_enable_gz  = DeclareLaunchArgument('enable_gz', default_value='True')

    return LaunchDescription([
        arg_robot_name,
        arg_enable_gz,
        OpaqueFunction(function = launch_gz),
    ])


def launch_gz(context, *args, **kwargs):
    robot_name = LaunchConfiguration('robot_name').perform(context)
    enable_gz  = LaunchConfiguration('enable_gz').perform(context)

    config = PathJoinSubstitution(
        [
            FindPackageShare("swerve_steering_controller"),
            "config",
            "swerve.yaml",
        ]
    )

    return [
        Node(
            package="swerve_steering_controller",
            executable="swerve_controller",
            name="swerve_controller",
            namespace=robot_name,
            parameters=[
                {'use_sim_time': enable_gz},
                config
            ],
            output="both",
        )
    ]
