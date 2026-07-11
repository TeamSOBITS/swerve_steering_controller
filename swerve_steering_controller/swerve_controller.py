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

from typing import List
import math
import rclpy
from rclpy.clock import Clock, Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_geometry_msgs import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_ros

from builtin_interfaces.msg import Duration as MsgDuration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tf_transformations import quaternion_from_euler

from .control_model import difference_between_angles
from .drive_module import DriveModule
from .geometry import Point
from .profile import SingleVariableLinearProfile, SingleVariableSCurveProfile, TransientVariableProfile
from .states import BodyMotion, DriveModuleMeasuredValues
from .steering_controller import DriveModuleDesiredValuesProfilePoint, ModuleFollowsBodySteeringController

class SwerveController(Node):
    def __init__(self):
        super().__init__("publisher_velocity_controller")
        # Declare all parameters
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("twist_topic", "cmd_vel")
        self.declare_parameter("enable_tf_prefix", False)

        self.declare_parameter("position_controller_name", "position_controller")
        self.declare_parameter("velocity_controller_name", "velocity_controller")
        self.declare_parameter("cycle_fequency", 50)
        self.declare_parameter("driving_status_threshold", 0.26)

        self.declare_parameter("steering_joints", ["joint1", "joint2"])
        self.declare_parameter("drive_joints", ["joint1", "joint2"])

        self.declare_parameter("mobile_base.wheel_radius", 0.04)
        self.declare_parameter("mobile_base.wheel_width", 0.08)
        self.declare_parameter("mobile_base.wheel_x_distance", 0.35)
        self.declare_parameter("mobile_base.wheel_y_distance", 0.35)
        self.declare_parameter("mobile_base.steer_max_vel", 10.0)
        self.declare_parameter("mobile_base.steer_min_acc", 0.1)
        self.declare_parameter("mobile_base.steer_max_acc", 1.0)
        self.declare_parameter("mobile_base.drive_max_vel", 10.0)
        self.declare_parameter("mobile_base.drive_min_acc", 0.1)
        self.declare_parameter("mobile_base.drive_max_acc", 1.0)

        self.get_logger().info(f'Initializing swerve controller ...')

        self.last_velocity_command: Twist = None

        # Set True once we have received a real joint_states message, so that the timer doesn't
        # command steer angles based on the initial (zeroed) drive module state and cause the
        # steer to jump from the robot's actual angle to 0.0 on the first command.
        self.received_joint_states = False

        self.node_namespace = self.get_namespace().replace("/", "")

        self.enable_tf_prefix = self.get_parameter("enable_tf_prefix").value

        self.robot_base_link = self.get_parameter("robot_base_frame").value
        self.get_logger().info(f'Using robot base link: {self.robot_base_link}')

        prefix = (self.node_namespace + "/") if (self.enable_tf_prefix and self.node_namespace != "") else ""
        self.odom_frame = prefix + "odom"
        self.base_frame = prefix + self.robot_base_link

        self.driving_status_threshold = self.get_parameter("driving_status_threshold").value

        self.mobile_base = {}
        self.mobile_base["wheel_radius"] = self.get_parameter("mobile_base.wheel_radius").value
        self.mobile_base["wheel_width"] = self.get_parameter("mobile_base.wheel_width").value
        self.mobile_base["wheel_x_distance"] = self.get_parameter("mobile_base.wheel_x_distance").value
        self.mobile_base["wheel_y_distance"] = self.get_parameter("mobile_base.wheel_y_distance").value
        self.mobile_base["steer_max_vel"] = self.get_parameter("mobile_base.steer_max_vel").value
        self.mobile_base["steer_min_acc"] = self.get_parameter("mobile_base.steer_min_acc").value
        self.mobile_base["steer_max_acc"] = self.get_parameter("mobile_base.steer_max_acc").value
        self.mobile_base["drive_max_vel"] = self.get_parameter("mobile_base.drive_max_vel").value
        self.mobile_base["drive_min_acc"] = self.get_parameter("mobile_base.drive_min_acc").value
        self.mobile_base["drive_max_acc"] = self.get_parameter("mobile_base.drive_max_acc").value

        self.get_logger().info(f'Initialized mobile base parameters: {self.mobile_base}')

        # publish the module steering angle
        position_controller_name = self.get_parameter("position_controller_name").value
        steering_angle_publish_topic = position_controller_name + "/" + "commands"
        self.drive_module_steering_angle_publisher = self.create_publisher(
            Float64MultiArray,
            steering_angle_publish_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))

        self.get_logger().info(
            f'Publishing steering angle changes on topic "{steering_angle_publish_topic}"'
        )

        # publish the module drive velocity
        velocity_controller_name = self.get_parameter("velocity_controller_name").value
        velocity_publish_topic = velocity_controller_name + "/" + "commands"
        self.drive_module_velocity_publisher = self.create_publisher(
            Float64MultiArray,
            velocity_publish_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))

        self.get_logger().info(
            f'Publishing drive velocity changes on topic "{velocity_publish_topic}"'
        )

        # publish odometry
        odom_topic = "odom"
        self.odometry_publisher = self.create_publisher(
            Odometry,
            odom_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))
        self.get_logger().info(
            f'Publishing odometry information on topic "{odom_topic}"'
        )

        # Define TF broadcaster 
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize odom TF 
        zero_odometry = Odometry()
        zero_odometry.header.stamp = self.get_clock().now().to_msg()
        zero_odometry.header.frame_id = self.odom_frame
        zero_odometry.child_frame_id = self.base_frame
        zero_odometry.pose.pose.position.x = 0.0
        zero_odometry.pose.pose.position.y = 0.0
        zero_odometry.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0.0, 0.0, 0.0)
        zero_odometry.pose.pose.orientation.x = quat[0]
        zero_odometry.pose.pose.orientation.y = quat[1]
        zero_odometry.pose.pose.orientation.z = quat[2]
        zero_odometry.pose.pose.orientation.w = quat[3]
        self.send_odom_transform(zero_odometry)

        # self.send_static_tf()

        # Create the controller that will determine the correct drive commands for the different drive modules
        # Create the controller before we subscribe to state changes so that the first change that comes in gets
        # registered
        self.get_logger().info(f'Storing drive module information...')
        self.drive_modules = self.get_drive_modules()
        self.controller = ModuleFollowsBodySteeringController(self.drive_modules, self.get_motion_profile, self.write_log)

        # initialize the time tracking variables after we get the controller up and running
        # so that we can initialize the controller at the same time.
        self.store_time_and_update_controller_time()
        self.last_control_update_send_at = self.last_recorded_time
        self.last_velocity_command_received_at = self.last_recorded_time

        # keep last position message to avoid inf value in steering angle data
        self.last_position_msg: Float64MultiArray = None

        # Create the timer that is used to ensure that we publish movement data regularly
        self.cycle_time_in_hertz = self.get_parameter("cycle_fequency").value
        self.get_logger().info(
            f'Publishing changes at fequency: "{self.cycle_time_in_hertz}" Hz'
        )

        self.timer = self.create_timer(
            1.0 / self.cycle_time_in_hertz,
            self.timer_callback,
            callback_group=None,
            clock=self.get_clock()
        )
        self.i = 0

        # Listen for state changes in the drive modules
        joint_state_topic = "joint_states"
        self.state_change_subscription = self.create_subscription(
            JointState,
            joint_state_topic,
            self.joint_states_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10)
        )

        self.get_logger().info(
            f'Listening for drive module state changes on "{joint_state_topic}"'
        )

        # Initialize the drive modules
        self.last_drive_module_state = self.initialize_drive_module_states(self.drive_modules)

        # Finally listen to the cmd_vel topic for movement commands. We could have a message incoming
        # at any point after we register so we set this subscription up last.
        twist_topic = self.get_parameter("twist_topic").value
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            twist_topic,
            self.cmd_vel_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))
        self.get_logger().info(
            f'Listening for movement commands on topic "{twist_topic}"'
        )

    def cmd_vel_callback(self, msg: Twist):
        if msg == None:
            return

        # If this twist message is the same as last time, then we don't need to do anything
        if self.last_velocity_command is not None:
            if msg.linear.x == self.last_velocity_command.linear.x and \
                msg.linear.y == self.last_velocity_command.linear.y and \
                msg.angular.z == self.last_velocity_command.angular.z:

                # The last command was the same as the current command. So just ignore it and move on.
                self.get_logger().info(
                    f'Received a Twist message that is the same as the last message. Taking no action. Message was: "{msg}"'
                )

                return

        self.get_logger().info(
            f'Received a Twist message that is different from the last command. Processing message: "{msg}"'
        )

        # Just record the latest twist and when it arrived. The timer_callback computes IK directly
        # from this latest twist on every tick (stateless), so there is no profile to (re)build here.
        self.last_velocity_command = msg
        self.last_velocity_command_received_at = self.last_recorded_time

    def get_drive_modules(self) -> List[DriveModule]:
        # store the steering joints
        steering_joint_names = self.get_parameter("steering_joints").value
        steering_joints = []
        for name in steering_joint_names:
            steering_joints.append(name)
            self.get_logger().info(
                f'Discovered steering joint: "{name}"'
            )

        # store the drive joints
        drive_joint_names = self.get_parameter("drive_joints").value
        drive_joints = []
        for name in drive_joint_names:
            drive_joints.append(name)
            self.get_logger().info(
                f'Discovered drive joint: "{name}"'
            )

        drive_modules: List[DriveModule] = []
        drive_module_name = "f_l"  # TODO: parameterize this
        left_front = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(
                0.5 * (self.mobile_base["wheel_x_distance"]),
                0.5 * (self.mobile_base["wheel_y_distance"]),
                0.0
            ),
            wheel_radius=self.mobile_base["wheel_radius"],
            wheel_width=self.mobile_base["wheel_width"],
            steering_motor_maximum_velocity=self.mobile_base["steer_max_vel"],
            steering_motor_minimum_acceleration=self.mobile_base["steer_min_acc"],
            steering_motor_maximum_acceleration=self.mobile_base["steer_max_acc"],
            drive_motor_maximum_velocity=self.mobile_base["drive_max_vel"],
            drive_motor_minimum_acceleration=self.mobile_base["drive_min_acc"],
            drive_motor_maximum_acceleration=self.mobile_base["drive_max_acc"],
        )
        drive_modules.append(left_front)

        self.get_logger().info(
            f'Configured drive module: "{left_front.name}" ' +
            f'with steering link: "{left_front.steering_link_name}" ' +
            f'and drive link: "{left_front.driving_link_name}" ' +
            f'and position: ["{left_front.steering_axis_xy_position.x}", "{left_front.steering_axis_xy_position.y}"]'
        )

        drive_module_name = "f_r"  # TODO: parameterize this
        right_front = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(
                0.5 * (self.mobile_base["wheel_x_distance"]),
                -0.5 * (self.mobile_base["wheel_y_distance"]),
                0.0
            ),
            wheel_radius=self.mobile_base["wheel_radius"],
            wheel_width=self.mobile_base["wheel_width"],
            steering_motor_maximum_velocity=self.mobile_base["steer_max_vel"],
            steering_motor_minimum_acceleration=self.mobile_base["steer_min_acc"],
            steering_motor_maximum_acceleration=self.mobile_base["steer_max_acc"],
            drive_motor_maximum_velocity=self.mobile_base["drive_max_vel"],
            drive_motor_minimum_acceleration=self.mobile_base["drive_min_acc"],
            drive_motor_maximum_acceleration=self.mobile_base["drive_max_acc"],
        )
        drive_modules.append(right_front)

        self.get_logger().info(
            f'Configured drive module: "{right_front.name}" ' +
            f'with steering link: "{right_front.steering_link_name}" ' +
            f'and drive link: "{right_front.driving_link_name}" ' +
            f'and position: ["{right_front.steering_axis_xy_position.x}", "{right_front.steering_axis_xy_position.y}"]'
        )

        drive_module_name = "b_l"  # TODO: parameterize this
        left_rear = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(
                -0.5 * (self.mobile_base["wheel_x_distance"]),
                0.5 * (self.mobile_base["wheel_y_distance"]),
                0.0
            ),
            wheel_radius=self.mobile_base["wheel_radius"],
            wheel_width=self.mobile_base["wheel_width"],
            steering_motor_maximum_velocity=self.mobile_base["steer_max_vel"],
            steering_motor_minimum_acceleration=self.mobile_base["steer_min_acc"],
            steering_motor_maximum_acceleration=self.mobile_base["steer_max_acc"],
            drive_motor_maximum_velocity=self.mobile_base["drive_max_vel"],
            drive_motor_minimum_acceleration=self.mobile_base["drive_min_acc"],
            drive_motor_maximum_acceleration=self.mobile_base["drive_max_acc"],
        )
        drive_modules.append(left_rear)

        self.get_logger().info(
            f'Configured drive module: "{left_rear.name}" ' +
            f'with steering link: "{left_rear.steering_link_name}" ' +
            f'and drive link: "{left_rear.driving_link_name}" ' +
            f'and position: ["{left_rear.steering_axis_xy_position.x}", "{left_rear.steering_axis_xy_position.y}"]'
        )

        drive_module_name = "b_r"  # TODO: parameterize this
        right_rear = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(
                -0.5 * (self.mobile_base["wheel_x_distance"]),
                -0.5 * (self.mobile_base["wheel_y_distance"]),
                0.0
            ),
            wheel_radius=self.mobile_base["wheel_radius"],
            wheel_width=self.mobile_base["wheel_width"],
            steering_motor_maximum_velocity=self.mobile_base["steer_max_vel"],
            steering_motor_minimum_acceleration=self.mobile_base["steer_min_acc"],
            steering_motor_maximum_acceleration=self.mobile_base["steer_max_acc"],
            drive_motor_maximum_velocity=self.mobile_base["drive_max_vel"],
            drive_motor_minimum_acceleration=self.mobile_base["drive_min_acc"],
            drive_motor_maximum_acceleration=self.mobile_base["drive_max_acc"],
        )
        drive_modules.append(right_rear)

        self.get_logger().info(
            f'Configured drive module: "{right_rear.name}" ' +
            f'with steering link: "{right_rear.steering_link_name}" ' +
            f'and drive link: "{right_rear.driving_link_name}" ' +
            f'and position: ["{right_rear.steering_axis_xy_position.x}", "{right_rear.steering_axis_xy_position.y}"]'
        )

        return drive_modules

    def get_motion_profile(self, start: float, end: float) -> TransientVariableProfile:
        # return SingleVariableSCurveProfile(start, end)

        return SingleVariableLinearProfile(start, end)

    def initialize_drive_module_states(self, drive_modules: List[DriveModule]) -> List[DriveModuleMeasuredValues]:
        measured_drive_states: List[DriveModuleMeasuredValues] = []
        for drive_module in self.drive_modules:

            value = DriveModuleMeasuredValues(
                drive_module.name,
                drive_module.steering_axis_xy_position.x,
                drive_module.steering_axis_xy_position.y,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            )
            measured_drive_states.append(value)

            self.get_logger().info(
                f'Initializing drive module state for module: "{drive_module.name}"'
            )

        self.store_time_and_update_controller_time()
        self.controller.on_state_update(measured_drive_states)

        return measured_drive_states

    def joint_states_callback(self, msg: JointState):
        if msg == None:
            return

        # self.get_logger().debug(
        #     f'Received a JointState message: "{msg}"'
        # )

        # It would be better if we stored this message and processed it during our own timer loop. That way
        # we wouldn't be blocking the callback.

        joint_names: List[str] = msg.name
        joint_positions: List[float] = [pos for pos in msg.position]
        joint_velocities: List[float] = [vel for vel in msg.velocity]

        measured_drive_states: List[DriveModuleMeasuredValues] = []
        for index, drive_module in enumerate(self.drive_modules):
            if drive_module.steering_link_name in joint_names and drive_module.driving_link_name in joint_names:
                steering_values_index = joint_names.index(drive_module.steering_link_name)
                drive_values_index = joint_names.index(drive_module.driving_link_name)

                value = DriveModuleMeasuredValues(
                    drive_module.name,
                    drive_module.steering_axis_xy_position.x,
                    drive_module.steering_axis_xy_position.y,
                    joint_positions[steering_values_index],
                    joint_velocities[steering_values_index],
                    0.0,
                    0.0,
                    joint_velocities[drive_values_index] * drive_module.wheel_radius,
                    0.0,
                    0.0
                )
                measured_drive_states.append(value)

                # self.get_logger().info(
                #     f'Updating joint states for: "{drive_module.name}" with: ' +
                #     f'[ steering angle: "{value.orientation_in_body_coordinates.z}", ' +
                #     f' steering velocity: "{value.orientation_velocity_in_body_coordinates.z}",' +
                #     f' velocity: "{value.drive_velocity_in_module_coordinates.x}" ] '
                # )
            else:
                # grab the previous state and just assume that's the one
                value = self.last_drive_module_state[index]
                measured_drive_states.append(value)

                # self.get_logger().debug(
                #     f'Updating joint states for: "{drive_module.name}" with: ' +
                #     f'[ steering angle: "{value.orientation_in_body_coordinates.z}", ' +
                #     f' steering velocity: "{value.orientation_velocity_in_body_coordinates.z}",' +
                #     f' velocity: "{value.drive_velocity_in_module_coordinates.x}" ] '
                # )

        # Ideally we would get the time from the message. And then check if we have gotten a more
        # recent message
        self.store_time_and_update_controller_time()
        self.controller.on_state_update(measured_drive_states)
        self.last_drive_module_state = measured_drive_states
        self.received_joint_states = True

    def publish_odometry(self):
        body_state = self.controller.body_state_at_current_time()

        msg = Odometry()
        msg.header.stamp = self.last_recorded_time.to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = body_state.position_in_world_coordinates.x
        msg.pose.pose.position.y = body_state.position_in_world_coordinates.y
        msg.pose.pose.position.z = body_state.position_in_world_coordinates.z

        quat = quaternion_from_euler(0.0, 0.0, body_state.orientation_in_world_coordinates.z)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        msg.twist.twist.linear.x = body_state.motion_in_body_coordinates.linear_velocity.x
        msg.twist.twist.linear.y = body_state.motion_in_body_coordinates.linear_velocity.y
        msg.twist.twist.linear.z = body_state.motion_in_body_coordinates.linear_velocity.z

        msg.twist.twist.angular.x = body_state.motion_in_body_coordinates.angular_velocity.x
        msg.twist.twist.angular.y = body_state.motion_in_body_coordinates.angular_velocity.y
        msg.twist.twist.angular.z = body_state.motion_in_body_coordinates.angular_velocity.z

        self.send_odom_transform(msg)

        # For now we ignore the covariances

        # self.get_logger().info(
        #     'Publishing odometry message {}'.format(msg)
        # )

        self.odometry_publisher.publish(msg)

    def send_odom_transform(self, odometry_msg: Odometry):
        transform = TransformStamped()
        transform.header.stamp = odometry_msg.header.stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = odometry_msg.pose.pose.position.x
        transform.transform.translation.y = odometry_msg.pose.pose.position.y
        transform.transform.translation.z = odometry_msg.pose.pose.position.z
        transform.transform.rotation.x = odometry_msg.pose.pose.orientation.x
        transform.transform.rotation.y = odometry_msg.pose.pose.orientation.y
        transform.transform.rotation.z = odometry_msg.pose.pose.orientation.z
        transform.transform.rotation.w = odometry_msg.pose.pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)

    def send_static_tf(self):
        tf_static_broadcaster = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        quat = quaternion_from_euler(0.0, 0.0, 0.0)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        tf_static_broadcaster.sendTransform(transform)

    def store_time_and_update_controller_time(self):
        time: Time = self.get_clock().now()
        seconds = time.nanoseconds * 1e-9
        self.controller.on_tick(seconds)
        self.last_recorded_time = time

    def timer_callback(self):
        self.store_time_and_update_controller_time()

        # always send out the odometry information
        self.publish_odometry()

        # Don't command steer angles until we have a real measured steer angle for each module,
        # otherwise the goal steer selection would be based on the initial (zeroed) state and could
        # jump the steer from the robot's actual angle to 0.0 on the first command.
        if not self.received_joint_states:
            return

        # Nothing to command yet if we haven't received a twist.
        if self.last_velocity_command is None:
            return

        # Stateless direct IK: compute the desired module states from the latest twist on every
        # tick. No profile is built or restarted here, so streamed cmd_vel (e.g. from Nav2) doesn't
        # cause per-tick ramp restarts.
        twist = self.last_velocity_command
        body_motion = BodyMotion(
            twist.linear.x, twist.linear.y, twist.angular.z,
            0, 0, 0,
            0, 0, 0)
        module_options = self.controller.control_model.state_of_wheel_modules_from_body_motion(body_motion)

        steering_angle_values = []
        drive_velocity_values = []
        for i, (forward_state, reverse_state) in enumerate(module_options):
            current_steer = self.last_drive_module_state[i].orientation_in_body_coordinates.z

            forward_diff = difference_between_angles(current_steer, forward_state.steering_angle_in_radians)
            reverse_diff = difference_between_angles(current_steer, reverse_state.steering_angle_in_radians)

            chosen_state = forward_state if abs(forward_diff) <= abs(reverse_diff) else reverse_state

            if math.isinf(chosen_state.steering_angle_in_radians):
                # Zero-velocity case: hold the current measured steer angle, drive at 0.
                chosen_steer_angle = current_steer
                chosen_drive_velocity_mps = 0.0
            else:
                chosen_steer_angle = chosen_state.steering_angle_in_radians
                chosen_drive_velocity_mps = chosen_state.drive_velocity_in_meters_per_second

            steering_angle_values.append(chosen_steer_angle)

            # The IK gives the velocity in meters per second, i.e. the velocity of the wheel at the
            # contact point with the ground. But ROS wants to know the rotational velocity of the wheel
            wheel_radius = self.mobile_base["wheel_radius"]
            drive_velocity_values.append(chosen_drive_velocity_mps / wheel_radius)

        # Steer-settle gate: don't let the drives run until the steer modules have reached (or are
        # closing in on) their goal angle. Mirrors the 3-state gate in the C++ swerve controller
        # (main.cpp control_callback): 1 = settled -> drive, 0/-1 = still turning -> hold drive at 0.
        steer_max_vel = self.mobile_base["steer_max_vel"]
        steering_state = 1
        for i in range(len(steering_angle_values)):
            current_steer = self.last_drive_module_state[i].orientation_in_body_coordinates.z
            err = abs(steering_angle_values[i] - current_steer)
            if steering_state != -1:
                if err > (self.driving_status_threshold + steer_max_vel / self.cycle_time_in_hertz):
                    steering_state = -1
                elif err > self.driving_status_threshold:
                    steering_state = 0

        if steering_state != 1:
            drive_velocity_values = [0.0 for _ in drive_velocity_values]

        position_msg = Float64MultiArray()
        position_msg.data = steering_angle_values

        velocity_msg = Float64MultiArray()
        velocity_msg.data = drive_velocity_values

        # if there are some inf values in data publish last position instead (or update last position message)
        if (any(math.isinf(x) for x in position_msg.data)) and not (self.last_position_msg is None):
            position_msg = self.last_position_msg
        else:
            self.last_position_msg = position_msg

        # Publish the next steering angle and the next velocity sets. Note that
        # The velocity is published (very) shortly after the position data, which means
        # that the velocity could lag in very tight update loops.
        #self.get_logger().info(f'Publishing steering angle data: "{position_msg}"')
        self.drive_module_steering_angle_publisher.publish(position_msg)

        #self.get_logger().info(f'Publishing velocity angle data: "{velocity_msg}"')
        self.drive_module_velocity_publisher.publish(velocity_msg)

        self.last_control_update_send_at = self.last_recorded_time

    def write_log(self, text: str):
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)

    pub = SwerveController()

    rclpy.spin(pub)
    pub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
