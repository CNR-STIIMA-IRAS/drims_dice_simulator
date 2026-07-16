from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "face_up",
                default_value="0",
                description="Face number facing upward (1–6, 0 = random)",
            ),
            DeclareLaunchArgument(
                "dice_size",
                default_value="0.027",
                description="Length of the dice edge (in meters)",
            ),
            DeclareLaunchArgument(
                "position",
                default_value="[0.6, 0.2, 0.0]",
                description="Initial dice position [x, y, z]",
            ),
            DeclareLaunchArgument(
                "random_position",
                default_value="false",
                description="Whether to spawn at a random position within bounds (true/false)",
            ),
            DeclareLaunchArgument(
                "x_min", default_value="0.40", description="Minimum X coordinate bounds"
            ),
            DeclareLaunchArgument(
                "x_max", default_value="0.80", description="Maximum X coordinate bounds"
            ),
            DeclareLaunchArgument(
                "y_min", default_value="-0.20", description="Minimum Y coordinate bounds"
            ),
            DeclareLaunchArgument(
                "y_max", default_value="0.40", description="Maximum Y coordinate bounds"
            ),
            DeclareLaunchArgument(
                "surface_height", default_value="0.0", description="Surface height (in meters)"
            ),
            Node(
                package="drims2_dice_simulator",
                executable="dice_spawner",
                output="screen",
                parameters=[
                    {
                        "face_up": LaunchConfiguration("face_up"),
                        "dice_size": LaunchConfiguration("dice_size"),
                        "position": LaunchConfiguration("position"),
                        "random_position": LaunchConfiguration("random_position"),
                        "x_min": LaunchConfiguration("x_min"),
                        "x_max": LaunchConfiguration("x_max"),
                        "y_min": LaunchConfiguration("y_min"),
                        "y_max": LaunchConfiguration("y_max"),
                        "surface_height": LaunchConfiguration("surface_height"),
                    }
                ],
            ),
        ]
    )
