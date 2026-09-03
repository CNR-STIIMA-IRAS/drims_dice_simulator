import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    params_file = context.perform_substitution(LaunchConfiguration("params_file"))

    param_names = [
        "face_up",
        "dice_size",
        "position",
        "yaw",
        "random_position",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "surface_height",
        "pips_distance",
        "pip_diameter",
    ]

    node_params = {}
    for name in param_names:
        try:
            val_str = context.perform_substitution(LaunchConfiguration(name))
            if val_str != "__default__":
                if name == "face_up":
                    node_params[name] = int(val_str)
                elif name == "random_position":
                    node_params[name] = val_str.lower() in ["true", "1", "yes"]
                elif name == "position":
                    cleaned = val_str.strip("[]")
                    if cleaned:
                        node_params[name] = [float(x.strip()) for x in cleaned.split(",")]
                else:
                    node_params[name] = float(val_str)
        except Exception:
            # If a launch configuration is not declared or fails substitution, ignore it
            pass

    params_list = []
    if os.path.exists(params_file):
        params_list.append(params_file)
    if node_params:
        params_list.append(node_params)

    node = Node(
        package="drims_dice_simulator",
        executable="dice_spawner",
        output="screen",
        parameters=params_list,
    )
    return [node]


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory("drims_dice_simulator"),
        "config",
        "dice_spawner_parameters.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Full path to the ROS 2 parameters YAML file to load",
            ),
            DeclareLaunchArgument(
                "face_up",
                default_value="__default__",
                description="Face number facing upward (1–6, 0 = random)",
            ),
            DeclareLaunchArgument(
                "dice_size",
                default_value="__default__",
                description="Length of the dice edge (in meters)",
            ),
            DeclareLaunchArgument(
                "position",
                default_value="__default__",
                description="Initial dice position [x, y, z]",
            ),
            DeclareLaunchArgument(
                "yaw",
                default_value="__default__",
                description="In-plane rotation angle around Z-axis (in radians)",
            ),
            DeclareLaunchArgument(
                "random_position",
                default_value="__default__",
                description="Whether to spawn at a random position within bounds",
            ),
            DeclareLaunchArgument(
                "x_min", default_value="__default__", description="Minimum X bounds"
            ),
            DeclareLaunchArgument(
                "x_max", default_value="__default__", description="Maximum X bounds"
            ),
            DeclareLaunchArgument(
                "y_min", default_value="__default__", description="Minimum Y bounds"
            ),
            DeclareLaunchArgument(
                "y_max", default_value="__default__", description="Maximum Y bounds"
            ),
            DeclareLaunchArgument(
                "surface_height", default_value="__default__", description="Surface height"
            ),
            DeclareLaunchArgument(
                "pips_distance", default_value="__default__", description="Pips distance bounds"
            ),
            DeclareLaunchArgument(
                "pip_diameter", default_value="__default__", description="Pips diameter bounds"
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
