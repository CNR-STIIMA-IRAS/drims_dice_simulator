from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    param_names = [
        "face_up",
        "dice_size",
        "position",
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
                    node_params[name] = [float(x.strip()) for x in cleaned.split(",")]
                else:
                    node_params[name] = float(val_str)
        except Exception:
            # If a launch configuration is not declared or fails substitution, ignore it
            pass

    node = Node(
        package="drims_dice_simulator",
        executable="dice_spawner",
        output="screen",
        parameters=[node_params],
    )
    return [node]


def generate_launch_description():
    return LaunchDescription(
        [
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
