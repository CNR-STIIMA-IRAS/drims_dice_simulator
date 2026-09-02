from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    param_names = [
        "face_up",
        "dice_size",
        "position",
        "orientation",
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
            if name == "face_up":
                node_params[name] = int(val_str)
            elif name == "random_position":
                node_params[name] = val_str.lower() in ["true", "1", "yes"]
            elif name in ["position", "orientation"]:
                cleaned = val_str.strip("[]")
                if cleaned:
                    node_params[name] = [float(x.strip()) for x in cleaned.split(",")]
                else:
                    node_params[name] = [0.0, 0.0, 0.0, 0.0]
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
                "orientation",
                default_value="[0.0, 0.0, 0.0, 0.0]",
                description="Initial dice orientation [roll, pitch, yaw] or [x, y, z, w]",
            ),
            DeclareLaunchArgument(
                "random_position",
                default_value="false",
                description="Whether to spawn at a random position within bounds",
            ),
            DeclareLaunchArgument(
                "x_min", default_value="0.40", description="Minimum X bounds"
            ),
            DeclareLaunchArgument(
                "x_max", default_value="0.80", description="Maximum X bounds"
            ),
            DeclareLaunchArgument(
                "y_min", default_value="-0.20", description="Minimum Y bounds"
            ),
            DeclareLaunchArgument(
                "y_max", default_value="0.40", description="Maximum Y bounds"
            ),
            DeclareLaunchArgument(
                "surface_height", default_value="0.0", description="Surface height"
            ),
            DeclareLaunchArgument(
                "pips_distance", default_value="0.26", description="Pips distance bounds"
            ),
            DeclareLaunchArgument(
                "pip_diameter", default_value="0.21", description="Pips diameter bounds"
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
