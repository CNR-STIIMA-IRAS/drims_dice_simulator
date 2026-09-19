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
        "selected_cell",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "surface_height",
        "pips_distance",
        "pip_diameter",
        "base_link",
        "single_face_detection",
    ]

    node_params = {}
    position_specified = False
    random_pos_specified = False

    for name in param_names:
        try:
            val_str = context.perform_substitution(LaunchConfiguration(name))
            if val_str != "":
                if name in ["face_up", "selected_cell"]:
                    node_params[name] = int(float(val_str))
                elif name == "random_position":
                    node_params[name] = val_str.lower() in ["true", "1", "yes"]
                    if "random_position" in context.launch_configurations:
                        random_pos_specified = True
                elif name == "position":
                    cleaned = val_str.strip("[]")
                    if cleaned:
                        node_params[name] = [float(x.strip()) for x in cleaned.split(",")]
                        position_specified = True
                elif name == "base_link":
                    node_params[name] = str(val_str)
                elif name == "single_face_detection":
                    node_params[name] = val_str.lower() in ["true", "1", "yes"]
                else:
                    node_params[name] = float(val_str)
        except Exception:
            # If a launch configuration is not declared or fails substitution, ignore it
            pass

    if position_specified and not random_pos_specified:
        node_params["random_position"] = False

    node_params["use_sim_time"] = True

    params_list = []
    if os.path.exists(params_file):
        params_list.append(params_file)
    if node_params:
        params_list.append(node_params)

    node = Node(
        package="drims_dice_simulator",
        executable="dice_spawner",
        name="dice_spawner_node",
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
                default_value="3",
                description="Face number facing upward (1–6, 0 = random)",
            ),
            DeclareLaunchArgument(
                "dice_size",
                default_value="0.03",
                description="Length of the dice edge (in meters)",
            ),
            DeclareLaunchArgument(
                "position",
                default_value="",
                description="Initial dice position [x, y, z] (overrides random_position if specified)",
            ),
            DeclareLaunchArgument(
                "yaw",
                default_value="0.5",
                description="In-plane rotation angle around Z-axis (in radians)",
            ),
            DeclareLaunchArgument(
                "random_position",
                default_value="",
                description="Whether to spawn at a random position within bounds",
            ),
            DeclareLaunchArgument(
                "selected_cell",
                default_value="4",
                description="Selected cell preset (1–5, 5 = TIAGo cell) to use for x/y bounds and surface height",
            ),
            DeclareLaunchArgument(
                "x_min", default_value="-0.30", description="Minimum X bounds"
            ),
            DeclareLaunchArgument(
                "x_max", default_value="0.20", description="Maximum X bounds"
            ),
            DeclareLaunchArgument(
                "y_min", default_value="0.50", description="Minimum Y bounds"
            ),
            DeclareLaunchArgument(
                "y_max", default_value="0.85", description="Maximum Y bounds"
            ),
            DeclareLaunchArgument(
                "surface_height", default_value="-0.02", description="Surface height"
            ),
            DeclareLaunchArgument(
                "pips_distance", default_value="0.26", description="Pips distance bounds"
            ),
            DeclareLaunchArgument(
                "pip_diameter", default_value="0.21", description="Pips diameter bounds"
            ),
            DeclareLaunchArgument(
                "base_link",
                default_value="",
                description="Base link frame name",
            ),
            DeclareLaunchArgument(
                "single_face_detection",
                default_value="",
                description="If true, DieIdentification3D only reports the top face; front_face/front_face_pips are returned as unknown, simulating a detector that only captures the top face",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
