import math
import os
import time
import random
import trimesh
import numpy as np

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Point, Vector3, Quaternion
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_msgs.msg import (
    PlanningScene,
    CollisionObject,
    ObjectColor,
)
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Int16, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from rcl_interfaces.srv import GetParameters
from std_srvs.srv import Trigger
from easy_motion_msgs.srv import DiceIdentification, AttachObject, DetachObject
from drims_dice_simulator.dice_spawner_parameters import dice_spawner_node


class DiceSpawner(Node):
    def __init__(self):
        super().__init__("dice_spawner_node")

        # Initialize parameter listener from generate_parameter_library
        self.param_listener = dice_spawner_node.ParamListener(self)
        self.params = self.param_listener.get_params()

        self.dice_name = "dice"
        self.dice_size = self.params.dice_size

        self.face_normals = {
            1: np.array([0, 0, -1]),
            2: np.array([-1, 0, 0]),
            3: np.array([0, 1, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([1, 0, 0]),
            6: np.array([0, 0, 1]),
        }

        # Initialize TF buffer and listener early so we can resolve base_link transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize internal node and executor early so spinning works
        self.internal_node = Node("dice_spawner_internal_node")
        self.internal_executor = MultiThreadedExecutor(num_threads=4)
        self.internal_executor.add_node(self.internal_node)

        self.get_group_name()

        if self.group_name == "manipulator":
            self.get_logger().info("Using 'world' as world frame.")
            self.world = "world"
        else:
            self.get_logger().info("Using 'base_footprint' as world frame.")
            self.world = "base_footprint"

        self.resolve_spawn_position()

        package_path = get_package_share_directory("drims_dice_simulator")
        self.dice_mesh_path = os.path.join(package_path, "urdf", "Dice.obj")

        self.dice_tf_spawned = False
        self.is_grasped = False

        self.service_callback_group = ReentrantCallbackGroup()
        self.get_scene_callback_group = ReentrantCallbackGroup()

        self.srv = self.create_service(
            DiceIdentification,
            "/dice_identification",
            self.get_dice_state_callback,
            callback_group=self.service_callback_group,
        )

        self.reset_srv = self.create_service(
            Trigger,
            "/reset_dice",
            self.reset_dice_callback,
            callback_group=self.service_callback_group,
        )

        self.dice_face_publisher_ = self.create_publisher(Int16, "/dice_face", 10)

        self.apply_scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        while not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /apply_planning_scene service...")

        self.apply_scene_client_sync = self.internal_node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=self.get_scene_callback_group,
        )
        while not self.apply_scene_client_sync.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /apply_planning_scene (sync) service...")

        self.get_scene_client = self.internal_node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
            callback_group=self.get_scene_callback_group,
        )
        while not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /get_planning_scene service...")

        self.add_client = self.create_client(AttachObject, "/attach_object")
        while not self.add_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /attach_object service...")

        self.detach_client = self.internal_node.create_client(
            DetachObject, "/detach_object", callback_group=self.get_scene_callback_group
        )
        while not self.detach_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /detach_object service...")

        # Precompute face transforms (offset and rotation) to optimize TF publishing loop
        self.face_transforms = {}
        for face_id, normal in self.face_normals.items():
            offset = (self.dice_size / 2.0) * normal
            q_face = self.get_quaternion_from_normal(normal)
            self.face_transforms[face_id] = (offset, q_face)

        # Initialize grasped state variables for dynamic TF publishing
        self.parent_frame = self.world
        self.current_pose = Pose()
        self.current_pose.orientation.w = 1.0

        # Marker publisher for the pips in RViz
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        self.marker_timer = self.create_timer(0.2, self.publish_pips_marker)

        # Dynamic transform publisher timer at 200Hz (every 0.005 seconds) for high-rate synchronization
        self.tf_timer = self.create_timer(0.005, self.publish_all_transforms)

        self.precompute_meshes()
        self.spawn_dice_with_mesh()
        self.dice_face_publisher_.publish(Int16(data=self.face))
        self.gravity_timer = self.create_timer(0.5, self.gravity_timer_callback)

    def get_group_name(self):
        # Retrieve 'group_name' from /motion_server_node

        self.group_name = None
        param_client = self.internal_node.create_client(
            GetParameters, "/motion_server_node/get_parameters"
        )
        while not param_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /motion_server_node/get_parameters service...")

        param_request = GetParameters.Request()
        param_request.names = ["move_group_name"]

        future = param_client.call_async(param_request)
        self.internal_executor.spin_until_future_complete(future, timeout_sec=5.0)

        if future.done() and future.result() is not None:
            values = future.result().values
            if values and values[0].string_value:
                self.group_name = values[0].string_value
                self.get_logger().info(f"Retrieved group_name: {self.group_name}")
            else:
                self.get_logger().warning("Parameter 'group_name' is empty or not set.")
                self.group_name = "default_group"
        else:
            raise RuntimeError(
                "Failed to get 'group_name' from /motion_server_node. Using default."
            )

    def publish_all_static_transforms(self):
        self.publish_all_transforms()

    def publish_all_transforms(self):
        transforms = []
        now = self.get_clock().now().to_msg()

        if self.is_grasped:
            parent_frame = self.parent_frame
            translation = Vector3(
                x=self.current_pose.position.x,
                y=self.current_pose.position.y,
                z=self.current_pose.position.z
            )
            rotation = self.current_pose.orientation
        else:
            parent_frame = self.world
            translation = Vector3(
                x=self.position.x, y=self.position.y, z=self.position.z
            )
            rotation = Quaternion(
                x=self.orientation_q[0],
                y=self.orientation_q[1],
                z=self.orientation_q[2],
                w=self.orientation_q[3]
            )

        tf_com = TransformStamped()
        tf_com.header.stamp = now
        tf_com.header.frame_id = parent_frame
        tf_com.child_frame_id = "dice_com_tf"
        tf_com.transform.translation = translation
        tf_com.transform.rotation = rotation
        transforms.append(tf_com)

        for face_id, (offset, q_face) in self.face_transforms.items():
            tf_face = TransformStamped()
            tf_face.header.stamp = now
            tf_face.header.frame_id = "dice_com_tf"
            tf_face.child_frame_id = f"face{face_id}_tf"
            tf_face.transform.translation.x = float(offset[0])
            tf_face.transform.translation.y = float(offset[1])
            tf_face.transform.translation.z = float(offset[2])
            tf_face.transform.rotation.x = q_face[0]
            tf_face.transform.rotation.y = q_face[1]
            tf_face.transform.rotation.z = q_face[2]
            tf_face.transform.rotation.w = q_face[3]
            transforms.append(tf_face)

        # Publish dice_tf (aligned with the upward face) only if spawned
        if self.dice_tf_spawned:
            tf_dice = TransformStamped()
            tf_dice.header.stamp = now
            tf_dice.header.frame_id = f"face{self.face}_tf"
            tf_dice.child_frame_id = "dice_tf"
            tf_dice.transform.translation.x = 0.0
            tf_dice.transform.translation.y = 0.0
            tf_dice.transform.translation.z = 0.0
            tf_dice.transform.rotation.x = 0.0
            tf_dice.transform.rotation.y = 0.0
            tf_dice.transform.rotation.z = 0.0
            tf_dice.transform.rotation.w = 1.0
            transforms.append(tf_dice)

        self.tf_broadcaster.sendTransform(transforms)

    def update_dice_tf_from_scene(self, result=None):
        try:
            if result is None:
                request = GetPlanningScene.Request()
                request.components.components = (
                    GetPlanningScene.Request().components.SCENE_SETTINGS
                    | GetPlanningScene.Request().components.WORLD_OBJECT_NAMES
                    | GetPlanningScene.Request().components.WORLD_OBJECT_GEOMETRY
                    | GetPlanningScene.Request().components.ROBOT_STATE_ATTACHED_OBJECTS
                )

                future = self.get_scene_client.call_async(request)
                self.internal_executor.spin_until_future_complete(future, timeout_sec=5.0)

                if not future.done():
                    self.get_logger().warning("Timeout while waiting for planning scene.")
                    return False

                result = future.result()

            # --- Check attachment state of dice ---
            dice_attached = False
            attached_link = None
            dice_touch_links = []

            for attached_obj in result.scene.robot_state.attached_collision_objects:
                if attached_obj.object.id == self.dice_name:
                    dice_attached = True
                    attached_link = attached_obj.link_name
                    dice_touch_links = attached_obj.touch_links

            self.is_grasped = dice_attached

            if dice_attached:
                # Dynamically construct the full gripper touch links list from ACM
                gripper_links = [attached_link]
                acm = result.scene.allowed_collision_matrix
                keywords = [
                    "finger", "knuckle", "tip", "pad", "hand", "gripper", "robotiq", "palm"
                ]
                for name in acm.entry_names:
                    if any(kw in name.lower() for kw in keywords):
                        if name not in gripper_links:
                            gripper_links.append(name)

                # Check if dice touch links need an update
                needs_dice_update = not all(link in dice_touch_links for link in gripper_links)

                if needs_dice_update:
                    self.get_logger().info("Coordinated Attachment: Updating dice grasp...")
                    # 1. Prepare dice attached object with updated touch links
                    attached_dice = None
                    for attached_obj in result.scene.robot_state.attached_collision_objects:
                        if attached_obj.object.id == self.dice_name:
                            import copy
                            attached_dice = copy.deepcopy(attached_obj)
                            attached_dice.object.operation = CollisionObject.ADD
                            attached_dice.touch_links = gripper_links
                            break

                    diff_scene = PlanningScene()
                    if attached_dice is not None:
                        diff_scene.robot_state.attached_collision_objects = [attached_dice]
                    diff_scene.is_diff = True

                    req = ApplyPlanningScene.Request(scene=diff_scene)
                    self.apply_scene_client.call_async(req)

            for obj in result.scene.world.collision_objects:
                if obj.id == self.dice_name:
                    self.publish_updated_dice_com_tf(obj.pose, obj.header.frame_id)
                    return True

            for attached_obj in result.scene.robot_state.attached_collision_objects:
                if attached_obj.object.id == self.dice_name:
                    self.publish_updated_dice_com_tf(
                        attached_obj.object.pose, attached_obj.object.header.frame_id
                    )
                    return True

            self.get_logger().warning(
                f"Dice object '{self.dice_name}' not found in planning scene."
            )
            return False

        except Exception as e:
            self.get_logger().error(f"Error in update_dice_tf_from_scene: {str(e)}")
            return False

    def publish_updated_dice_com_tf(self, pose: Pose, parent_frame: str):
        if parent_frame == self.world:
            self.position.x = pose.position.x
            self.position.y = pose.position.y
            self.position.z = pose.position.z
            self.orientation_q = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        else:
            self.parent_frame = parent_frame
            self.current_pose = pose
        self.publish_all_transforms()

    def precompute_meshes(self):
        mesh = trimesh.load(self.dice_mesh_path, force="mesh")

        # Build monolithic Mesh message for the body
        self.body_mesh = Mesh()
        self.body_mesh.triangles = [
            MeshTriangle(vertex_indices=tri.tolist()) for tri in mesh.faces
        ]
        for v in mesh.vertices:
            point = Point()
            point.x, point.y, point.z = v * self.dice_size
            self.body_mesh.vertices.append(point)

    def publish_pips_marker(self):
        marker_array = MarkerArray()

        S = self.dice_size
        D = self.params.pips_distance * S
        pip_diameter = self.params.pip_diameter * S
        h = 0.001  # 1 mm height of cylinder
        # We place the cylinder top 1.0 mm above the face surface
        # to ensure it is always rendered in front of the dice
        z_offset = -h / 2.0 + 0.0010

        face_layouts = {
            1: [(0.0, 0.0)],
            2: [(-D, -D), (D, D)],
            3: [(-D, -D), (0.0, 0.0), (D, D)],
            4: [(-D, -D), (-D, D), (D, -D), (D, D)],
            5: [(-D, -D), (-D, D), (D, -D), (D, D), (0.0, 0.0)],
            6: [(-D, -D), (-D, 0.0), (-D, D), (D, -D), (D, 0.0), (D, D)]
        }

        pip_idx = 0
        stamp_zero = rclpy.time.Time().to_msg()

        for face_id, pips in face_layouts.items():
            for (dx, dy) in pips:
                marker = Marker()
                marker.header.frame_id = f"face{face_id}_tf"
                marker.header.stamp = stamp_zero
                marker.ns = "dice_pips"
                marker.id = pip_idx
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD

                marker.pose.position.x = dx
                marker.pose.position.y = dy
                marker.pose.position.z = z_offset
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = pip_diameter
                marker.scale.y = pip_diameter
                marker.scale.z = h

                marker.color.r = 0.1
                marker.color.g = 0.1
                marker.color.b = 0.1
                marker.color.a = 1.0

                marker_array.markers.append(marker)
                pip_idx += 1

        self.marker_pub.publish(marker_array)

    def spawn_dice_with_mesh(self):
        pose = PoseStamped()
        pose.header.frame_id = "dice_com_tf"
        pose.pose.position.y = 0.0
        pose.pose.orientation.w = 1.0

        # Body collision object (yellow-ochre)
        obj = CollisionObject()
        obj.id = self.dice_name
        obj.header = pose.header
        obj.meshes = [self.body_mesh]
        obj.mesh_poses = [pose.pose]
        obj.operation = CollisionObject.ADD

        color = ObjectColor()
        color.id = self.dice_name
        color.color = ColorRGBA(r=0.85, g=0.65, b=0.25, a=1.0)

        # Publish the marker for the pips in RViz
        self.publish_pips_marker()

        scene = PlanningScene()
        scene.world.collision_objects = [obj]
        scene.object_colors = [color]
        scene.is_diff = True

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.apply_scene_client.call_async(req)
        future.add_done_callback(self.spawn_dice_result)

        q_str = f"[{self.orientation_q[0]:.4f}, {self.orientation_q[1]:.4f}, {self.orientation_q[2]:.4f}, {self.orientation_q[3]:.4f}]"
        self.get_logger().info(
            f"Spawned dice with:\n - face {self.face} up \n - position ["
            f"{self.position.x:.4f}, {self.position.y:.4f}, {self.position.z:.4f}] \n - orientation {q_str} \n - size {self.dice_size}"
        )

    def spawn_dice_result(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"AddObject response: {response}")
        except Exception as e:
            self.get_logger().error(f"Error while spawning dice: {str(e)}")

    def get_dice_state_callback(self, request, response):
        self.get_logger().info("Received dice identification request")
        try:
            success = self.update_dice_tf_from_scene()
            if not success:
                self.get_logger().warning("Failed to update dice transform from planning scene.")
                response.success = False
                return response

            time.sleep(0.5)

            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_face = None
            best_dot = -1.0
            best_tf = None

            for face_id in range(1, 7):
                tf = self.tf_buffer.lookup_transform(self.world, f"face{face_id}_tf", now)
                q = tf.transform.rotation
                q_np = np.array([q.x, q.y, q.z, q.w])
                z_local = np.array([0, 0, 1])
                z_world_face = self.rotate_vector(z_local, q_np)
                dot = np.dot(z_world_face, z_world)
                if dot > best_dot:
                    best_dot = dot
                    best_face = face_id
                    best_tf = tf

            self.face = best_face
            self.dice_face_publisher_.publish(Int16(data=best_face))

            pose = PoseStamped()
            pose.header = best_tf.header
            pose.pose.position = Point(
                x=best_tf.transform.translation.x,
                y=best_tf.transform.translation.y,
                z=best_tf.transform.translation.z,
            )
            pose.pose.orientation = best_tf.transform.rotation

            self.get_logger().info(f"Detected face up: {best_face}")
            self.get_logger().info(
                f"Position: x={pose.pose.position.x:.3f}, "
                f"y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}"
            )
            self.get_logger().info(
                f"Orientation (quaternion): x={pose.pose.orientation.x:.3f}, "
                f"y={pose.pose.orientation.y:.3f}, z={pose.pose.orientation.z:.3f}, "
                f"w={pose.pose.orientation.w:.3f}"
            )

            self.dice_tf_spawned = True
            self.publish_all_static_transforms()

            response.pose = pose
            response.face_number = best_face
            response.success = True
            return response

        except Exception as e:
            self.get_logger().error(f"get_dice_state_callback error: {e}")
            response.success = False
            return response

    def reset_dice_callback(self, request, response):
        self.get_logger().info("Received reset simulation request")
        try:
            # 1. Detach dice from gripper using easy_motion's /detach_object service
            detach_req = DetachObject.Request()
            detach_req.object_id = self.dice_name
            future_detach = self.detach_client.call_async(detach_req)
            self.internal_executor.spin_until_future_complete(future_detach, timeout_sec=5.0)

            # 2. Remove dice from world in the planning scene
            remove_dice = CollisionObject()
            remove_dice.id = self.dice_name
            remove_dice.operation = CollisionObject.REMOVE

            scene = PlanningScene()
            scene.world.collision_objects = [remove_dice]
            scene.is_diff = True

            req = ApplyPlanningScene.Request(scene=scene)
            future = self.apply_scene_client_sync.call_async(req)
            self.internal_executor.spin_until_future_complete(future, timeout_sec=5.0)

            # 2. Reset state variables
            self.dice_tf_spawned = False
            self.is_grasped = False
            self.parent_frame = self.world

            # 3. Resolve spawn position and orientation w.r.t base_link and transform to world
            self.resolve_spawn_position()

            # 5. Publish new static transforms (this moves dice_com_tf and face_tfs)
            self.publish_all_static_transforms()

            # 6. Re-spawn the collision objects in the planning scene
            self.spawn_dice_with_mesh()

            # 7. Identify the face up immediately to update/spawn the dice_tf
            time.sleep(0.5)
            self.update_dice_tf_from_scene()

            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_face = None
            best_dot = -1.0

            for face_id in range(1, 7):
                try:
                    tf = self.tf_buffer.lookup_transform(self.world, f"face{face_id}_tf", now)
                    q_tf = tf.transform.rotation
                    q_np = np.array([q_tf.x, q_tf.y, q_tf.z, q_tf.w])
                    z_local = np.array([0, 0, 1])
                    z_world_face = self.rotate_vector(z_local, q_np)
                    dot = np.dot(z_world_face, z_world)
                    if dot > best_dot:
                        best_dot = dot
                        best_face = face_id
                except Exception as ex:
                    self.get_logger().warning(
                        f"Could not lookup transform for face {face_id} during reset: {ex}"
                    )

            if best_face is not None:
                self.face = best_face
                self.dice_face_publisher_.publish(Int16(data=best_face))
                self.dice_tf_spawned = True
                self.publish_all_static_transforms()
                self.get_logger().info(f"Immediately identified face up after reset: {best_face}")

            response.success = True
            response.message = "Simulation reset successfully."
            return response

        except Exception as e:
            self.get_logger().error(f"Error in reset_dice_callback: {str(e)}")
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

    def get_orientation_for_face(self, face):
        face_to_rpy = {
            1: (0, math.pi, 0),
            2: (0, math.pi / 2, 0),
            3: (math.pi / 2, 0, 0),
            4: (-math.pi / 2, 0, 0),
            5: (0, -math.pi / 2, 0),
            6: (0, 0, 0),
        }
        rpy = face_to_rpy.get(face, (0, 0, 0))
        return quaternion_from_euler(*rpy)

    def resolve_spawn_position(self):
        # 1. Spawning position/bounds validation w.r.t base_link
        random_pos = self.params.random_position
        x_min = self.params.x_min
        x_max = self.params.x_max
        y_min = self.params.y_min
        y_max = self.params.y_max
        surface_height = self.params.surface_height
        pos_param = self.params.position

        if random_pos:
            x_spawn = random.uniform(x_min, x_max)
            y_spawn = random.uniform(y_min, y_max)
            self.get_logger().info(
                "Random position spawning enabled (w.r.t base_link). "
                f"Generated: x={x_spawn:.4f}, y={y_spawn:.4f}"
            )
        else:
            x_spawn = pos_param[0]
            y_spawn = pos_param[1]
            if not (x_min <= x_spawn <= x_max) or not (y_min <= y_spawn <= y_max):
                error_msg = (
                    f"Specified position [{x_spawn:.4f}, {y_spawn:.4f}] is outside the bounds: "
                    f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}] (w.r.t base_link)"
                )
                self.get_logger().error(error_msg)
                raise SystemExit(error_msg)

        clearance = (
            0.0005  # 0.5 mm padding to prevent false collision reports with the table surface
        )
        target_z = surface_height + (self.dice_size / 2.0) + clearance
        input_z = pos_param[2]
        if input_z < target_z - 1e-5:
            warn_msg = (
                f"Input Z coordinate {input_z:.4f} is below the target Z {target_z:.4f} "
                f"(surface_height {surface_height:.4f} + dice_size/2 {self.dice_size/2.0:.4f}) "
                "w.r.t base_link. Adjusting Z coordinate to place bottom of dice on surface."
            )
            self.get_logger().warning(warn_msg)
            z_spawn = target_z
        else:
            z_spawn = input_z

        # 2. Transform the position from base_link to self.world
        start_time = time.time()
        transform = None
        while transform is None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world, "base_link", rclpy.time.Time()
                )
            except Exception as e:
                if time.time() - start_time > 5.0:
                    self.get_logger().error(
                        f"Timeout waiting for transform from base_link to {self.world}: {e}"
                    )
                    raise SystemExit("Transform lookup failed")
                # Spin internal node/executor to allow TF updates to process
                rclpy.spin_once(self, timeout_sec=0.1)

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w

        rotated = self.rotate_vector([x_spawn, y_spawn, z_spawn], [qx, qy, qz, qw])
        self.position = Point(x=rotated[0] + tx, y=rotated[1] + ty, z=rotated[2] + tz)
        self.get_logger().info(
            f"Spawning position resolved in internal frame '{self.world}': "
            f"x={self.position.x:.4f}, y={self.position.y:.4f}, z={self.position.z:.4f}"
        )

        # Transform the surface height from base_link to self.world
        surface_pt_base = [0.0, 0.0, surface_height]
        rotated_surface = self.rotate_vector(surface_pt_base, [qx, qy, qz, qw])
        self.surface_height_world = rotated_surface[2] + tz

        # 3. Transform and resolve the spawn orientation
        self.resolve_spawn_orientation(qx, qy, qz, qw)

    def resolve_spawn_orientation(self, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.params = self.param_listener.get_params()
        orientation_param = (
            list(self.params.orientation)
            if hasattr(self.params, "orientation") and self.params.orientation
            else []
        )
        is_nonzero = any(v != 0 for v in orientation_param)

        if is_nonzero and len(orientation_param) == 3:
            r, p, y = orientation_param[0], orientation_param[1], orientation_param[2]
            q_base = quaternion_from_euler(r, p, y)
        elif is_nonzero and len(orientation_param) == 4:
            q_base = np.array(orientation_param, dtype=float)
            norm = np.linalg.norm(q_base)
            if norm > 1e-6:
                q_base = q_base / norm
            else:
                q_base = np.array([0.0, 0.0, 0.0, 1.0])
        elif is_nonzero:
            error_msg = (
                f"Invalid orientation parameter length {len(orientation_param)}. "
                "Expected 3 elements [roll, pitch, yaw] or 4 elements [x, y, z, w]."
            )
            self.get_logger().error(error_msg)
            raise SystemExit(error_msg)
        else:
            face = self.params.face_up
            self.face = face if 1 <= face <= 6 else random.randint(1, 6)
            q_base = self.get_orientation_for_face(self.face)

        q_world = quaternion_multiply([qx, qy, qz, qw], q_base)
        self.orientation_q = [
            float(q_world[0]),
            float(q_world[1]),
            float(q_world[2]),
            float(q_world[3]),
        ]

        best_face = None
        best_dot = -1.0
        for face_id, normal in self.face_normals.items():
            rotated_normal = self.rotate_vector(normal, self.orientation_q)
            dot = rotated_normal[2]
            if dot > best_dot:
                best_dot = dot
                best_face = face_id
        if best_face is not None:
            self.face = best_face

    def get_quaternion_from_normal(self, normal):
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, normal)
        c = np.dot(z_axis, normal)
        if np.linalg.norm(v) < 1e-6:
            return (0.0, 0.0, 0.0, 1.0) if c > 0 else quaternion_from_euler(math.pi, 0, 0)
        vx, vy, vz = v / np.linalg.norm(v)
        return (
            vx * math.sin(math.acos(c) / 2),
            vy * math.sin(math.acos(c) / 2),
            vz * math.sin(math.acos(c) / 2),
            math.cos(math.acos(c) / 2),
        )

    def get_aligned_pose(self, pose: Pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Local axes vectors in world
        x_axis = np.array(self.rotate_vector([1.0, 0.0, 0.0], q))
        y_axis = np.array(self.rotate_vector([0.0, 1.0, 0.0], q))
        z_axis = np.array(self.rotate_vector([0.0, 0.0, 1.0], q))

        dx = abs(x_axis[2])
        dy = abs(y_axis[2])
        dz = abs(z_axis[2])

        max_d = max(dx, dy, dz)
        if max_d == dx:
            closest_vec = x_axis
        elif max_d == dy:
            closest_vec = y_axis
        else:
            closest_vec = z_axis

        target = np.array([0.0, 0.0, 1.0 if closest_vec[2] > 0 else -1.0])

        cross = np.cross(closest_vec, target)
        dot = np.dot(closest_vec, target)

        if np.linalg.norm(cross) < 1e-6:
            q_align = [0.0, 0.0, 0.0, 1.0]
        else:
            axis = cross / np.linalg.norm(cross)
            angle = math.acos(np.clip(dot, -1.0, 1.0))
            s = math.sin(angle / 2.0)
            q_align = [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)]

        q_new = quaternion_multiply(q_align, q)

        aligned_pose = Pose()
        aligned_pose.position.x = pose.position.x
        aligned_pose.position.y = pose.position.y
        aligned_pose.position.z = self.surface_height_world + (self.dice_size / 2.0) + 0.0005
        aligned_pose.orientation.x = q_new[0]
        aligned_pose.orientation.y = q_new[1]
        aligned_pose.orientation.z = q_new[2]
        aligned_pose.orientation.w = q_new[3]
        return aligned_pose, max_d

    def gravity_timer_callback(self):
        try:
            request = GetPlanningScene.Request()
            request.components.components = (
                GetPlanningScene.Request().components.SCENE_SETTINGS
                | GetPlanningScene.Request().components.WORLD_OBJECT_NAMES
                | GetPlanningScene.Request().components.WORLD_OBJECT_GEOMETRY
                | GetPlanningScene.Request().components.ROBOT_STATE_ATTACHED_OBJECTS
            )

            future = self.get_scene_client.call_async(request)
            self.internal_executor.spin_until_future_complete(future, timeout_sec=1.0)

            if not future.done():
                return

            result = future.result()

            # Sync pips attachment state and update TFs
            self.update_dice_tf_from_scene(result)

            if self.is_grasped:
                return

            dice_obj = None
            for obj in result.scene.world.collision_objects:
                if obj.id == self.dice_name:
                    dice_obj = obj
                    break

            if dice_obj is None:
                return

            current_pose = dice_obj.pose
            parent_frame = dice_obj.header.frame_id

            if parent_frame != self.world:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.world, parent_frame, rclpy.time.Time()
                    )
                    tx = tf.transform.translation.x
                    ty = tf.transform.translation.y
                    tz = tf.transform.translation.z
                    qx = tf.transform.rotation.x
                    qy = tf.transform.rotation.y
                    qz = tf.transform.rotation.z
                    qw = tf.transform.rotation.w

                    p_rot = self.rotate_vector(
                        [
                            current_pose.position.x,
                            current_pose.position.y,
                            current_pose.position.z,
                        ],
                        [qx, qy, qz, qw],
                    )
                    current_pose.position.x = p_rot[0] + tx
                    current_pose.position.y = p_rot[1] + ty
                    current_pose.position.z = p_rot[2] + tz

                    q_new = quaternion_multiply(
                        [qx, qy, qz, qw],
                        [
                            current_pose.orientation.x,
                            current_pose.orientation.y,
                            current_pose.orientation.z,
                            current_pose.orientation.w,
                        ],
                    )
                    current_pose.orientation.x = q_new[0]
                    current_pose.orientation.y = q_new[1]
                    current_pose.orientation.z = q_new[2]
                    current_pose.orientation.w = q_new[3]
                    parent_frame = self.world
                except Exception as e:
                    self.get_logger().error(f"Failed to lookup transform in gravity timer: {e}")
                    return

            aligned_pose, max_d = self.get_aligned_pose(current_pose)

            target_z = self.surface_height_world + (self.dice_size / 2.0) + 0.0005
            z_diff = abs(current_pose.position.z - target_z)

            if max_d >= 0.999 and z_diff < 1e-4:
                return

            self.get_logger().info(
                f"Applying gravity & snapping: drop from Z={current_pose.position.z:.4f} "
                f"to surface Z={target_z:.4f}"
            )

            # Update the local variables of the node
            self.position.x = aligned_pose.position.x
            self.position.y = aligned_pose.position.y
            self.position.z = aligned_pose.position.z
            self.orientation_q = [
                aligned_pose.orientation.x,
                aligned_pose.orientation.y,
                aligned_pose.orientation.z,
                aligned_pose.orientation.w,
            ]

            best_face = None
            best_dot = -1.0
            for face_id, normal in self.face_normals.items():
                rotated_normal = self.rotate_vector(normal, self.orientation_q)
                dot = rotated_normal[2]
                if dot > best_dot:
                    best_dot = dot
                    best_face = face_id
            self.face = best_face
            self.dice_face_publisher_.publish(Int16(data=self.face))

            # Publish the updated static transforms using the proper hierarchy
            # (world -> dice_com_tf)
            self.publish_all_static_transforms()

            # Re-spawn the collision objects relative to the new dice_com_tf
            # in MoveIt planning scene
            self.spawn_dice_with_mesh()

        except Exception as e:
            self.get_logger().error(f"Error in gravity_timer_callback: {e}")

    def rotate_vector(self, v, q):
        v_q = (v[0], v[1], v[2], 0.0)
        q_conj = (-q[0], -q[1], -q[2], q[3])
        result = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
        return result[:3]


def main(args=None):
    rclpy.init(args=args)
    node = DiceSpawner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()
