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
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Point, Vector3
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject, ObjectColor
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Int16, ColorRGBA
from rcl_interfaces.srv import GetParameters

from easy_motion_msgs.srv import DiceIdentification, AttachObject
from drims2_dice_simulator.dice_spawner_parameters import dice_spawner_node


class DiceSpawner(Node):
    def __init__(self):
        super().__init__('dice_spawner_node')

        # Initialize parameter listener from generate_parameter_library
        self.param_listener = dice_spawner_node.ParamListener(self)
        self.params = self.param_listener.get_params()

        face = self.params.face_up
        self.face = face if 1 <= face <= 6 else random.randint(1, 6)
        q = self.get_orientation_for_face(self.face)
        self.orientation_q = [q[0], q[1], q[2], q[3]]

        self.dice_name = "dice"
        self.dice_size = self.params.dice_size

        # Initialize TF buffer and listener early so we can resolve base_link transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Initialize internal node and executor early so spinning works
        self.internal_node = Node('dice_spawner_internal_node')
        self.internal_executor = MultiThreadedExecutor(num_threads=4)
        self.internal_executor.add_node(self.internal_node)

        self.get_group_name()

        if self.group_name == "manipulator":
            self.get_logger().info("Using 'world' as world frame.")
            self.world = "world"
        else:
            self.get_logger().info("Using 'base_footprint' as world frame.")
            self.world = "base_footprint"

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
            self.get_logger().info(f"Random position spawning enabled (w.r.t base_link). Generated: x={x_spawn:.4f}, y={y_spawn:.4f}")
        else:
            x_spawn = pos_param[0]
            y_spawn = pos_param[1]
            if not (x_min <= x_spawn <= x_max) or not (y_min <= y_spawn <= y_max):
                error_msg = f"Specified position [{x_spawn:.4f}, {y_spawn:.4f}] is outside the bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}] (w.r.t base_link)"
                self.get_logger().error(error_msg)
                raise SystemExit(error_msg)

        target_z = surface_height + (self.dice_size / 2.0)
        input_z = pos_param[2]
        if input_z < target_z - 1e-5:
            warn_msg = f"Input Z coordinate {input_z:.4f} is below the target Z {target_z:.4f} (surface_height {surface_height:.4f} + dice_size/2 {self.dice_size/2.0:.4f}) w.r.t base_link. Adjusting Z coordinate to place bottom of dice on surface."
            self.get_logger().warning(warn_msg)
            z_spawn = target_z
        else:
            z_spawn = input_z

        # 2. Transform the position from base_link to self.world
        start_time = time.time()
        transform = None
        while transform is None:
            try:
                transform = self.tf_buffer.lookup_transform(self.world, 'base_link', rclpy.time.Time())
            except Exception as e:
                if time.time() - start_time > 5.0:
                    self.get_logger().error(f"Timeout waiting for transform from base_link to {self.world}: {e}")
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
        self.position = Point(
            x=rotated[0] + tx,
            y=rotated[1] + ty,
            z=rotated[2] + tz
        )
        self.get_logger().info(f"Spawning position resolved in internal frame '{self.world}': x={self.position.x:.4f}, y={self.position.y:.4f}, z={self.position.z:.4f}")

        # Transform the surface height from base_link to self.world
        surface_pt_base = [0.0, 0.0, surface_height]
        rotated_surface = self.rotate_vector(surface_pt_base, [qx, qy, qz, qw])
        self.surface_height_world = rotated_surface[2] + tz

        package_path = get_package_share_directory('drims2_dice_simulator')
        self.dice_mesh_path = os.path.join(package_path, 'urdf', 'Dice.obj')

        self.service_callback_group = ReentrantCallbackGroup()
        self.get_scene_callback_group = ReentrantCallbackGroup()

        self.srv = self.create_service(
            DiceIdentification,
            '/dice_identification',
            self.get_dice_state_callback,
            callback_group=self.service_callback_group
        )

        self.dice_face_publisher_ = self.create_publisher(Int16, '/dice_face', 10)

        self.apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        self.get_scene_client = self.internal_node.create_client(
            GetPlanningScene,
            '/get_planning_scene',
            callback_group=self.get_scene_callback_group
        )
        while not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /get_planning_scene service...")

        self.add_client = self.create_client(AttachObject, '/attach_object')
        while not self.add_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /attach_object service...")

        self.face_normals = {
            1: np.array([0, 0, -1]),
            2: np.array([-1, 0, 0]),
            3: np.array([0, 1, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([1, 0, 0]),
            6: np.array([0, 0, 1]),
        }

        self.publish_all_static_transforms()
        time.sleep(1.0)
        self.spawn_dice_with_mesh()
        self.dice_face_publisher_.publish(Int16(data=self.face))
        self.gravity_timer = self.create_timer(0.5, self.gravity_timer_callback)

    def get_group_name(self):
        # Retrieve 'group_name' from /motion_server_node

        self.group_name = None
        param_client = self.internal_node.create_client(GetParameters, '/motion_server_node/get_parameters')
        while not param_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /motion_server_node/get_parameters service...")

        param_request = GetParameters.Request()
        param_request.names = ['move_group_name']

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
            raise RuntimeError("Failed to get 'group_name' from /motion_server_node. Using default.")

    def publish_all_static_transforms(self):
        transforms = []

        tf_base = TransformStamped()
        tf_base.header.stamp = self.get_clock().now().to_msg()
        tf_base.header.frame_id = self.world
        tf_base.child_frame_id = "dice_base_tf"
        tf_base.transform.translation = Vector3(
            x=self.position.x,
            y=self.position.y,
            z=self.position.z
        )
        tf_base.transform.rotation.x = 0.0
        tf_base.transform.rotation.y = 0.0
        tf_base.transform.rotation.z = 0.0
        tf_base.transform.rotation.w = 1.0
        transforms.append(tf_base)

        tf_rot = TransformStamped()
        tf_rot.header.stamp = self.get_clock().now().to_msg()
        tf_rot.header.frame_id = "dice_base_tf"
        tf_rot.child_frame_id = "dice_rotated_tf"
        tf_rot.transform.translation.x = 0.0
        tf_rot.transform.translation.y = 0.0
        tf_rot.transform.translation.z = 0.0
        tf_rot.transform.rotation.x = self.orientation_q[0]
        tf_rot.transform.rotation.y = self.orientation_q[1]
        tf_rot.transform.rotation.z = self.orientation_q[2]
        tf_rot.transform.rotation.w = self.orientation_q[3]
        transforms.append(tf_rot)

        for face_id, normal in self.face_normals.items():
            offset = (self.dice_size / 2.0) * normal
            q_face = self.get_quaternion_from_normal(normal)
            tf_face = TransformStamped()
            tf_face.header.stamp = self.get_clock().now().to_msg()
            tf_face.header.frame_id = "dice_rotated_tf"
            tf_face.child_frame_id = f"face{face_id}_tf"
            tf_face.transform.translation.x = float(offset[0])
            tf_face.transform.translation.y = float(offset[1])
            tf_face.transform.translation.z = float(offset[2])
            tf_face.transform.rotation.x = q_face[0]
            tf_face.transform.rotation.y = q_face[1]
            tf_face.transform.rotation.z = q_face[2]
            tf_face.transform.rotation.w = q_face[3]
            transforms.append(tf_face)

        tf_dice = TransformStamped()
        tf_dice.header.stamp = self.get_clock().now().to_msg()
        tf_dice.header.frame_id = "dice_rotated_tf"
        tf_dice.child_frame_id = "dice_tf"
        tf_dice.transform.translation.x = 0.0
        tf_dice.transform.translation.y = 0.0
        tf_dice.transform.translation.z = 0.0
        tf_dice.transform.rotation.x = 0.0
        tf_dice.transform.rotation.y = 0.0
        tf_dice.transform.rotation.z = 0.0
        tf_dice.transform.rotation.w = 1.0
        transforms.append(tf_dice)

        self.static_tf_broadcaster.sendTransform(transforms)

    def update_dice_tf_from_scene(self):
        try:
            request = GetPlanningScene.Request()
            request.components.components = (
                GetPlanningScene.Request().components.SCENE_SETTINGS |
                GetPlanningScene.Request().components.WORLD_OBJECT_NAMES |
                GetPlanningScene.Request().components.WORLD_OBJECT_GEOMETRY |
                GetPlanningScene.Request().components.ROBOT_STATE_ATTACHED_OBJECTS
            )

            future = self.get_scene_client.call_async(request)
            self.internal_executor.spin_until_future_complete(future, timeout_sec=5.0)

            if not future.done():
                self.get_logger().warning("Timeout while waiting for planning scene.")
                return False

            result = future.result()

            for obj in result.scene.world.collision_objects:
                if obj.id == self.dice_name:
                    self.publish_updated_dice_rotated_tf(obj.pose, obj.header.frame_id)
                    return True

            for attached_obj in result.scene.robot_state.attached_collision_objects:
                if attached_obj.object.id == self.dice_name:
                    self.publish_updated_dice_rotated_tf(attached_obj.object.pose, attached_obj.object.header.frame_id)
                    return True

            self.get_logger().warning(f"Dice object '{self.dice_name}' not found in planning scene.")
            return False

        except Exception as e:
            self.get_logger().error(f'Error in update_dice_tf_from_scene: {str(e)}')
            return False

    def publish_updated_dice_rotated_tf(self, pose: Pose, parent_frame: str):
        if parent_frame == self.world:
            self.position.x = pose.position.x
            self.position.y = pose.position.y
            self.position.z = pose.position.z
            self.orientation_q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            self.publish_all_static_transforms()
            return

        transforms = []

        tf_rot = TransformStamped()
        tf_rot.header.stamp = self.get_clock().now().to_msg()
        tf_rot.header.frame_id = parent_frame
        tf_rot.child_frame_id = "dice_rotated_tf"
        tf_rot.transform.translation = Vector3(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z
        )
        tf_rot.transform.rotation = pose.orientation
        transforms.append(tf_rot)

        for face_id, normal in self.face_normals.items():
            offset = (self.dice_size / 2.0) * normal
            q = self.get_quaternion_from_normal(normal)
            tf_face = TransformStamped()
            tf_face.header.stamp = self.get_clock().now().to_msg()
            tf_face.header.frame_id = "dice_rotated_tf"
            tf_face.child_frame_id = f"face{face_id}_tf"
            tf_face.transform.translation.x = float(offset[0])
            tf_face.transform.translation.y = float(offset[1])
            tf_face.transform.translation.z = float(offset[2])
            tf_face.transform.rotation.x = q[0]
            tf_face.transform.rotation.y = q[1]
            tf_face.transform.rotation.z = q[2]
            tf_face.transform.rotation.w = q[3]
            transforms.append(tf_face)

        self.static_tf_broadcaster.sendTransform(transforms)

    def spawn_dice_with_mesh(self):
        pose = PoseStamped()
        pose.header.frame_id = "dice_rotated_tf"
        pose.pose.position.y = 0.0
        pose.pose.orientation.w = 1.0

        mesh = trimesh.load(self.dice_mesh_path, force='mesh')

        pip_centers = [
            (0.0, 0.0),
            (-0.2, 0.2),
            (0.2, 0.2),
            (-0.2, -0.2),
            (0.2, -0.2),
            (-0.2, 0.0),
            (0.2, 0.0),
        ]

        def is_pip_vertex(v):
            for axis in range(3):
                for sign in [-1, 1]:
                    coord = v[axis]
                    if sign * coord > 0.35:
                        proj_axes = [i for i in range(3) if i != axis]
                        p = (v[proj_axes[0]], v[proj_axes[1]])
                        for c in pip_centers:
                            if np.linalg.norm(np.array(p) - np.array(c)) < 0.25:
                                if abs(coord) < 0.499:
                                    return True
            return False

        # Classify vertices
        is_pip = np.array([is_pip_vertex(v) for v in mesh.vertices])

        # Separate triangles into body and pips
        body_triangles = []
        pip_triangles = []
        for tri in mesh.faces:
            if sum(is_pip[tri]) >= 2:
                pip_triangles.append(MeshTriangle(vertex_indices=tri.tolist()))
            else:
                body_triangles.append(MeshTriangle(vertex_indices=tri.tolist()))

        # Build Mesh messages
        body_mesh = Mesh()
        body_mesh.triangles = body_triangles
        for v in mesh.vertices:
            point = Point()
            point.x, point.y, point.z = v * self.dice_size
            body_mesh.vertices.append(point)

        pip_mesh = Mesh()
        pip_mesh.triangles = pip_triangles
        for v in mesh.vertices:
            point = Point()
            point.x, point.y, point.z = v * self.dice_size
            pip_mesh.vertices.append(point)

        # Body collision object (yellow-ochre)
        obj = CollisionObject()
        obj.id = self.dice_name
        obj.header = pose.header
        obj.meshes = [body_mesh]
        obj.mesh_poses = [pose.pose]
        obj.operation = CollisionObject.ADD

        color = ObjectColor()
        color.id = self.dice_name
        color.color = ColorRGBA(r=0.85, g=0.65, b=0.25, a=1.0)

        # Pips collision object (black)
        obj_pips = CollisionObject()
        obj_pips.id = self.dice_name + "_pips"
        obj_pips.header = pose.header
        obj_pips.meshes = [pip_mesh]
        obj_pips.mesh_poses = [pose.pose]
        obj_pips.operation = CollisionObject.ADD

        color_pips = ObjectColor()
        color_pips.id = self.dice_name + "_pips"
        color_pips.color = ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0)

        scene = PlanningScene()
        scene.world.collision_objects = [obj, obj_pips]
        scene.object_colors = [color, color_pips]
        scene.is_diff = True

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.apply_scene_client.call_async(req)
        future.add_done_callback(self.spawn_dice_result)

        self.get_logger().info(f"Spawned dice with:\n - face {self.face} up \n - position [{self.position.x}, {self.position.y}, {self.position.z}] \n - size {self.dice_size}")

    def spawn_dice_result(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"AddObject response: {response}")
        except Exception as e:
            self.get_logger().error(f'Error while spawning dice: {str(e)}')

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
                tf = self.tf_buffer.lookup_transform(self.world, f'face{face_id}_tf', now)
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

            # Now lookup 'dice_rotated_tf' (center of the dice) relative to the world
            dice_tf_lookup = self.tf_buffer.lookup_transform(self.world, 'dice_rotated_tf', now)
            pose = PoseStamped()
            pose.header = dice_tf_lookup.header
            pose.pose.position = Point(
                x=dice_tf_lookup.transform.translation.x,
                y=dice_tf_lookup.transform.translation.y,
                z=dice_tf_lookup.transform.translation.z
            )
            pose.pose.orientation = dice_tf_lookup.transform.rotation

            self.orientation_q = [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w
            ]

            self.get_logger().info(f"Detected face up: {best_face}")
            self.get_logger().info(f"Position: x={pose.pose.position.x:.3f}, y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}")
            self.get_logger().info(f"Orientation (quaternion): x={pose.pose.orientation.x:.3f}, y={pose.pose.orientation.y:.3f}, z={pose.pose.orientation.z:.3f}, w={pose.pose.orientation.w:.3f}")

            dice_tf = TransformStamped()
            dice_tf.header.stamp = self.get_clock().now().to_msg()
            dice_tf.header.frame_id = "dice_rotated_tf"
            dice_tf.child_frame_id = "dice_tf"
            dice_tf.transform.translation.x = 0.0
            dice_tf.transform.translation.y = 0.0
            dice_tf.transform.translation.z = 0.0
            dice_tf.transform.rotation.x = 0.0
            dice_tf.transform.rotation.y = 0.0
            dice_tf.transform.rotation.z = 0.0
            dice_tf.transform.rotation.w = 1.0
            self.static_tf_broadcaster.sendTransform([dice_tf])

            response.pose = pose
            response.face_number = best_face
            response.success = True
            return response

        except Exception as e:
            self.get_logger().error(f"get_dice_state_callback error: {e}")
            response.success = False
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

    def get_quaternion_from_normal(self, normal):
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, normal)
        c = np.dot(z_axis, normal)
        if np.linalg.norm(v) < 1e-6:
            return (0.0, 0.0, 0.0, 1.0) if c > 0 else quaternion_from_euler(math.pi, 0, 0)
        s = math.sqrt((1 + c) * 2)
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
        aligned_pose.position.z = self.surface_height_world + (self.dice_size / 2.0)
        aligned_pose.orientation.x = q_new[0]
        aligned_pose.orientation.y = q_new[1]
        aligned_pose.orientation.z = q_new[2]
        aligned_pose.orientation.w = q_new[3]
        return aligned_pose, max_d

    def gravity_timer_callback(self):
        try:
            request = GetPlanningScene.Request()
            request.components.components = (
                GetPlanningScene.Request().components.SCENE_SETTINGS |
                GetPlanningScene.Request().components.WORLD_OBJECT_NAMES |
                GetPlanningScene.Request().components.WORLD_OBJECT_GEOMETRY |
                GetPlanningScene.Request().components.ROBOT_STATE_ATTACHED_OBJECTS
            )

            future = self.get_scene_client.call_async(request)
            self.internal_executor.spin_until_future_complete(future, timeout_sec=1.0)

            if not future.done():
                return

            result = future.result()

            is_grasped = False
            for attached_obj in result.scene.robot_state.attached_collision_objects:
                if attached_obj.object.id == self.dice_name:
                    is_grasped = True
                    break

            if is_grasped:
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
                    tf = self.tf_buffer.lookup_transform(self.world, parent_frame, rclpy.time.Time())
                    tx = tf.transform.translation.x
                    ty = tf.transform.translation.y
                    tz = tf.transform.translation.z
                    qx = tf.transform.rotation.x
                    qy = tf.transform.rotation.y
                    qz = tf.transform.rotation.z
                    qw = tf.transform.rotation.w
                    
                    p_rot = self.rotate_vector([current_pose.position.x, current_pose.position.y, current_pose.position.z], [qx, qy, qz, qw])
                    current_pose.position.x = p_rot[0] + tx
                    current_pose.position.y = p_rot[1] + ty
                    current_pose.position.z = p_rot[2] + tz
                    
                    q_new = quaternion_multiply([qx, qy, qz, qw], [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
                    current_pose.orientation.x = q_new[0]
                    current_pose.orientation.y = q_new[1]
                    current_pose.orientation.z = q_new[2]
                    current_pose.orientation.w = q_new[3]
                    parent_frame = self.world
                except Exception as e:
                    self.get_logger().error(f"Failed to lookup transform in gravity timer: {e}")
                    return

            aligned_pose, max_d = self.get_aligned_pose(current_pose)
            
            target_z = self.surface_height_world + (self.dice_size / 2.0)
            z_diff = abs(current_pose.position.z - target_z)
            
            if max_d >= 0.999 and z_diff < 1e-4:
                return

            self.get_logger().info(f"Applying gravity & snapping: drop from Z={current_pose.position.z:.4f} to surface Z={target_z:.4f}")

            # Update the local variables of the node
            self.position.x = aligned_pose.position.x
            self.position.y = aligned_pose.position.y
            self.position.z = aligned_pose.position.z
            self.orientation_q = [
                aligned_pose.orientation.x,
                aligned_pose.orientation.y,
                aligned_pose.orientation.z,
                aligned_pose.orientation.w
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

            # Publish the updated static transforms using the proper hierarchy (world -> dice_base_tf -> dice_rotated_tf)
            self.publish_all_static_transforms()

            # Re-spawn the collision objects relative to the new dice_rotated_tf in MoveIt planning scene
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
