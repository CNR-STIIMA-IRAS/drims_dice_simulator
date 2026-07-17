import rclpy
from rclpy.node import Node
import random
import math
import numpy as np
import time
from geometry_msgs.msg import Pose, Quaternion
from tf2_ros import Buffer, TransformListener
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from easy_motion_msgs.srv import DiceIdentification
from std_msgs.msg import ColorRGBA
from moveit_msgs.msg import ObjectColor


class RealignmentIntegrationTester(Node):
    def __init__(self):
        super().__init__("realignment_tester_node")
        self.declare_parameter("x_min", 0.4)
        self.declare_parameter("x_max", 0.8)
        self.declare_parameter("y_min", -0.2)
        self.declare_parameter("y_max", 0.4)

        self.x_min = self.get_parameter("x_min").get_parameter_value().double_value
        self.x_max = self.get_parameter("x_max").get_parameter_value().double_value
        self.y_min = self.get_parameter("y_min").get_parameter_value().double_value
        self.y_max = self.get_parameter("y_max").get_parameter_value().double_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.apply_scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        while not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /apply_planning_scene service...")

        self.get_scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        while not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /get_planning_scene service...")

    def generate_random_quaternion(self):
        u = np.random.rand(3)
        return Quaternion(
            x=math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
            y=math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
            z=math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
            w=math.sqrt(u[0]) * math.cos(2 * math.pi * u[2]),
        )

    def rotate_vector(self, v, q):
        v_q = (v[0], v[1], v[2], 0.0)
        q_conj = (-q[0], -q[1], -q[2], q[3])

        def q_m(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return [x, y, z, w]

        res = q_m(q_m(q, v_q), q_conj)
        return res[:3]

    def run_teleport_test(self):
        # 1. Query the planning scene to find the current dice collision object
        self.get_logger().info("Querying current planning scene to find the dice...")
        req = GetPlanningScene.Request()
        req.components.components = (
            GetPlanningScene.Request().components.SCENE_SETTINGS
            | GetPlanningScene.Request().components.WORLD_OBJECT_NAMES
            | GetPlanningScene.Request().components.WORLD_OBJECT_GEOMETRY
        )

        future = self.get_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done() or future.result() is None:
            self.get_logger().error("Failed to query planning scene.")
            return False

        scene = future.result().scene

        body_obj = None
        for obj in scene.world.collision_objects:
            if obj.id == "dice":
                body_obj = obj

        if body_obj is None:
            self.get_logger().error(
                "Could not find 'dice' in MoveIt planning scene. "
                "Please launch spawn_dice.launch.py first!"
            )
            return False

        # 2. Generate random starting pose in the air
        # We assume target world frame is the frame of body_obj (usually 'world' or
        # 'base_footprint'). We query the transform from base_link to world_frame to
        # express the random spawn position w.r.t base_link bounds.
        world_frame = body_obj.header.frame_id

        tx, ty, tz = 0.0, 0.0, 0.0
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        try:
            transform = None
            start_time = time.time()
            while transform is None:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        world_frame, "base_link", rclpy.time.Time()
                    )
                except Exception as e:
                    if time.time() - start_time > 5.0:
                        self.get_logger().error(
                            f"Timeout waiting for transform from base_link to {world_frame}: {e}"
                        )
                        return False
                    rclpy.spin_once(self, timeout_sec=0.1)

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
        except Exception as e:
            self.get_logger().error(
                f"Error resolving transform from base_link to {world_frame}: {e}"
            )
            return False

        x_base = random.uniform(self.x_min, self.x_max)
        y_base = random.uniform(self.y_min, self.y_max)
        # 0.1 meters base link Z height serves as a clean starting height offset
        pos_base = [x_base, y_base, 0.1]
        pos_rot = self.rotate_vector(pos_base, [qx, qy, qz, qw])

        start_pose = Pose()
        start_pose.position.x = pos_rot[0] + tx
        start_pose.position.y = pos_rot[1] + ty
        # Spawn the dice high in the air (0.15 - 0.5 meters above surface)
        start_pose.position.z = (pos_rot[2] + tz) + random.uniform(0.15, 0.5)
        start_pose.orientation = self.generate_random_quaternion()

        self.get_logger().info(
            f"Teleporting dice to random pose: frame={world_frame}, pos=["
            f"{start_pose.position.x:.3f}, {start_pose.position.y:.3f}, "
            f"{start_pose.position.z:.3f}]"
        )

        # 3. Create update collision objects at the random pose (ADD replaces them)
        body_update = CollisionObject()
        body_update.id = "dice"
        body_update.header.frame_id = world_frame
        body_update.meshes = body_obj.meshes
        body_update.mesh_poses = [start_pose]
        body_update.operation = CollisionObject.ADD

        # Setup colors
        color = ObjectColor()
        color.id = "dice"
        color.color = ColorRGBA(r=0.85, g=0.65, b=0.25, a=1.0)

        # Apply diff
        diff_scene = PlanningScene()
        diff_scene.world.collision_objects = [body_update]
        diff_scene.object_colors = [color]
        diff_scene.is_diff = True

        apply_req = ApplyPlanningScene.Request(scene=diff_scene)
        apply_future = self.apply_scene_client.call_async(apply_req)
        rclpy.spin_until_future_complete(self, apply_future, timeout_sec=5.0)

        self.get_logger().info("Teleported successfully! Waiting for gravity snapping timer...")

        # 4. Wait for 2 seconds to let gravity node trigger, drop and snap the dice
        time.sleep(2.0)

        # 5. Query planning scene again to check alignment
        self.get_logger().info("Verifying final snapped pose...")
        future2 = self.get_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future2, timeout_sec=5.0)

        if not future2.done() or future2.result() is None:
            self.get_logger().error("Failed to query final planning scene.")
            return False

        final_scene = future2.result().scene
        final_body = None
        for obj in final_scene.world.collision_objects:
            if obj.id == "dice":
                final_body = obj
                break

        if final_body is None:
            self.get_logger().error("Dice object vanished from scene!")
            return False

        final_pose = final_body.mesh_poses[0]
        self.get_logger().info(
            f"Final aligned pose: pos=[{final_pose.position.x:.3f}, "
            f"{final_pose.position.y:.3f}, {final_pose.position.z:.3f}]"
        )

        # Since it is spawned relative to world_frame, get the surface_height_world
        # from dice_spawner if possible, or we just check if the height is stable
        # (less than the start height)
        if final_pose.position.z >= start_pose.position.z:
            self.get_logger().error("FAIL: Dice did not drop!")
            return False

        # Check orientation alignment
        q = [
            final_pose.orientation.x,
            final_pose.orientation.y,
            final_pose.orientation.z,
            final_pose.orientation.w,
        ]

        x_axis = np.array(self.rotate_vector([1.0, 0.0, 0.0], q))
        y_axis = np.array(self.rotate_vector([0.0, 1.0, 0.0], q))
        z_axis = np.array(self.rotate_vector([0.0, 0.0, 1.0], q))

        dx = abs(x_axis[2])
        dy = abs(y_axis[2])
        dz = abs(z_axis[2])

        max_d = max(dx, dy, dz)
        if not math.isclose(max_d, 1.0, abs_tol=1.0e-3):
            self.get_logger().error(
                f"FAIL: Orientation not aligned! Max vertical axis projection = {max_d:.6f}"
            )
            return False

        self.get_logger().info("SUCCESS: Dice dropped and snapped perfectly flat on the surface!")

        # 6. Call the /dice_identification service to trigger the spawning of dice_tf (Z-up)
        self.get_logger().info("Calling /dice_identification service...")
        identify_client = self.create_client(DiceIdentification, "/dice_identification")
        if not identify_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("FAIL: DiceIdentification service not available!")
            return False

        req_ident = DiceIdentification.Request()
        future_ident = identify_client.call_async(req_ident)
        rclpy.spin_until_future_complete(self, future_ident, timeout_sec=5.0)

        if not future_ident.done() or future_ident.result() is None:
            self.get_logger().error("FAIL: DiceIdentification service call failed!")
            return False

        if not future_ident.result().success:
            self.get_logger().error("FAIL: DiceIdentification service returned success=False!")
            return False

        self.get_logger().info(
            f"Service returned face: {future_ident.result().face_number}"
        )

        # 7. Verify that dice_tf is now spawned, Z-up and matches the face orientation
        # Wait a short moment to let TF process
        time.sleep(0.5)
        try:
            tf_dice = self.tf_buffer.lookup_transform(
                world_frame, "dice_tf", rclpy.time.Time()
            )
            q_dt = [
                tf_dice.transform.rotation.x,
                tf_dice.transform.rotation.y,
                tf_dice.transform.rotation.z,
                tf_dice.transform.rotation.w,
            ]
            z_axis_dt = np.array(self.rotate_vector([0.0, 0.0, 1.0], q_dt))
            # The Z-axis of dice_tf must point straight up
            if not math.isclose(z_axis_dt[2], 1.0, abs_tol=1e-3):
                self.get_logger().error(
                    f"FAIL: dice_tf Z-axis not straight up! Z = {z_axis_dt[2]:.6f}"
                )
                return False
            self.get_logger().info("SUCCESS: dice_tf is spawned and pointing straight Z-up!")
        except Exception as e:
            self.get_logger().error(f"FAIL: Could not lookup dice_tf: {e}")
            return False

        return True


def main(args=None):
    rclpy.init(args=args)
    tester = RealignmentIntegrationTester()

    tester.get_logger().info("Starting Realignment Integration Test...")
    success = tester.run_teleport_test()

    tester.destroy_node()
    rclpy.shutdown()

    if success:
        print("\n=== INTEGRATION TEST PASSED ===\n")
    else:
        print("\n=== INTEGRATION TEST FAILED ===\n")


if __name__ == "__main__":
    main()
