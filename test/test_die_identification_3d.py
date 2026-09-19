import unittest
import numpy as np


class TestDieIdentification3DLogic(unittest.TestCase):
    """Test 3D die top and front face resolution logic without needing a running ROS 2 node."""

    def rotate_vector(self, v, q):
        v_q = (v[0], v[1], v[2], 0.0)
        q_conj = (-q[0], -q[1], -q[2], q[3])

        def q_mult(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return [x, y, z, w]

        res = q_mult(q_mult(q, v_q), q_conj)
        return np.array(res[:3])

    def get_top_and_front_faces(self, face_normals_world, die_position_world):
        z_world = np.array([0.0, 0.0, 1.0])

        # Top face: face with max dot product w.r.t +Z_world
        best_top_face = None
        best_top_dot = -float("inf")
        for face_id, normal_w in face_normals_world.items():
            dot = np.dot(normal_w, z_world)
            if dot > best_top_dot:
                best_top_dot = dot
                best_top_face = face_id

        # Front face: face pointing towards robot base origin (0, 0, 0)
        to_robot = -np.array(die_position_world)
        norm = np.linalg.norm(to_robot)
        to_robot_dir = to_robot / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])

        best_front_face = None
        best_front_dot = -float("inf")
        for face_id, normal_w in face_normals_world.items():
            if face_id == best_top_face:
                continue
            dot = np.dot(normal_w, to_robot_dir)
            if dot > best_front_dot:
                best_front_dot = dot
                best_front_face = face_id

        return best_top_face, best_front_face

    def test_identity_rotation_face_6_up(self):
        # Face 6 is +Z (up), Face 2 is -X, Face 5 is +X, Face 3 is +Y, Face 4 is -Y
        # If die is at (0.5, 0.0, 0.0), to_robot vector is (-0.5, 0.0, 0.0) -> pointing in -X direction
        # So Face 2 (-X) is pointing towards robot!
        normals_world = {
            1: np.array([0.0, 0.0, -1.0]),
            2: np.array([-1.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([0.0, -1.0, 0.0]),
            5: np.array([1.0, 0.0, 0.0]),
            6: np.array([0.0, 0.0, 1.0]),
        }
        top, front = self.get_top_and_front_faces(normals_world, [0.5, 0.0, 0.0])
        self.assertEqual(top, 6)
        self.assertEqual(front, 2)

    def test_die_at_y_axis(self):
        # If die is at (0.0, 0.5, 0.0), to_robot is (0.0, -0.5, 0.0) -> -Y direction
        # Face 4 is -Y, so Face 4 should be front face!
        normals_world = {
            1: np.array([0.0, 0.0, -1.0]),
            2: np.array([-1.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([0.0, -1.0, 0.0]),
            5: np.array([1.0, 0.0, 0.0]),
            6: np.array([0.0, 0.0, 1.0]),
        }
        top, front = self.get_top_and_front_faces(normals_world, [0.0, 0.5, 0.0])
        self.assertEqual(top, 6)
        self.assertEqual(front, 4)


if __name__ == "__main__":
    unittest.main()
