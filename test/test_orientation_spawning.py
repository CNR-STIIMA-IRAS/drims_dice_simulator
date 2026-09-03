import math
import numpy as np
from tf_transformations import quaternion_from_euler, quaternion_multiply


class DummyParamListener:
    def __init__(self, params):
        self._params = params

    def get_params(self):
        return self._params


class DummyParams:
    def __init__(self, face_up=0, yaw=0.0):
        self.face_up = face_up
        self.yaw = yaw


class DummyNode:
    def __init__(self, params):
        self.param_listener = DummyParamListener(params)
        self.params = params
        self.face_normals = {
            1: np.array([0, 0, -1]),
            2: np.array([-1, 0, 0]),
            3: np.array([0, 1, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([1, 0, 0]),
            6: np.array([0, 0, 1]),
        }
        self.face = 0
        self.orientation_q = [0.0, 0.0, 0.0, 1.0]

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

    def rotate_vector(self, v, q):
        v_q = (v[0], v[1], v[2], 0.0)
        q_conj = (-q[0], -q[1], -q[2], q[3])
        result = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
        return result[:3]

    def resolve_spawn_orientation(self, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.params = self.param_listener.get_params()

        face = self.params.face_up
        self.face = face if 1 <= face <= 6 else 6
        q_face = self.get_orientation_for_face(self.face)

        yaw = float(getattr(self.params, "yaw", 0.0))
        q_yaw = quaternion_from_euler(0.0, 0.0, yaw)

        q_base = quaternion_multiply(q_yaw, q_face)
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


def test_face_up_with_zero_yaw():
    # Face 2 should be up
    params = DummyParams(face_up=2, yaw=0.0)
    node = DummyNode(params)
    node.resolve_spawn_orientation()

    assert node.face == 2


def test_face_up_with_in_plane_yaw():
    # Face 3 should still be up after rotating in-plane by 90 degrees (pi/2)
    params = DummyParams(face_up=3, yaw=math.pi / 2)
    node = DummyNode(params)
    node.resolve_spawn_orientation()

    assert node.face == 3


if __name__ == "__main__":
    test_face_up_with_zero_yaw()
    test_face_up_with_in_plane_yaw()
    print("ALL YAW ORIENTATION TESTS PASSED SUCCESSFULLY!")
