import math
import numpy as np
import pytest
from tf_transformations import quaternion_from_euler, quaternion_multiply


class DummyParamListener:
    def __init__(self, params):
        self._params = params

    def get_params(self):
        return self._params


class DummyParams:
    def __init__(self, face_up=0, orientation=None):
        self.face_up = face_up
        self.orientation = orientation if orientation is not None else []


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
            raise ValueError(
                f"Invalid orientation parameter length {len(orientation_param)}. "
                "Expected 3 elements [roll, pitch, yaw] or 4 elements [x, y, z, w]."
            )
        else:
            face = self.params.face_up
            self.face = face if 1 <= face <= 6 else 6
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


def test_rpy_orientation():
    # Test specifying orientation as RPY [roll, pitch, yaw]
    # Pitch = pi/2 -> Face 2 should be up
    params = DummyParams(orientation=[0.0, math.pi / 2, 0.0])
    node = DummyNode(params)
    node.resolve_spawn_orientation()

    assert node.face == 2
    assert math.isclose(node.orientation_q[1], math.sin(math.pi / 4), abs_tol=1e-5)


def test_quaternion_orientation():
    # Test specifying orientation as Quaternion [x, y, z, w]
    # Roll = pi/2 -> Quaternion [sin(pi/4), 0, 0, cos(pi/4)] -> Face 3 should be up
    q = quaternion_from_euler(math.pi / 2, 0.0, 0.0)
    params = DummyParams(orientation=list(q))
    node = DummyNode(params)
    node.resolve_spawn_orientation()

    assert node.face == 3


def test_empty_orientation_fallback():
    # Test fallback to face_up when orientation is empty
    params = DummyParams(face_up=5, orientation=[])
    node = DummyNode(params)
    node.resolve_spawn_orientation()

    assert node.face == 5


def test_invalid_orientation_length():
    # Test error raised when orientation parameter has invalid length
    params = DummyParams(orientation=[1.0, 2.0])
    node = DummyNode(params)
    with pytest.raises(ValueError):
        node.resolve_spawn_orientation()


if __name__ == "__main__":
    test_rpy_orientation()
    test_quaternion_orientation()
    test_empty_orientation_fallback()
    test_invalid_orientation_length()
    print("ALL ORIENTATION TESTS PASSED SUCCESSFULLY!")

