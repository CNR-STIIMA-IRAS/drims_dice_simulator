import numpy as np
import math

# Implementation of the realignment logic to be tested

face_normals = {
    1: np.array([0.0, 0.0, -1.0]),
    2: np.array([-1.0, 0.0, 0.0]),
    3: np.array([0.0, 1.0, 0.0]),
    4: np.array([0.0, -1.0, 0.0]),
    5: np.array([1.0, 0.0, 0.0]),
    6: np.array([0.0, 0.0, 1.0]),
}


def rotate_vector(v, q):
    # Quaternion rotation: q * v_q * q_conj
    v_q = (v[0], v[1], v[2], 0.0)
    q_conj = (-q[0], -q[1], -q[2], q[3])
    # Quaternion multiplication helper

    def q_mult(q1, q2):
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [x, y, z, w]

    result = q_mult(q_mult(q, v_q), q_conj)
    return result[:3]


def get_aligned_pose(q_in, pos_in, surface_height_world, dice_size):
    # Local axes vectors in world
    x_axis = np.array(rotate_vector([1.0, 0.0, 0.0], q_in))
    y_axis = np.array(rotate_vector([0.0, 1.0, 0.0], q_in))
    z_axis = np.array(rotate_vector([0.0, 0.0, 1.0], q_in))

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

    # q_new = q_align * q_in
    def q_mult(q1, q2):
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [x, y, z, w]

    q_new = q_mult(q_align, q_in)

    # Normalize q_new to be safe
    q_new = np.array(q_new) / np.linalg.norm(q_new)

    z_out = surface_height_world + (dice_size / 2.0) + 0.0005
    pos_out = [pos_in[0], pos_in[1], z_out]
    return q_new, pos_out, max_d


def generate_random_quaternion():
    # Uniform random quaternion generation (Shoemake algorithm)
    u = np.random.rand(3)
    q = [
        math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
        math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
        math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
        math.sqrt(u[0]) * math.cos(2 * math.pi * u[2]),
    ]
    return q


def run_tests(num_samples=1000):
    surface_height_world = 0.78
    dice_size = 0.06
    target_z = surface_height_world + (dice_size / 2.0) + 0.0005

    print(f"Running {num_samples} random realignment test cases...")
    print(f"Target Z height: {target_z:.4f}")

    failures = 0
    for i in range(num_samples):
        # Generate random start state
        q_in = generate_random_quaternion()
        pos_in = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0.8, 2.0)]

        # Run logic
        q_out, pos_out, max_d = get_aligned_pose(q_in, pos_in, surface_height_world, dice_size)

        # Test Z height convergence
        if not math.isclose(pos_out[2], target_z, abs_tol=1e-6):
            print(f"FAIL: Height convergence error! Expected {target_z:.4f}, got {pos_out[2]:.4f}")
            failures += 1
            continue

        # Test X and Y position preservation
        if pos_out[0] != pos_in[0] or pos_out[1] != pos_in[1]:
            print(f"FAIL: X or Y changed! In: {pos_in[:2]}, Out: {pos_out[:2]}")
            failures += 1
            continue

        # Test orientation alignment
        # In the output orientation, the closest axis should be perfectly aligned
        # with world Z (i.e. absolute Z coordinate should be 1.0)
        x_axis_out = np.array(rotate_vector([1.0, 0.0, 0.0], q_out))
        y_axis_out = np.array(rotate_vector([0.0, 1.0, 0.0], q_out))
        z_axis_out = np.array(rotate_vector([0.0, 0.0, 1.0], q_out))

        dx_out = abs(x_axis_out[2])
        dy_out = abs(y_axis_out[2])
        dz_out = abs(z_axis_out[2])

        max_d_out = max(dx_out, dy_out, dz_out)

        if not math.isclose(max_d_out, 1.0, abs_tol=1e-5):
            print(f"FAIL: Axis not aligned with Z! Max absolute Z projection: {max_d_out:.6f}")
            failures += 1
            continue

        # Check that the face up matches the closest face normal in starting pose
        best_face = None
        best_dot = -1.0
        for face_id, normal in face_normals.items():
            rotated_normal = rotate_vector(normal, q_out)
            dot = rotated_normal[2]
            if dot > best_dot:
                best_dot = dot
                best_face = face_id

        if not math.isclose(best_dot, 1.0, abs_tol=1e-5):
            print(
                f"FAIL: Target face not facing up! Face {best_face} "
                f"dot product = {best_dot:.6f}"
            )
            failures += 1
            continue

    if failures == 0:
        print(f"SUCCESS: All {num_samples} test cases passed successfully!")
    else:
        raise AssertionError(f"Realignment math test failed with {failures} failures.")


def test_realignment():
    run_tests(num_samples=100)


if __name__ == "__main__":
    run_tests()
