import math

from drims_dice_simulator.dice_spawner import generate_random_spawn_pose


def test_generate_random_spawn_pose_randomizes_position_and_orientation():
    x, y, yaw, face = generate_random_spawn_pose(-0.30, 0.15, 0.50, 0.85)

    assert -0.30 <= x <= 0.15
    assert 0.50 <= y <= 0.85
    assert 1 <= face <= 6
    assert -math.pi <= yaw <= math.pi
