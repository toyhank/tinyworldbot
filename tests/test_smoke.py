import mujoco
import numpy as np

from tinyworldbot.env import SO101PushEnv
from tinyworldbot.pickplace import GRASP_BANK


def test_model_loads_and_has_cameras():
    env = SO101PushEnv(seed=0)
    assert env.model.ncam >= 2
    assert env.model.camera("scene_camera").id >= 0
    assert env.model.camera("wrist_camera").id >= 0
    assert env.joint_positions.shape == (6,)


def test_object_truth_is_separate_from_robot_state():
    env = SO101PushEnv(seed=1)
    assert env.state_without_object().shape == (7,)
    assert env.true_object_xy().shape == (2,)


def test_pickplace_grasp_bank_is_small_and_bounded():
    assert GRASP_BANK.shape == (6, 5)
    # dx/dy, descend z, wrist roll, gripper target
    assert np.all(np.abs(GRASP_BANK[:, :2]) < 0.03)
    assert np.all((GRASP_BANK[:, 2] > 0.015) & (GRASP_BANK[:, 2] < 0.04))
    assert np.all(np.abs(GRASP_BANK[:, 3]) < 2.0)
