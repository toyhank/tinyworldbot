import mujoco

from tinyworldbot.env import SO101PushEnv


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
