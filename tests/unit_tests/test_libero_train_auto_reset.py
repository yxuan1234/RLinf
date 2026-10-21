from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from rlinf.envs.libero.libero_env import LiberoEnv


def test_train_auto_reset_uses_existing_reset_state_vector():
    env = LiberoEnv.__new__(LiberoEnv)
    env.num_envs = 8
    env.use_fixed_reset_state_ids = True
    env.cfg = SimpleNamespace(use_ordered_reset_state_ids=True)
    env.reset_state_ids = np.arange(100, 108, dtype=np.int64)
    env.update_reset_state_ids = Mock(
        side_effect=lambda: setattr(
            env, "reset_state_ids", np.arange(108, 116, dtype=np.int64)
        )
    )
    env.reset = Mock(return_value=({"obs": "reset"}, {}))

    dones = np.array([False, False, True, False, False, False, False, False])
    obs, infos = env._handle_train_auto_reset(dones, {"obs": "final"}, {})

    np.testing.assert_array_equal(
        env.reset_state_ids,
        np.arange(108, 116, dtype=np.int64),
    )
    env.update_reset_state_ids.assert_called_once_with()
    reset_kwargs = env.reset.call_args.kwargs
    np.testing.assert_array_equal(reset_kwargs["env_idx"], np.array([2]))
    np.testing.assert_array_equal(
        reset_kwargs["reset_state_ids"], np.array([110])
    )
    assert obs == {"obs": "reset"}
    assert infos["final_observation"] == {"obs": "final"}


def test_train_auto_reset_does_not_allocate_per_completed_slot():
    env = LiberoEnv.__new__(LiberoEnv)
    env.num_envs = 8
    env.use_fixed_reset_state_ids = True
    env.cfg = SimpleNamespace(use_ordered_reset_state_ids=True)
    env.reset_state_ids = np.arange(200, 208, dtype=np.int64)
    env.update_reset_state_ids = Mock(
        side_effect=lambda: setattr(
            env, "reset_state_ids", np.arange(208, 216, dtype=np.int64)
        )
    )
    env.reset = Mock(return_value=({}, {}))

    dones = np.array([True, False, False, True, False, False, True, False])
    env._handle_train_auto_reset(dones, {}, {})

    np.testing.assert_array_equal(
        env.reset_state_ids,
        np.arange(208, 216, dtype=np.int64),
    )
    env.update_reset_state_ids.assert_called_once_with()
    np.testing.assert_array_equal(
        env.reset.call_args.kwargs["env_idx"], np.array([0, 3, 6])
    )
    np.testing.assert_array_equal(
        env.reset.call_args.kwargs["reset_state_ids"], np.array([208, 211, 214])
    )
