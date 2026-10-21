from types import MethodType, SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.workers.actor import fsdp_opd_policy_worker
from rlinf.workers.actor.fsdp_opd_policy_worker import EmbodiedOPDFSDPPolicy


class _TinyOpenPi(OpenPi0ForRLActionPrediction):
    """Minimal OpenPI shell exercising the NFT prefix/suffix control flow."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(action_chunk=2, add_value_head=False)
        self.prefix_calls = 0
        self.velocity_calls = 0

    def input_transform(self, forward_inputs, transpose=False):
        del transpose
        return forward_inputs

    def _preprocess_observation(self, observation, train=False):
        del train
        batch_size = observation.state.shape[0]
        return (
            [torch.zeros(batch_size, 1)],
            [torch.ones(batch_size, dtype=torch.bool)],
            torch.zeros(batch_size, 1, dtype=torch.long),
            torch.ones(batch_size, 1, dtype=torch.bool),
            observation.state,
        )

    def _build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
        del images, img_masks, lang_tokens, lang_masks
        self.prefix_calls += 1
        return self.scale, torch.ones(1, 1, dtype=torch.bool), self.scale

    def get_velocity(self, state, x_t, timestep, prefix_pad_masks, past_key_values):
        del state, prefix_pad_masks
        self.velocity_calls += 1
        velocity = x_t * past_key_values + timestep[:, None, None]
        return velocity, velocity


def _forward_inputs(batch_size):
    return {
        "image": {"cam": torch.zeros(batch_size, 1, 1, 3)},
        "image_mask": {"cam": torch.ones(batch_size, dtype=torch.bool)},
        "state": torch.zeros(batch_size, 1),
    }


def test_multistep_nft_reuses_prefix_and_matches_single_step_gradients():
    batch_size, num_steps = 2, 3
    x_t = torch.arange(batch_size * num_steps * 2 * 2, dtype=torch.float32).reshape(
        batch_size, num_steps, 2, 2
    )
    timesteps = torch.linspace(1.0, 0.25, num_steps).repeat(batch_size, 1)

    multi_model = _TinyOpenPi()
    multi_output = multi_model.nft_forward(
        _forward_inputs(batch_size),
        nft_inputs={"x_t": x_t, "timesteps": timesteps},
    )["v_theta"]
    multi_output.sum().backward()

    single_model = _TinyOpenPi()
    single_outputs = []
    for step_idx in range(num_steps):
        single_outputs.append(
            single_model.nft_forward(
                _forward_inputs(batch_size),
                nft_inputs={
                    "x_t": x_t[:, step_idx],
                    "timesteps": timesteps[:, step_idx],
                },
            )["v_theta"]
        )
    single_output = torch.stack(single_outputs, dim=1)
    single_output.sum().backward()

    torch.testing.assert_close(multi_output, single_output)
    torch.testing.assert_close(multi_model.scale.grad, single_model.scale.grad)
    assert multi_model.prefix_calls == 1
    assert multi_model.velocity_calls == num_steps
    assert single_model.prefix_calls == num_steps


def test_streamed_suffix_backward_reuses_prefix_and_matches_mean_backward():
    batch_size, num_steps = 2, 3
    x_t = torch.arange(batch_size * num_steps * 2 * 2, dtype=torch.float32).reshape(
        batch_size, num_steps, 2, 2
    )
    timesteps = torch.linspace(1.0, 0.25, num_steps).repeat(batch_size, 1)

    mean_model = _TinyOpenPi()
    mean_output = mean_model.nft_forward(
        _forward_inputs(batch_size),
        nft_inputs={"x_t": x_t, "timesteps": timesteps},
    )["v_theta"]
    mean_output.square().mean(dim=(0, 2, 3)).mean().backward()

    streamed_model = _TinyOpenPi()
    prefix_cache = None
    streamed_outputs = []
    for step_idx in range(num_steps):
        output = streamed_model.nft_forward(
            _forward_inputs(batch_size),
            nft_inputs={
                "x_t": x_t[:, step_idx],
                "timesteps": timesteps[:, step_idx],
                "prefix_cache": prefix_cache,
                "return_prefix_cache": True,
            },
        )
        prefix_cache = output["prefix_cache"]
        step_output = output["v_theta"]
        streamed_outputs.append(step_output.detach())
        (step_output.square().mean() / num_steps).backward()

    streamed_output = torch.stack(streamed_outputs, dim=1)
    torch.testing.assert_close(mean_output, streamed_output)
    torch.testing.assert_close(mean_model.scale.grad, streamed_model.scale.grad)
    assert streamed_model.prefix_calls == 1
    assert streamed_model.velocity_calls == num_steps


class _BatchedTeacher:
    def __init__(self, *, offset=0.0, fail=False):
        self.calls = 0
        self.moves = []
        self.offset = offset
        self.fail = fail

    def to(self, device):
        self.moves.append(str(device))
        return self

    def eval(self):
        return self

    def __call__(self, *, nft_inputs, **kwargs):
        del kwargs
        self.calls += 1
        if self.fail:
            raise RuntimeError("teacher forward failed")
        x_t = nft_inputs["x_t"]
        timesteps = nft_inputs["timesteps"][:, :, None, None]
        return {
            "v_theta": x_t * 0.25 + timesteps + self.offset,
            "x_t": x_t,
        }


def test_teacher_targets_batch_all_timesteps_per_prefix(monkeypatch):
    # This test exercises CPU target batching, not CUDA allocator cleanup.
    monkeypatch.setattr(fsdp_opd_policy_worker, "clear_memory", lambda **kwargs: None)
    batch_size, num_steps = 5, 3
    chains = torch.arange(batch_size * num_steps * 2 * 2, dtype=torch.float32).reshape(
        batch_size, num_steps, 2, 2
    )
    teacher_0 = _BatchedTeacher()
    teacher_1 = _BatchedTeacher(offset=10.0)
    inactive_teacher = _BatchedTeacher()
    teacher_ids = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)
    policy = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(action_chunk=2, num_steps=num_steps)
        ),
        opd_train_step_indices=list(range(num_steps)),
        device=torch.device("cpu"),
        teacher_models=[teacher_0, teacher_1, inactive_teacher],
        opd_teacher_micro_batch_size=2,
    )
    policy._build_ode_schedule = MethodType(
        EmbodiedOPDFSDPPolicy._build_ode_schedule, policy
    )
    policy._compute_ode_step_mean = MethodType(
        EmbodiedOPDFSDPPolicy._compute_ode_step_mean, policy
    )
    policy._index_nested_tensors = MethodType(
        EmbodiedOPDFSDPPolicy._index_nested_tensors, policy
    )

    means = EmbodiedOPDFSDPPolicy._compute_teacher_means(
        policy,
        model_forward_inputs={"obs": torch.zeros(batch_size, 1)},
        chains=chains,
        teacher_ids=teacher_ids,
    )

    schedule = torch.linspace(1.0, 0.0, num_steps + 1)
    for step_idx in range(num_steps):
        x_t = chains[:, step_idx]
        teacher_offsets = teacher_ids[:, None, None] * 10.0
        velocity = x_t * 0.25 + schedule[step_idx] + teacher_offsets
        expected = x_t - (schedule[step_idx] - schedule[step_idx + 1]) * velocity
        torch.testing.assert_close(means[:, step_idx], expected)
    assert teacher_0.calls == 1
    assert teacher_1.calls == 2
    assert teacher_0.moves == ["cpu", "cpu"]
    assert teacher_1.moves == ["cpu", "cpu"]
    assert inactive_teacher.moves == []


def test_active_teacher_is_offloaded_when_precompute_fails(monkeypatch):
    teacher = _BatchedTeacher(fail=True)
    policy = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(action_chunk=2, num_steps=1)),
        opd_train_step_indices=[0],
        device=torch.device("cpu"),
        teacher_models=[teacher],
        opd_teacher_micro_batch_size=1,
    )
    policy._build_ode_schedule = MethodType(
        EmbodiedOPDFSDPPolicy._build_ode_schedule, policy
    )
    policy._compute_ode_step_mean = MethodType(
        EmbodiedOPDFSDPPolicy._compute_ode_step_mean, policy
    )
    policy._index_nested_tensors = MethodType(
        EmbodiedOPDFSDPPolicy._index_nested_tensors, policy
    )
    monkeypatch.setattr(fsdp_opd_policy_worker, "clear_memory", lambda sync=False: None)

    with pytest.raises(RuntimeError, match="teacher forward failed"):
        EmbodiedOPDFSDPPolicy._compute_teacher_means(
            policy,
            model_forward_inputs={"obs": torch.zeros(1, 1)},
            chains=torch.zeros(1, 1, 2, 2),
            teacher_ids=torch.zeros(1, dtype=torch.long),
        )

    assert teacher.moves == ["cpu", "cpu"]
