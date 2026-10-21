from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.actor import fsdp_opd_policy_worker as worker


@pytest.mark.parametrize("expert_only", [True, False])
@pytest.mark.parametrize("padding", [False, True])
def test_routed_target_is_the_only_loss(expert_only, padding):
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.config = SimpleNamespace(action_chunk=1, num_steps=2)

        def forward(self, *, nft_inputs, **kwargs):
            return {
                "v_theta": torch.ones_like(nft_inputs["x_t"]) * self.weight,
                "prefix_cache": None,
            }

    model = Student()
    policy = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        cfg=OmegaConf.create(
            {"actor": {"model": {"openpi": {"train_expert_only": expert_only}}}}
        ),
        opd_train_step_indices=[0, 1],
        gradient_accumulation=2,
        teacher_models=[None, None],
        before_micro_batch=lambda *a, **kw: nullcontext(),
        amp_context=nullcontext(),
        grad_scaler=SimpleNamespace(scale=lambda x: x),
    )
    for name in (
        "_extract_model_forward_inputs",
        "_build_ode_schedule",
        "_compute_ode_step_mean",
        "_masked_step_mean",
    ):
        setattr(policy, name, MethodType(getattr(worker.EmbodiedOPDFSDPPolicy, name), policy))
    # Distinct cached targets represent specialist and Base-SFT routed rows.
    # Student mean is -1 at both steps: row losses are 0.5 and 2.0.
    inputs = {
        "chains": torch.zeros(2, 2, 1, 1),
        "teacher_ids": torch.tensor([0, 1]),
        "opd_teacher_means": torch.tensor([0.0, 1.0]).view(2, 1, 1, 1).expand(2, 2, 1, 1),
    }
    metrics = {}
    worker.EmbodiedOPDFSDPPolicy.train_micro_batch(
        policy,
        {"forward_inputs": inputs, "loss_mask": torch.full((2, 1), not padding)},
        metrics,
        is_last=True,
    )
    scale = 0 if padding else 1
    assert metrics["actor/total_loss"][0] == pytest.approx(scale * 1.25)
    assert model.weight.grad.item() == pytest.approx(scale * 0.375)


def test_precompute_preserves_one_teacher_per_sample():
    chains = torch.ones(2, 2, 1, 1)
    ids = torch.tensor([1, 0])
    inputs = {"chains": chains, "teacher_ids": ids}
    calls = []

    def compute(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(chains)

    policy = SimpleNamespace(
        rollout_batch={"forward_inputs": inputs},
        log_info=lambda _: None,
        _compute_teacher_means=compute,
        _extract_model_forward_inputs=lambda _: {},
    )
    worker.EmbodiedOPDFSDPPolicy._precompute_rollout_teacher_means(policy)
    assert len(calls) == 1
    assert torch.equal(calls[0]["teacher_ids"], ids)
    assert set(inputs) == {"chains", "teacher_ids", "opd_teacher_means"}
