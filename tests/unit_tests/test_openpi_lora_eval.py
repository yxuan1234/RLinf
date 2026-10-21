from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

from rlinf.models import (
    _merge_openpi_lora_for_eval,
    _reload_fsdp_checkpoint_after_lora,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self, rank=2):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = get_peft_model(
            torch.nn.Sequential(torch.nn.Linear(4, 4)),
            LoraConfig(r=rank, lora_alpha=rank, target_modules=["0"]),
        )
        self.action_head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.action_head(self.paligemma_with_expert.paligemma(x))


@pytest.mark.parametrize("actor_subdir", [False, True])
def test_restored_adapter_and_action_head_survive_merge(tmp_path, actor_subdir):
    torch.manual_seed(7)
    trained = TinyPolicy().eval()
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if "lora_" in name:
                param.fill_(0.2)
    x = torch.randn(3, 4)
    expected = trained(x).detach()
    path = tmp_path / "actor" if actor_subdir else tmp_path
    (path / "model_state_dict").mkdir(parents=True)
    torch.save(trained.state_dict(), path / "model_state_dict/full_weights.pt")

    restored = TinyPolicy().eval()
    cfg = OmegaConf.create(
        {"model_path": "unused_base", "checkpoint_path": str(tmp_path)}
    )
    _reload_fsdp_checkpoint_after_lora(restored, cfg)
    torch.testing.assert_close(restored(x), expected)
    _merge_openpi_lora_for_eval(restored)
    torch.testing.assert_close(restored(x), expected)
    assert not any("lora_" in name for name in restored.state_dict())
    assert not restored.training
    assert all(not p.requires_grad for p in restored.parameters())


def test_missing_explicit_checkpoint_is_an_error(tmp_path):
    cfg = OmegaConf.create(
        {"model_path": "unused_base", "checkpoint_path": str(tmp_path)}
    )
    with pytest.raises(FileNotFoundError, match="full_weights.pt"):
        _reload_fsdp_checkpoint_after_lora(TinyPolicy(), cfg)


def test_wrong_adapter_rank_is_an_error(tmp_path):
    (tmp_path / "model_state_dict").mkdir()
    torch.save(TinyPolicy().state_dict(), tmp_path / "model_state_dict/full_weights.pt")
    cfg = OmegaConf.create(
        {"model_path": "unused_base", "checkpoint_path": str(tmp_path)}
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        _reload_fsdp_checkpoint_after_lora(TinyPolicy(rank=3), cfg)


def test_merge_requires_nested_adapter():
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(paligemma=torch.nn.Linear(4, 4))
    )
    with pytest.raises(ValueError, match="PEFT adapter"):
        _merge_openpi_lora_for_eval(model)


def test_merge_rejects_checkpoint_without_adapter(tmp_path):
    (tmp_path / "model_state_dict").mkdir()
    torch.save({}, tmp_path / "model_state_dict/full_weights.pt")
    cfg = OmegaConf.create(
        {
            "model_path": "unused_base",
            "checkpoint_path": str(tmp_path),
            "merge_lora_for_eval": True,
        }
    )
    with pytest.raises(ValueError, match="no LoRA tensors"):
        _reload_fsdp_checkpoint_after_lora(TinyPolicy(), cfg)


@pytest.mark.parametrize(
    "suite,horizon", [("spatial", 240), ("object", 280), ("goal", 320), ("10", 520)]
)
def test_eval_configs_match_suite_and_training_recipe(monkeypatch, suite, horizon):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setenv("PI05_SFT_MODEL_DIR", "/path/to/base")
    monkeypatch.setenv("PI05_OPD_CHECKPOINT_DIR", "/path/to/checkpoint")
    monkeypatch.setenv("OPD_GPU_PLACEMENT", "0")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "evaluations/libero")
    ):
        cfg = compose(config_name=f"libero_{suite}_opd_vlm_lora_eval")
        OmegaConf.to_container(cfg, resolve=True)
        assert cfg.env.eval.task_suite_name == f"libero_{suite}"
        assert cfg.env.eval.max_episode_steps == horizon
        assert cfg.env.eval.total_num_envs == 50
        assert cfg.env.eval.max_steps_per_rollout_epoch == horizon * 10
        assert cfg.env.eval.is_eval and cfg.env.eval.auto_reset
        assert not cfg.env.eval.video_cfg.save_video
        assert cfg.rollout.model.is_lora and cfg.rollout.model.merge_lora_for_eval
        assert cfg.rollout.model.lora_scope == "language_model"
        assert cfg.rollout.model.num_steps == 5
        assert cfg.rollout.model.num_action_chunks == 10
        assert cfg.rollout.model.openpi.noise_method == "flow_ode"
