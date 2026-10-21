# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Callable, Optional

from omegaconf import DictConfig
import torch

from rlinf.config import EMBODIED_MODEL, SupportedModel, torch_dtype_from_precision
from rlinf.scheduler import Worker

ModelBuilder = Callable[[DictConfig, Optional[object]], object]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}
logger = logging.getLogger(__name__)


def register_model(
    model_type: str,
    model_builder: ModelBuilder,
    category: str = "embodied",
    force: bool = False,
):
    """Register a model builder for cfg.model_type."""
    if not model_type:
        raise ValueError("model_type must be a non-empty string.")
    if not callable(model_builder):
        raise TypeError("model_builder must be callable.")
    if not force and model_type in _MODEL_REGISTRY:
        raise ValueError(
            f"Model type `{model_type}` is already registered. "
            "Set force=True to override it."
        )
    _MODEL_REGISTRY[model_type] = model_builder
    SupportedModel.register(model_type, force=force)
    if category == "embodied":
        EMBODIED_MODEL.add(SupportedModel(model_type))


def _register_builtin_models():
    def _build_openvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_openvla_oft(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla_oft import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_pi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_pi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_dm0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_dm0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_mlp_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.mlp_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_cnn_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.cnn_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_flow_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.flow_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_lingbotvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.lingbotvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_abot_m0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.abot_m0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_starvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.starvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_dreamzero(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dreamzero import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t_n1d6(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t_n1d7(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi_cfg(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi_cfg import get_model

        return get_model(cfg, torch_dtype)

    def _build_value_model(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.value_model import get_model

        return get_model(cfg, torch_dtype)

    register_model(
        SupportedModel.OPENVLA.value,
        _build_openvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENVLA_OFT.value,
        _build_openvla_oft,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENPI.value,
        _build_openpi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_PI.value,
        _build_dexbotic_pi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_DM0.value,
        _build_dexbotic_dm0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.MLP_POLICY.value,
        _build_mlp_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T.value,
        _build_gr00t,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CNN_POLICY.value,
        _build_cnn_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.FLOW_POLICY.value,
        _build_flow_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.LINGBOTVLA.value,
        _build_lingbotvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.ABOT_M0.value,
        _build_abot_m0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.STARVLA.value,
        _build_starvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DREAMZERO.value,
        _build_dreamzero,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CFG_MODEL.value,
        _build_openpi_cfg,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.VALUE_MODEL.value,
        _build_value_model,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T_N1D6.value,
        _build_gr00t_n1d6,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T_N1D7.value,
        _build_gr00t_n1d7,
        category="embodied",
        force=True,
    )


_register_builtin_models()


def get_model(cfg: DictConfig):
    model_type = str(cfg.model_type)
    model_builder = _MODEL_REGISTRY.get(model_type)
    if model_builder is None:
        return None

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = model_builder(cfg, torch_dtype)

    if (
        Worker.torch_platform is not None
        and Worker.torch_platform.is_available()
        and cfg.get("load_to_device", True)
    ):
        model = model.to(Worker.torch_device_type)

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_scope = cfg.get("lora_scope", "all")
            if lora_scope not in ("all", "language_model"):
                raise ValueError(
                    f"Unsupported lora_scope={lora_scope!r}; expected 'all' or "
                    "'language_model'."
                )
            # The language-only OpenPI setup used by OPD training adapts the
            # Gemma attention and MLP projections, and nothing else.
            language_model_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_target_modules = list(language_model_target_modules)
            if lora_scope == "all":
                lora_target_modules.extend(
                    [
                        "proj",
                        "qkv",
                        "fc1",
                        "fc2",  # vision
                        "q",
                        "kv",
                        "fc3",
                        "out_proj",  # projector
                        "lm_head",
                    ]
                )
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=lora_target_modules,
                # This is a second guard against matching similarly named
                # modules in SigLIP when language-only LoRA is requested.
                exclude_modules=(
                    r".*vision_tower.*"
                    if lora_scope == "language_model"
                    else None
                ),
                init_lora_weights="gaussian",
            )
            if SupportedModel(model_type) in (
                SupportedModel.OPENPI,
                SupportedModel.CFG_MODEL,
            ):
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
                _reload_fsdp_checkpoint_after_lora(model, cfg)
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    if cfg.get("merge_lora_for_eval", False):
        if not cfg.is_lora:
            raise ValueError("merge_lora_for_eval requires is_lora=True.")
        if SupportedModel(model_type) != SupportedModel.OPENPI:
            raise ValueError("merge_lora_for_eval currently supports OpenPI only.")
        _merge_openpi_lora_for_eval(model)

    return model


def _reload_fsdp_checkpoint_after_lora(model, cfg: DictConfig) -> None:
    """Load FSDP full_weights again after PEFT LoRA modules are attached.

    The OpenPI model builder loads checkpoint weights before the generic LoRA
    wrapper in this module is applied. For FSDP checkpoints that contain LoRA
    tensors, e.g. ``*.lora_A.default.weight``, those keys are ignored by the
    pre-LoRA ``strict=False`` load. Reloading after PEFT injection restores the
    trained adapter weights while keeping normal base-model initialization
    unchanged when ``model_path`` is not an FSDP checkpoint.
    """

    checkpoint_dir = str(cfg.get("checkpoint_path") or cfg.model_path)
    full_weights_candidates = [
        os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"),
    ]
    for full_weights_path in full_weights_candidates:
        if os.path.exists(full_weights_path):
            model_state_dict = torch.load(full_weights_path, map_location="cpu")
            checkpoint_lora_keys = {
                key for key in model_state_dict if ".lora_" in key
            }
            if cfg.get("merge_lora_for_eval", False) and not checkpoint_lora_keys:
                raise ValueError(
                    "The evaluation checkpoint contains no LoRA tensors. "
                    "For an already merged model, set is_lora=False and "
                    "merge_lora_for_eval=False."
                )
            model_keys = set(model.state_dict())
            unmatched_lora_keys = checkpoint_lora_keys - model_keys
            if unmatched_lora_keys:
                sample = sorted(unmatched_lora_keys)[:5]
                raise RuntimeError(
                    f"LoRA checkpoint structure does not match the model: "
                    f"{len(unmatched_lora_keys)}/{len(checkpoint_lora_keys)} "
                    f"adapter tensors are unmatched. Examples: {sample}. "
                    "Check is_lora, lora_rank, and lora_scope."
                )
            model.load_state_dict(model_state_dict, strict=False)
            if checkpoint_lora_keys:
                logger.info(
                    "Restored %d LoRA tensors from %s",
                    len(checkpoint_lora_keys),
                    full_weights_path,
                )
            return

    if cfg.get("checkpoint_path"):
        raise FileNotFoundError(
            f"No FSDP full_weights.pt found in {checkpoint_dir}. "
            "Set checkpoint_path to the checkpoint root or its actor directory."
        )


def _merge_openpi_lora_for_eval(model) -> None:
    """Merge the nested VLM adapter after restoring its trained FSDP weights."""
    from peft import PeftModel

    vlm = model.paligemma_with_expert.paligemma
    if not isinstance(vlm, PeftModel):
        raise ValueError("Expected an OpenPI VLM PEFT adapter before merging.")
    model.paligemma_with_expert.paligemma = vlm.merge_and_unload(safe_merge=True)
    model.requires_grad_(False)
    model.eval()
    logger.info("Merged OpenPI VLM LoRA for evaluation; all parameters are frozen.")


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
