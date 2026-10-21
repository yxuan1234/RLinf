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

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    concat_batch,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedOPDFSDPPolicy(EmbodiedFSDPActor):
    """Pure ODE OPD training worker for OpenPI."""

    def init_worker(self) -> None:
        super().init_worker()
        if str(self.cfg.actor.model.model_type) != "openpi":
            raise ValueError("embodied_opd currently only supports actor.model=openpi.")
        if str(self.model.config.noise_method) != "flow_ode":
            raise ValueError(
                "embodied_opd currently only supports OpenPI noise_method=flow_ode."
            )

        self.opd_train_step_indices = self._build_opd_train_step_indices()
        self.opd_teacher_micro_batch_size = int(
            self.cfg.algorithm.get(
                "opd_teacher_micro_batch_size", self.cfg.actor.micro_batch_size
            )
        )
        self.teacher_models, self.teacher_names = self._build_teacher_models()
        self.log_info(
            "Loaded OPD teachers: "
            + ", ".join(f"{idx}:{name}" for idx, name in enumerate(self.teacher_names))
        )
        train_expert_only = bool(
            self.cfg.actor.model.openpi.get("train_expert_only", True)
        )
        self.log_info(
            "OPD Student timestep backward mode: "
            + (
                "streamed suffixes with one frozen prefix cache"
                if train_expert_only
                else "joint multi-timestep backward for trainable VLM prefix"
            )
        )

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        return compute_rollout_metrics(self.rollout_batch)

    def _use_rollout_epoch_schedule(self) -> bool:
        return self.cfg.algorithm.get("rollout_epoch_schedule", None) is not None

    def _get_scheduled_rollout_channel_key(self) -> str:
        return f"scheduled_actor_rank_{self._rank}"

    def _flatten_block_batch(self, nested_dict: dict[str, torch.Tensor]) -> dict:
        flattened = {}
        for key, value in nested_dict.items():
            if key in {"dones", "terminations", "truncations", "prev_values"}:
                value = value[:-1]

            if value is None:
                flattened[key] = None
            elif isinstance(value, torch.Tensor):
                flattened[key] = value.reshape(-1, *value.shape[2:])
            elif isinstance(value, dict):
                flattened[key] = self._flatten_block_batch(value)
            elif isinstance(value, list):
                flattened[key] = value
        return flattened

    @staticmethod
    def _mask_successful_episodes(
        loss_mask: torch.Tensor,
        terminations: torch.Tensor,
    ) -> torch.Tensor:
        """Keep OPD loss only for episodes that never succeeded.

        Scheduled embodied rollouts have shape ``[T, B, ...]`` and LIBERO
        reports task success through ``terminations``.  Reducing every axis
        except the environment-slot axis therefore matches the episode-level
        ``success_once`` semantics.  The resulting failure mask is broadcast
        over every chunk row belonging to that slot.
        """
        if terminations is None or terminations.ndim < 2:
            raise ValueError(
                "algorithm.opd_failed_episodes_only requires trajectory "
                "terminations with shape [T, B, ...]."
            )
        if loss_mask.ndim < 2:
            raise ValueError(
                f"Expected loss_mask with shape [T, B, ...], got {loss_mask.shape}."
            )
        if loss_mask.shape[1] != terminations.shape[1]:
            raise ValueError(
                "loss_mask and terminations disagree on environment batch size: "
                f"{loss_mask.shape[1]} != {terminations.shape[1]}."
            )

        reduce_dims = tuple(dim for dim in range(terminations.ndim) if dim != 1)
        episode_succeeded = terminations.bool().any(dim=reduce_dims)
        failed_episode_mask = ~episode_succeeded
        broadcast_shape = [1, failed_episode_mask.numel()] + [1] * (loss_mask.ndim - 2)
        return loss_mask & failed_episode_mask.reshape(broadcast_shape)

    def _build_scheduled_rollout_batch(
        self, trajectories: list[Trajectory]
    ) -> dict[str, torch.Tensor]:
        flattened_batches = []
        failed_episodes_only = bool(
            self.cfg.algorithm.get("opd_failed_episodes_only", False)
        )
        for trajectory in trajectories:
            block_batch = convert_trajectories_to_batch([trajectory])

            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                loss_mask, _ = compute_loss_mask(block_batch["dones"])
                if self.cfg.algorithm.reward_type == "chunk_level":
                    loss_mask = loss_mask.any(dim=-1, keepdim=True)
                if failed_episodes_only:
                    loss_mask = self._mask_successful_episodes(
                        loss_mask,
                        block_batch.get("terminations", None),
                    )
                block_batch["loss_mask"] = loss_mask

            flattened = self._flatten_block_batch(block_batch)

            flattened_batches.append(flattened)

        merged_batch = flattened_batches[0]
        for batch in flattened_batches[1:]:
            merged_batch = concat_batch(merged_batch, batch)
        return merged_batch



    def _index_train_samples(
        self, nested_dict: dict[str, torch.Tensor], indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        indexed = {}
        for key, value in nested_dict.items():
            if value is None:
                indexed[key] = None
            elif isinstance(value, torch.Tensor):
                indexed[key] = value.index_select(0, indices)
            elif isinstance(value, dict):
                indexed[key] = self._index_train_samples(value, indices)
            elif isinstance(value, list):
                indexed[key] = [value[int(index)] for index in indices.cpu().tolist()]
        return indexed

    @staticmethod
    def _pad_train_samples(
        nested_batch: dict[str, torch.Tensor],
        pad_count: int,
        *,
        _top_level: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Pad a flattened OPD batch with masked rows for static micro-batches."""
        if pad_count <= 0:
            return nested_batch
        padded: dict[str, torch.Tensor] = {}
        for key, value in nested_batch.items():
            if isinstance(value, torch.Tensor):
                pad_shape = (pad_count, *value.shape[1:])
                padded[key] = torch.cat(
                    [
                        value,
                        torch.zeros(pad_shape, dtype=value.dtype, device=value.device),
                    ],
                    dim=0,
                )
            elif isinstance(value, dict):
                padded[key] = EmbodiedOPDFSDPPolicy._pad_train_samples(
                    value, pad_count, _top_level=False
                )
            elif isinstance(value, list):
                padded[key] = value + (
                    [value[-1]] * pad_count if value else [None] * pad_count
                )
            else:
                padded[key] = value
        if not _top_level:
            return padded
        if "loss_mask" in padded:
            # Padded rows must not contribute to OPD loss, while retaining the
            # static batch size required by the FSDP accumulation schedule.
            padded["loss_mask"][-pad_count:] = False
        else:
            padded["loss_mask"] = torch.cat(
                [
                    torch.ones(
                        (nested_batch["forward_inputs"]["chains"].shape[0], 1),
                        dtype=torch.bool,
                    ),
                    torch.zeros((pad_count, 1), dtype=torch.bool),
                ],
                dim=0,
            )
        return padded

    @staticmethod
    def _scheduled_target_rollout_size(
        local_rollout_size: int,
        max_rollout_size: int,
        batch_size_per_rank: int,
    ) -> int:
        """Round the cross-rank maximum row count to a batch boundary."""
        del local_rollout_size
        return (
            (max_rollout_size + batch_size_per_rank - 1) // batch_size_per_rank
        ) * batch_size_per_rank

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        if not self._use_rollout_epoch_schedule():
            await super().recv_rollout_trajectories(input_channel)
            return

        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        num_blocks = int(self.cfg.env.train.rollout_epoch)

        recv_list = []
        for _ in range(split_num * num_blocks):
            trajectory: Trajectory = await input_channel.get(
                key=self._get_scheduled_rollout_channel_key(),
                async_op=True,
            ).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = self._build_scheduled_rollout_batch(recv_list)

    def _build_opd_train_step_indices(self) -> list[int]:
        num_steps = int(self.model.config.num_steps)
        timestep_fraction = float(self.cfg.algorithm.get("opd_timestep_fraction", 1.0))
        if timestep_fraction <= 0 or timestep_fraction > 1.0:
            raise ValueError(
                f"opd_timestep_fraction must be in (0, 1], got {timestep_fraction}"
            )
        num_train_timesteps = max(1, int(num_steps * timestep_fraction))
        num_train_timesteps = min(num_steps, num_train_timesteps)
        return list(range(num_train_timesteps))

    def _build_teacher_models(self):
        teacher_cfgs = self.cfg.algorithm.get("opd_teachers", None)
        if not teacher_cfgs:
            raise ValueError(
                "embodied_opd requires algorithm.opd_teachers with routed teacher checkpoints."
            )

        teacher_models = []
        teacher_names = []
        for teacher_idx, teacher_cfg in enumerate(teacher_cfgs):
            teacher_name = str(teacher_cfg.get("name", f"teacher_{teacher_idx}"))
            teacher_model_path = teacher_cfg.get("model_path", None)
            if not teacher_model_path:
                raise ValueError(
                    f"algorithm.opd_teachers[{teacher_idx}].model_path is required."
                )

            teacher_model_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg.actor.model, resolve=True)
            )
            teacher_model_cfg.model_path = teacher_model_path
            # A Student checkpoint must never be inherited by a routed Teacher.
            # Teachers are fixed policies loaded from their own model_path.
            teacher_model_cfg.checkpoint_path = None
            teacher_model_cfg.load_to_device = False
            teacher_model_cfg.add_value_head = False
            # Student-side LoRA is an optimization target, not part of the
            # fixed OPD Teacher policy.
            teacher_model_cfg.is_lora = False
            if teacher_model_cfg.get("openpi", None) is not None:
                teacher_model_cfg.openpi.add_value_head = False
                teacher_model_cfg.openpi.noise_method = "flow_ode"
                # OPD compares Student and Teacher predictions in the same
                # normalized action space. Unless a Teacher explicitly opts
                # into different statistics, reuse the Student checkpoint's
                # normalization assets.
                norm_stats_dir = teacher_cfg.get("norm_stats_dir", None)
                if not norm_stats_dir:
                    norm_stats_dir = self.cfg.actor.model.model_path
                teacher_model_cfg.openpi.norm_stats_dir = norm_stats_dir

            try:
                teacher_model = get_model(teacher_model_cfg)
            except FileNotFoundError as exc:
                if "norm_stats" not in str(exc):
                    raise
                raise FileNotFoundError(
                    "Could not load normalization statistics for the OPD Teacher. "
                    "By default they are loaded from the Student model_path, which "
                    "must contain 'physical-intelligence/libero/norm_stats.json'. "
                    f"Teacher '{teacher_name}' was configured with: "
                    f"{teacher_model_path}; Student model_path is: "
                    f"{self.cfg.actor.model.model_path}. "
                    "To use different statistics, set the optional "
                    "algorithm.opd_teachers[].norm_stats_dir."
                ) from exc
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")

            teacher_models.append(teacher_model)
            teacher_names.append(teacher_name)

        return teacher_models, teacher_names

    def _extract_model_forward_inputs(self, forward_inputs: dict) -> dict:
        model_forward_inputs = {}
        for key, value in forward_inputs.items():
            if "/" in key or key in ("tokenized_prompt", "tokenized_prompt_mask"):
                model_forward_inputs[key] = value
        return model_forward_inputs

    def _index_nested_tensors(self, nested_dict: dict, indices: torch.Tensor) -> dict:
        indexed = {}
        for key, value in nested_dict.items():
            if isinstance(value, torch.Tensor):
                indexed[key] = value.index_select(0, indices)
            elif isinstance(value, dict):
                indexed[key] = self._index_nested_tensors(value, indices)
            else:
                indexed[key] = value
        return indexed

    def _build_ode_schedule(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.linspace(
            1.0,
            0.0,
            int(self.model.config.num_steps) + 1,
            device=device,
            dtype=dtype,
        )

    def _compute_ode_step_mean(
        self, x_t: torch.Tensor, v_theta: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return x_t.float() - delta.float() * v_theta.float()

    def _masked_step_mean(
        self,
        per_element_value: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if loss_mask is None:
            return per_element_value.mean()

        if loss_mask.dim() == 1:
            loss_mask = loss_mask[:, None]
        if loss_mask.shape[1] == 1 and per_element_value.shape[1] > 1:
            loss_mask = loss_mask.expand(-1, per_element_value.shape[1])
        else:
            loss_mask = loss_mask[:, : per_element_value.shape[1]]

        expanded_mask = (
            loss_mask.unsqueeze(-1)
            .expand(-1, per_element_value.shape[1], per_element_value.shape[2])
            .to(dtype=per_element_value.dtype)
        )
        denom = expanded_mask.sum().clamp_min(1.0)
        return (per_element_value * expanded_mask).sum() / denom

    @torch.no_grad()
    def _compute_teacher_means(
        self,
        model_forward_inputs: dict,
        chains: torch.Tensor,
        teacher_ids: torch.Tensor,
        models=None,
    ) -> torch.Tensor:
        """Precompute routed Teacher targets for the complete rollout batch."""
        models = self.teacher_models if models is None else models
        schedule = self._build_ode_schedule(device=chains.device, dtype=chains.dtype)
        batch_size = chains.shape[0]
        action_chunk = int(self.model.config.action_chunk)
        latent_action_dim = int(chains.shape[-1])

        teacher_means = torch.zeros(
            batch_size,
            len(self.opd_train_step_indices),
            action_chunk,
            latent_action_dim,
            device=chains.device,
            dtype=torch.float32,
        )

        train_step_indices = torch.tensor(
            self.opd_train_step_indices,
            device=chains.device,
            dtype=torch.long,
        )
        train_timesteps = schedule.index_select(0, train_step_indices)
        train_deltas = train_timesteps - schedule.index_select(
            0, train_step_indices + 1
        )

        unique_teacher_ids = torch.unique(teacher_ids).tolist()
        for teacher_id in unique_teacher_ids:
            teacher_id = int(teacher_id)
            if teacher_id < 0 or teacher_id >= len(models):
                raise ValueError(
                    f"teacher_id {teacher_id} is out of range for {len(models)} configured OPD teachers."
                )

            teacher_model = models[teacher_id].to(self.device)
            teacher_model.eval()

            teacher_indices = torch.nonzero(
                teacher_ids == teacher_id, as_tuple=False
            ).flatten()
            try:
                for start in range(
                    0, teacher_indices.numel(), self.opd_teacher_micro_batch_size
                ):
                    end = min(
                        start + self.opd_teacher_micro_batch_size,
                        teacher_indices.numel(),
                    )
                    batch_indices = teacher_indices[start:end]
                    selected_chains = chains.index_select(0, batch_indices)
                    x_t = selected_chains.index_select(1, train_step_indices)[
                        :, :, :action_chunk
                    ]
                    timesteps = (
                        train_timesteps.to(dtype=x_t.dtype)
                        .unsqueeze(0)
                        .expand(x_t.shape[0], -1)
                    )
                    teacher_forward_inputs = self._index_nested_tensors(
                        model_forward_inputs, batch_indices
                    )
                    teacher_forward_inputs = put_tensor_device(
                        teacher_forward_inputs, self.device
                    )
                    teacher_output = teacher_model(
                        forward_type=ForwardType.NFT,
                        forward_inputs=teacher_forward_inputs,
                        nft_inputs={"x_t": x_t, "timesteps": timesteps},
                        compute_values=False,
                    )
                    if os.environ.get("DEBUG_CONDITION", "0") == "1":
                        student_model = self.model
                        student_model.eval()
                        _ = student_model(
                            forward_type=ForwardType.NFT,
                            forward_inputs=teacher_forward_inputs,
                            nft_inputs={"x_t": x_t, "timesteps": timesteps},
                            compute_values=False,
                        )
                        student_prefix = student_model._debug_prefix_output
                        teacher_prefix = teacher_model._debug_prefix_output
                        prefix_diff = (student_prefix - teacher_prefix).abs()
                        print(
                            "condition:",
                            "shape=",
                            tuple(student_prefix.shape),
                            "max_abs_diff=",
                            prefix_diff.max().item(),
                            "mean_abs_diff=",
                            prefix_diff.mean().item(),
                            "cosine=",
                            F.cosine_similarity(
                                student_prefix.flatten(),
                                teacher_prefix.flatten(),
                                dim=0,
                            ).item(),
                            flush=True,
                        )
                        for layer_id, ((sk, sv), (tk, tv)) in enumerate(
                            zip(
                                student_model._debug_prefix_kv,
                                teacher_model._debug_prefix_kv,
                            )
                        ):
                            print(
                                "kv_layer",
                                layer_id,
                                "key_max_diff=",
                                (sk - tk).abs().max().item(),
                                "key_cosine=",
                                F.cosine_similarity(
                                    sk.flatten(), tk.flatten(), dim=0
                                ).item(),
                                "value_max_diff=",
                                (sv - tv).abs().max().item(),
                                "value_cosine=",
                                F.cosine_similarity(
                                    sv.flatten(), tv.flatten(), dim=0
                                ).item(),
                                flush=True,
                            )
                        raise SystemExit("condition comparison finished")

                    teacher_velocities = teacher_output["v_theta"]
                    teacher_x_t = teacher_output["x_t"]
                    batch_teacher_means = self._compute_ode_step_mean(
                        x_t=teacher_x_t[
                            :,
                            :,
                            : teacher_velocities.shape[2],
                            : teacher_velocities.shape[3],
                        ],
                        v_theta=teacher_velocities,
                        delta=train_deltas.to(self.device).reshape(1, -1, 1, 1),
                    )
                    teacher_means.index_copy_(
                        0,
                        batch_indices.to(teacher_means.device),
                        batch_teacher_means.to(teacher_means.device),
                    )
            finally:
                teacher_model.to("cpu")
                clear_memory(sync=False)

        return teacher_means

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)

        forward_inputs = micro_batch["forward_inputs"]
        model_forward_inputs = self._extract_model_forward_inputs(forward_inputs)
        chains = forward_inputs["chains"]
        teacher_ids = forward_inputs.get("teacher_ids", None)
        if teacher_ids is None:
            raise ValueError(
                "forward_inputs.teacher_ids is required for embodied_opd teacher routing."
            )
        teacher_ids = teacher_ids.long().reshape(-1)
        loss_mask = micro_batch.get("loss_mask", None)
        cached_teacher_means = forward_inputs.get("opd_teacher_means", None)
        if cached_teacher_means is None:
            raise ValueError(
                "forward_inputs.opd_teacher_means must be precomputed before "
                "Student OPD updates."
            )
        schedule = self._build_ode_schedule(device=chains.device, dtype=chains.dtype)
        action_chunk = int(self.model.config.action_chunk)
        train_step_indices = torch.tensor(
            self.opd_train_step_indices,
            device=chains.device,
            dtype=torch.long,
        )
        train_timesteps = schedule.index_select(0, train_step_indices)
        train_deltas = train_timesteps - schedule.index_select(
            0, train_step_indices + 1
        )
        x_t = chains.index_select(1, train_step_indices)[:, :, :action_chunk]
        timesteps = (
            train_timesteps.to(dtype=x_t.dtype).unsqueeze(0).expand(x_t.shape[0], -1)
        )

        train_expert_only = bool(
            self.cfg.actor.model.openpi.get("train_expert_only", True)
        )
        step_losses = []
        teacher_step_losses = []
        with backward_ctx:
            if train_expert_only:
                # Reuse the frozen observation prefix, but stream the five
                # action-expert suffixes through forward/backward one at a
                # time. This preserves the exact mean-timestep objective while
                # keeping only one suffix activation graph alive.
                prefix_cache = None
                num_train_steps = len(self.opd_train_step_indices)
                for output_idx in range(num_train_steps):
                    with self.amp_context:
                        step_x_t = x_t[:, output_idx]
                        step_timesteps = timesteps[:, output_idx]
                        output_dict = self.model(
                            forward_type=ForwardType.NFT,
                            forward_inputs=model_forward_inputs,
                            nft_inputs={
                                "x_t": step_x_t,
                                "timesteps": step_timesteps,
                                "prefix_cache": prefix_cache,
                                "return_prefix_cache": True,
                            },
                            compute_values=False,
                        )
                        prefix_cache = output_dict.pop("prefix_cache")
                        student_velocity = output_dict["v_theta"]
                        student_means = self._compute_ode_step_mean(
                            x_t=step_x_t[
                                :,
                                : student_velocity.shape[1],
                                : student_velocity.shape[2],
                            ],
                            v_theta=student_velocity,
                            delta=train_deltas[output_idx],
                        )
                        teacher_means = cached_teacher_means[
                            :,
                            output_idx,
                            : student_means.shape[1],
                            : student_means.shape[2],
                        ]
                        per_step_kl = 0.5 * (student_means - teacher_means).pow(2)
                        step_loss = self._masked_step_mean(per_step_kl, loss_mask)
                        teacher_step_losses.append(step_loss.detach())
                        scaled_step_loss = step_loss / (
                            num_train_steps * self.gradient_accumulation
                        )

                    self.grad_scaler.scale(scaled_step_loss).backward()
                    step_losses.append(step_loss.detach())
                    del (
                        output_dict,
                        student_velocity,
                        student_means,
                        teacher_means,
                        per_step_kl,
                        step_loss,
                        scaled_step_loss,
                        step_x_t,
                        step_timesteps,
                    )

                final_loss = torch.stack(step_losses).mean()
                final_loss_value = final_loss.item()
                del prefix_cache, final_loss
            else:
                # A trainable VLM makes the prefix cache part of the gradient
                # graph. Keep the existing single-backward path until shared
                # prefix multi-backward is validated with FSDP.
                with self.amp_context:
                    output_dict = self.model(
                        forward_type=ForwardType.NFT,
                        forward_inputs=model_forward_inputs,
                        nft_inputs={"x_t": x_t, "timesteps": timesteps},
                        compute_values=False,
                    )
                    student_velocities = output_dict["v_theta"]
                    for output_idx in range(len(self.opd_train_step_indices)):
                        step_x_t = x_t[:, output_idx]
                        student_velocity = student_velocities[:, output_idx]
                        student_means = self._compute_ode_step_mean(
                            x_t=step_x_t[
                                :,
                                : student_velocity.shape[1],
                                : student_velocity.shape[2],
                            ],
                            v_theta=student_velocity,
                            delta=train_deltas[output_idx],
                        )
                        teacher_means = cached_teacher_means[
                            :,
                            output_idx,
                            : student_means.shape[1],
                            : student_means.shape[2],
                        ]
                        per_step_kl = 0.5 * (student_means - teacher_means).pow(2)
                        step_loss = self._masked_step_mean(per_step_kl, loss_mask)
                        teacher_step_losses.append(step_loss.detach())
                        step_losses.append(step_loss)

                final_loss = torch.stack(step_losses).mean()
                scaled_loss = final_loss / self.gradient_accumulation
                self.grad_scaler.scale(scaled_loss).backward()
                final_loss_value = final_loss.detach().item()

                del (
                    output_dict,
                    student_velocities,
                    student_means,
                    teacher_means,
                    per_step_kl,
                    step_loss,
                    scaled_loss,
                )

        metrics_data = {
            "actor/opd_loss": final_loss_value,
            "actor/opd_num_timesteps": float(len(self.opd_train_step_indices)),
            "actor/opd_reverse_kl": torch.stack(teacher_step_losses).mean().item(),
        }
        if isinstance(loss_mask, torch.Tensor):
            effective_rows = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=1)
            metrics_data["actor/opd_effective_row_fraction"] = (
                effective_rows.float().mean().item()
            )
        teacher_denom = max(teacher_ids.numel(), 1)
        teacher_counts = torch.bincount(teacher_ids, minlength=len(self.teacher_models))
        # Log every configured teacher each sub-batch so the final averaged
        # fractions remain normalized and sum to 1 across teachers.
        for teacher_id, teacher_count in enumerate(teacher_counts.tolist()):
            metrics_data[f"actor/teacher_{teacher_id}_fraction"] = float(
                teacher_count / teacher_denom
            )

        metrics_data["actor/total_loss"] = final_loss_value
        append_to_dict(metrics, metrics_data)

    def _precompute_rollout_teacher_means(self) -> None:
        """Cache routed Teacher targets before shuffle and Student updates."""
        forward_inputs = self.rollout_batch["forward_inputs"]
        chains = forward_inputs.get("chains", None)
        teacher_ids = forward_inputs.get("teacher_ids", None)
        if not isinstance(chains, torch.Tensor) or not isinstance(
            teacher_ids, torch.Tensor
        ):
            raise ValueError(
                "Scheduled OPD rollout must contain tensor chains and teacher_ids."
            )
        active_teacher_ids = sorted(
            int(teacher_id) for teacher_id in torch.unique(teacher_ids).tolist()
        )
        self.log_info(
            "Precomputing OPD Teacher targets for active teacher_ids="
            f"{active_teacher_ids}, rows={chains.shape[0]}"
        )
        model_forward_inputs = self._extract_model_forward_inputs(forward_inputs)
        forward_inputs["opd_teacher_means"] = self._compute_teacher_means(
            model_forward_inputs=model_forward_inputs,
            chains=chains,
            teacher_ids=teacher_ids.long().reshape(-1),
        )

    def run_training(self) -> dict[str, float]:
        if not self._use_rollout_epoch_schedule():
            return super().run_training()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        self._precompute_rollout_teacher_means()
        return self._run_scheduled_training()

    def _run_scheduled_training(self) -> dict[str, float]:
        """Run Student updates using cached rollout-level Teacher targets."""
        # Keep the fixed slot x chunk-step layout through shuffle and global
        # batch splitting. Episode padding (and, optionally, every row from a
        # successful episode) remains structural and contributes zero through
        # loss_mask. This mirrors the original Pi-OPD batching behavior.
        rollout_size = self.rollout_batch["forward_inputs"]["chains"].shape[0]
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        # Scheduled horizons can produce different row counts per Env/Actor
        # rank. FSDP collectives require every rank to execute the same number
        # of optimizer batches, so pad all ranks to the global maximum local
        # rollout size before rounding to the per-rank batch size.
        max_rollout_size = rollout_size
        if dist.is_initialized():
            max_size_tensor = torch.tensor(
                [rollout_size], device=self.device, dtype=torch.long
            )
            dist.all_reduce(max_size_tensor, op=dist.ReduceOp.MAX)
            max_rollout_size = int(max_size_tensor.item())
        target_rollout_size = self._scheduled_target_rollout_size(
            rollout_size, max_rollout_size, batch_size_per_rank
        )
        pad_count = target_rollout_size - rollout_size
        if pad_count:
            self.rollout_batch = self._pad_train_samples(self.rollout_batch, pad_count)
        rollout_size = target_rollout_size
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = self._index_train_samples(
                self.rollout_batch, shuffle_id
            )

        assert rollout_size % batch_size_per_rank == 0, (
            "Scheduled OPD rollout size "
            f"{rollout_size} is not divisible by per-rank global batch size "
            f"{batch_size_per_rank}."
        )

        metrics = {}
        optimizer_steps = 0
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                train_global_batch_size = train_global_batch["forward_inputs"][
                    "chains"
                ].shape[0]
                assert train_global_batch_size == batch_size_per_rank, (
                    f"{train_global_batch_size=} != {batch_size_per_rank=}"
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size=}"
                )

                batch_loss_mask = train_global_batch.get("loss_mask", None)
                if isinstance(batch_loss_mask, torch.Tensor):
                    has_effective_rows = torch.tensor(
                        [bool(batch_loss_mask.any().item())],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    if dist.is_initialized():
                        dist.all_reduce(has_effective_rows, op=dist.ReduceOp.MAX)
                    if not bool(has_effective_rows.item()):
                        append_to_dict(
                            metrics, {"actor/opd_skipped_global_batches": 1.0}
                        )
                        continue

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                optimizer_steps += 1
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                append_to_dict(metrics, data)

        if optimizer_steps > 0:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {
            key: sum(value) / len(value) for key, value in metrics.items() if value
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict
