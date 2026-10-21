# Copyright 2025 The RLinf Authors.
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

import asyncio
import copy
import gc
import os
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, Worker, split_channel_message
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.only_eval = cfg.runner.get("only_eval", False)
        self.algorithm_cfg = cfg.get("algorithm", {})
        if str(self.algorithm_cfg.get("loss_type", "")) == "embodied_opd":
            if self.algorithm_cfg.get("opd_occupancy", {}).get("enabled", False):
                raise ValueError("Release OPD supports only online Student rollouts; set opd_occupancy.enabled=false.")
            if cfg.rollout.get("expert_model", None) is not None:
                raise ValueError("Release OPD uses the Student to act; configure target policies under algorithm.opd_teachers.")
        self.model_cfg = cfg.rollout.model if self.only_eval else cfg.actor.model
        self.actor_group_name = (
            cfg.actor.get("group_name", None)
            if cfg.get("actor", None) is not None
            else None
        )
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0
        train_env_cfg = cfg.env.get("train", None)
        eval_env_cfg = cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None

        self.total_num_train_envs = (
            cfg.env.train.total_num_envs if self.enable_train else 0
        )
        self.total_num_eval_envs = (
            cfg.env.eval.total_num_envs if self.enable_eval else 0
        )
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = self.total_num_train_envs // self.num_pipeline_stages
        self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages

        self.per_node_train_batch_size = (
            self.train_batch_size // self._world_size if self.enable_train else 0
        )
        self.per_node_eval_batch_size = (
            self.eval_batch_size // self._world_size if self.enable_eval else 0
        )

        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // self.model_cfg.num_action_chunks
            if self.enable_train
            else 0
        )
        self.train_rollout_epoch_chunk_steps_schedule = (
            self._build_train_rollout_epoch_chunk_steps_schedule()
            if self.enable_train
            else []
        )
        self.train_rollout_epoch_episode_chunk_steps_schedule = (
            self._build_train_rollout_epoch_episode_chunk_steps_schedule()
            if self.enable_train
            else []
        )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None

        self.weight_syncer = None
        self._sync_weight_comm_options = None
        if not self.only_eval:
            weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
            assert weight_syncer_cfg is not None, (
                "rollout.weight_syncer config must be provided"
            )
            self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
            self._sync_weight_comm_options = self.weight_syncer.comm_options

        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # save the run-time imformation in communicate channel for decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {
                "rollout_results": [],
            }
        self.rollout_queue_size = self.cfg.rollout.get("rollout_queue_size", 0)

    def _should_enable_rollout_epoch_schedule(self) -> bool:
        if self.algorithm_cfg.get("rollout_epoch_schedule", None) is None:
            return False

        return str(self.algorithm_cfg.get("loss_type", "")) == "embodied_opd"

    def _build_train_rollout_epoch_chunk_steps_schedule(self) -> list[int]:
        raw_schedule = self.algorithm_cfg.get("rollout_epoch_schedule", None)
        if raw_schedule is None or not self._should_enable_rollout_epoch_schedule():
            return []

        schedule: list[int] = []
        for block_idx, raw_block in enumerate(raw_schedule):
            block_cfg = OmegaConf.to_container(raw_block, resolve=True)
            if "max_steps_per_rollout_epoch" not in block_cfg:
                raise ValueError(
                    "Each rollout_epoch_schedule block must set max_steps_per_rollout_epoch."
                )

            max_steps_per_rollout_epoch = int(block_cfg["max_steps_per_rollout_epoch"])
            if (
                max_steps_per_rollout_epoch
                % self.model_cfg.num_action_chunks
                != 0
            ):
                raise ValueError(
                    "rollout_epoch_schedule["
                    f"{block_idx}].max_steps_per_rollout_epoch="
                    f"{max_steps_per_rollout_epoch} must be divisible by "
                    f"actor.model.num_action_chunks={self.model_cfg.num_action_chunks}."
                )

            schedule.append(
                max_steps_per_rollout_epoch // self.model_cfg.num_action_chunks
            )

        rollout_epoch = int(self.rollout_epoch)
        if len(schedule) == rollout_epoch:
            return schedule

        if rollout_epoch % len(schedule) != 0:
            raise ValueError(
                "env.train.rollout_epoch must either match rollout_epoch_schedule "
                "length or be an integer multiple of it: "
                f"{rollout_epoch} vs {len(schedule)}."
            )

        return schedule * (rollout_epoch // len(schedule))

    def _get_train_chunk_steps_for_epoch(self, epoch: int) -> int:
        if not self.train_rollout_epoch_chunk_steps_schedule:
            return self.n_train_chunk_steps
        return int(self.train_rollout_epoch_chunk_steps_schedule[epoch])

    def _build_train_rollout_epoch_episode_chunk_steps_schedule(self) -> list[int]:
        """Return fixed OPD episode rows from suite max episode steps."""
        raw_schedule = self.algorithm_cfg.get("rollout_epoch_schedule", None)
        chunk_size = int(self.model_cfg.num_action_chunks)
        if raw_schedule is None or not self._should_enable_rollout_epoch_schedule():
            max_episode_steps = int(self.cfg.env.train.max_episode_steps)
            return [max(1, (max_episode_steps + chunk_size - 1) // chunk_size)] * int(
                self.rollout_epoch
            )

        schedule = []
        for block_idx, raw_block in enumerate(raw_schedule):
            block_cfg = OmegaConf.to_container(raw_block, resolve=True)
            max_episode_steps = int(
                block_cfg.get("max_episode_steps", block_cfg["max_steps_per_rollout_epoch"])
            )
            if max_episode_steps <= 0:
                raise ValueError(
                    f"rollout_epoch_schedule[{block_idx}].max_episode_steps must be positive."
                )
            schedule.append(max(1, (max_episode_steps + chunk_size - 1) // chunk_size))

        if len(schedule) == self.rollout_epoch:
            return schedule
        if self.rollout_epoch % len(schedule) != 0:
            raise ValueError(
                "env.train.rollout_epoch must either match rollout_epoch_schedule "
                "length or be an integer multiple of it: "
                f"{self.rollout_epoch} vs {len(schedule)}."
            )
        return schedule * (self.rollout_epoch // len(schedule))

    def _get_train_episode_chunk_steps_for_epoch(self, epoch: int) -> int:
        return int(self.train_rollout_epoch_episode_chunk_steps_schedule[epoch])

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.model_cfg)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)


        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # sampling parameters for rollout
        sampling_params = self.cfg.rollout.get("sampling_params", None)
        if sampling_params is not None:
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            self._train_sampling_params = {
                "do_sample": sampling_params["do_sample"],
                "temperature": sampling_params["temperature_train"]
                if sampling_params["do_sample"]
                else 1.0,
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
            self._eval_sampling_params = {
                "do_sample": True
                if sampling_params.get("temperature_eval", -1) > 0
                else False,
                "temperature": sampling_params["temperature_eval"],
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
        else:
            self._train_sampling_params = {}
            self._eval_sampling_params = {}

        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.algorithm_cfg.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.algorithm_cfg.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: Any,
        *,
        tag: str,
        batch_size: int,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
    ):
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=output_channel,
            data=rollout_result,
            tag=tag,
            async_op=True,
            batch_size=batch_size,
            split_fn=split_fn,
        )

    async def recv_from_and_record_batch_routes_with_timeout(
        self,
        group_name: str,
        channel: Any | None,
        *,
        route_key: Any = None,
        tag: str | None = None,
        batch_size: int | None = None,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
        infer_batch_size_fn: Optional[Callable[[Any], int]] = None,
        timeout_time: float = 0.02,
        recv_queue_size: int = 0,
    ):
        """Receive routed batch shards and record their return routes.

        This method is used in env-decoupled mode. It builds a receive plan for the
        source worker group, receives shard messages from ``channel`` one by one, and
        stops when all planned items are received or ``timeout_time`` is reached.

        Each received channel item must be a dict with ``batch_index`` (route
        metadata) and ``batch`` (payload shard).

        The ``batch_index`` values are stored in ``self.batch_router[tag]`` so a
        later send call can split the response and send each shard back to the original
        source rank. The received payload shards are validated, then merged with
        ``merge_fn`` or the default ``merge_batches``.

        Args:
            group_name: Source worker group name.
            channel: Channel used to receive routed batch shards.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag used to build receive keys and index recorded routes.
            batch_size: Expected batch size for each planned receive entry.
            merge_fn: Optional custom function for merging received shards.
            infer_batch_size_fn: Optional function used to infer shard batch size during
                validation.
            timeout_time: Maximum time in seconds to wait before finalizing partial
                results.
            recv_queue_size: Number of receive queue entries used when building the
                receive plan.

        Returns:

            A merged payload and its split sizes. If only one shard is received, the
            current implementation returns that shard directly.
        """
        from rlinf.scheduler import (
            decoupled_build_recv_plan,
            get_batch_size,
            get_group_world_size,
            merge_batches,
        )

        world_size = get_group_world_size(self._manager_proxy, group_name)
        plan = decoupled_build_recv_plan(
            src_group_name=group_name,
            dst_group_name=self.worker_address.root_group_name,
            recv_rank=None,
            src_world_size=self._world_size,
            dst_world_size=world_size,
            tag=tag,
            route_key=route_key,
            batch_size=batch_size,
            recv_queue_size=recv_queue_size,
        )

        def _finalize(received_items: list[Any]):
            if not received_items:
                assert False, "received_items is empty"

            # get the tag from the received_items
            _, _, _, tag = split_channel_message(received_items[0]["batch_index"])

            assert tag in self.batch_router, (
                f"{tag=} need to be already in the batch_router"
            )
            # Save the batch_index to the batch_router.
            list_received_items = []
            for item in received_items:
                batch_index = item["batch_index"]
                received_item = item["batch"]
                list_received_items.append(received_item)
                # Save the batch_index to the batch_router.
                self.batch_router[tag].append(batch_index)
            received_items = list_received_items
            split_sizes = [
                get_batch_size(item, infer_batch_size_fn) for item in received_items
            ]

            if merge_fn is not None:
                return merge_fn(received_items), split_sizes
            if len(received_items) == 1:
                return received_items[0]
            return merge_batches(received_items), split_sizes

        timeout_time = timeout_time + time.time()
        get_items = None
        max_item_num = len(plan.entries)
        get_item_num = 0
        received_items = []
        while get_item_num < max_item_num:
            # get the items
            if get_items is None:
                get_items = channel.get(
                    key=plan.entries[get_item_num].key, async_op=True
                )
            else:
                # Now, the worker is getting a item, sleep to wait
                await asyncio.sleep(0.0001)

            # handle the get_items finish
            if get_items.done():
                # save the data and init the get_items to get next data
                received_items.append(await get_items.async_wait())
                get_items = None
                get_item_num = get_item_num + 1

            # handle the timeout case
            if time.time() >= timeout_time:
                max_item_num = get_item_num
                if get_items is not None:
                    received_items.append(await get_items.async_wait())
                    get_items = None
                    get_item_num = get_item_num + 1

        return _finalize(received_items)

    def send_to_recorded_batch_routes(
        self,
        group_name: str,
        channel: Any | None,
        data: Any,
        *,
        route_key: Any = None,
        tag: str | None = None,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
        split_sizes: list[int],
    ):
        """Send split batch results back using recorded batch routes.

        This method is used after a previous receive call has populated
        ``self.batch_router[tag]`` with batch indices from incoming messages.
        The outgoing ``data`` is split according to ``split_sizes`` and each shard is
        sent to the rank encoded in the corresponding recorded batch index.

        Each outgoing channel item is a dict with ``batch_index`` (recorded batch
        index) and ``batch`` (payload shard).

        After all shards are queued, the recorded batch indices for ``tag`` are cleared
        to avoid reusing stale routes.

        Args:
            group_name: Destination worker group name.
            channel: Channel used to send the split payloads.
            data: Payload to split and send.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag whose recorded batch indices should be consumed.
            split_fn: Optional custom splitter. If omitted, ``split_batch`` is used.
            split_sizes: Batch sizes used to split ``data``. Must have the same length
                as ``self.batch_router[tag]``.

        Returns:

            AsyncRouteWork wrapping the async channel put operations.
        """
        from rlinf.scheduler import build_send_key, split_batch
        from rlinf.scheduler.collective import AsyncRouteWork

        assert tag in self.batch_router, (
            f"{tag=} need to be already in the batch_router"
        )

        assert len(self.batch_router[tag]) > 0, f"{self.batch_router[tag]=} is empty"
        assert len(self.batch_router[tag]) == len(split_sizes), (
            f"{self.batch_router[tag]=} length should equal {split_sizes=} length"
        )

        payloads = (
            split_fn(data, split_sizes)
            if split_fn is not None
            else split_batch(data, split_sizes)
        )

        works = []
        for i, payload in enumerate(payloads):
            batch_index = self.batch_router[tag][i]
            send_rank, _, mode, _ = split_channel_message(batch_index)
            # After enabling env_decoupled_mode, the data sending format is as follows:
            # {
            #     "batch_index": batch_index,
            #     "batch": batch,
            # }
            # The batch_index is the index of the batch in the data.
            # The batch is the data to send.
            # batch_index: {send_rank}_{batch_idx}_{mode}_{tag}
            # The send_rank is the rank of the worker that originally sent the data.
            # The batch_idx is the index of the batch in the data.
            # The tag is the tag of the data.
            senditem = {
                "batch_index": batch_index,
                "batch": payload,
            }
            key = build_send_key(
                src_group_name=self.worker_address.root_group_name,
                dst_group_name=group_name,
                src_rank=None,
                dst_rank=send_rank,
                tag=tag if mode is None else f"{mode}_{tag}",
                route_key=route_key,
            )
            work = channel.put(
                item=senditem,
                key=key,
                async_op=True,
            )
            works.append(work)

        # clear the batch_router for the next send
        self.batch_router[tag] = []
        return AsyncRouteWork(works, lambda _: None)

    async def recv_env_output(
        self,
        input_channel: Channel,
        *,
        tag: str,
        batch_size: int,
    ):
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=input_channel,
            tag=tag,
            async_op=True,
            batch_size=batch_size,
            merge_fn=self._merge_obs_batches,
            infer_batch_size_fn=self._infer_env_batch_size,
        ).async_wait()

    def update_dagger_beta(self):
        if self.expert_model is None:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )


    @staticmethod
    def _to_cpu_nested(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, dict):
            return {key: MultiStepRolloutWorker._to_cpu_nested(item) for key, item in value.items()}
        if isinstance(value, list):
            return [MultiStepRolloutWorker._to_cpu_nested(item) for item in value]
        return value


    @staticmethod
    def _replace_rows(
        base: Any, replacement: Any, mask: torch.Tensor
    ) -> Any:
        """Replace selected batch rows with cached rows recursively."""
        if isinstance(base, torch.Tensor) and isinstance(replacement, torch.Tensor):
            if base.ndim == 0 or base.shape[0] != mask.numel():
                return base
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                return base
            output = base.clone()
            output.index_copy_(
                0,
                indices.to(output.device),
                replacement.index_select(0, indices.to(replacement.device)).to(
                    device=output.device, dtype=output.dtype
                ),
            )
            return output
        if isinstance(base, dict) and isinstance(replacement, dict):
            return {
                key: MultiStepRolloutWorker._replace_rows(
                    base_value, replacement.get(key, base_value), mask
                )
                for key, base_value in base.items()
            }
        if isinstance(base, list) and isinstance(replacement, list):
            if len(base) != mask.numel() or len(replacement) != mask.numel():
                return base
            output = list(base)
            for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                output[index] = replacement[index]
            return output
        return base




    @Worker.timer("predict")
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
        ]:
            loss_type = self.algorithm_cfg.get("loss_type", "actor")
            if loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.algorithm_cfg.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(
        self,
        input_channel: Channel,
        output_channel: Channel,
        n_train_chunk_steps: int | None = None,
    ):
        self.update_dagger_beta()
        if n_train_chunk_steps is None:
            n_train_chunk_steps = self.n_train_chunk_steps
        local_batch_size = int(
            getattr(self, "per_node_train_batch_size", self.train_batch_size)
        )
        inactive_masks = [
            torch.zeros(local_batch_size, dtype=torch.bool)
            for _ in range(self.num_pipeline_stages)
        ]
        last_actions: list[torch.Tensor | None] = [None] * self.num_pipeline_stages
        last_results: list[dict[str, Any] | None] = [None] * self.num_pipeline_stages

        for _ in range(n_train_chunk_steps):
            for stage_idx in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(
                    input_channel=input_channel,
                    tag="train_rollout_results",
                    batch_size=self.train_batch_size,
                )
                env_dones = env_output.get("dones", None)
                if env_dones is not None:
                    inactive_masks[stage_idx] |= env_dones.any(dim=-1).cpu()
                actions, result = self.predict(env_output["obs"])
                padding_mask = inactive_masks[stage_idx]
                if (
                    padding_mask.any()
                    and last_actions[stage_idx] is not None
                    and last_results[stage_idx] is not None
                ):
                    actions = self._replace_rows(
                        actions,
                        last_actions[stage_idx],
                        padding_mask,
                    )
                    result["forward_inputs"] = self._replace_rows(
                        result["forward_inputs"],
                        last_results[stage_idx]["forward_inputs"],
                        padding_mask,
                    )
                    for result_key in ("prev_logprobs", "prev_values"):
                        if result.get(result_key) is not None:
                            result[result_key] = self._replace_rows(
                                result[result_key],
                                last_results[stage_idx].get(result_key),
                                padding_mask,
                            )
                last_actions[stage_idx] = actions.detach().cpu().contiguous()
                last_results[stage_idx] = self._to_cpu_nested(
                    {
                        "forward_inputs": result["forward_inputs"],
                        "prev_logprobs": result.get("prev_logprobs"),
                        "prev_values": result.get("prev_values"),
                    }
                )

                save_flags = None
                if result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.model_cfg.num_action_chunks),
                        True,
                        dtype=torch.bool,
                        device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result["prev_values"]
                    if self.collect_prev_infos
                    else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    forward_inputs=result["forward_inputs"],
                    versions=(
                        torch.full_like(
                            result["prev_logprobs"],
                            float(self.version),
                            dtype=torch.float32,
                        )
                        if result.get("prev_logprobs") is not None
                        else None
                    ),
                )
                self.send_rollout_result(
                    output_channel=output_channel,
                    rollout_result=rollout_result,
                    tag="train_rollout_results",
                    batch_size=self.train_batch_size,
                    split_fn=self._split_rollout_result,
                )
        for stage_idx in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(
                input_channel=input_channel,
                tag="train_rollout_results",
                batch_size=self.train_batch_size,
            )
            env_dones = env_output.get("dones", None)
            if env_dones is not None:
                inactive_masks[stage_idx] |= env_dones.any(dim=-1).cpu()
            actions, result = self.predict(env_output["obs"])
            padding_mask = inactive_masks[stage_idx]
            if (
                padding_mask.any()
                and last_actions[stage_idx] is not None
                and last_results[stage_idx] is not None
            ):
                actions = self._replace_rows(
                    actions,
                    last_actions[stage_idx],
                    padding_mask,
                )
                result["forward_inputs"] = self._replace_rows(
                    result["forward_inputs"],
                    last_results[stage_idx]["forward_inputs"],
                    padding_mask,
                )
                for result_key in ("prev_logprobs", "prev_values"):
                    if result.get(result_key) is not None:
                        result[result_key] = self._replace_rows(
                            result[result_key],
                            last_results[stage_idx].get(result_key),
                            padding_mask,
                        )

            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                forward_inputs=result["forward_inputs"],
            )
            self.send_rollout_result(
                output_channel=output_channel,
                rollout_result=rollout_result,
                tag="train_rollout_results",
                batch_size=self.train_batch_size,
                split_fn=self._split_rollout_result,
            )

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            n_train_chunk_steps = self._get_train_chunk_steps_for_epoch(_)
            await self.generate_one_epoch(
                input_channel,
                output_channel,
                n_train_chunk_steps=n_train_chunk_steps,

            )

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        if self.env_decoupled_mode:
            while True:
                (
                    env_output,
                    split_sizes,
                ) = await self.recv_from_and_record_batch_routes_with_timeout(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    batch_size=self.eval_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                    timeout_time=0.02,
                    recv_queue_size=self.rollout_queue_size,
                )
                actions, _ = self.predict(env_output["obs"], mode="eval")
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                self.send_to_recorded_batch_routes(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=actions,
                    tag="rollout_results",
                    split_sizes=split_sizes,
                )
        else:
            for _ in tqdm(
                range(self.eval_rollout_epoch),
                desc="Evaluating Rollout Epochs",
                disable=(self._rank != 0),
            ):
                for _ in range(self.n_eval_chunk_steps):
                    for _ in range(self.num_pipeline_stages):
                        env_output = await self.recv_env_output(
                            input_channel=input_channel,
                            tag="eval_rollout_results",
                            batch_size=self.eval_batch_size,
                        )
                        actions, _ = self.predict(env_output["obs"], mode="eval")
                        if isinstance(actions, torch.Tensor):
                            actions = actions.detach().cpu().contiguous()
                        self.send_rollout_result(
                            output_channel=output_channel,
                            rollout_result=actions,
                            tag="eval_rollout_results",
                            batch_size=self.eval_batch_size,
                        )

            if self.enable_offload:
                self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)
        dones = [obs_batch.get("dones", None) for obs_batch in obs_batches]
        merged_dones = None
        if any(done is not None for done in dones):
            if any(done is None for done in dones):
                raise ValueError("Inconsistent env output: some shards omit dones.")
            merged_dones = torch.cat(dones, dim=0)

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "dones": merged_dones,
        }

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )
        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
