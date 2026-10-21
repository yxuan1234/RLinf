# Evaluating LIBERO OPD VLM LoRA checkpoints

Use the dedicated configs in `evaluations/libero/`:

| Suite | Config | Horizon |
| --- | --- | --- |
| Spatial | `libero_spatial_opd_vlm_lora_eval` | 240 |
| Object | `libero_object_opd_vlm_lora_eval` | 280 |
| Goal | `libero_goal_opd_vlm_lora_eval` | 320 |
| LIBERO-10 | `libero_10_opd_vlm_lora_eval` | 520 |

These recipes match language-model-only VLM LoRA rank 32, five flow-ODE
denoising steps and ten-action chunks. Match these settings to the training
run if you change the recipe. Each suite defaults to 500 episodes with fixed
reset state IDs and video disabled. The archived gs500 command overrides the
sampler to `flow_sde` with noise level 0.3 to match the reported run.

## Checkpoint inputs

```bash
export PI05_SFT_MODEL_DIR=/path/to/RLinf-Pi05-LIBERO-SFT
export PI05_OPD_CHECKPOINT_DIR=/path/to/checkpoints/global_step_200/actor
export EMBODIED_PATH="$PWD/examples/embodiment"
export OPD_GPU_PLACEMENT=0
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

`PI05_OPD_CHECKPOINT_DIR` must contain `model_state_dict/full_weights.pt`,
or be its parent checkpoint directory containing `actor/model_state_dict/full_weights.pt`.
It must be the full unmerged FSDP checkpoint, including trained LoRA tensors
and action-expert weights. `model_path` remains the original base-model directory;
normalization statistics are read from that base directory.

The loader constructs the model, injects language-only LoRA, restores the full
checkpoint again after injection, then calls `merge_and_unload(safe_merge=True)`
on the nested `paligemma_with_expert.paligemma` PEFT model. It freezes the merged
model for inference. The merge happens in memory and does not overwrite the
training checkpoint. An invalid explicit checkpoint path, incompatible adapter
shape, or a merge-evaluation checkpoint with no LoRA tensors raises an error.

## Launch

Connect to your own explicitly addressed Ray cluster, configured with the same
physical GPUs as `OPD_GPU_PLACEMENT`. On shared servers use an isolated Ray
port, GPU allocation and temporary directory. Then invoke the entry point:

```bash
export RAY_ADDRESS=127.0.0.1:YOUR_RAY_PORT
python evaluations/eval_embodied_agent.py \
  --config-path libero \
  --config-name libero_spatial_opd_vlm_lora_eval \
  runner.logger.log_path=/path/to/eval-results
```

To compare unmerged PEFT inference with merged inference, set
`rollout.model.merge_lora_for_eval=false`; keep `is_lora=true` and the same
checkpoint. For an already merged safetensors export instead, set `model_path`
to that export, `checkpoint_path=null`, `is_lora=false`, and
`merge_lora_for_eval=false` to avoid creating a new random adapter.
