# VLA-MOPD: On-Policy Distillation for Multi-Task VLA Policy Consolidation

**On-policy distillation for multi-task VLA policy consolidation.** The Student collects its own LIBERO rollouts; task-routed frozen specialists provide transition targets at intermediate flow states. The release includes the global-step-500 Pi0.5 Student checkpoint with a trained language-model-only LoRA adapter (rank 32).

[English](README.md) · [简体中文](README.zh-CN.md) · [Method and evaluation details](docs/opd-gs500.md) · [Training configs](examples/embodiment/config/opd/libero/)

## Framework

![VLA-MOPD framework: student rollout, routed teacher transition targets, and student update](docs/libero-opd-method.png)

## LIBERO results

Success-once rate (%) for the released global-step-500 OPD Student, with 500 episodes per suite.

| Method | Spatial | Object | Goal | LIBERO-10 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| **VLA-MOPD (OPD Student, gs500)** | **99.4** | **99.0** | **98.0** | **99.0** | **98.85** |

The unweighted four-suite mean is 98.85% (1977/2000 successful episodes). Results are from one training seed. The metric is success-once. Archived results and evaluation configs: [evaluation details](docs/opd-gs500.md).

## Quick start: evaluate the released Student

### Released checkpoints

| Role | Hugging Face model | Used for |
| --- | --- | --- |
| Student initialization / Base SFT | [`RLinf/RLinf-Pi05-LIBERO-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT) | Student initialization and shared LIBERO normalization statistics |
| S/O/G Teacher | [`yxuan1234/Pi05-LIBERO-SOG-Teacher`](https://huggingface.co/yxuan1234/Pi05-LIBERO-SOG-Teacher) | Shared frozen PPO Teacher for Spatial, Object, and Goal |
| LIBERO-10 Teacher | [`yxuan1234/Pi05-LIBERO10-Teacher`](https://huggingface.co/yxuan1234/Pi05-LIBERO10-Teacher) | Frozen PPO Teacher for LIBERO-10 |
| OPD Student, global step 500 | [`yxuan1234/Pi05-LIBERO-OPD-Student-gs500`](https://huggingface.co/yxuan1234/Pi05-LIBERO-OPD-Student-gs500) | Released distilled policy and the evaluation results above |

The three `yxuan1234` model repositories are private and require authorized HF access. The RLinf Base SFT is public.

### 1. Install RLinf OpenPI + LIBERO

Follow the [official RLinf installation guide](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html), then from the repository root run:

```bash
bash requirements/install.sh embodied --model openpi --env libero
```

Prepare LIBERO assets and initial states following the [RLinf LIBERO guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html).

### 2. Download checkpoints and configure paths

Download the base and both Teachers for training, or only the Base SFT and OPD Student for evaluation. The LIBERO-10 Teacher checkpoint is stored at the repository root (under `model_state_dict/`), so use that local repository directory as its model path.

```bash
hf download RLinf/RLinf-Pi05-LIBERO-SFT \
  --local-dir /path/to/RLinf-Pi05-LIBERO-SFT
hf download yxuan1234/Pi05-LIBERO-SOG-Teacher \
  --local-dir /path/to/Pi05-LIBERO-SOG-Teacher
hf download yxuan1234/Pi05-LIBERO10-Teacher \
  --local-dir /path/to/Pi05-LIBERO10-Teacher
hf download yxuan1234/Pi05-LIBERO-OPD-Student-gs500 \
  --revision 282d6ce8dd9c1ce34ea2bfe06fb88a40cee52cdf \
  --local-dir /path/to/Pi05-LIBERO-OPD-Student-gs500

export PI05_SFT_MODEL_DIR=/path/to/RLinf-Pi05-LIBERO-SFT
export PI05_SOG_TEACHER_DIR=/path/to/Pi05-LIBERO-SOG-Teacher
export PI05_LIBERO10_PPO_TEACHER_DIR=/path/to/Pi05-LIBERO10-Teacher
export PI05_OPD_CHECKPOINT_DIR=/path/to/Pi05-LIBERO-OPD-Student-gs500/actor
export EMBODIED_PATH="$PWD/examples/embodiment"
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The Base SFT supplies the Student initialization and LIBERO normalization statistics. Connect to a Ray cluster allocated to your job; placement values are physical GPU ranks.

### Reproduce OPD training through global step 500

The main routed-teacher recipe is `libero_sog_plus10_opd_ppo120_teacher_vlm_save50_x5lr`. It routes one S/O/G Teacher to Spatial/Object/Goal and the LIBERO-10 Teacher to LIBERO-10.

```bash
export RAY_ADDRESS=127.0.0.1:YOUR_RAY_PORT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPD_GPU_PLACEMENT=0-7

python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name opd/libero/libero_sog_plus10_opd_ppo120_teacher_vlm_save50_x5lr \
  runner.max_steps=500 \
  runner.logger.log_path=/path/to/training-results
```

This starts from the public Base SFT and trains the language-only LoRA32 Student while the full action expert is trainable. The config saves checkpoints every 50 steps. Use the same model, simulator, and rollout settings for a comparable run; exact results may vary by environment and seed.

### Teacher-as-reference OPD: Spatial adaptation with Object retention

Config: [`spatial_opd_objkl.yaml`](examples/embodiment/config/opd/libero/spatial_opd_objkl.yaml).
This example collects online Student rollouts on Spatial and Object and selects
one frozen Teacher per sample:

| Suite | Routed Teacher | Purpose |
| --- | --- | --- |
| Spatial | `PI05_SOG_TEACHER_DIR` (PPO specialist) | Learn the specialist's Spatial capability |
| Object | `PI05_SFT_MODEL_DIR` (frozen Base SFT) | Retain the starting policy's Object capability |

Both suites use the same OPD step-mean matching loss. Each sample has exactly
one routed target; no additional reference loss is added. The historical
`objkl` config name refers to Base SFT acting as the Object Teacher.

After installing the environment and downloading the Base SFT and S/O/G
Teacher above, run:

```bash
export RAY_ADDRESS=127.0.0.1:YOUR_RAY_PORT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPD_GPU_PLACEMENT=0-7

python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name opd/libero/spatial_opd_objkl \
  runner.logger.log_path=/path/to/reference-opd-results
```

The config sets the suite schedule, episode horizons, training budget, and
50-step checkpoint interval. This is a separate adaptation/retention recipe
from the four-suite gs500 consolidation run. See [reference routing details](docs/opd-reference-kl.md).

### 3. Evaluate all four suites

Each suite config in `evaluations/libero/` already defines its horizon and rollout length. This command supplies the archived sampler settings:

```bash
for suite in spatial object goal 10; do
  python evaluations/eval_embodied_agent.py \
    --config-path libero \
    --config-name "libero_${suite}_opd_vlm_lora_eval" \
    env.eval.seed=0 \
    env.eval.use_ordered_reset_state_ids=false \
    rollout.model.openpi.noise_method=flow_sde \
    rollout.model.openpi.noise_level=0.3 \
    rollout.model.openpi.train_expert_only=true \
    runner.logger.log_path="/path/to/evaluation-results/gs500/$suite" || break
done
```

The evaluator restores the full checkpoint and merges LoRA in memory; no separate conversion or merge step is needed. See [the full evaluation guide](docs/opd-gs500.md) and [training configs](docs/opd-training-launch.md).

This implementation is built on [RLinf](https://github.com/RLinf/RLinf); see the [RLinf paper](https://arxiv.org/abs/2509.15965) for the upstream framework. The VLA-MOPD preprint link will be added when available.
