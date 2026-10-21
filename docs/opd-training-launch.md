# OPD training: direct launch

Launch the common RLinf entry point with an OPD config under
`examples/embodiment/config/opd/libero/`. Make the selected GPUs, placement,
runtime environment, config, and overrides explicit.

## Standard eight-GPU launch

Run from the repository root:

```bash
unset DISPLAY WAYLAND_DISPLAY PYTHONHOME PYTHONPATH

VENV_PATH=/path/to/your/venv \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OPD_GPU_PLACEMENT=0-7 \
RAY_BIN=/path/to/your/venv/bin/ray \
RAY_PORT=6669 \
DASHBOARD_PORT=8992 \
RESTART_RAY=0 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
ROBOT_PLATFORM=LIBERO \
LIBERO_TYPE=standard \
bash examples/embodiment/run_embodiment.sh \
  <hydra-config> LIBERO \
  env.train.video_cfg.save_video=false
```

`run_embodiment.sh` activates `VENV_PATH`, starts Ray when `RESTART_RAY=1`,
creates the timestamped log directory, and invokes Hydra. Do not start a
second such job on the same host while it is using the same Ray port.
The example assumes an existing cluster and an explicitly exported
`RAY_ADDRESS=127.0.0.1:YOUR_RAY_PORT`. `RESTART_RAY=1` invokes the wrapper's
global Ray cleanup; use `RESTART_RAY=0` for an existing shared-machine cluster.

## Retained LIBERO OPD configs

| Purpose | `<hydra-config>` | Extra overrides |
| --- | --- | --- |
| Main four-suite, routed PPO Teachers | `opd/libero/libero_sog_plus10_opd_ppo120_teacher_vlm_save50_x5lr` | — |
| Spatial OPD with reference preservation | `opd/libero/spatial_opd_objkl` | — |

The main recipe is the four-suite S/O/G + LIBERO-10 run. The reference configs
use frozen Base SFT as a routed Teacher, with one OPD loss per sample. Evaluation
configs live under `evaluations/libero/`.
