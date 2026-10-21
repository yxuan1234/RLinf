# Configuring local paths for release examples

The release configs do not embed developer-machine paths. Model, dataset, and
asset locations are supplied through Hydra environment resolvers. Set only the
variables required by the config you run.

## LIBERO simulator setup

The release does not vendor simulator repositories or large benchmark assets.
Install the official RLinf LIBERO environment into the active virtual
environment from the repository root:

```bash
bash requirements/install.sh embodied --model openpi --env libero
```

The installer clones `https://github.com/RLinf/LIBERO.git` into the virtual
environment by default. To reuse an existing official checkout, set
`LIBERO_PATH` before running the installer. Follow the LIBERO documentation for
benchmark assets and initial-state data; do not commit those assets into this
repository.

## Pi05 LIBERO OPD

```bash
export PI05_SFT_MODEL_DIR=/path/to/RLinf-Pi05-LIBERO-SFT
export PI05_SOG_TEACHER_DIR=/path/to/Pi05-LIBERO-SOG-Teacher
export PI05_LIBERO10_PPO_TEACHER_DIR=/path/to/Pi05-LIBERO10-Teacher
export EMBODIED_PATH="$PWD/examples/embodiment"
```

The main routed-teacher config uses `PI05_SOG_TEACHER_DIR` for
Spatial/Object/Goal and `PI05_LIBERO10_PPO_TEACHER_DIR` for LIBERO-10. Each
sample uses only the OPD loss of its routed Teacher. For Spatial
adaptation with Object retention, use `spatial_opd_objkl`: Spatial routes to
the S/O/G PPO Teacher and Object routes to frozen Base SFT.

## Evaluation configs

Set the corresponding model directory variables before composing an evaluation
config. For an OPD checkpoint, pass the checkpoint separately through
`checkpoint_path`; keep `model_path` pointed at the base model and
`norm_stats_dir` pointed at its normalization assets.

## Release scope

This release exposes the OPD implementation and training recipes for LIBERO.
The shared RLinf environment support remains in the repository, but RoboTwin
OPD training configurations are intentionally excluded from this release.
