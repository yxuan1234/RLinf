#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_async.py"
export RAY_BIN="${RAY_BIN:-ray}"
export RAY_PORT="${RAY_PORT:-6669}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8992}"
export RAY_START_LOG="${REPO_PATH}/logs/ray_start.log"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

cleanup_ray() {
    echo "[INFO] Cleaning old Ray processes..."
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "raylet|gcs_server|dashboard|monitor.py|runtime_env|plasma" >/dev/null 2>&1 || true
    rm -rf /tmp/ray /dev/shm/ray >/dev/null 2>&1 || true
    sleep 5
}

wait_ray_ready() {
    local retries="${1:-60}"
    local interval="${2:-2}"

    for ((i=1; i<=retries; i++)); do
        if "${RAY_BIN}" status >/dev/null 2>&1; then
            return 0
        fi
        sleep "${interval}"
    done

    return 1
}

start_ray() {
    echo "[INFO] Starting Ray head on port ${RAY_PORT}..."
    mkdir -p "${REPO_PATH}/logs"
    "${RAY_BIN}" start --head \
        --port="${RAY_PORT}" \
        --dashboard-port="${DASHBOARD_PORT}" \
        --include-dashboard=True \
        --disable-usage-stats \
        >"${RAY_START_LOG}" 2>&1

    if wait_ray_ready 60 2; then
        echo "[INFO] Ray cluster is ready."
    else
        echo "[ERROR] Ray cluster failed to become healthy."
        echo "========== ${RAY_START_LOG} =========="
        cat "${RAY_START_LOG}" || true
        echo "====================================="
        exit 1
    fi

    sleep 5
}

if [ "${RESTART_RAY:-1}" = "1" ]; then
    cleanup_ray
    start_ray
fi

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_sac_mlp_async"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
