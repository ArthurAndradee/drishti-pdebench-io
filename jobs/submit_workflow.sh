#!/bin/bash
set -euo pipefail

# Model & Training Params
MODEL_NAME="Unet"
TRAIN_VARIANT="ARB"   # STD | ARB | BRO | MPI
DATASET="advection"
DATASET_PREFIX="1D_Advection_Sols_beta"
CONFIG_FILE="config_Adv.yaml"
BATCH_SIZE="512"
EPOCHS_PER_FILE="500"

# Paths & Directories
DATA_PATH="/scratch/aadsilva/erad-2026/PDEBench/pdebench/data_download/advection/1D/Advection/Train/"
TRAINED_MODELS_DIR="trained-models"
DARSHAN_ROOT="/scratch/aadsilva/darshan/install"
DARSHAN_LOGDIR="darshan-logs"  # Directory to store darshan logs
CONDA_ENV_PATH="/scratch/aadsilva/conda/envs/pde_env"

# Job Control
RUN_NAME="${MODEL_NAME}_${DATASET}_${TRAIN_VARIANT}"
NUM_RUNS=10

# Create local directories if they don't exist (to avoid failures on compute nodes)
mkdir -p "${DARSHAN_LOGDIR}"
mkdir -p "${TRAINED_MODELS_DIR}"

PREV_TRAIN_JOB=""

for RUN_INDEX in $(seq 1 "${NUM_RUNS}"); do
    echo "🚀 Submitting training run ${RUN_INDEX}/${NUM_RUNS} (${TRAIN_VARIANT})"

    # We export ALL variables defined above to the train script
    TRAIN_JOB_ID=$(sbatch \
        ${PREV_TRAIN_JOB:+--dependency=afterok:${PREV_TRAIN_JOB}} \
        --job-name="${RUN_NAME}_train" \
        --export=ALL,\
RUN_NAME="${RUN_NAME}",\
MODEL_NAME="${MODEL_NAME}",\
DATASET_PREFIX="${DATASET_PREFIX}",\
RUN_INDEX="${RUN_INDEX}",\
TRAIN_VARIANT="${TRAIN_VARIANT}",\
CONFIG_FILE="${CONFIG_FILE}",\
BATCH_SIZE="${BATCH_SIZE}",\
EPOCHS_PER_FILE="${EPOCHS_PER_FILE}",\
DATA_PATH="${DATA_PATH}",\
TRAINED_MODELS_DIR="${TRAINED_MODELS_DIR}",\
DARSHAN_ROOT="${DARSHAN_ROOT}",\
DARSHAN_LOGDIR="${DARSHAN_LOGDIR}",\
CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
        train.sh | awk '{print $4}')

    echo "🧠 Training job ID: ${TRAIN_JOB_ID}"

    PREV_TRAIN_JOB="${TRAIN_JOB_ID}"
done

echo "✅ Workflow submitted (${NUM_RUNS} sequential training runs)"