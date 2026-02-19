#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --partition=tupi
#SBATCH --nodelist=tupi6

set -euo pipefail

# -------------------------
# Required variables (Enforce passing from submit_workflow.sh)
# -------------------------
: "${RUN_NAME:?RUN_NAME not set}"
: "${RUN_INDEX:?RUN_INDEX not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
: "${DATASET_PREFIX:?DATASET_PREFIX not set}"
: "${TRAIN_VARIANT:?TRAIN_VARIANT not set}"
: "${CONFIG_FILE:?CONFIG_FILE not set}"
: "${DATA_PATH:?DATA_PATH not set}"
: "${BATCH_SIZE:?BATCH_SIZE not set}"
: "${EPOCHS_PER_FILE:?EPOCHS_PER_FILE not set}"
: "${TRAINED_MODELS_DIR:?TRAINED_MODELS_DIR not set}"
: "${DARSHAN_ROOT:?DARSHAN_ROOT not set}"
: "${DARSHAN_LOGDIR:?DARSHAN_LOGDIR not set}"
: "${CONDA_ENV_PATH:?CONDA_ENV_PATH not set}"

echo "🧪 Starting training run ${RUN_INDEX}"
echo "🔎 Darshan log dir: ${DARSHAN_LOGDIR}"
echo "📂 Data path: ${DATA_PATH}"

# -------------------------
# Environment
# -------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /scratch/aadsilva/erad-2026/PDEBench/pdebench/models
mkdir -p "${TRAINED_MODELS_DIR}"

# -------------------------
# Darshan Setup
# -------------------------
export LD_LIBRARY_PATH="$DARSHAN_ROOT/lib:${LD_LIBRARY_PATH:-}"
export DARSHAN_LOGDIR="${DARSHAN_LOGDIR}"
export DARSHAN_ENABLE_NONMPI=1
export DXT_ENABLE_IO_TRACE=1
export LD_PRELOAD="$DARSHAN_ROOT/lib/libdarshan.so"
# Add Job ID and Run Index to log hints to prevent overwrites
export DARSHAN_LOG_HINTS="jobid=${SLURM_JOB_ID}_run${RUN_INDEX}"

# -------------------------
# Dataset discovery
# -------------------------
mapfile -t DATAFILES < <(
  ls "${DATA_PATH}/${DATASET_PREFIX}"*.h5 \
     "${DATA_PATH}/${DATASET_PREFIX}"*.hdf5 2>/dev/null | sort
)

if [[ ${#DATAFILES[@]} -eq 0 ]]; then
  echo "❌ No data files found in ${DATA_PATH} with prefix ${DATASET_PREFIX}"
  exit 1
fi

COMMON_ARGS=(
    "+args=${CONFIG_FILE}"
    "++args.model_name=${MODEL_NAME}"
    "++args.data_path=${DATA_PATH}"
    "++args.batch_size=${BATCH_SIZE}"
)

TRAIN_SCRIPT="train_models_forward_${TRAIN_VARIANT}.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "❌ Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

echo "🧠 Training variant : ${TRAIN_VARIANT}"
echo "📄 Training script  : ${TRAIN_SCRIPT}"

# -------------------------
# Independent Training Loop
# -------------------------
for (( i=0; i<${#DATAFILES[@]}; i++ )); do
    FULL_PATH="${DATAFILES[$i]}"
    FILENAME="$(basename "${FULL_PATH}")"
    BASE_NAME="${FILENAME%.*}" # Remove extension (.h5/.hdf5)
    
    echo "---------------------------------------------------"
    echo "🔄 Training Independent Model for: ${FILENAME}"
    echo "---------------------------------------------------"

    # Execute training forcing fresh start
    python "${TRAIN_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        "++args.filename=${FILENAME}" \
        "++args.epochs=${EPOCHS_PER_FILE}" \
        "++args.if_training=true" \
        "++args.continue_training=false"

    # Capture generated model
    GENERATED_MODEL=$(ls -t *.pt | head -n 1)

    if [[ -f "${GENERATED_MODEL}" ]]; then
        TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
        
        # Rename logic: Architecture_DatasetFile_Timestamp_RunID.pt
        FINAL_NAME="${MODEL_NAME}_${BASE_NAME}_${TIMESTAMP}_Run${RUN_INDEX}.pt"
        
        mv "${GENERATED_MODEL}" "${TRAINED_MODELS_DIR}/${FINAL_NAME}"
        echo "✅ Model saved: ${FINAL_NAME}"
    else
        echo "⚠️ Warning: No .pt file found after training ${FILENAME}"
    fi
    
    # Cleanup pickles
    rm -f *.pickle
done

echo "✅ All independent trainings completed for Run ${RUN_INDEX}"