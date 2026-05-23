#!/bin/bash
#
# Single-node 32B benchmark — TP=4, 4 GPUs, intra-node tensor parallelism
#
# Usage: ./launch_32b.sh <mode> [steps] [--liger] [--flash-attn] [--compile] [--cce]
#
# Modes:  throughput  (50 steps)
#         train       (N steps)
#
# Examples:
#   ./launch_32b.sh throughput
#   ./launch_32b.sh throughput 50 --liger
#   ./launch_32b.sh throughput 50 --liger --flash-attn

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch_32b.sh <mode> [steps] [--flags]}

# Parse optional flags
USE_LIGER=false
USE_FLASH_ATTN=false
USE_COMPILE=false
USE_CCE=false
USE_NSYS=false
for arg in "${@:2}"; do
    case $arg in
        --liger) USE_LIGER=true ;;
        --flash-attn) USE_FLASH_ATTN=true ;;
        --compile) USE_COMPILE=true ;;
        --cce) USE_CCE=true ;;
        --nsys) USE_NSYS=true ;;
    esac
done

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=50
        NODES=1
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${2:?Usage: ./launch_32b.sh train <steps>}
        NODES=1
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ 32B model config (TP=4) ################
# Architecture: ~32B params, Llama-style
# hidden=8192, ffn=28672, 32 layers, 64 heads, 8 kv_heads
# With TP=4: each GPU handles hidden/4=2048 dims per head group
MODEL_SIZE="32b"
NUM_LAYERS=32
HIDDEN=8192
FFN=28672
HEADS=64
KV_HEADS=8
MBS=1       # small MBS due to 32B model size
GBS=256
SEQ_LEN=4096
TP=4        # intra-node tensor parallelism

# Build job name suffix
OPT_SUFFIX=""
[ "$USE_FLASH_ATTN" = true ] && OPT_SUFFIX="${OPT_SUFFIX}-fa"
[ "$USE_LIGER" = true ] && OPT_SUFFIX="${OPT_SUFFIX}-liger"
[ "$USE_COMPILE" = true ] && OPT_SUFFIX="${OPT_SUFFIX}-compile"
[ "$USE_CCE" = true ] && OPT_SUFFIX="${OPT_SUFFIX}-cce"
[ "$USE_NSYS" = true ] && OPT_SUFFIX="${OPT_SUFFIX}-nsys"
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-tp${TP}-${TRAINING_STEPS}s-${NODES}n${OPT_SUFFIX}"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: $(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-tp${TP}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

# Install Liger-Kernel (fast, cached after first install)
pip install liger-kernel --quiet

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --sequence-parallel
    --recompute-granularity selective
    --recompute-method uniform
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << RESOLVED_FLAGS
# Optimization flags resolved at submission time
USE_LIGER=${USE_LIGER}
USE_FLASH_ATTN=${USE_FLASH_ATTN}
USE_COMPILE=${USE_COMPILE}
USE_CCE=${USE_CCE}
USE_NSYS=${USE_NSYS}
export MEGATRON_LM_DIR=${WORKDIR}/Megatron-LM
RESOLVED_FLAGS

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

FLASH_ATTN_FLAG=""
[ "${USE_FLASH_ATTN}" = true ] && FLASH_ATTN_FLAG="--use-flash-attn"

COMPILE_FLAG=""

if [ "${USE_CCE}" = true ]; then
    TRAIN_SCRIPT="$WORKDIR/pretrain_gpt_cce.py"
elif [ "${USE_COMPILE}" = true ]; then
    TRAIN_SCRIPT="$WORKDIR/pretrain_gpt_compile.py"
    [ "${USE_LIGER}" = true ] && export USE_LIGER=true
elif [ "${USE_LIGER}" = true ]; then
    TRAIN_SCRIPT="$WORKDIR/pretrain_gpt_liger.py"
else
    TRAIN_SCRIPT="$MEGATRON_LM_DIR/pretrain_gpt.py"
fi

TORCHRUN_CMD="torchrun ${TORCHRUN_ARGS[@]} $TRAIN_SCRIPT \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    $FLASH_ATTN_FLAG \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

if [ "${USE_NSYS}" = true ]; then
    NSYS_OUT="$LOG_DIR/nsys/trace_rank$SLURM_PROCID"
    TRAINING_CMD="nsys profile \
        --output=$NSYS_OUT \
        --trace=cuda,nvtx \
        --force-overwrite=true \
        $TORCHRUN_CMD"
else
    TRAINING_CMD="$TORCHRUN_CMD"
fi

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
