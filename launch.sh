#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [options]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode; positional after model_size
# Nodes:     optional positional after steps (default 1)
#
# Options (can appear in any order after mode and model_size):
#   --tp N              Tensor parallel size (default: 1)
#   --sp                Enable sequence parallelism (requires --tp > 1)
#   --tp-overlap        Enable TP communication/GEMM overlap (requires --sp)
#   --fp8               Enable FP8 training via TransformerEngine
#   --fp8-opt           Enable FP8 optimizer states (stacks on --fp8)
#   --fa                Enable FlashAttention
#   --no-jit            Disable JIT fuser (torch.compile for kernel fusions)
#   --gbs N             Override global batch size (default: 256)
#   --mbs N             Override micro-batch size (model-specific default)
#   --seq-len N         Override sequence length (default: 4096)
#   --profile           Enable NSYS profiling (steps 10-20)
#   --zero {2,3}        Enable ZeRO-2 or ZeRO-3 via Megatron FSDP
#   --recompute         Enable selective activation recompute (attention softmax only)
#   --full-recompute    Enable full activation recompute per layer (~30% throughput cost)
#   --opt-cpu-offload   Offload optimizer states (m/v/master) to CPU — fixes OOM at 32B
#   --fg-offload        Fine-grained TE activation offload to CPU (GH200: ~14x faster via NVLink-C2C)
#   --layer-offload N   Offload activations of N transformer layers to CPU
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 1.5b 50 1 --tp 4 --sp --fp8
#            ./launch.sh throughput 3b 50 1 --fp8 --fp8-opt --mbs 8
#            ./launch.sh train 760m 5000 --tp 2
#            ./launch.sh throughput 1.5b --profile

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [options]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [options]}
shift 2

################ Parse remaining args ################
TP=1
ENABLE_SP=false
ENABLE_TP_OVERLAP=false
ENABLE_FP8=false
ENABLE_FP8_OPT=false
ENABLE_FA=false
ENABLE_NO_JIT=false
ENABLE_PROFILE=false
ENABLE_RECOMPUTE=false
ENABLE_FULL_RECOMPUTE=false
ENABLE_OPT_CPU_OFFLOAD=false
ENABLE_FG_OFFLOAD=false
LAYER_OFFLOAD=""
GBS=256
MBS_OVERRIDE=""
SEQ_LEN_OVERRIDE=""
ZERO_STAGE=""

_POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp)           TP="${2:?--tp requires N}"; shift 2;;
        --sp)           ENABLE_SP=true; shift;;
        --tp-overlap)   ENABLE_TP_OVERLAP=true; shift;;
        --fp8)          ENABLE_FP8=true; shift;;
        --fp8-opt)      ENABLE_FP8_OPT=true; shift;;
        --fa)           ENABLE_FA=true; shift;;
        --no-jit)       ENABLE_NO_JIT=true; shift;;
        --profile)      ENABLE_PROFILE=true; shift;;
        --recompute)         ENABLE_RECOMPUTE=true; shift;;
        --full-recompute)    ENABLE_FULL_RECOMPUTE=true; shift;;
        --opt-cpu-offload)   ENABLE_OPT_CPU_OFFLOAD=true; shift;;
        --fg-offload)        ENABLE_FG_OFFLOAD=true; shift;;
        --layer-offload)     LAYER_OFFLOAD="${2:?--layer-offload requires N}"; shift 2;;
        --gbs)          GBS="${2:?--gbs requires N}"; shift 2;;
        --mbs)          MBS_OVERRIDE="${2:?--mbs requires N}"; shift 2;;
        --seq-len)      SEQ_LEN_OVERRIDE="${2:?--seq-len requires N}"; shift 2;;
        --zero)
            ZERO_STAGE="${2:?--zero requires 2 or 3}"
            if [[ "$ZERO_STAGE" != "2" && "$ZERO_STAGE" != "3" ]]; then
                echo "--zero must be 2 or 3"; exit 1
            fi
            shift 2;;
        --*) echo "Unknown option: $1"; exit 1;;
        *) _POSITIONAL+=("$1"); shift;;
    esac
done

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${_POSITIONAL[0]:-15}
        NODES=${_POSITIONAL[1]:-1}
        TIME=00:17:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${_POSITIONAL[0]:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${_POSITIONAL[1]:-1}
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

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    32b)
        NUM_LAYERS=64; HIDDEN=5120; FFN=27648; HEADS=40; KV_HEADS=8
        MBS=1
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b, 32b"
        exit 1
        ;;
esac

[[ -n "$MBS_OVERRIDE" ]] && MBS="$MBS_OVERRIDE"

SEQ_LEN=4096
[[ -n "$SEQ_LEN_OVERRIDE" ]] && SEQ_LEN="$SEQ_LEN_OVERRIDE"

################ Build experiment tag ################
EXP_TAGS="${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-tp${TP}-seq${SEQ_LEN}"
$ENABLE_SP         && EXP_TAGS="${EXP_TAGS}-sp"
$ENABLE_TP_OVERLAP && EXP_TAGS="${EXP_TAGS}-tpoverlap"
$ENABLE_FP8        && EXP_TAGS="${EXP_TAGS}-fp8"
$ENABLE_FP8_OPT    && EXP_TAGS="${EXP_TAGS}-fp8opt"
$ENABLE_FA         && EXP_TAGS="${EXP_TAGS}-fa"
$ENABLE_NO_JIT     && EXP_TAGS="${EXP_TAGS}-nojit"
$ENABLE_RECOMPUTE        && EXP_TAGS="${EXP_TAGS}-recompute"
$ENABLE_FULL_RECOMPUTE   && EXP_TAGS="${EXP_TAGS}-fullrecompute"
$ENABLE_OPT_CPU_OFFLOAD  && EXP_TAGS="${EXP_TAGS}-optcpuoffload"
$ENABLE_FG_OFFLOAD       && EXP_TAGS="${EXP_TAGS}-fgoffload"
[[ -n "$LAYER_OFFLOAD" ]] && EXP_TAGS="${EXP_TAGS}-layeroffload${LAYER_OFFLOAD}"
$ENABLE_PROFILE    && EXP_TAGS="${EXP_TAGS}-profile"
[[ -n "$ZERO_STAGE" ]] && EXP_TAGS="${EXP_TAGS}-zero${ZERO_STAGE}"
JOB_NAME="gipfel-${EXP_TAGS}"

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

################ Build optional args for sbatch ################
EXTRA_ARGS=""
$ENABLE_FA              && EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
$ENABLE_NO_JIT          && EXTRA_ARGS="$EXTRA_ARGS --disable-jit-fuser"
$ENABLE_RECOMPUTE       && EXTRA_ARGS="$EXTRA_ARGS --recompute-activations"
$ENABLE_FULL_RECOMPUTE  && EXTRA_ARGS="$EXTRA_ARGS --recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
$ENABLE_OPT_CPU_OFFLOAD && EXTRA_ARGS="$EXTRA_ARGS --optimizer-cpu-offload"
$ENABLE_FG_OFFLOAD      && EXTRA_ARGS="$EXTRA_ARGS --fine-grained-activation-offloading"
[[ -n "$LAYER_OFFLOAD" ]] && EXTRA_ARGS="$EXTRA_ARGS --cpu-offloading-num-layers ${LAYER_OFFLOAD}"

if $ENABLE_FP8; then
    EXTRA_ARGS="$EXTRA_ARGS --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
fi
if $ENABLE_FP8_OPT; then
    EXTRA_ARGS="$EXTRA_ARGS --exp-avg-dtype fp8 --exp-avg-sq-dtype fp8"
fi
if [[ -n "$ZERO_STAGE" ]]; then
    case $ZERO_STAGE in
        2) EXTRA_ARGS="$EXTRA_ARGS --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads";;
        3) EXTRA_ARGS="$EXTRA_ARGS --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params";;
    esac
fi

################ Build DISTRIBUTED_ARGS content ################
DISTRIBUTED_ARGS_CONTENT="    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather"
$ENABLE_SP         && DISTRIBUTED_ARGS_CONTENT="${DISTRIBUTED_ARGS_CONTENT}
    --sequence-parallel"
$ENABLE_TP_OVERLAP && DISTRIBUTED_ARGS_CONTENT="${DISTRIBUTED_ARGS_CONTENT}
    --tp-comm-overlap"

################ Fine-grained offload env var (TE >= 2.10 requires NVTE_CPU_OFFLOAD_V1=1) ################
FG_OFFLOAD_ENV=""
$ENABLE_FG_OFFLOAD && FG_OFFLOAD_ENV="export NVTE_CPU_OFFLOAD_V1=1"

################ NSYS profiling ################
NSYS_CMD=""
NSYS_MKDIR=""
if $ENABLE_PROFILE; then
    NSYS_OUTPUT="/iopsstor/scratch/cscs/\${USER}/gipfelsturm/nsys/${JOB_NAME}-\${SLURM_PROCID}"
    NSYS_CMD="nsys profile --trace=cuda,nvtx,osrt --output=${NSYS_OUTPUT} --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true"
    NSYS_MKDIR="mkdir -p /iopsstor/scratch/cscs/\${USER}/gipfelsturm/nsys"
    EXTRA_ARGS="$EXTRA_ARGS --profile --profile-step-start 10 --profile-step-end 20"
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
EXP_NAME=${EXP_TAGS}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << SETUP_ENV
${FG_OFFLOAD_ENV}
SETUP_ENV

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

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

REST

cat >> "$SCRIPT" << DISTRIBUTED_BLOCK

DISTRIBUTED_ARGS=(
${DISTRIBUTED_ARGS_CONTENT}
)
DISTRIBUTED_BLOCK

if $ENABLE_FG_OFFLOAD; then
cat >> "$SCRIPT" << 'FG_OFFLOAD_BLOCK'

OFFLOAD_MODULES_ARGS=(
    --offload-modules
    core_attn
    attn_proj
    qkv_linear
    attn_norm
    mlp_norm
)
FG_OFFLOAD_BLOCK
else
cat >> "$SCRIPT" << 'FG_OFFLOAD_BLOCK'

OFFLOAD_MODULES_ARGS=()
FG_OFFLOAD_BLOCK
fi

cat >> "$SCRIPT" << 'LOGGING_HEAD'

LOGGING_ARGS=(
    --log-throughput
    --log-progress
LOGGING_HEAD

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

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

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OFFLOAD_MODULES_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << EXTRA_ARGS_SECTION

TRAINING_CMD="\$TRAINING_CMD ${EXTRA_ARGS}"
EXTRA_ARGS_SECTION

cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << FOOTER

echo "CMD: \$TRAINING_CMD"
${NSYS_MKDIR}
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 ${NSYS_CMD} \$TRAINING_CMD"

echo "END TIME: \$(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
