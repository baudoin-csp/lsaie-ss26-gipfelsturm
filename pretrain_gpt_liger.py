"""Wrapper around Megatron-LM pretrain_gpt.py that applies Liger-Kernel fused ops before training."""

import os
import sys

# Apply Liger-Kernel fused kernels before any Megatron modules are imported.
# This monkey-patches Megatron's RMSNorm, SwiGLU, and Cross-Entropy classes
# with Triton-based fused implementations that reduce HBM memory roundtrips.
#
# Note: when --transformer-impl transformer_engine is active, TransformerEngine
# handles attention and MLP internals. Liger's cross-entropy always applies;
# RMSNorm and SwiGLU gains depend on how much TE defers to Megatron's classes.
try:
    from liger_kernel.transformers import apply_liger_kernel_to_megatron
    apply_liger_kernel_to_megatron(
        rms_norm=True,
        swiglu=True,
        cross_entropy=True,
        fused_linear_cross_entropy=False,
    )
    print("[Liger-Kernel] Applied: RMSNorm=True, SwiGLU=True, CrossEntropy=True", flush=True)
except ImportError as e:
    print(f"[Liger-Kernel] ImportError: {e}", flush=True)
except Exception as e:
    print(f"[Liger-Kernel] Error: {type(e).__name__}: {e}", flush=True)

# Run pretrain_gpt.py as __main__ with the same sys.argv
import runpy

megatron_dir = os.environ.get("MEGATRON_LM_DIR")
if megatron_dir is None:
    raise RuntimeError("MEGATRON_LM_DIR environment variable not set")

pretrain_script = os.path.join(megatron_dir, "pretrain_gpt.py")
if not os.path.exists(pretrain_script):
    raise RuntimeError(f"pretrain_gpt.py not found at: {pretrain_script}")

runpy.run_path(pretrain_script, run_name="__main__")
