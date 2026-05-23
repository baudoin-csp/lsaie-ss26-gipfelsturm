"""Wrapper applying Liger-Kernel with fused linear cross-entropy (Cut Cross-Entropy style).

Unlike pretrain_gpt_liger.py which uses standard fused CE, this uses
fused_linear_cross_entropy=True which fuses the vocab projection (lm_head)
and cross-entropy into a single chunked kernel. This avoids materializing
the full [B*T x vocab_size] logit matrix in HBM, saving ~2-4 GB for 8b models.
"""

import os
import runpy

try:
    from liger_kernel.transformers import apply_liger_kernel_to_megatron
    apply_liger_kernel_to_megatron(
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )
    print("[Liger-CCE] Applied: RMSNorm=True, SwiGLU=True, FusedLinearCrossEntropy=True", flush=True)
except ImportError:
    print("[Liger-CCE] liger-kernel not installed — run: pip install liger-kernel", flush=True)

megatron_dir = os.environ.get("MEGATRON_LM_DIR")
if megatron_dir is None:
    raise RuntimeError("MEGATRON_LM_DIR environment variable not set")

pretrain_script = os.path.join(megatron_dir, "pretrain_gpt.py")
if not os.path.exists(pretrain_script):
    raise RuntimeError(f"pretrain_gpt.py not found at: {pretrain_script}")

runpy.run_path(pretrain_script, run_name="__main__")
