"""Wrapper that applies torch.compile to Megatron model via monkey-patching.

torch.compile is not a Megatron-LM CLI flag in core_v0.16.1.
This wrapper patches setup_model_and_optimizer() to wrap the model
with torch.compile(mode='reduce-overhead') after creation.

Optionally also applies Liger-Kernel if USE_LIGER=true in env.
"""

import os
import sys
import torch
import runpy

megatron_dir = os.environ.get("MEGATRON_LM_DIR")
if megatron_dir is None:
    raise RuntimeError("MEGATRON_LM_DIR not set")
if megatron_dir not in sys.path:
    sys.path.insert(0, megatron_dir)

# Optionally apply Liger first
if os.environ.get("USE_LIGER", "false").lower() == "true":
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_megatron
        apply_liger_kernel_to_megatron(rms_norm=True, swiglu=True, cross_entropy=True)
        print("[Compile+Liger] Liger applied", flush=True)
    except ImportError:
        print("[Compile+Liger] liger-kernel not installed", flush=True)

# Patch setup_model_and_optimizer to wrap model with torch.compile
import megatron.training.training as _mtt

_original_setup = _mtt.setup_model_and_optimizer

def _setup_with_compile(*args, **kwargs):
    model, optimizer, lr_scheduler = _original_setup(*args, **kwargs)
    model_list = model if isinstance(model, list) else [model]
    compiled = []
    for m in model_list:
        try:
            # dynamic=True avoids recompilation on shape changes
            # fullgraph=False allows partial compilation compatible with DDP hooks
            compiled.append(torch.compile(m, fullgraph=False, dynamic=True))
            print("[torch.compile] Model compiled successfully", flush=True)
        except Exception as e:
            print(f"[torch.compile] Failed: {e} — using uncompiled model", flush=True)
            compiled.append(m)
    result = compiled if isinstance(model, list) else compiled[0]
    return result, optimizer, lr_scheduler

_mtt.setup_model_and_optimizer = _setup_with_compile
print("[torch.compile] Patched setup_model_and_optimizer", flush=True)

runpy.run_path(os.path.join(megatron_dir, "pretrain_gpt.py"), run_name="__main__")
