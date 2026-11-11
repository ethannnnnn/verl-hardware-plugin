# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Selective activation checkpointing flags of the TPU engine (``activation_checkpoint=selective``).

``selective`` builds torchtitan's ``TPUSelectiveAC``: core ``SelectiveAC`` (saves the outputs of the compute-intensive
ops -- mm, bmm, ... -- except every second mm / linear, and recomputes everything else in backward) minus the
collectives it would save. That one policy class has three flags, off by default, and any combination composes:

* ``recompute_bmm`` (``engine_kwargs.tpu.selective_ac_recompute_bmm``): recompute ``aten.bmm`` outputs in backward
  instead of saving them. Qwen3.5's XLA GatedDeltaNet path unrolls its inter-chunk recurrence into three ``[heads,
  64..128, 64..128]`` bmm per 64-token chunk, so saving them hands ``27 + 3 * seq_len / 64`` tensors from each
  GatedDeltaNet block's compiled forward program to its backward program; recomputed, the loop's products come from the
  26 tensors that stay saved (the loop's forward bmm FLOPs once more, about 1% of the model's FLOPs).
* ``save_pallas_attention`` (default :data:`SAVE_PALLAS_ATTENTION`): also save the Pallas splash-attention forward's
  output and logsumexp, so backward runs the splash backward kernel on them instead of re-running the forward kernel
  first. Costs one bf16 ``[1, H, T, D]`` + f32 ``[1, H, T]`` per attention layer.
* ``save_triangular_solve`` (default :data:`SAVE_TRIANGULAR_SOLVE`): also save the GatedDeltaNet UT transform's
  triangular solve (torchtitan's ``torchtitan_tpu::gdn_unit_solve_triangular`` custom op) instead of re-running it in
  backward: one solve per GatedDeltaNet layer less, one f32 ``[1, heads, seq_len / 64, 64, key_dim + value_dim]`` saved
  per layer. Not ``aten.linalg_solve_triangular``, whose meta kernel's column-major strides differ from torch-tpu's
  row-major output: a saved tensor reaches the compiled backward with its real strides while that program was compiled
  for its fake ones.

The values do not change: a saved output and a recomputed one come from the same op on the same inputs. On CPU the
forward output and every gradient are bitwise identical for every flag combination. On TPU the recomputed ops are
compiled into the backward program, where XLA may fuse the surrounding elementwise ops differently, as for every op
selective AC recomputes.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The flags that have no engine option yet: their values for every selective-AC engine of this plugin tree.
SAVE_PALLAS_ATTENTION = True
SAVE_TRIANGULAR_SOLVE = True
FLAGS = ("recompute_bmm", "save_pallas_attention", "save_triangular_solve")


def selective_ac_config(
    ac_config: Any,
    ac_mode: str,
    *,
    recompute_bmm: bool,
    save_pallas_attention: Optional[bool] = None,
    save_triangular_solve: Optional[bool] = None,
) -> Any:
    """Sets the selective-AC flags on the engine's activation-checkpointing config.

    Args:
        ac_config: The config the engine built for ``activation_checkpoint=ac_mode`` (``TPUSelectiveAC.Config()`` for
            ``selective``).
        ac_mode: The engine's ``activation_checkpoint`` mode.
        recompute_bmm: ``engine_kwargs.tpu.selective_ac_recompute_bmm``.
        save_pallas_attention: Overrides :data:`SAVE_PALLAS_ATTENTION` (None: the default).
        save_triangular_solve: Overrides :data:`SAVE_TRIANGULAR_SOLVE` (None: the default).

    Returns:
        ``ac_config`` itself when ``ac_mode`` is not ``selective`` or no flag is set. Otherwise a copy of ``ac_config``
        (every field kept) with the three flags set.

    Raises:
        ValueError: ``recompute_bmm`` with an ``activation_checkpoint`` other than ``selective``, or a selective config
            whose policy has no such flags (a torchtitan ``TPUSelectiveAC`` without them).
    """
    flags = {
        "recompute_bmm": bool(recompute_bmm),
        "save_pallas_attention": SAVE_PALLAS_ATTENTION
        if save_pallas_attention is None
        else bool(save_pallas_attention),
        "save_triangular_solve": SAVE_TRIANGULAR_SOLVE
        if save_triangular_solve is None
        else bool(save_triangular_solve),
    }
    if ac_mode != "selective":
        if flags["recompute_bmm"]:
            raise ValueError(
                "engine_kwargs.tpu.selective_ac_recompute_bmm=True needs activation_checkpoint=selective "
                f"(got {ac_mode!r}, config {type(ac_config).__name__})"
            )
        return ac_config
    if not any(flags.values()):
        return ac_config
    names = {f.name for f in dataclasses.fields(ac_config)} if dataclasses.is_dataclass(ac_config) else set()
    if not set(FLAGS) <= names:
        raise ValueError(
            f"activation_checkpoint=selective with flags {flags}: config {type(ac_config).__name__} has no "
            f"{sorted(set(FLAGS) - names)} (needs torchtitan's TPUSelectiveAC with flags)"
        )
    config = dataclasses.replace(ac_config, **flags)
    owner = getattr(type(config), "_owner", None)
    logger.warning(
        "selective AC policy: %s(%s)",
        getattr(owner, "__name__", type(config).__qualname__),
        ", ".join(f"{k}={v}" for k, v in flags.items()),
    )
    return config
