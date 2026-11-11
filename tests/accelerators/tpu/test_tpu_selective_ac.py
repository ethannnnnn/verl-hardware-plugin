# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for ``engines/tpu_selective_ac.py`` (``engine_kwargs.tpu.selective_ac_recompute_bmm`` and the defaults of
the other selective-AC flags)."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from verl_hardware_plugin.accelerators.tpu.engines import tpu_selective_ac


@dataclasses.dataclass(kw_only=True)
class _FlagsConfig:
    """Stands in for torchtitan's ``TPUSelectiveAC.Config``: the three flags plus an unrelated field."""

    preserve_rng_state: bool = True
    recompute_bmm: bool = False
    save_pallas_attention: bool = False
    save_triangular_solve: bool = False


@dataclasses.dataclass(kw_only=True)
class _NoFlagsConfig:
    """A selective config whose policy has no flags (torchtitan's core ``SelectiveAC.Config``)."""

    preserve_rng_state: bool = True


def _flags(config) -> tuple[bool, bool, bool]:
    return tuple(getattr(config, name) for name in tpu_selective_ac.FLAGS)


def test_flag_names_and_defaults():
    assert tpu_selective_ac.FLAGS == ("recompute_bmm", "save_pallas_attention", "save_triangular_solve")
    assert tpu_selective_ac.SAVE_PALLAS_ATTENTION is True
    assert tpu_selective_ac.SAVE_TRIANGULAR_SOLVE is True


def test_option_sets_recompute_bmm_on_a_copy_and_logs_the_policy(caplog):
    source = _FlagsConfig(preserve_rng_state=False)
    with caplog.at_level(logging.WARNING, logger=tpu_selective_ac.logger.name):
        config = tpu_selective_ac.selective_ac_config(source, "selective", recompute_bmm=True)
    assert config is not source
    assert _flags(config) == (True, True, True)
    assert config.preserve_rng_state is False  # every other field is kept
    assert _flags(source) == (False, False, False)  # the input is untouched
    assert "selective AC policy:" in caplog.text
    assert "recompute_bmm=True, save_pallas_attention=True, save_triangular_solve=True" in caplog.text


def test_without_the_option_the_save_defaults_still_apply():
    config = tpu_selective_ac.selective_ac_config(_FlagsConfig(), "selective", recompute_bmm=False)
    assert _flags(config) == (False, True, True)


def test_overrides_and_all_flags_off():
    config = tpu_selective_ac.selective_ac_config(
        _FlagsConfig(), "selective", recompute_bmm=True, save_pallas_attention=False, save_triangular_solve=False
    )
    assert _flags(config) == (True, False, False)
    source = _FlagsConfig()
    same = tpu_selective_ac.selective_ac_config(
        source, "selective", recompute_bmm=False, save_pallas_attention=False, save_triangular_solve=False
    )
    assert same is source


@pytest.mark.parametrize("mode, config", [("full", object()), ("none", None)])
def test_other_modes_are_left_alone_and_refuse_the_option(mode, config):
    assert tpu_selective_ac.selective_ac_config(config, mode, recompute_bmm=False) is config
    with pytest.raises(ValueError, match="needs activation_checkpoint=selective"):
        tpu_selective_ac.selective_ac_config(config, mode, recompute_bmm=True)


def test_a_selective_policy_without_the_flags_fails_closed():
    with pytest.raises(ValueError, match="has no"):
        tpu_selective_ac.selective_ac_config(_NoFlagsConfig(), "selective", recompute_bmm=True)
    with pytest.raises(ValueError, match="has no"):
        tpu_selective_ac.selective_ac_config(_NoFlagsConfig(), "selective", recompute_bmm=False)


def _torchtitan_tpu_selective_ac():
    activation_checkpoint = pytest.importorskip("torchtitan.experiments.tpu.activation_checkpoint")
    policy = activation_checkpoint.TPUSelectiveAC
    names = {f.name for f in dataclasses.fields(policy.Config)}
    if not set(tpu_selective_ac.FLAGS) <= names:
        pytest.skip("this torchtitan's TPUSelectiveAC has no selective-AC flags")
    return policy


def test_torchtitan_policy_built_from_the_engine_config():
    """With torchtitan's TPUSelectiveAC, the returned config builds that policy with the flags set."""
    torch = pytest.importorskip("torch")
    policy_cls = _torchtitan_tpu_selective_ac()
    config = tpu_selective_ac.selective_ac_config(
        policy_cls.Config(preserve_rng_state=False), "selective", recompute_bmm=True
    )
    assert type(config) is policy_cls.Config
    policy = config.build(dump_folder="")
    assert type(policy) is policy_cls
    save_ops = policy.get_save_ops()
    assert torch.ops.aten.bmm.default not in save_ops
    assert torch.ops.torchtitan_tpu.gdn_unit_solve_triangular.default in save_ops
    assert torch.ops.aten.linalg_solve_triangular.default not in save_ops

    class _SplashForward:
        def name(self):
            return "pallas::splash_attn_fwd_0123456789abcdef"

    assert _SplashForward() in save_ops
