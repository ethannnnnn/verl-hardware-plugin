# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the TorchTitan engine's deferred host syncs in training passes.

The engine's real training path runs on CPU with a toy model in place of the decoder: ``forward_backward_batch`` with
fixed-shape micro-batches and pre-warm, ``forward_step`` and verl's bypass-mode ``ppo_loss``, two updates. The TPU
primitives are recorders: ``tpu_synchronize`` (tensors or device-wide, wait flag) and the micro-batch fence's device
events.
"""

from __future__ import annotations

import contextlib
import logging
import random
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu as engine_module
from verl_hardware_plugin.accelerators.tpu.engines import tpu_fixed_shapes as fs

VOCAB = 64
ROWS = "256:512"
_RECORD_DEVICE_EVENT = engine_module._record_device_event  # before any test replaces it


def _nested(ts):
    return torch.nested.nested_tensor(ts, layout=torch.jagged)


def _mini_batch(n, seed):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    lengths = [(rng.randint(2, 20), 40 if rng.random() < 0.15 else rng.randint(1, 40)) for _ in range(n)]
    data = TensorDict(
        {
            "input_ids": _nested([torch.randint(1, VOCAB, (p + r,), generator=gen) for p, r in lengths]),
            "position_ids": _nested([torch.arange(p + r) for p, r in lengths]),
            "prompts": _nested([torch.randint(1, VOCAB, (p,), generator=gen) for p, _ in lengths]),
            "responses": _nested([torch.randint(1, VOCAB, (r,), generator=gen) for _, r in lengths]),
            "response_mask": _nested([torch.ones(r, dtype=torch.int64) for _, r in lengths]),
            "loss_mask": _nested(
                [torch.cat([torch.zeros(p, dtype=torch.int64), torch.ones(r, dtype=torch.int64)]) for p, r in lengths]
            ),
            "old_log_probs": _nested([(-3.5 - torch.rand(r, generator=gen)) for _, r in lengths]),
            "advantages": _nested([torch.randn(r, generator=gen) for _, r in lengths]),
        },
        batch_size=[n],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=True,
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        max_token_len_per_gpu=512,
        dp_size=1,
        global_batch_size=n,
        batch_num_tokens=sum(r for _, r in lengths),
        temperature=1.0,
    )
    return data


def _loss_function(model_output, data, dp_group=None):
    correction = RolloutCorrectionConfig(
        rollout_is="token",
        rollout_is_threshold=3.0,
        rollout_is_batch_normalize=False,
        rollout_rs=None,
        bypass_mode=True,
        loss_type="ppo_clip",
    )
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=16,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=1.0e9,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=correction),
    )
    return ppo_loss(config, model_output, data)


class _ToyLM(torch.nn.Module):
    """logits[t] = (E[token_t] + P[min(position_t, 63)]) @ W^T, in place of the decoder and the LM head."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(7)
        self.emb = torch.nn.Parameter(0.3 * torch.randn(VOCAB, 16, generator=gen))
        self.pos = torch.nn.Parameter(0.1 * torch.randn(64, 16, generator=gen))
        self.out = torch.nn.Parameter(0.3 * torch.randn(VOCAB, 16, generator=gen))

    def forward(self, tokens, positions=None):
        return (self.emb[tokens] + self.pos[positions.clamp(max=63)]) @ self.out.t()


class _Recorder:
    """The TPU primitives the engine calls, logged with the current phase and micro-batch."""

    def __init__(self):
        self.log = []
        self.phase = "init"
        self.micro_batch = -1
        self.events_fail = False


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()

    def tpu_synchronize(tensors=None, *, wait=True, strict=False):
        kind = "tensor" if isinstance(tensors, torch.Tensor) else "device"
        rec.log.append(("sync", kind, bool(wait), rec.phase, rec.micro_batch))

    class _Event:
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            rec.log.append(("wait", self.index, rec.phase, rec.micro_batch))

    def record_device_event():
        if rec.events_fail:
            raise RuntimeError("no device events on this backend")
        rec.log.append(("record", rec.micro_batch, rec.phase, rec.micro_batch))
        return _Event(rec.micro_batch)

    monkeypatch.setattr(engine_module, "tpu_synchronize", tpu_synchronize)
    monkeypatch.setattr(engine_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(engine_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(engine_module, "_record_device_event", record_device_event)
    monkeypatch.setattr(engine_module._MicroBatchFence, "_disabled", False)

    cls = engine_module.TorchTitanTPUEngine
    micro_batch_fn, prewarm_fn = cls._forward_backward_micro_batch, cls._prewarm_fixed_shapes

    def micro_batch(self, *args, **kwargs):
        rec.micro_batch += 1
        rec.log.append(("mb", rec.phase, rec.micro_batch))
        return micro_batch_fn(self, *args, **kwargs)

    def prewarm(self, *args, **kwargs):
        previous, rec.phase = rec.phase, "prewarm"
        try:
            return prewarm_fn(self, *args, **kwargs)
        finally:
            rec.phase, rec.micro_batch = previous, -1

    monkeypatch.setattr(cls, "_forward_backward_micro_batch", micro_batch)
    monkeypatch.setattr(cls, "_prewarm_fixed_shapes", prewarm)
    # The pre-warm prices rows by wall time and the layout planner reads the prices: pin them so that every run
    # partitions identically.
    fs_prewarm = fs.prewarm

    def pinned_prewarm(data, config, run_micro_batch, zero_grad, *args, **kwargs):
        fs_prewarm(data, config, run_micro_batch, zero_grad, *args, **kwargs)
        config.costs = {row: fs.default_cost(row) for row in config.rows}

    monkeypatch.setattr(fs, "prewarm", pinned_prewarm)
    return rec


def _engine(model):
    engine = object.__new__(engine_module.TorchTitanTPUEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        engine_kwargs={
            "tpu": {
                "seq_bucket_size": 32,
                "fixed_shape_rows": ROWS,
                "fixed_shape_max_seqs": 8,
                "fixed_shape_response_len": 48,
                "fixed_shape_prewarm": True,
            }
        },
        tensor_parallel_size=1,
        context_parallel_size=1,
        data_parallel_shard_size=1,
        data_parallel_replicate_size=1,
    )
    engine._fixed_shapes = fs.FixedShapeConfig.from_engine_config(engine.engine_config)
    engine.parallel_dims = SimpleNamespace(cp_enabled=False, pp_enabled=False, tp_enabled=False)
    engine._use_simple_fsdp, engine._tpu_eager_mode, engine._simple_fsdp_loss_scale = True, None, None
    engine._tpu_kernels = frozenset()
    engine._embed_head_accum = False
    engine.module = [model]
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=VOCAB))
    engine.trainer = SimpleNamespace(
        train_context=contextlib.nullcontext,
        config=SimpleNamespace(parallelism=SimpleNamespace(spmd_backend=None)),
    )
    engine.get_data_parallel_group = lambda: None

    def zero_grad():
        for p in model.parameters():
            p.grad = None

    engine.optimizer_zero_grad = zero_grad
    return engine


def _comparable(output):
    def value(x):
        return (type(x).__name__, x.hex() if isinstance(x, float) else repr(x))

    metrics = {k: [value(x) for x in getattr(v, "values", [v])] for k, v in output["metrics"].items()}
    log_probs = output["model_output"]["log_probs"]
    return [value(x) for x in output["loss"]], metrics, [t.clone() for t in log_probs.unbind()]


def _train(rec, monkeypatch, mode):
    """Two training updates in host-scalar mode ``mode``: outputs and gradients per update."""
    monkeypatch.setattr(engine_module, "_HOST_SCALARS_MODE", mode)
    rec.log.clear()
    model = _ToyLM()
    engine = _engine(model)
    updates = []
    for update, seed in enumerate((11, 12)):
        rec.phase, rec.micro_batch = f"train{update}", -1
        output = engine.forward_backward_batch(_mini_batch(24, seed), _loss_function, forward_only=False)
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        updates.append((_comparable(output), grads))
        engine.optimizer_zero_grad()
    return updates


def _micro_batches(rec, phase):
    return sorted({e[2] for e in rec.log if e[0] == "mb" and e[1] == phase})


def _device_syncs(rec, phase, index):
    return [e[2] for e in rec.log if e[0] == "sync" and e[1] == "device" and e[3] == phase and e[4] == index]


def test_training_micro_batches_drain_only_at_the_end_of_the_loop(recorder, monkeypatch):
    _train(recorder, monkeypatch, "cut")
    for phase in ("train0", "train1"):
        indices = _micro_batches(recorder, phase)
        assert len(indices) >= 2
        for index in indices[:-1]:
            assert _device_syncs(recorder, phase, index) == [False]
        assert _device_syncs(recorder, phase, indices[-1]) == [True]
    prewarm = _micro_batches(recorder, "prewarm")
    assert prewarm
    for index in prewarm:  # pre-warm runs are timed: each one drains
        assert _device_syncs(recorder, "prewarm", index) == [True]


def test_the_fence_bounds_the_run_ahead_to_one_micro_batch(recorder, monkeypatch):
    _train(recorder, monkeypatch, "cut")
    for phase in ("train0", "train1"):
        indices = _micro_batches(recorder, phase)
        events = [e for e in recorder.log if e[0] in ("record", "wait") and e[2] == phase]
        recorded = [e[1] for e in events if e[0] == "record"]
        assert recorded == indices[:-1]
        # After recording micro-batch k the host waits for k - 1 before it dispatches k + 1.
        waits = [(e[1], e[3]) for e in events if e[0] == "wait"]
        assert waits == [(k - 1, k) for k in indices[1:-1]]


@pytest.mark.parametrize("mode", ["cut", "fuse"])
def test_deferred_modes_match_the_eager_reads_bitwise(recorder, monkeypatch, mode):
    eager = _train(recorder, monkeypatch, "off")
    deferred = _train(recorder, monkeypatch, mode)
    for (out_e, grads_e), (out_d, grads_d) in zip(eager, deferred, strict=True):
        assert out_d[0] == out_e[0]  # losses: type and bits
        assert out_d[1] == out_e[1]  # every metric value
        assert all(torch.equal(a, b) for a, b in zip(out_d[2], out_e[2], strict=True))  # log-probs
        assert all(torch.equal(grads_d[n], grads_e[n]) for n in grads_e)


def test_cut_mode_materializes_where_the_host_reads_were(recorder, monkeypatch):
    """``cut`` keeps a non-blocking materialization at every former read, ``fuse`` and ``off`` only the loss's."""
    per_mode = {}
    for mode in ("off", "cut", "fuse"):
        _train(recorder, monkeypatch, mode)
        index = _micro_batches(recorder, "train0")[0]
        per_mode[mode] = [
            e[2] for e in recorder.log if e[0] == "sync" and e[1] == "tensor" and e[3] == "train0" and e[4] == index
        ]
    assert per_mode["off"] == per_mode["fuse"] == [False]  # tpu_synchronize(loss, wait=False) before backward
    assert len(per_mode["cut"]) > 10 and not any(per_mode["cut"])


def test_without_device_events_every_micro_batch_drains(recorder, monkeypatch, caplog):
    reference = _train(recorder, monkeypatch, "cut")
    recorder.events_fail = True
    with caplog.at_level(logging.WARNING, logger=engine_module.logger.name):
        fallback = _train(recorder, monkeypatch, "cut")
    assert "draining after every micro-batch instead" in caplog.text
    for phase in ("train0", "train1"):
        for index in _micro_batches(recorder, phase)[:-1]:
            assert _device_syncs(recorder, phase, index) == [False, True]
    for (out_r, grads_r), (out_f, grads_f) in zip(reference, fallback, strict=True):
        assert out_f[:2] == out_r[:2]
        assert all(torch.equal(grads_f[n], grads_r[n]) for n in grads_r)


def test_without_a_tpu_device_every_micro_batch_drains(recorder, monkeypatch, caplog):
    """Only TPU events are used: elsewhere (here: CPU) the fence drains after every micro-batch from the start."""
    reference = _train(recorder, monkeypatch, "cut")
    monkeypatch.setattr(engine_module, "_record_device_event", _RECORD_DEVICE_EVENT)
    with pytest.raises(RuntimeError, match="need a TPU device, not cpu"):
        _RECORD_DEVICE_EVENT()
    with caplog.at_level(logging.WARNING, logger=engine_module.logger.name):
        on_cpu = _train(recorder, monkeypatch, "cut")
    assert "draining after every micro-batch instead" in caplog.text
    for phase in ("train0", "train1"):
        for index in _micro_batches(recorder, phase)[:-1]:
            assert _device_syncs(recorder, phase, index) == [False, True]
    for (out_r, grads_r), (out_c, grads_c) in zip(reference, on_cpu, strict=True):
        assert out_c[:2] == out_r[:2]
        assert all(torch.equal(grads_c[n], grads_r[n]) for n in grads_r)
