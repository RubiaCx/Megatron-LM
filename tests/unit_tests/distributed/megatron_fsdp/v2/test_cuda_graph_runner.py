# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Megatron-FSDP v2 CUDA graph capture hooks."""

import copy
import gc
import importlib
import sys
import weakref
from contextlib import nullcontext
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.cuda_graph_runner import (
    CudaGraphRunner,
    _build_input_output_aliases,
    _capture_module_topology,
    _is_direct_autograd_alias,
    _make_bwd_post_hook,
    _make_bwd_pre_hook,
    _make_module_topology_preflight,
    _module_uses_delayed_wgrad,
    _prepare_compiled_modules_for_capture,
    _restore_compiled_modules_after_capture_failure,
    _snapshot_capture_mutable_state,
    _validate_activation_recompute_lifetime,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks import (
    _register_forward_hook,
    _pre_backward_setup,
    mfsdp_forward_pre_hook,
    mfsdp_post_backward_final_callback,
    mfsdp_post_forward_hook,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    _MFSDP_CAPTURE_CAPABILITIES,
    _get_tracked_cuda_generators,
    _get_compatible_main_grad_buffer,
    _graph_context_wrapper,
    _get_static_grad_buffers,
    _none_grad_context_wrapper,
    _refresh_module_parameter_surface,
    _require_torch,
    _static_dgrad_metadata,
    _static_grad_context_wrapper,
    graph_safe_rng_available,
    is_graph_capturing,
    make_weak_ref,
    make_graphed_callables,
)
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce






def test_input_output_aliases_preserve_detach_boundaries():
    """Do not reconnect a detached input to a grad-producing output."""
    first = torch.nn.Identity()
    second = torch.nn.Identity()
    modules = (first, second)
    sample = torch.ones(2, 4, requires_grad=True)
    produced = first(sample)
    sample_outputs = {id(first): produced, id(second): produced.detach()}
    sample_args = {id(first): (sample,), id(second): (produced.detach(),)}
    sample_kwargs = {id(first): {}, id(second): {}}

    aliases = _build_input_output_aliases(
        modules, sample_outputs, sample_args, sample_kwargs
    )
    assert aliases == ({}, {})


def test_input_output_aliases_accept_direct_autograd_view():
    """Reconnect the view inserted by the M-FSDP backward hook."""
    first = torch.nn.Linear(4, 4, bias=False)
    second = torch.nn.Identity()
    modules = (first, second)
    sample = torch.ones(2, 4, requires_grad=True)
    produced = first(sample)
    wrapped = produced.view_as(produced)
    sample_outputs = {id(first): produced, id(second): wrapped}
    sample_args = {id(first): (sample,), id(second): (wrapped,)}
    sample_kwargs = {id(first): {}, id(second): {}}
    assert _is_direct_autograd_alias(wrapped, produced)

    aliases = _build_input_output_aliases(
        modules, sample_outputs, sample_args, sample_kwargs
    )

    assert aliases == ({}, {0: (0, 0)})

    sample_args[id(second)] = (produced,)
    aliases = _build_input_output_aliases(
        modules, sample_outputs, sample_args, sample_kwargs
    )
    assert aliases == ({}, {0: (0, 0)})

    releaf = produced.detach().requires_grad_()
    sample_args[id(second)] = (releaf,)
    aliases = _build_input_output_aliases(
        modules, sample_outputs, sample_args, sample_kwargs
    )
    assert aliases == ({}, {})


def test_input_output_aliases_reject_ambiguous_and_nonidentity_views():
    """Reject duplicate outputs, conjugate views, and empty storage keys."""
    first = torch.nn.Identity()
    second = torch.nn.Identity()
    modules = (first, second)
    sample_kwargs = {id(first): {}, id(second): {}}

    produced = torch.ones(2, 4, requires_grad=True)
    aliases = _build_input_output_aliases(
        modules,
        {id(first): (produced, produced), id(second): produced},
        {id(first): (produced,), id(second): (produced,)},
        sample_kwargs,
    )
    assert aliases == ({}, {})

    complex_leaf = torch.ones(2, 4, dtype=torch.complex64, requires_grad=True)
    complex_output = complex_leaf * (1 + 0j)
    conjugate_input = complex_output.conj()
    aliases = _build_input_output_aliases(
        modules,
        {id(first): complex_output, id(second): conjugate_input},
        {id(first): (complex_output,), id(second): (conjugate_input,)},
        sample_kwargs,
    )
    assert aliases == ({}, {})

    empty = torch.empty(0, requires_grad=True)
    aliases = _build_input_output_aliases(
        modules,
        {id(first): empty, id(second): empty},
        {id(first): (empty,), id(second): (empty,)},
        sample_kwargs,
    )
    assert aliases == ({}, {})


def test_input_output_aliases_follow_reused_storage_in_execution_order():
    """Match each consumer before a later output reuses the same storage."""
    modules = tuple(torch.nn.Identity() for _ in range(4))
    reused_storage = torch.ones(2, 4, requires_grad=True)
    other = torch.zeros_like(reused_storage, requires_grad=True)
    sample_outputs = {
        id(modules[0]): reused_storage,
        id(modules[1]): other,
        id(modules[2]): reused_storage,
        id(modules[3]): other,
    }
    sample_args = {
        id(modules[0]): (other,),
        id(modules[1]): (reused_storage,),
        id(modules[2]): (other,),
        id(modules[3]): (reused_storage,),
    }
    sample_kwargs = {id(module): {} for module in modules}

    aliases = _build_input_output_aliases(
        modules, sample_outputs, sample_args, sample_kwargs
    )

    assert aliases == ({}, {0: (0, 0)}, {0: (1, 0)}, {0: (2, 0)})










def test_cuda_graph_runner_rejects_repeated_module_instance():
    """Reject a repeated module occurrence before capture installs an invalid graph."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    sample = torch.ones(2, 4)
    runner = CudaGraphRunner(graph_pool=None)

    runner.record_module(module, (sample,), {})
    with pytest.raises(RuntimeError, match="same module instance more than once"):
        runner.record_module(module, (sample,), {})






def test_module_topology_preflight_rejects_added_or_deleted_buffer_slots():
    """Reject registered-buffer key changes that surface-only checks miss."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module[0].register_buffer("scale", torch.ones(1))
    preflight = _make_module_topology_preflight(_capture_module_topology(module))

    module[0].register_buffer("offset", torch.zeros(1))
    with pytest.raises(RuntimeError, match="registered buffer topology changed"):
        preflight()

    del module[0]._buffers["offset"]
    preflight()
    del module[0]._buffers["scale"]
    with pytest.raises(RuntimeError, match="registered buffer topology changed"):
        preflight()


def test_module_topology_preflight_rejects_child_replacement():
    """Reject replacing a child whose captured graph still uses the old module."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    preflight = _make_module_topology_preflight(_capture_module_topology(module))

    module[0] = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="child module topology changed"):
        preflight()


def test_module_topology_preflight_uses_cached_owners_and_delegates():
    """Avoid recursive replay walks while preserving the runtime preflight."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    topology = _capture_module_topology(module)
    calls = []
    preflight = _make_module_topology_preflight(topology, lambda: calls.append("delegate"))

    def fail_recursive_walk(*args, **kwargs):
        """Reject an unexpected replay-time recursive module walk."""
        del args, kwargs
        raise AssertionError("replay walked named_modules")

    module.named_modules = fail_recursive_walk
    preflight()

    assert calls == ["delegate"]










@pytest.mark.parametrize(
    "events",
    [
        [("forward", 0), ("forward", 1), ("backward", 0), ("backward", 1)],
        [("forward", 0), ("forward", 1), ("backward", 1)],
        [("forward", 0), ("backward", 0), ("forward", 1), ("backward", 1)],
    ],
)
def test_activation_recompute_rejects_non_linear_lifetime(events):
    """Reject branched, missing-backward, and interleaved lifetime traces."""
    with pytest.raises(RuntimeError, match="complete linear F0..Fn,Bn..B0"):
        _validate_activation_recompute_lifetime(events, module_count=2)


def test_activation_recompute_accepts_complete_linear_lifetime():
    """Accept one forward chain followed by exact reverse backward order."""
    _validate_activation_recompute_lifetime(
        [("forward", 0), ("forward", 1), ("backward", 1), ("backward", 0)],
        module_count=2,
    )


def test_cuda_graph_runner_reset_clears_capture_observations():
    """Start lifetime and autocast discovery from scratch after reset."""
    module = torch.nn.Linear(4, 4, bias=False)
    module._fsdp_param_groups = ()
    sample = torch.ones(2, 4)
    runner = CudaGraphRunner(graph_pool=None)

    runner.record_module(module, (sample,), {})
    runner.record_module_backward(module)
    runner._captured = True
    runner.reset()

    assert runner._autocast_states == {}
    assert runner._lifetime_events == []
    assert runner._backward_lifetime_modules == set()

    runner.record_module(module, (sample,), {})
    runner.record_module_backward(module)
    assert runner._lifetime_events == [("forward", 0), ("backward", 0)]








def test_delayed_wgrad_detection_includes_te_wgrad_store():
    """Reject TE modules that expose delayed wgrad only through wgrad_store."""
    module = SimpleNamespace(
        delay_wgrad_compute=False,
        config=SimpleNamespace(delay_wgrad_compute=False),
        wgrad_store=SimpleNamespace(delay_wgrad_compute=lambda: True),
    )
    assert _module_uses_delayed_wgrad(module)


def test_capture_backward_post_hook_clears_only_unsharded_parameter_grads():
    """Capture-only grads must not keep their TracePool slot active."""
    full_param = torch.nn.Parameter(torch.ones(4))
    full_param.grad = torch.full_like(full_param, 2)
    full_param.main_grad = torch.full_like(full_param, 3)

    dist_param = torch.nn.Parameter(torch.ones(2))
    dist_param.grad = torch.full_like(dist_param, 4)

    reshard_calls = []
    release_calls = []
    module = SimpleNamespace(
        _fsdp_param_groups=[
            SimpleNamespace(
                params=[full_param],
                dist_params=[dist_param],
                release_grad_buffer=lambda: release_calls.append(True),
            )
        ],
        reshard=lambda: reshard_calls.append(True),
    )

    _make_bwd_post_hook(module)(module, (), ())

    assert full_param.grad is None
    assert torch.equal(full_param.main_grad, torch.full_like(full_param, 3))
    assert torch.equal(dist_param.grad, torch.full_like(dist_param, 4))
    assert reshard_calls == [True]
    assert release_calls == [True]


def test_capture_mutable_state_restores_buffers_and_persistent_main_grad():
    """Undo warmup mutations before the real optimizer consumes gradients."""
    module = torch.nn.Linear(4, 4, bias=False)
    module.register_buffer("running_value", torch.tensor([3.0]))
    param = module.weight
    param.grad = torch.full_like(param, 2)
    param.grad_added_to_main_grad = True
    dist_param = torch.nn.Parameter(torch.ones(4))
    dist_grad = object()
    grad_buffer = SimpleNamespace(
        data=torch.full((param.numel(),), 5.0),
        is_distributed=False,
        _unsharded_buffer=None,
        reshard=lambda: None,
    )
    param_group = SimpleNamespace(
        params=[param],
        dist_params=[dist_param],
        dist_grads=[dist_grad],
        main_grad_buffer=grad_buffer,
        _grad_buffer_is_fresh=False,
        _main_grad_buffer_has_unreduced_data=True,
    )
    param._mfsdp_recorded_te_wgrad = True
    module._fsdp_param_groups = [param_group]

    original_grad = param.grad
    state = _snapshot_capture_mutable_state((module,))
    module.running_value.fill_(9)
    grad_buffer.data.zero_()
    param.grad = None
    param.grad_added_to_main_grad = False
    param.overwrite_main_grad = True
    param_group.dist_grads = [None]
    param_group._grad_buffer_is_fresh = True
    param_group._main_grad_buffer_has_unreduced_data = False
    del param._mfsdp_recorded_te_wgrad
    state.restore()

    torch.testing.assert_close(module.running_value, torch.tensor([3.0]))
    torch.testing.assert_close(grad_buffer.data, torch.full_like(grad_buffer.data, 5.0))
    assert param.grad is original_grad
    assert param.grad_added_to_main_grad is True
    assert "overwrite_main_grad" not in param.__dict__
    assert param_group.dist_grads == [dist_grad]
    assert param_group._grad_buffer_is_fresh is False
    assert param_group._main_grad_buffer_has_unreduced_data is True
    assert param._mfsdp_recorded_te_wgrad is True


def test_capture_mutable_state_rejects_live_distributed_grad_buffer():
    """Do not destroy a caller-owned unsharded main-grad allocation."""
    module = torch.nn.Linear(2, 2, bias=False)
    module._fsdp_param_groups = [
        SimpleNamespace(
            params=[module.weight],
            dist_params=[],
            dist_grads=[],
            main_grad_buffer=SimpleNamespace(
                data=torch.zeros(1),
                is_distributed=True,
                _unsharded_buffer=torch.zeros(1),
            ),
            _grad_buffer_is_fresh=False,
        )
    ]

    with pytest.raises(RuntimeError, match="to be resharded"):
        _snapshot_capture_mutable_state((module,))


def test_final_callback_clears_only_already_reduced_compute_grads():
    """Remove late main-grad aliases without dropping deferred gradients."""
    reduced_param = torch.nn.Parameter(torch.ones(4))
    reduced_param.grad = torch.full_like(reduced_param, 2)
    deferred_param = torch.nn.Parameter(torch.ones(4))
    deferred_param.grad = torch.full_like(deferred_param, 3)
    reduced_module = SimpleNamespace(
        post_backward_issued=True,
        _fsdp_pre_backward_done=True,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=False),
        _fsdp_param_groups=[
            SimpleNamespace(sharding_strategy="optim_grads_params", params=[reduced_param])
        ],
    )
    deferred_module = SimpleNamespace(
        post_backward_issued=True,
        _fsdp_pre_backward_done=True,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=False),
        _fsdp_param_groups=[
            SimpleNamespace(sharding_strategy="no_shard", params=[deferred_param])
        ],
    )
    context = SimpleNamespace(
        cuda_graph_active=False,
        forward_order=[reduced_module, deferred_module],
        reduce_grad_buckets={},
        rs_stream=object(),
        backward_phase=True,
        backward_module=id(reduced_module),
        backward_done_modules=set(),
        bucket_allocator=None,
        cuda_graph_runner=None,
    )
    root = SimpleNamespace(
        _fsdp_state=SimpleNamespace(_is_root=True, _post_backward_callback_queued=True),
        _fsdp_root_context=context,
    )
    current_stream = SimpleNamespace(wait_stream=lambda stream: None)

    hook_module = "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks"
    with (
        patch(f"{hook_module}.FSDPModule", SimpleNamespace),
        patch("torch.cuda.current_stream", return_value=current_stream),
    ):
        mfsdp_post_backward_final_callback(root)

    assert reduced_param.grad is None
    torch.testing.assert_close(deferred_param.grad, torch.full_like(deferred_param, 3))


def test_recompute_fetches_forward_buffer_without_forward_prefetch():
    """Fetch both compute buffers without forward-order prefetch during recompute."""
    unshard_calls = []
    target = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(
            backward_phase=True, cuda_graph_active=False, enable_unshard_prefetch=True
        ),
        _fsdp_state=SimpleNamespace(_is_root=False),
        _fsdp_cg_activation_recompute=True,
        _fsdp_param_groups=(),
        unshard=lambda **kwargs: unshard_calls.append(kwargs),
    )

    with patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks._find_fsdp_target",
        return_value=target,
    ):
        mfsdp_forward_pre_hook(object(), (), {})

    assert unshard_calls == [
        {"async_op": True, "bwd_pass": True},
        {"async_op": False, "bwd_pass": False},
    ]


def test_checkpoint_recompute_uses_only_backward_weight_buffer():
    """Do not fetch a second weight buffer for ordinary checkpoint recompute."""
    unshard_calls = []
    target = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(
            backward_phase=True, cuda_graph_active=False, enable_unshard_prefetch=True
        ),
        _fsdp_state=SimpleNamespace(_is_root=False),
        _fsdp_param_groups=(),
        unshard=lambda **kwargs: unshard_calls.append(kwargs),
    )

    with patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks._find_fsdp_target",
        return_value=target,
    ):
        mfsdp_forward_pre_hook(object(), (), {})

    assert unshard_calls == [{"async_op": True, "bwd_pass": True}]


def test_recompute_preflight_rejects_before_unshard():
    """Leave parameter buffers sharded when a second forward is rejected."""
    calls = []

    def reject():
        calls.append("preflight")
        raise RuntimeError("backward to finish before the next forward")

    target = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(cuda_graph_active=False),
        _cuda_graph_preflight=reject,
        unshard=lambda **kwargs: calls.append(("unshard", kwargs)),
    )
    with (
        patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks._find_fsdp_target",
            return_value=target,
        ),
        pytest.raises(RuntimeError, match="backward to finish before the next forward"),
    ):
        mfsdp_forward_pre_hook(object(), (), {})

    assert calls == ["preflight"]


def test_forward_exception_always_reshards_current_recompute_module():
    """Reshard a backward-recompute module when its forward raises."""

    class FailingFSDPModule(FSDPModule, torch.nn.Module):
        """Minimal FSDP module whose forward always fails."""

        def forward(self, input_tensor):
            """Raise after the pre-forward hook has unsharded parameters."""
            del input_tensor
            raise RuntimeError("forward failed")

    module = FailingFSDPModule()
    reshard_calls = []
    module._fsdp_root_context = SimpleNamespace(
        cuda_graph_active=False,
        backward_phase=True,
        backward_module=-1,
    )
    module.reshard = lambda: reshard_calls.append(True)
    _register_forward_hook(module)

    with pytest.raises(RuntimeError, match="forward failed"):
        module(torch.ones(1))

    assert reshard_calls == [True]


def test_checkpoint_early_stop_keeps_recompute_module_unsharded():
    """Treat non-reentrant checkpoint early-stop as a successful recompute."""
    from torch.utils.checkpoint import _StopRecomputationError

    class EarlyStopFSDPModule(FSDPModule, torch.nn.Module):
        """Minimal module that models checkpoint's internal early-stop."""

        def forward(self, input_tensor):
            """Raise the checkpoint control-flow exception."""
            del input_tensor
            raise _StopRecomputationError

    module = EarlyStopFSDPModule()
    reshard_calls = []
    module._fsdp_root_context = SimpleNamespace(
        cuda_graph_active=False,
        backward_phase=False,
        backward_module=-1,
    )
    module.reshard = lambda: reshard_calls.append(True)
    _register_forward_hook(module)

    with pytest.raises(_StopRecomputationError):
        module(torch.ones(1))

    assert reshard_calls == []


def test_post_forward_reshards_noncurrent_recompute_module():
    """Release a recomputed module that is not next in backward order."""

    class DummyFSDPModule(FSDPModule, torch.nn.Module):
        """Provide an FSDP module for direct post-forward-hook testing."""

    module = DummyFSDPModule()
    reshard_calls = []
    module._fsdp_root_context = SimpleNamespace(
        cuda_graph_active=False,
        backward_phase=True,
        backward_module=-1,
        cuda_graph_runner=None,
    )
    module._fsdp_state = SimpleNamespace(enable_cuda_graph=True)
    module.reshard = lambda: reshard_calls.append(True)

    mfsdp_post_forward_hook(module, (), torch.ones(1))

    assert reshard_calls == [True]


def test_post_forward_keeps_current_recompute_module_unsharded_on_success():
    """Keep a recomputed module unsharded until its backward runs."""

    class DummyFSDPModule(FSDPModule, torch.nn.Module):
        """Minimal FSDP module for direct post-forward-hook testing."""

    module = DummyFSDPModule()
    reshard_calls = []
    module._fsdp_root_context = SimpleNamespace(
        cuda_graph_active=False,
        backward_phase=True,
        backward_module=id(module),
        cuda_graph_runner=None,
    )
    module._fsdp_state = SimpleNamespace(enable_cuda_graph=True)
    module.reshard = lambda: reshard_calls.append(True)

    mfsdp_post_forward_hook(module, (), torch.ones(1))

    assert reshard_calls == []


def test_vendored_runtime_declares_release_safe_mfsdp_capture():
    """Require installed runtimes to match the capture buffer lifetime contract."""
    assert {
        "activation_recompute",
        "capture_grad_buffer_release",
        "activation_recompute_argument_binding",
        "activation_recompute_discard_tape",
        "activation_recompute_preflight",
        "fp8_activation_recompute_metadata",
        "parameter_surface_refresh",
        "registered_buffer_validation",
        "static_dgrad_reuse",
        "static_fwd_reuse",
        "static_grad_binding",
    }.issubset(_MFSDP_CAPTURE_CAPABILITIES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reduce_grad_skips_aliased_main_grad_copy():
    """Skip copying a parameter gradient that already aliases main grad."""
    param = torch.nn.Parameter(torch.ones(4, device="cuda"))
    main_grad = torch.zeros_like(param)
    param.grad = main_grad
    param.get_main_grad = lambda: main_grad

    reduce_calls = []
    release_calls = []
    dist_param = SimpleNamespace(dtype=param.dtype, grad=None)
    param_group = SimpleNamespace(
        requires_grad=True,
        params=(param,),
        dist_params=(dist_param,),
        dist_grads=(None,),
        mp_policy=SimpleNamespace(use_decoupled_grad=False),
        _init_dist_grads=lambda: None,
        reduce_grad=lambda: reduce_calls.append(True),
        release_grad_buffer=lambda: release_calls.append(True),
    )
    module = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(rs_stream=None),
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        _named_param_groups=[(("weight",), param_group)],
        _wait_for_previous_async_reduce_grad=lambda: None,
    )

    with patch.object(torch, "_foreach_copy_") as copy_mock:
        FSDPModule.reduce_grad(module)

    copy_mock.assert_not_called()
    assert param.grad is None
    assert reduce_calls == [True]
    assert release_calls == [True]


def test_module_compile_is_converted_to_compiled_forward_for_capture():
    module = torch.nn.Linear(2, 2)
    original_forward = module.forward
    compiled_call_impl = object()
    compiled_forward = object()
    module._compiled_call_impl = compiled_call_impl

    with patch.object(torch, "compile", return_value=compiled_forward) as compile_mock:
        saved = _prepare_compiled_modules_for_capture([module])

    compile_mock.assert_called_once_with(
        original_forward, dynamic=False, options={"triton.cudagraphs": False}
    )
    assert module._compiled_call_impl is None
    assert module.forward is compiled_forward

    _restore_compiled_modules_after_capture_failure(saved)
    assert module._compiled_call_impl is compiled_call_impl
    assert module.forward == original_forward


def test_module_compile_is_normalized_when_first_forward_is_recorded():
    module = torch.nn.Linear(2, 2)
    module._compiled_call_impl = object()

    def compiled_forward(input):
        return input

    runner = CudaGraphRunner(graph_pool=None)
    sample = torch.ones(1, 2)

    with patch.object(torch, "compile", return_value=compiled_forward):
        runner.record_module(module, (sample,), {})

    assert module._compiled_call_impl is None
    assert module.forward is compiled_forward
    assert len(runner._compiled_module_state) == 1


def test_cuda_graph_runner_reset_before_capture_restores_module_compile():
    """Restore Module.compile state when capture has not succeeded."""
    module = torch.nn.Linear(2, 2)
    original_forward = module.forward
    compiled_call_impl = object()

    def compiled_forward(input_tensor):
        """Return the pre-capture reset test input unchanged.

        :param input_tensor: Test input.
        :type input_tensor: torch.Tensor
        :return: Unchanged test input.
        :rtype: torch.Tensor
        """
        return input_tensor

    module._compiled_call_impl = compiled_call_impl
    runner = CudaGraphRunner(graph_pool=None)

    with patch.object(torch, "compile", return_value=compiled_forward):
        runner.record_module(module, (torch.ones(1, 2),), {})
    runner.reset()

    assert module.forward == original_forward
    assert module._compiled_call_impl is compiled_call_impl


def test_cuda_graph_runner_reset_after_capture_keeps_compiled_forward():
    """Keep the installed compiled-forward semantics after graph release."""
    module = torch.nn.Linear(2, 2)
    compiled_call_impl = object()

    def compiled_forward(input_tensor):
        """Return the compiled test input unchanged.

        :param input_tensor: Test input.
        :type input_tensor: torch.Tensor
        :return: Unchanged test input.
        :rtype: torch.Tensor
        """
        return input_tensor

    module._compiled_call_impl = compiled_call_impl
    runner = CudaGraphRunner(graph_pool=None)
    with patch.object(torch, "compile", return_value=compiled_forward):
        runner.record_module(module, (torch.ones(1, 2),), {})
    module.forward = lambda input_tensor: input_tensor + 1
    runner._captured = True

    runner.reset()

    assert module.forward is compiled_forward
    assert module._compiled_call_impl is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_capture_failure_restores_module_compile_state():
    """Preserve saved Module.compile state across reset during failure recovery."""
    module = torch.nn.Linear(2, 2, device="cuda")
    module._fsdp_param_groups = ()
    module.reshard = lambda: None
    original_forward = module.forward
    compiled_call_impl = object()
    module._compiled_call_impl = compiled_call_impl

    def compiled_forward(input_tensor):
        """Return the capture test output.

        :param input_tensor: Capture test input.
        :type input_tensor: torch.Tensor
        :return: Linear output.
        :rtype: torch.Tensor
        """
        return torch.nn.functional.linear(input_tensor, module.weight, module.bias)

    def fail_capture(*args, use_main_grad=False, **kwargs):
        """Fail the installed-runtime capture call.

        :param args: Positional capture arguments.
        :type args: Any
        :param use_main_grad: Static main-grad binding flag, defaults to False.
        :type use_main_grad: bool, optional
        :param kwargs: Keyword capture arguments.
        :type kwargs: Any
        :raises RuntimeError: Always, to exercise failure recovery.
        """
        del args, use_main_grad, kwargs
        raise RuntimeError("capture failed")

    fake_runtime = ModuleType("te_graph_runtime")
    fake_runtime.make_graphed_callables = fail_capture
    fake_graph = ModuleType("te_graph_runtime.graph")
    fake_graph._MFSDP_CAPTURE_CAPABILITIES = {
        "capture_grad_buffer_release",
        "parameter_surface_refresh",
        "registered_buffer_validation",
        "static_grad_binding",
    }
    fake_graph._get_compatible_main_grad_buffer = lambda input_tensor: None
    fake_graph._refresh_module_parameter_surface = lambda func, inputs: ((), inputs)

    runner = CudaGraphRunner(graph_pool=None)
    sample = torch.ones(1, 2, device="cuda")
    with patch.object(torch, "compile", return_value=compiled_forward):
        runner.record_module(module, (sample,), {})
    runner.record_module_output(module, module(sample))

    with (
        patch.dict(
            sys.modules,
            {"te_graph_runtime": fake_runtime, "te_graph_runtime.graph": fake_graph},
        ),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        runner.capture_and_install(module)

    assert module.forward == original_forward
    assert module._compiled_call_impl is compiled_call_impl


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "missing_capability",
    [
        "activation_recompute_argument_binding",
        "activation_recompute_discard_tape",
        "activation_recompute_preflight",
        "fp8_activation_recompute_metadata",
        "static_dgrad_reuse",
        "static_fwd_reuse",
    ],
)
def test_activation_recompute_falls_back_without_required_capability(missing_capability):
    """Use vendored recompute when an installed correctness capability is missing."""
    module = torch.nn.Linear(2, 2, device="cuda")
    module._fsdp_param_groups = ()
    module.reshard = lambda: None
    module.gradient_checkpointing = True
    sample = torch.ones(1, 2, device="cuda")
    runner = CudaGraphRunner(graph_pool=None)
    runner.record_module(module, (sample,), {})
    runner.record_module_output(module, module(sample))
    runner.record_module_backward(module)

    def installed_runtime(*args, use_main_grad=False, _activation_recompute=False, **kwargs):
        del args, use_main_grad, _activation_recompute, kwargs
        raise AssertionError("old installed runtime selected")

    def vendored_runtime(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("vendored runtime selected")

    fake_runtime = ModuleType("te_graph_runtime")
    fake_runtime.make_graphed_callables = installed_runtime
    fake_graph = ModuleType("te_graph_runtime.graph")
    fake_graph._MFSDP_CAPTURE_CAPABILITIES = set(_MFSDP_CAPTURE_CAPABILITIES) - {
        missing_capability
    }
    fake_graph._get_compatible_main_grad_buffer = lambda input_tensor: None
    fake_graph._refresh_module_parameter_surface = lambda func, inputs: ((), inputs)
    with (
        patch.dict(
            sys.modules,
            {"te_graph_runtime": fake_runtime, "te_graph_runtime.graph": fake_graph},
        ),
        patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime."
            "make_graphed_callables",
            new=vendored_runtime,
        ),
        pytest.raises(RuntimeError, match="vendored runtime selected"),
    ):
        runner.capture_and_install(module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_runner_falls_back_without_registered_buffer_validation():
    """Select vendored runtime when an installed correctness capability is absent."""
    module = torch.nn.Linear(2, 2, device="cuda")
    module._fsdp_param_groups = ()
    module.reshard = lambda: None
    sample = torch.ones(1, 2, device="cuda")
    runner = CudaGraphRunner(graph_pool=None)
    runner.record_module(module, (sample,), {})
    runner.record_module_output(module, module(sample))

    def installed_runtime(*args, use_main_grad=False, **kwargs):
        """Reject selection of the incomplete installed runtime.

        :param args: Positional capture arguments.
        :type args: Any
        :param use_main_grad: Static main-grad flag, defaults to False.
        :type use_main_grad: bool, optional
        :param kwargs: Keyword capture arguments.
        :type kwargs: Any
        :raises AssertionError: Always.
        """
        del args, use_main_grad, kwargs
        raise AssertionError("old installed runtime selected")

    def vendored_runtime(*args, **kwargs):
        """Mark successful vendored-runtime selection.

        :param args: Positional capture arguments.
        :type args: Any
        :param kwargs: Keyword capture arguments.
        :type kwargs: Any
        :raises RuntimeError: Always after selection.
        """
        del args, kwargs
        raise RuntimeError("vendored runtime selected")

    capabilities = set(_MFSDP_CAPTURE_CAPABILITIES) - {
        "registered_buffer_validation"
    }
    fake_runtime = ModuleType("te_graph_runtime")
    fake_runtime.make_graphed_callables = installed_runtime
    fake_graph = ModuleType("te_graph_runtime.graph")
    fake_graph._MFSDP_CAPTURE_CAPABILITIES = capabilities
    fake_graph._get_compatible_main_grad_buffer = lambda input_tensor: None
    fake_graph._refresh_module_parameter_surface = lambda func, inputs: ((), inputs)
    with (
        patch.dict(
            sys.modules,
            {"te_graph_runtime": fake_runtime, "te_graph_runtime.graph": fake_graph},
        ),
        patch(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime."
            "make_graphed_callables",
            new=vendored_runtime,
        ),
        pytest.raises(RuntimeError, match="vendored runtime selected"),
    ):
        runner.capture_and_install(module)










def test_graph_context_wrapper_restores_python_gc_after_failure():
    """Re-enable Python GC when a graph capture body raises."""
    _require_torch()
    gc_was_enabled = gc.isenabled()
    gc.enable()
    try:
        with patch.object(torch.cuda, "graph", return_value=nullcontext()):
            with pytest.raises(RuntimeError, match="capture failed"):
                with _graph_context_wrapper():
                    assert not gc.isenabled()
                    raise RuntimeError("capture failed")
        assert gc.isenabled()
    finally:
        if not gc_was_enabled:
            gc.disable()


def test_cuda_graph_rejects_legacy_tensor_rng_tracker_state():
    """Reject legacy tensor RNG tracker values before graph warmup."""
    rng_patch = patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph."
        "get_all_rng_states",
        return_value={"legacy": torch.get_rng_state()},
    )
    with rng_patch, pytest.raises(RuntimeError, match="Legacy tensor RNG tracker states"):
        _get_tracked_cuda_generators()


def test_cuda_graph_uses_rng_fallback_for_legacy_tracker_without_recompute():
    """Select the CUDA RNG-state fallback for a legacy non-recompute tracker."""
    rng_patch = patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph."
        "get_all_rng_states",
        return_value={"legacy": torch.get_rng_state()},
    )
    with rng_patch:
        assert _get_tracked_cuda_generators(require_generators=False) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_retained_graph_discovers_legacy_tracker_for_cuda_rng_fallback():
    """Discover a legacy tracker before selecting the CUDA RNG fallback."""
    module = torch.nn.Linear(2, 2, device="cuda")
    sample = torch.ones(1, 2, device="cuda")
    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )
    with (
        patch(f"{graph_module}._prepare_runtime", return_value=False),
        patch(f"{graph_module}.graph_safe_rng_available", return_value=False),
        patch(f"{graph_module}._get_tracked_cuda_generators", return_value=None) as discover,
    ):
        graphed = make_graphed_callables(
            module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
        )
        graphed(input=sample).sum().backward()

    discover.assert_called_once_with(require_generators=False)
    assert not hasattr(graphed, "_cuda_graph_static_io_storage_records")




@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_make_graphed_callables_restores_capture_state_after_failure():
    """Restore RNG, class dispatch, FP8 state, and capture state after failure."""
    module = torch.nn.Linear(2, 2, device="cuda")
    module_class = type(module)
    original_call = module_class.__call__
    original_rng_state = torch.cuda.get_rng_state()
    saved_fp8_tensors = [object()]
    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )

    def fail_after_rng_use(*args, **kwargs):
        """Consume CUDA RNG and fail the mocked graph capture.

        :param args: Positional capture arguments.
        :type args: Any
        :param kwargs: Keyword capture arguments.
        :type kwargs: Any
        :raises RuntimeError: Always, after consuming CUDA RNG.
        """
        del args, kwargs
        torch.rand(1, device="cuda")
        raise RuntimeError("capture failed")

    with (
        patch(f"{graph_module}.save_fp8_tensors", return_value=saved_fp8_tensors),
        patch(f"{graph_module}.restore_fp8_tensors") as restore_fp8,
        patch(f"{graph_module}._make_graphed_callables", side_effect=fail_after_rng_use),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        make_graphed_callables(
            module,
            (),
            sample_kwargs={"input": torch.ones(1, 2, device="cuda")},
            num_warmup_iters=1,
        )

    assert module_class.__call__ is original_call
    assert not is_graph_capturing()
    torch.testing.assert_close(torch.cuda.get_rng_state(), original_rng_state)
    restore_fp8.assert_called_once_with((module,), saved_fp8_tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_failure_restores_discovered_generator_without_graph_safe_rng():
    """Restore tracked Generator state when recompute capture rejects graph-safe RNG."""
    module = torch.nn.Linear(2, 2, device="cuda")
    tracked = torch.Generator(device="cuda").manual_seed(1234)
    tracked_state = tracked.get_state().clone()
    default_state = torch.cuda.get_rng_state().clone()
    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )

    def fail_after_rng_use(*args, **kwargs):
        """Advance both generators and fail capture.

        :param args: Positional capture arguments.
        :type args: Any
        :param kwargs: Keyword capture arguments.
        :type kwargs: Any
        :raises RuntimeError: Always after advancing RNG state.
        """
        del args, kwargs
        torch.rand(1, device="cuda", generator=tracked)
        torch.rand(1, device="cuda")
        raise RuntimeError("capture failed")

    with (
        patch(f"{graph_module}._prepare_runtime", return_value=False),
        patch(f"{graph_module}.graph_safe_rng_available", return_value=False),
        patch(f"{graph_module}._get_tracked_cuda_generators", return_value=(tracked,)),
        patch(f"{graph_module}._make_graphed_callables", side_effect=fail_after_rng_use),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        make_graphed_callables(
            module,
            (),
            sample_kwargs={"input": torch.ones(1, 2, device="cuda")},
            num_warmup_iters=1,
            _activation_recompute=True,
        )

    torch.testing.assert_close(tracked.get_state(), tracked_state)
    torch.testing.assert_close(torch.cuda.get_rng_state(), default_state)


def test_none_grad_context_restores_leaf_grad_after_exception():
    """Restore direct-runtime leaf gradients when capture backward fails."""
    param = torch.nn.Parameter(torch.ones(4))
    original_grad = torch.full_like(param, 3)
    param.grad = original_grad

    with pytest.raises(RuntimeError, match="backward failed"):
        with _none_grad_context_wrapper((param,)):
            assert param.grad is None
            raise RuntimeError("backward failed")

    assert param.grad is original_grad


def test_parameter_surface_refresh_uses_current_registered_parameters():
    """Use parameters installed by a capture-time replacement hook."""
    module = torch.nn.Linear(2, 2)
    replacement_weight = torch.nn.Parameter(torch.full_like(module.weight, 2))
    replacement_bias = torch.nn.Parameter(torch.full_like(module.bias, 3))
    module.weight = replacement_weight
    module.bias = replacement_bias
    user_inputs = (torch.ones(1, 2),)

    module_params, input_surface = _refresh_module_parameter_surface(
        module, user_inputs, parameter_indices=(0,)
    )

    assert len(module_params) == 1
    assert module_params[0] is replacement_weight
    assert input_surface[0] is user_inputs[0]
    assert input_surface[1] is replacement_weight


def test_static_grad_context_uses_main_grad_and_restores_leaf_state():
    """Bind a main-grad buffer during capture and restore the original leaf state."""
    param = torch.nn.Parameter(torch.ones(4))
    original_grad = torch.full_like(param, 5)
    main_grad = torch.full_like(param, 7)
    param.grad = original_grad
    param.get_main_grad = lambda: main_grad

    grad_buffers = _get_static_grad_buffers((param,))
    with _static_grad_context_wrapper((param,), grad_buffers):
        assert param.grad is main_grad
        assert torch.count_nonzero(main_grad) == 0

    assert param.grad is original_grad


def test_static_grad_buffer_rejects_incompatible_gradient_contract():
    """Keep incompatible main grads on the normal autograd-owned path."""
    param = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))

    param.get_main_grad = lambda: torch.zeros(2, 2, dtype=torch.float32)
    assert _get_compatible_main_grad_buffer(param) is None

    param.get_main_grad = lambda: torch.zeros(4, dtype=torch.bfloat16)
    assert _get_compatible_main_grad_buffer(param) is None

    transposed_main_grad = torch.zeros(2, 2, dtype=torch.bfloat16).t()
    param.get_main_grad = lambda: transposed_main_grad
    assert _get_compatible_main_grad_buffer(param) is None


def test_static_grad_buffer_skips_mfsdp_accumulation_strategy():
    """Keep accumulated M-FSDP gradients on the autograd-owned path."""
    param = torch.nn.Parameter(torch.ones(4))
    main_grad = torch.zeros_like(param)
    getter_calls = []
    param.__fsdp_param__ = True
    param.overwrite_main_grad = False
    param.get_main_grad = lambda: getter_calls.append(True) or main_grad

    assert _get_compatible_main_grad_buffer(param) is None
    assert getter_calls == []


def test_static_grad_buffer_skips_recorded_te_fused_wgrad():
    """Keep TE dummy wgrad separate from its fused main-grad destination."""
    param = torch.nn.Parameter(torch.ones(4))
    main_grad = torch.zeros_like(param)
    getter_calls = []
    param._mfsdp_recorded_te_wgrad = True
    param.get_main_grad = lambda: getter_calls.append(True) or main_grad

    assert _get_compatible_main_grad_buffer(param) is None
    assert getter_calls == []


def test_static_grad_buffer_checks_mfsdp_dtype_before_fetch():
    """Reject mixed-dtype M-FSDP main grad before materializing its buffer."""
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    main_grad = torch.zeros(4, dtype=torch.float32)
    getter_calls = []
    param.__fsdp_param__ = True
    param.overwrite_main_grad = True
    param._gbuf = SimpleNamespace(dtype=torch.float32)
    param.get_main_grad = lambda: getter_calls.append(True) or main_grad

    assert _get_compatible_main_grad_buffer(param) is None
    assert getter_calls == []


def test_capture_backward_pre_hook_prefetches_only_te_fused_wgrad():
    """Materialize capture buffers only for recorded TE fused-wgrad groups."""
    fused_param = torch.nn.Parameter(torch.ones(4))
    fused_param._mfsdp_recorded_te_wgrad = True
    fused_main_grad = torch.full_like(fused_param, 2)
    fused_param.get_main_grad = lambda: fused_main_grad
    regular_param = torch.nn.Parameter(torch.ones(4))
    fused_init_calls = []
    fused_fetch_calls = []
    regular_init_calls = []
    regular_fetch_calls = []
    fused_group = SimpleNamespace(
        params=(fused_param,),
        sharding_strategy="optim_grads_params",
        main_grad_buffer=SimpleNamespace(fetch_buffer=lambda: fused_fetch_calls.append(True)),
        _init_dist_grads=lambda: fused_init_calls.append(True),
    )
    regular_group = SimpleNamespace(
        params=(regular_param,),
        sharding_strategy="optim_grads_params",
        main_grad_buffer=SimpleNamespace(fetch_buffer=lambda: regular_fetch_calls.append(True)),
        _init_dist_grads=lambda: regular_init_calls.append(True),
    )
    unshard_calls = []
    module = SimpleNamespace(
        _fsdp_param_groups=(fused_group, regular_group),
        unshard=lambda **kwargs: unshard_calls.append(kwargs),
    )

    _make_bwd_pre_hook(module)(module, ())

    assert unshard_calls == [{"bwd_pass": True}]
    assert fused_init_calls == [True]
    assert fused_fetch_calls == [True]
    assert fused_param.main_grad is fused_main_grad
    assert regular_init_calls == []
    assert regular_fetch_calls == []


def test_trace_prefetches_static_main_grad_before_backward():
    """Trace the main-grad lifetime used by CUDA graph replay."""
    param = torch.nn.Parameter(torch.ones(4))
    init_calls = []
    fetch_calls = []
    unshard_calls = []
    param_group = SimpleNamespace(
        params=(param,),
        requires_grad=True,
        sharding_strategy="optim_grads_params",
        main_grad_buffer=SimpleNamespace(
            dtype=param.dtype, fetch_buffer=lambda: fetch_calls.append(True)
        ),
        _init_dist_grads=lambda: init_calls.append(True),
    )
    module = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(cuda_graph_active=False, enable_unshard_prefetch=False),
        _fsdp_state=SimpleNamespace(_is_root=False, enable_cuda_graph=True),
        _fsdp_param_groups=(param_group,),
        unshard=lambda **kwargs: unshard_calls.append(kwargs),
    )

    _pre_backward_setup(module)

    assert unshard_calls == [{"async_op": False, "bwd_pass": True}]
    assert param.overwrite_main_grad
    assert init_calls == [True]
    assert fetch_calls == [True]


def test_trace_prefetches_mixed_dtype_te_fused_wgrad_before_backward():
    """Fetch a mixed-dtype main-grad buffer used by recorded TE fused wgrad."""
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    param._mfsdp_recorded_te_wgrad = True
    main_grad = torch.zeros(4, dtype=torch.float32)
    param.get_main_grad = lambda: main_grad
    init_calls = []
    fetch_calls = []
    param_group = SimpleNamespace(
        params=(param,),
        requires_grad=True,
        sharding_strategy="optim_grads_params",
        main_grad_buffer=SimpleNamespace(
            dtype=torch.float32, fetch_buffer=lambda: fetch_calls.append(True)
        ),
        _init_dist_grads=lambda: init_calls.append(True),
    )
    module = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(cuda_graph_active=False, enable_unshard_prefetch=False),
        _fsdp_state=SimpleNamespace(_is_root=False, enable_cuda_graph=True),
        _fsdp_param_groups=(param_group,),
        unshard=lambda **kwargs: None,
    )

    _pre_backward_setup(module)

    assert init_calls == [True]
    assert fetch_calls == [True]
    assert param.main_grad is main_grad


def test_static_grad_context_accumulates_into_main_grad_buffer():
    """Accumulate an autograd result directly into a compatible main-grad buffer."""
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    main_grad = torch.zeros_like(param)
    param.get_main_grad = lambda: main_grad

    grad_buffers = _get_static_grad_buffers((param,))
    with _static_grad_context_wrapper((param,), grad_buffers):
        param.square().sum().backward()
        assert param.grad is main_grad
        torch.testing.assert_close(main_grad, torch.tensor([2.0, 4.0]))

    assert param.grad is None


def test_make_graphed_callables_rejects_non_bool_use_main_grad():
    """Reject ambiguous main-grad capture configuration."""
    with pytest.raises(TypeError, match="use_main_grad must be a bool"):
        make_graphed_callables(torch.nn.Linear(2, 2), (), use_main_grad=None)


def test_make_graphed_callables_rejects_non_bool_activation_recompute():
    """Reject ambiguous activation-recompute configuration."""
    with pytest.raises(TypeError, match="_activation_recompute must be a bool"):
        make_graphed_callables(
            torch.nn.Linear(2, 2), (), _activation_recompute=None
        )






@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_activation_recompute_matches_eager_chain():
    """Discard the initial forward tape and recompute each block in backward."""
    torch.manual_seed(1234)
    first = torch.nn.Sequential(
        torch.nn.Linear(8, 16, device="cuda"),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 8, device="cuda"),
    )
    second = copy.deepcopy(first)
    eager_first = copy.deepcopy(first)
    eager_second = copy.deepcopy(second)
    grad_modes = {id(first): [], id(second): []}

    for module in (first, second):
        original_forward = module.forward

        def record_grad_mode(input_tensor, module=module, original_forward=original_forward):
            grad_modes[id(module)].append(torch.is_grad_enabled())
            return original_forward(input_tensor)

        module.forward = record_grad_mode

    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    first_graph, second_graph = make_graphed_callables(
        (first, second),
        ((), ()),
        sample_kwargs=(
            {"input_tensor": sample},
            {"input_tensor": sample.detach().clone().requires_grad_()},
        ),
        num_warmup_iters=1,
        _input_output_aliases=({}, {0: (0, 0)}),
        _activation_recompute=True,
        _reuse_graph_input_output_buffers=True,
    )

    graph_input = torch.randn_like(sample, requires_grad=True)
    eager_input = graph_input.detach().clone().requires_grad_()
    graph_output = second_graph(input_tensor=first_graph(input_tensor=graph_input))
    eager_output = eager_second(eager_first(eager_input))
    graph_output.square().mean().backward()
    eager_output.square().mean().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, eager_output)
    torch.testing.assert_close(graph_input.grad, eager_input.grad)
    for graph_param, eager_param in zip(
        tuple(first.parameters()) + tuple(second.parameters()),
        tuple(eager_first.parameters()) + tuple(eager_second.parameters()),
    ):
        torch.testing.assert_close(graph_param.grad, eager_param.grad)
    assert grad_modes[id(first)][-2:] == [True, True]
    assert grad_modes[id(second)][-2:] == [True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_weakens_internal_forward_boundary():
    """Release a linear internal boundary after its consumer backward capture."""
    first = torch.nn.Linear(8, 8, device="cuda")
    second = torch.nn.Linear(8, 8, device="cuda")
    sample = torch.randn(3, 8, device="cuda", requires_grad=True)
    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )

    with patch(f"{graph_module}.make_weak_ref", wraps=make_weak_ref) as weak_ref:
        first_graph, second_graph = make_graphed_callables(
            (first, second),
            ((sample,), (sample.detach().clone().requires_grad_(),)),
            num_warmup_iters=1,
            _input_output_aliases=({}, {0: (0, 0)}),
            _activation_recompute=True,
            _reuse_graph_input_output_buffers=True,
        )

    graph_input = torch.randn_like(sample, requires_grad=True)
    second_graph(first_graph(graph_input)).square().sum().backward()
    torch.cuda.synchronize()

    assert weak_ref.call_count == 2
    assert graph_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_reuses_adjacent_dgrad_ping_pong():
    """Use two alternating user-dgrad slots across a four-layer chain."""
    torch.manual_seed(1234)
    modules = tuple(torch.nn.Linear(8, 8, device="cuda") for _ in range(4))
    eager_modules = tuple(copy.deepcopy(module) for module in modules)
    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    observed_user_grad_buffers = []

    def record_static_grad_buffers(inputs, grad_buffers):
        """Record the explicit user-dgrad slot passed into backward capture."""
        observed_user_grad_buffers.append(
            grad_buffers[0].data_ptr() if grad_buffers[0] is not None else None
        )
        return _static_grad_context_wrapper(inputs, grad_buffers)

    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )
    with patch(
        f"{graph_module}._static_grad_context_wrapper",
        new=record_static_grad_buffers,
    ):
        graphed_modules = make_graphed_callables(
            modules,
            ((), (), (), ()),
            sample_kwargs=tuple(
                {"input": sample.detach().clone().requires_grad_()} for _ in modules
            ),
            num_warmup_iters=1,
            _input_output_aliases=({}, {0: (0, 0)}, {0: (1, 0)}, {0: (2, 0)}),
            _activation_recompute=True,
        )

    graph_input = torch.randn_like(sample, requires_grad=True)
    eager_input = graph_input.detach().clone().requires_grad_()
    graph_output = graph_input
    eager_output = eager_input
    for graphed_module, eager_module in zip(graphed_modules, eager_modules):
        graph_output = graphed_module(input=graph_output)
        eager_output = eager_module(eager_output)
    graph_output.square().mean().backward()
    eager_output.square().mean().backward()
    torch.cuda.synchronize()

    assert len(observed_user_grad_buffers) == 4
    assert observed_user_grad_buffers[0] is not None
    assert observed_user_grad_buffers[1] is not None
    assert observed_user_grad_buffers[0] == observed_user_grad_buffers[2]
    assert observed_user_grad_buffers[0] != observed_user_grad_buffers[1]
    assert observed_user_grad_buffers[3] is None
    torch.testing.assert_close(graph_output, eager_output)
    torch.testing.assert_close(graph_input.grad, eager_input.grad)
    for module, eager_module in zip(modules, eager_modules):
        for param, eager_param in zip(module.parameters(), eager_module.parameters()):
            torch.testing.assert_close(param.grad, eager_param.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_keeps_independent_dgrad_chains_separate():
    """Do not share ping-pong slots between independent dependency chains."""
    modules = tuple(torch.nn.Linear(4, 4, bias=False, device="cuda") for _ in range(4))
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    observed_user_grad_buffers = []

    def record_static_grad_buffers(inputs, grad_buffers):
        """Record the explicit user-dgrad slots used during capture."""
        observed_user_grad_buffers.append(
            grad_buffers[0].data_ptr() if grad_buffers[0] is not None else None
        )
        return _static_grad_context_wrapper(inputs, grad_buffers)

    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )
    with patch(
        f"{graph_module}._static_grad_context_wrapper",
        new=record_static_grad_buffers,
    ):
        make_graphed_callables(
            modules,
            ((), (), (), ()),
            sample_kwargs=tuple(
                {"input": sample.detach().clone().requires_grad_()} for _ in modules
            ),
            num_warmup_iters=1,
            _input_output_aliases=({}, {0: (0, 0)}, {}, {0: (2, 0)}),
            _activation_recompute=True,
        )

    reused_buffers = [pointer for pointer in observed_user_grad_buffers if pointer is not None]
    assert len(reused_buffers) == 2
    assert len(set(reused_buffers)) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_dgrad_reuse_falls_back_for_fanout():
    """Keep separate dgrad allocations when one output has two consumers."""
    modules = tuple(torch.nn.Linear(4, 4, bias=False, device="cuda") for _ in range(3))
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    observed_user_grad_buffers = []

    def record_static_grad_buffers(inputs, grad_buffers):
        """Record the first explicit user-dgrad slot."""
        observed_user_grad_buffers.append(
            grad_buffers[0].data_ptr() if grad_buffers[0] is not None else None
        )
        return _static_grad_context_wrapper(inputs, grad_buffers)

    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )
    with patch(
        f"{graph_module}._static_grad_context_wrapper",
        new=record_static_grad_buffers,
    ):
        make_graphed_callables(
            modules,
            ((), (), ()),
            sample_kwargs=tuple(
                {"input": sample.detach().clone().requires_grad_()} for _ in modules
            ),
            num_warmup_iters=1,
            _input_output_aliases=({}, {0: (0, 0)}, {0: (0, 0)}),
            _activation_recompute=True,
        )

    assert observed_user_grad_buffers == [None, None, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_dgrad_reuse_skips_unused_input():
    """Do not bind a buffer for an aliased input whose warmup grad is None."""

    class IgnoreSecond(torch.nn.Module):
        """Use only the first of two tensor inputs."""

        def forward(self, used, unused):
            """Square the used input and ignore the second input."""
            del unused
            return used.square()

    producer = torch.nn.Linear(4, 4, bias=False, device="cuda")
    consumer = IgnoreSecond().cuda()
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    observed_grad_buffers = []

    def record_static_grad_buffers(inputs, grad_buffers):
        """Record explicit buffers in reverse capture order."""
        observed_grad_buffers.append(tuple(grad_buffers[: len(inputs)]))
        return _static_grad_context_wrapper(inputs, grad_buffers)

    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )
    with patch(
        f"{graph_module}._static_grad_context_wrapper",
        new=record_static_grad_buffers,
    ):
        make_graphed_callables(
            (producer, consumer),
            (
                (sample,),
                (
                    sample.detach().clone().requires_grad_(),
                    sample.detach().clone().requires_grad_(),
                ),
            ),
            num_warmup_iters=1,
            allow_unused_input=True,
            _input_output_aliases=({}, {1: (0, 0)}),
            _activation_recompute=True,
        )

    consumer_buffers = observed_grad_buffers[0]
    assert consumer_buffers[0] is None
    assert consumer_buffers[1] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_dgrad_reuse_rejects_views_and_overlapping_tensors():
    """Reject layouts that cannot safely use an independent strided grad slot."""
    tensor = torch.ones(4, 4, device="cuda")
    assert _static_dgrad_metadata(tensor) is not None
    assert _static_dgrad_metadata(tensor[:, 1:]) is None
    assert _static_dgrad_metadata(tensor[:1].expand(4, 4)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_activation_recompute_rejects_concurrent_microbatches():
    """Reject F1 before overwrite, finish F0, then accept a serial F2."""
    module = torch.nn.Linear(4, 4, device="cuda")
    eager = copy.deepcopy(module)
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )

    first_input = torch.randn_like(sample, requires_grad=True)
    eager_first_input = first_input.detach().clone().requires_grad_()
    first_output = graphed(input=first_input)
    eager_first_output = eager(eager_first_input)
    saved_first_output = first_output.detach().clone()
    with pytest.raises(RuntimeError, match="backward to finish before the next forward"):
        graphed(input=torch.randn_like(sample, requires_grad=True))
    with torch.no_grad(), pytest.raises(
        RuntimeError, match="backward to finish before the next forward"
    ):
        graphed(input=torch.randn_like(sample))
    torch.testing.assert_close(first_output, saved_first_output)

    first_output.square().sum().backward()
    eager_first_output.square().sum().backward()
    torch.testing.assert_close(first_input.grad, eager_first_input.grad)
    torch.testing.assert_close(module.weight.grad, eager.weight.grad)

    module.zero_grad(set_to_none=True)
    second_input = torch.randn_like(sample, requires_grad=True)
    second_output = graphed(input=second_input)
    second_output.sum().backward()
    assert second_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_forward_failure_allows_retry():
    """Clear pending-backward state when the first forward replay raises."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )
    failures = []

    def fail_first_forward():
        """Fail once after the pending-backward state is set."""
        if not failures:
            failures.append(True)
            raise RuntimeError("injected forward replay failure")
        return False

    graph_module = (
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph"
    )
    with patch(
        f"{graph_module}.FP8GlobalStateManager.is_first_fp8_module",
        new=fail_first_forward,
    ):
        with pytest.raises(RuntimeError, match="injected forward replay failure"):
            graphed(input=torch.full_like(sample, 2.0, requires_grad=True))
        retry_input = torch.full_like(sample, 3.0, requires_grad=True)
        graphed(input=retry_input).sum().backward()

    assert retry_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_bypasses_no_grad_then_runs_normal_and_reentrant():
    """Match eager gradients through normal and reentrant checkpoint paths."""
    module = torch.nn.Linear(4, 4, device="cuda")
    eager_module = copy.deepcopy(module)
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )

    with torch.no_grad():
        no_grad_output = graphed(input=torch.full_like(sample, 2.0))
    assert no_grad_output.grad_fn is None

    normal_input = torch.full_like(sample, 3.0, requires_grad=True)
    graphed(input=normal_input).sum().backward()
    assert normal_input.grad is not None

    module.zero_grad(set_to_none=True)
    reentrant_input = torch.full_like(sample, 4.0, requires_grad=True)
    eager_input = reentrant_input.detach().clone().requires_grad_()
    reentrant_output = torch.utils.checkpoint.checkpoint(
        lambda value: graphed(input=value), reentrant_input, use_reentrant=True
    )
    eager_output = torch.utils.checkpoint.checkpoint(
        eager_module, eager_input, use_reentrant=True
    )
    reentrant_output.sum().backward()
    eager_output.sum().backward()

    torch.testing.assert_close(reentrant_output, eager_output)
    torch.testing.assert_close(reentrant_input.grad, eager_input.grad)
    for graph_param, eager_param in zip(module.parameters(), eager_module.parameters()):
        torch.testing.assert_close(graph_param.grad, eager_param.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_reentrant_checkpoint_runs_two_forwards():
    """Avoid a third forward inside the recomputed-forward-plus-backward graph."""

    class StatefulLinear(torch.nn.Module):
        """Count every eager or graphed forward on a registered CUDA buffer."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, device="cuda")
            self.register_buffer("forward_count", torch.zeros((), device="cuda"))

        def forward(self, input_tensor):
            self.forward_count.add_(1)
            return self.linear(input_tensor)

    module = StatefulLinear()
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input_tensor": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )
    module.forward_count.zero_()

    runtime_input = torch.full_like(sample, 2.0, requires_grad=True)
    output = torch.utils.checkpoint.checkpoint(
        lambda value: graphed(input_tensor=value),
        runtime_input,
        use_reentrant=True,
    )
    output.sum().backward()
    torch.cuda.synchronize()

    assert module.forward_count.item() == 2
    assert runtime_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_releases_abandoned_output():
    """Release the replay slot after an unused backward node is destroyed."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )

    abandoned = graphed(input=torch.full_like(sample, 2.0, requires_grad=True))
    backward_node = weakref.ref(abandoned.grad_fn)
    del abandoned
    gc.collect()
    assert backward_node() is None

    runtime_input = torch.full_like(sample, 3.0, requires_grad=True)
    graphed(input=runtime_input).sum().backward()
    assert runtime_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_keeps_consumed_output_pending_backward():
    """Keep a replay slot while a downstream edge retains its backward node."""
    torch.manual_seed(1234)
    module = torch.nn.Linear(4, 4, device="cuda")
    eager = copy.deepcopy(module)
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )

    runtime_input = torch.randn_like(sample, requires_grad=True)
    eager_input = runtime_input.detach().clone().requires_grad_()
    output = graphed(input=runtime_input)
    backward_node = weakref.ref(output.grad_fn)
    derived = output.square()
    del output
    gc.collect()

    assert backward_node() is not None
    with pytest.raises(RuntimeError, match="backward to finish before the next forward"):
        graphed(input=torch.randn_like(sample, requires_grad=True))

    derived.sum().backward()
    eager(eager_input).square().sum().backward()
    torch.testing.assert_close(runtime_input.grad, eager_input.grad)
    for graph_param, eager_param in zip(module.parameters(), eager.parameters()):
        torch.testing.assert_close(graph_param.grad, eager_param.grad)

    module.zero_grad(set_to_none=True)
    later_input = torch.randn_like(sample, requires_grad=True)
    graphed(input=later_input).sum().backward()
    assert later_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_rejects_custom_capture_order():
    """Reject custom-order capture before sharing the recompute activation pool."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)

    with pytest.raises(ValueError, match="does not support a custom capture order"):
        make_graphed_callables(
            module,
            (),
            sample_kwargs={"input": sample},
            num_warmup_iters=1,
            _activation_recompute=True,
            _order=[1],
            _num_layers_per_chunk=[1],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_runner_matches_autocast_eager_on_first_replay():
    """Capture and first-replay a module under its recorded BF16 autocast state."""
    torch.manual_seed(1234)
    module = torch.nn.Sequential(
        torch.nn.Linear(8, 16, device="cuda"),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 4, device="cuda"),
    )
    eager = copy.deepcopy(module)
    module._fsdp_param_groups = ()
    module.unshard = lambda **kwargs: None
    module.reshard = lambda: None
    module.gradient_checkpointing = True
    runner = CudaGraphRunner(graph_pool=None)
    sample = torch.randn(3, 8, device="cuda", requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        runner.record_module(module, (sample,), {})
        sample_output = module(sample)
    runner.record_module_output(module, sample_output)
    runner.record_module_backward(module)
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.weight = torch.nn.Parameter(layer.weight.detach().clone())
            layer.bias = torch.nn.Parameter(layer.bias.detach().clone())
    del sample_output, sample
    gc.collect()
    torch.cuda.synchronize()

    runner.capture_and_install(module)

    graph_input = torch.randn(3, 8, device="cuda", requires_grad=True)
    eager_input = graph_input.detach().clone().requires_grad_()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        graph_output = module(graph_input)
        eager_output = eager(eager_input)
    graph_output.float().square().sum().backward()
    eager_output.float().square().sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(graph_input.grad, eager_input.grad, rtol=0, atol=0)
    for graph_param, eager_param in zip(module.parameters(), eager.parameters()):
        torch.testing.assert_close(graph_param.grad, eager_param.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_rejects_runtime_requires_grad_change():
    """Reject a false-to-true input requires-grad change at replay."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=False)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )

    with pytest.raises(RuntimeError, match="tensor metadata changed"):
        graphed(input=sample.detach().clone().requires_grad_())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_rebinds_recorded_kwargs_from_positional_call():
    """Resolve tensor kwargs by signature and validate interleaved static defaults."""

    class StaticFlagModule(torch.nn.Module):
        """Add two tensors while exposing a static flag between them."""

        def forward(self, input, use_bias=False, context=None, metadata=None):
            """Add the input and context with an optional scalar bias."""
            assert metadata is None
            output = input + context
            return output + 1 if use_bias else output

    module = StaticFlagModule().cuda()
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    context = torch.full_like(sample, 2, requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample, "context": context, "metadata": None},
        num_warmup_iters=1,
    )

    runtime_input = torch.full_like(sample, 3, requires_grad=True)
    runtime_context = torch.full_like(context, 4, requires_grad=True)
    torch.testing.assert_close(
        graphed(runtime_input, False, runtime_context, None),
        runtime_input + runtime_context,
    )
    with pytest.raises(RuntimeError, match="static metadata changed"):
        graphed(runtime_input, True, runtime_context, None)
    with pytest.raises(RuntimeError, match="structure or static metadata changed"):
        graphed(runtime_input, False, runtime_context, runtime_input)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_reconstructs_positional_pytree_inputs():
    """Replay positional tensor PyTrees in their captured leaf order."""

    class PairSum(torch.nn.Module):
        """Sum a nested pair of positional tensors."""

        def forward(self, pair):
            return pair[0] + pair[1]["value"]

    module = PairSum().cuda()
    left = torch.ones(2, 4, device="cuda", requires_grad=True)
    right = torch.full_like(left, 2, requires_grad=True)
    graphed = make_graphed_callables(
        module,
        ((left, {"value": right}),),
        num_warmup_iters=1,
    )

    runtime_left = torch.full_like(left, 3, requires_grad=True)
    runtime_right = torch.full_like(right, 4, requires_grad=True)
    output = graphed((runtime_left, {"value": runtime_right}))
    output.sum().backward()

    torch.testing.assert_close(output, runtime_left + runtime_right)
    torch.testing.assert_close(runtime_left.grad, torch.ones_like(runtime_left))
    torch.testing.assert_close(runtime_right.grad, torch.ones_like(runtime_right))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_activation_recompute_restores_default_rng():
    """Use the forward dropout mask again during recompute."""
    torch.manual_seed(1234)
    module = torch.nn.Dropout(0.5).cuda()
    eager_module = copy.deepcopy(module)
    sample = torch.ones(4, 8, device="cuda", requires_grad=True)
    if not graph_safe_rng_available():
        with pytest.raises(RuntimeError, match="graph-safe generator state"):
            make_graphed_callables(
                    module,
                    (),
                    sample_kwargs={"input": sample},
                    num_warmup_iters=1,
                _activation_recompute=True,
            )
        return

    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        _activation_recompute=True,
    )

    for value in (1.0, 2.0):
        graph_input = torch.full_like(sample, value, requires_grad=True)
        eager_input = graph_input.detach().clone().requires_grad_()
        forward_rng_state = torch.cuda.get_rng_state()

        graph_output = graphed(input=graph_input)
        graph_output.square().mean().backward()
        graph_final_rng_state = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(forward_rng_state)
        eager_output = eager_module(eager_input)
        eager_output.square().mean().backward()
        eager_final_rng_state = torch.cuda.get_rng_state()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output)
        torch.testing.assert_close(graph_input.grad, eager_input.grad)
        for graph_param, eager_param in zip(module.parameters(), eager_module.parameters()):
            torch.testing.assert_close(graph_param.grad, eager_param.grad)
        torch.testing.assert_close(graph_final_rng_state, eager_final_rng_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_activation_recompute_rejects_zero_warmup():
    """Require warmup to discover output metadata, RNG, and delayed wgrad."""
    sample = torch.ones(4, 8, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="at least one warmup iteration"):
        make_graphed_callables(
            torch.nn.Dropout(0.5).cuda(),
            (),
            sample_kwargs={"input": sample},
            num_warmup_iters=0,
            _activation_recompute=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_activation_recompute_restores_tracked_rng():
    """Use the same tracked-generator values in forward and recompute."""
    if not graph_safe_rng_available():
        pytest.skip("Tracked recompute RNG requires graph-safe generator state")

    class TrackedRandom(torch.nn.Module):
        def __init__(self, generator):
            super().__init__()
            self.generator = generator

        def forward(self, input_tensor):
            random_values = torch.rand(
                input_tensor.shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
                generator=self.generator,
            )
            return input_tensor * random_values

    graph_generator = torch.Generator(device="cuda").manual_seed(4321)
    eager_generator = graph_generator.clone_state()
    module = TrackedRandom(graph_generator)
    eager_module = TrackedRandom(eager_generator)
    sample = torch.ones(4, 8, device="cuda", requires_grad=True)
    rng_patch = patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph."
        "get_all_rng_states",
        return_value={"tracked": graph_generator},
    )
    with rng_patch:
        graphed = make_graphed_callables(
            module,
            (),
            sample_kwargs={"input_tensor": sample},
            num_warmup_iters=1,
            _activation_recompute=True,
        )

    for value in (1.0, 2.0):
        graph_input = torch.full_like(sample, value, requires_grad=True)
        eager_input = graph_input.detach().clone().requires_grad_()
        eager_generator.set_state(graph_generator.get_state())

        graph_output = graphed(input_tensor=graph_input)
        graph_output.sum().backward()
        eager_output = eager_module(eager_input)
        eager_output.sum().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output)
        torch.testing.assert_close(graph_input.grad, eager_input.grad)
        torch.testing.assert_close(graph_generator.get_state(), eager_generator.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_activation_recompute_restores_delayed_scaling_fp8_meta():
    """Match eager delayed-scaling state, outputs, gradients, and updates."""
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch.distributed import (
        activation_recompute_forward as te_activation_recompute_forward,
    )
    from transformer_engine.pytorch.quantization import (
        FP8GlobalStateManager as TEFP8GlobalStateManager,
    )
    from transformer_engine.pytorch.quantization import autocast as te_autocast

    def run_steps(use_graph):
        """Run two deterministic delayed-scaling optimizer steps."""
        TEFP8GlobalStateManager.reset()
        torch.manual_seed(1234)
        recipe = DelayedScaling(amax_history_len=4)
        module = te.Linear(
            16,
            16,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
        if use_graph:
            sample = torch.ones(
                16, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            module = make_graphed_callables(
                module,
                (sample,),
                num_warmup_iters=1,
                enabled=True,
                recipe=recipe,
                _activation_recompute=True,
            )
            assert "global_fp8_buffer_pos_fwd_recompute" not in module.fp8_meta
            assert not TEFP8GlobalStateManager.quantization_state.fp8_tensors_recompute_buffer
            assert not te_activation_recompute_forward._is_first_fp8_module

        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        results = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            input_tensor = torch.full(
                (16, 16),
                step + 1.0,
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            with te_autocast(enabled=True, recipe=recipe):
                output = module(input_tensor)
            output.float().square().mean().backward()
            optimizer.step()
            torch.cuda.synchronize()
            results.append(
                (
                    output.detach().clone(),
                    input_tensor.grad.detach().clone(),
                    tuple(param.grad.detach().clone() for param in module.parameters()),
                    tuple(param.detach().clone() for param in module.parameters()),
                    module.fp8_meta["scaling_fwd"].amax_history.detach().clone(),
                    module.fp8_meta["scaling_fwd"].scale.detach().clone(),
                )
            )
        return results

    graph_results = run_steps(use_graph=True)
    eager_results = run_steps(use_graph=False)
    for graph_step, eager_step in zip(graph_results, eager_results):
        for graph_value, eager_value in zip(graph_step, eager_step):
            if isinstance(graph_value, tuple):
                for graph_tensor, eager_tensor in zip(graph_value, eager_value):
                    torch.testing.assert_close(graph_tensor, eager_tensor, rtol=0, atol=0)
            else:
                torch.testing.assert_close(graph_value, eager_value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_fp8_recompute_rejects_missing_metadata_support_before_capture():
    """Reject an incompatible TE runtime before CUDA graph capture starts."""
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch.quantization import (
        FP8GlobalStateManager as TEFP8GlobalStateManager,
    )

    module = te.Linear(16, 16, params_dtype=torch.bfloat16, device="cuda")
    sample = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with patch.object(TEFP8GlobalStateManager, "restore_fp8_meta_tensors", None):
        with pytest.raises(RuntimeError, match="missing FP8GlobalStateManager methods"):
            make_graphed_callables(
                module,
                (sample,),
                num_warmup_iters=1,
                enabled=True,
                recipe=DelayedScaling(amax_history_len=4),
                _activation_recompute=True,
            )
    assert not is_graph_capturing()




@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_recompute_rejects_delayed_wgrad_before_capture_hooks():
    """Reject delayed wgrad before FSDP capture hooks can release buffers."""

    class DelayedWgradModule(torch.nn.Linear):
        def __init__(self):
            super().__init__(4, 4, device="cuda")
            self.config = SimpleNamespace(delay_wgrad_compute=True)

    events = []
    capture_hooks = {
        "forward_pre_hooks": {0: lambda *args: events.append("forward")},
        "forward_pre_hooks_with_kwargs": {0: True},
        "forward_hooks": {},
        "forward_hooks_with_kwargs": {},
        "backward_pre_hooks": {0: lambda *args: events.append("backward")},
        "backward_hooks": {},
    }
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)

    with pytest.raises(RuntimeError, match="delayed backward-wgrad"):
        make_graphed_callables(
            DelayedWgradModule(),
            (),
            sample_kwargs={"input": sample},
            num_warmup_iters=1,
            capture_time_hooks=[capture_hooks],
            _activation_recompute=True,
        )
    assert events == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_replay_can_disable_main_grad_binding():
    """Keep parameter gradients on the normal autograd path when disabled."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda")
    main_grad = torch.zeros_like(module.weight)
    getter_calls = []
    module.weight.get_main_grad = lambda: getter_calls.append(True) or main_grad
    sample = torch.ones(2, 4, device="cuda")

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, use_main_grad=False
    )
    graphed(input=torch.full_like(sample, 2.0)).sum().backward()
    torch.cuda.synchronize()

    assert getter_calls == []
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 4.0))
    torch.testing.assert_close(main_grad, torch.zeros_like(main_grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_replay_preserves_mfsdp_microbatch_accumulation():
    """Accumulate two M-FSDP microbatches without static main-grad binding."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda")
    main_grad = torch.zeros_like(module.weight)
    module.weight.__fsdp_param__ = True
    module.weight.overwrite_main_grad = False
    module.weight.get_main_grad = lambda: main_grad
    sample = torch.ones(2, 4, device="cuda")

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, use_main_grad=True
    )

    for value in (2.0, 3.0):
        graphed(input=torch.full_like(sample, value)).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 10.0))
    torch.testing.assert_close(main_grad, torch.zeros_like(main_grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_replay_keeps_mixed_dtype_main_grad_lazy():
    """Return BF16 grads without fetching an incompatible FP32 main-grad buffer."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda", dtype=torch.bfloat16)
    main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
    getter_calls = []
    module.weight.__fsdp_param__ = True
    module.weight.overwrite_main_grad = True
    module.weight._gbuf = SimpleNamespace(dtype=torch.float32)
    module.weight.get_main_grad = lambda: getter_calls.append(True) or main_grad
    sample = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, use_main_grad=True
    )
    graphed(input=torch.full_like(sample, 2.0)).sum().backward()
    torch.cuda.synchronize()

    assert getter_calls == []
    assert module.weight.grad.dtype == torch.bfloat16
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_links_adjacent_static_surfaces():
    """Reuse a producer output as its consumer's static input."""
    first = torch.nn.Linear(4, 4, bias=False, device="cuda")
    second = torch.nn.Linear(4, 4, bias=False, device="cuda")
    with torch.no_grad():
        first.weight.copy_(torch.eye(4, device="cuda") * 2)
        second.weight.copy_(torch.eye(4, device="cuda") * 3)

    second_forward = second.forward

    def record_input(input_tensor):
        """Record the consumer input address during capture.

        :param input_tensor: Consumer input.
        :type input_tensor: torch.Tensor
        :return: Consumer output.
        :rtype: torch.Tensor
        """
        second.capture_input_ptr = input_tensor.data_ptr()
        return second_forward(input_tensor)

    second.forward = record_input
    samples = (
        torch.ones(2, 4, device="cuda", requires_grad=True),
        torch.ones(2, 4, device="cuda", requires_grad=True),
    )
    first_graph, second_graph = make_graphed_callables(
        (first, second),
        ((), ()),
        sample_kwargs=({"input": samples[0]}, {"input_tensor": samples[1]}),
        num_warmup_iters=1,
        _input_output_aliases=({}, {0: (0, 0)}),
    )

    runtime_input = torch.full_like(samples[0], 5, requires_grad=True)
    first_output = first_graph(input=runtime_input)
    assert first_output.data_ptr() == second.capture_input_ptr
    second_graph(input_tensor=first_output).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(runtime_input.grad, torch.full_like(runtime_input, 6))














@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_rejects_registered_buffer_address_change():
    """Reject replay after a registered buffer is rebound to new storage."""
    module = torch.nn.Linear(4, 4, bias=False, device="cuda")
    module.register_buffer("scale", torch.ones(1, device="cuda"))
    sample = torch.ones(2, 4, device="cuda")
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )

    module.scale = module.scale.clone()
    with pytest.raises(RuntimeError, match="registered buffer metadata or address changed"):
        graphed(input=sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_rejects_registered_buffer_autograd_metadata_change():
    """Reject requires-grad mutation on a captured registered-buffer slot."""

    class BufferedScale(torch.nn.Module):
        """Scale an input by one registered tensor.

        :param scale: Initial scale tensor.
        :type scale: torch.Tensor
        """

        def __init__(self, scale):
            """Register the scale tensor.

            :param scale: Initial scale tensor.
            :type scale: torch.Tensor
            """
            super().__init__()
            self.register_buffer("scale", scale)

        def forward(self, input_tensor):
            """Apply the registered scale.

            :param input_tensor: Input tensor.
            :type input_tensor: torch.Tensor
            :return: Scaled tensor.
            :rtype: torch.Tensor
            """
            return input_tensor * self.scale

    module = BufferedScale(torch.ones(1, device="cuda"))
    sample = torch.ones(2, 4, device="cuda")
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input_tensor": sample}, num_warmup_iters=1
    )

    module.scale.requires_grad_(True)
    with pytest.raises(RuntimeError, match="registered buffer metadata or address changed"):
        graphed(input_tensor=sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_bufferless_replay_skips_recursive_module_walk():
    """Use the empty registered-buffer fast path during replay."""
    module = torch.nn.Linear(4, 4, bias=False, device="cuda")
    sample = torch.ones(2, 4, device="cuda")
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )

    def fail_recursive_walk(*args, **kwargs):
        """Reject an unexpected replay-time recursive walk.

        :param args: Positional walk arguments.
        :type args: Any
        :param kwargs: Keyword walk arguments.
        :type kwargs: Any
        :raises AssertionError: Always.
        """
        del args, kwargs
        raise AssertionError("replay walked named_modules")

    module.named_modules = fail_recursive_walk
    graphed(input=sample)




@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
@pytest.mark.parametrize(
    ("overwrite_main_grad", "values", "expected"),
    [
        pytest.param(True, (2.0,), 4.0, id="overwrite"),
        pytest.param(False, (2.0, 3.0), 10.0, id="accumulate"),
    ],
)
@pytest.mark.parametrize("activation_recompute", [False, True], ids=["retained", "recompute"])
def test_cuda_graph_replay_te_fused_wgrad_main_grad(
    overwrite_main_grad, values, expected, activation_recompute
):
    """Write TE fused wgrad with the M-FSDP microbatch policy.

    :param overwrite_main_grad: Whether each microbatch replaces main grad.
    :type overwrite_main_grad: bool
    :param values: Runtime input values for consecutive microbatches.
    :type values: Tuple[float, ...]
    :param expected: Expected final value in every main-grad element.
    :type expected: float
    :param activation_recompute: Whether backward replays a recomputed forward.
    :type activation_recompute: bool
    """
    module = torch.nn.Linear(4, 3, bias=False, device="cuda", dtype=torch.bfloat16)
    main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
    module.weight.__fsdp_param__ = True
    module.weight.grad_added_to_main_grad = False
    module.weight.overwrite_main_grad = overwrite_main_grad
    module.weight._mfsdp_recorded_te_wgrad = True
    module.weight.get_main_grad = lambda: main_grad
    module.forward = partial(
        linear_with_grad_accumulation_and_async_allreduce,
        weight=module.weight,
        bias=None,
        gradient_accumulation_fusion=True,
        allreduce_dgrad=False,
        sequence_parallel=False,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        tp_group=None,
    )
    sample = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        use_main_grad=True,
        _activation_recompute=activation_recompute,
    )

    main_grad.zero_()
    module.weight.grad = None
    for value in values:
        graphed(input=torch.full_like(sample, value)).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(main_grad, torch.full_like(main_grad, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_graph_replay_restores_leaf_grad_and_reuses_main_grad(dtype):
    """Replay once per input without accumulating the main grad twice.

    :param dtype: Parameter and main-gradient dtype under test.
    :type dtype: torch.dtype
    """
    module = torch.nn.Linear(4, 3, bias=False, device="cuda", dtype=dtype)
    main_grad = torch.zeros_like(module.weight)
    module.weight.get_main_grad = lambda: main_grad
    sample = torch.ones(2, 4, device="cuda", dtype=dtype)

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, use_main_grad=True
    )

    assert module.weight.grad is None
    for value in (2.0, 3.0):
        runtime_input = torch.full_like(sample, value)
        graphed(input=runtime_input).sum().backward()
        torch.cuda.synchronize()

        expected = torch.full_like(main_grad, 2.0 * value)
        torch.testing.assert_close(main_grad, expected)
        assert module.weight.grad is not None
        assert module.weight.grad.data_ptr() == main_grad.data_ptr()
        module.weight.grad = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_capture_does_not_refetch_main_grad_for_clone_policy():
    """Reuse the capture-time main-grad view when selecting returned clone slots."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda")
    main_grad = torch.zeros_like(module.weight)
    getter_calls = []
    module.weight.get_main_grad = lambda: getter_calls.append(True) or main_grad
    sample = torch.ones(2, 4, device="cuda")

    make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, use_main_grad=True
    )

    assert getter_calls == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_capture_refreshes_parameter_after_backward_pre_hook():
    """Capture the parameter installed by a backward pre-hook."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda")
    sharded_weight = module.weight
    compute_weight = torch.nn.Parameter(torch.full_like(sharded_weight, 2))
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)

    def install_compute_weight(_module, *_args):
        """Install the unsharded compute parameter."""
        module.weight = compute_weight

    def install_sharded_weight(_module, *_args):
        """Restore the optimizer-facing parameter."""
        module.weight = sharded_weight

    capture_hooks = {
        "forward_pre_hooks": {0: install_compute_weight},
        "forward_pre_hooks_with_kwargs": {0: True},
        "forward_hooks": {0: install_sharded_weight},
        "forward_hooks_with_kwargs": {0: True},
        "backward_pre_hooks": {0: install_compute_weight},
        "backward_hooks": {0: install_sharded_weight},
    }
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        capture_time_hooks=[capture_hooks],
    )

    assert module.weight is sharded_weight
    runtime_input = torch.full_like(sample, 3, requires_grad=True)
    graphed(input=runtime_input).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(compute_weight.grad, torch.full_like(compute_weight, 6))
    assert sharded_weight.grad is None
