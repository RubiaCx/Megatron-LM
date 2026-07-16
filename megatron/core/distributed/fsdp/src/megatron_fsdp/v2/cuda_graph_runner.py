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

"""CUDA graph capture and replay for M-FSDP v2 modules.

Built on ``te_graph_runtime.make_graphed_callables`` which supports
``capture_time_hooks`` — hooks that run outside CUDA graph capture (for
FSDP unshard / reshard) and are not replayed.  ``sample_kwargs`` is used
so modules receive keyword arguments natively.

A single ``CudaGraphRunner`` instance is stored on the root context and
orchestrates:

  1. Recording sample args for each eligible FSDP module during the
     first optimized forward pass.
  2. Calling ``make_graphed_callables`` with all modules and
     ``capture_time_hooks`` that perform unshard / reshard.
"""  # noqa: E501

import dataclasses
import gc
import inspect
import logging
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_map

logger = logging.getLogger(__name__)

_CUDA_GRAPH_RUNTIME_ATTRS = (
    "backward_dw",
    "reset",
    "_cuda_graph_preflight",
)


_MISSING_CAPTURE_ATTRIBUTE = object()


@dataclasses.dataclass
class _CaptureMutableState:
    """Snapshot Python and tensor state mutated by graph warmup and capture."""

    buffer_values: List[Tuple[torch.Tensor, torch.Tensor]]
    parameter_states: List[Tuple[torch.nn.Parameter, Any, Dict[str, Any]]]
    param_group_states: List[
        Tuple[Any, Any, Any, Optional[torch.Tensor], Any, Any, Any]
    ]

    @torch.no_grad()
    def restore(self) -> None:
        """Restore state in place and release all one-time backup tensors."""
        for buffer, value in self.buffer_values:
            buffer.copy_(value)
        for param, grad, attributes in self.parameter_states:
            param.grad = grad
            for name, value in attributes.items():
                if value is _MISSING_CAPTURE_ATTRIBUTE:
                    param.__dict__.pop(name, None)
                else:
                    setattr(param, name, value)
        for (
            param_group,
            grad_buffer,
            data,
            data_value,
            dist_grads,
            fresh,
            has_unreduced_data,
        ) in self.param_group_states:
            if grad_buffer is not None:
                grad_buffer.reshard()
                grad_buffer.data = data
                if data_value is not None:
                    data.copy_(data_value)
            if dist_grads is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop("dist_grads", None)
            else:
                param_group.dist_grads = dist_grads
            if fresh is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop("_grad_buffer_is_fresh", None)
            else:
                param_group._grad_buffer_is_fresh = fresh
            if has_unreduced_data is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop(
                    "_main_grad_buffer_has_unreduced_data", None
                )
            else:
                param_group._main_grad_buffer_has_unreduced_data = has_unreduced_data
        self.buffer_values.clear()
        self.parameter_states.clear()
        self.param_group_states.clear()


def _snapshot_capture_mutable_state(
    modules: Tuple[torch.nn.Module, ...]
) -> _CaptureMutableState:
    """Save state that warmup or capture may update before the real optimizer step.

    :param modules: FSDP modules about to be captured.
    :type modules: Tuple[torch.nn.Module, ...]
    :return: Restorable capture transaction state.
    :rtype: _CaptureMutableState
    :raises RuntimeError: If a distributed main-grad buffer is already unsharded.
    """
    buffer_values = []
    seen_buffers = set()
    for module in modules:
        for buffer in module.buffers():
            if id(buffer) in seen_buffers:
                continue
            seen_buffers.add(id(buffer))
            buffer_values.append((buffer, buffer.detach().clone()))

    parameter_states = []
    param_group_states = []
    seen_params = set()
    seen_param_groups = set()
    for module in modules:
        for param_group in module._fsdp_param_groups:
            if id(param_group) not in seen_param_groups:
                seen_param_groups.add(id(param_group))
                grad_buffer = getattr(param_group, "main_grad_buffer", None)
                if (
                    grad_buffer is not None
                    and grad_buffer.is_distributed
                    and grad_buffer._unsharded_buffer is not None
                ):
                    raise RuntimeError(
                        "CUDA graph capture requires distributed main-grad buffers "
                        "to be resharded"
                    )
                data = grad_buffer.data if grad_buffer is not None else None
                data_value = (
                    data.detach().to(device="cpu", copy=True)
                    if data is not None and not grad_buffer.is_distributed
                    else None
                )
                param_group_states.append(
                    (
                        param_group,
                        grad_buffer,
                        data,
                        data_value,
                        getattr(param_group, "dist_grads", _MISSING_CAPTURE_ATTRIBUTE),
                        getattr(
                            param_group,
                            "_grad_buffer_is_fresh",
                            _MISSING_CAPTURE_ATTRIBUTE,
                        ),
                        getattr(
                            param_group,
                            "_main_grad_buffer_has_unreduced_data",
                            _MISSING_CAPTURE_ATTRIBUTE,
                        ),
                    )
                )
            for param in (
                *getattr(param_group, "params", ()),
                *getattr(param_group, "dist_params", ()),
            ):
                if id(param) in seen_params:
                    continue
                seen_params.add(id(param))
                attributes = {
                    name: param.__dict__.get(name, _MISSING_CAPTURE_ATTRIBUTE)
                    for name in (
                        "main_grad",
                        "grad_added_to_main_grad",
                        "overwrite_main_grad",
                        "_mfsdp_recorded_te_wgrad",
                    )
                }
                parameter_states.append((param, param.grad, attributes))
    return _CaptureMutableState(buffer_values, parameter_states, param_group_states)


def _cuda_autocast_state() -> Tuple[bool, Optional[torch.dtype], bool]:
    """Return the current CUDA autocast state.

    :return: Autocast enabled flag, active dtype, and cache flag.
    :rtype: Tuple[bool, Optional[torch.dtype], bool]
    """
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    cache_enabled = torch.is_autocast_cache_enabled()
    if not enabled:
        return False, None, cache_enabled
    try:
        dtype = torch.get_autocast_dtype("cuda")
    except AttributeError:
        dtype = torch.get_autocast_gpu_dtype()
    return True, dtype, cache_enabled


def _capture_module_topology(module: torch.nn.Module) -> Tuple[Tuple[Any, ...], ...]:
    """Capture direct buffer and child-module slots for every module owner.

    :param module: Root module whose recursive owner set is captured.
    :type module: torch.nn.Module
    :return: Qualified owner names, owner references, buffer keys, and child identities.
    :rtype: Tuple[Tuple[Any, ...], ...]
    """
    return tuple(
        (
            module_name,
            owner,
            tuple(owner._buffers),
            tuple(
                (child_name, id(child) if child is not None else None)
                for child_name, child in owner._modules.items()
            ),
        )
        for module_name, owner in module.named_modules(remove_duplicate=False)
    )


def _make_module_topology_preflight(
    topology: Tuple[Tuple[Any, ...], ...], delegate: Optional[Callable] = None
) -> Callable[[], None]:
    """Build a replay check without a recursive module walk.

    :param topology: Module-owner topology captured before CUDA graph capture.
    :type topology: Tuple[Tuple[Any, ...], ...]
    :param delegate: Existing runtime preflight callback, if any.
    :type delegate: Optional[Callable]
    :return: Callback that rejects buffer-slot or child replacement changes.
    :rtype: Callable[[], None]
    """

    def preflight() -> None:
        for module_name, owner, expected_buffer_keys, expected_children in topology:
            owner_name = module_name or "<root>"
            if tuple(owner._buffers) != expected_buffer_keys:
                raise RuntimeError(
                    "CUDA graph registered buffer topology changed after capture "
                    f"at module {owner_name!r}"
                )
            current_children = tuple(
                (child_name, id(child) if child is not None else None)
                for child_name, child in owner._modules.items()
            )
            if current_children != expected_children:
                raise RuntimeError(
                    "CUDA graph child module topology changed after capture "
                    f"at module {owner_name!r}"
                )
        if callable(delegate):
            delegate()

    return preflight


def _tensor_storage_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identify a tensor storage view.

    :param tensor: Tensor to identify.
    :type tensor: torch.Tensor
    :return: Storage address and view metadata.
    :rtype: Tuple[Any, ...]
    """
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.layout,
        tensor.device,
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _is_direct_autograd_alias(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> bool:
    """Return whether an input is the producer output or its direct autograd view.

    :param input_tensor: Consumer tensor to classify.
    :type input_tensor: torch.Tensor
    :param output_tensor: Earlier module output sharing the same storage view.
    :type output_tensor: torch.Tensor
    :return: Whether reconnecting their dgrad surfaces preserves the traced edge.
    :rtype: bool
    """
    if input_tensor.numel() == 0 or output_tensor.numel() == 0:
        return False
    if input_tensor is output_tensor:
        return True
    if input_tensor.requires_grad != output_tensor.requires_grad:
        return False
    input_grad_fn = input_tensor.grad_fn
    output_grad_fn = output_tensor.grad_fn
    if input_grad_fn is None or output_grad_fn is None:
        return False
    return any(next_fn is output_grad_fn for next_fn, _ in input_grad_fn.next_functions)


def _validate_activation_recompute_lifetime(
    lifetime_events: List[Tuple[str, int]], module_count: int
) -> None:
    """Require one complete linear forward and reverse-backward lifetime.

    :param lifetime_events: Recorded phase and module-index events.
    :type lifetime_events: List[Tuple[str, int]]
    :param module_count: Number of captured modules.
    :type module_count: int
    :raises RuntimeError: If the trace is missing, branched, or not strictly linear.
    """
    expected = tuple(
        [("forward", module_idx) for module_idx in range(module_count)]
        + [("backward", module_idx) for module_idx in reversed(range(module_count))]
    )
    if tuple(lifetime_events) != expected:
        raise RuntimeError(
            "Activation-recompute CUDA graphs require a complete linear "
            "F0..Fn,Bn..B0 lifetime order"
        )


# ---------------------------------------------------------------------------
# NVML memory helper (real GPU memory, not just torch allocator view)
# ---------------------------------------------------------------------------


def _nvml_device_memory(device: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return (used_MiB, total_MiB) from NVML, or None if unavailable."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        if device is None:
            device = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (info.used // (1024 * 1024), info.total // (1024 * 1024))
    except Exception:
        return None


def _mem_snapshot() -> Dict[str, int]:
    """Capture a snapshot of memory counters across torch and NVML."""
    snap = {
        "torch_alloc": torch.cuda.memory_allocated() // 1_000_000,
        "torch_reserved": torch.cuda.memory_reserved() // 1_000_000,
    }
    nvml = _nvml_device_memory()
    if nvml is not None:
        snap["nvml_used"] = nvml[0]
        snap["nvml_total"] = nvml[1]
    return snap


def _fmt_mem_snapshot(before: Dict[str, int], after: Dict[str, int], peak_alloc: int) -> str:
    """Format memory diff as a human-readable string."""
    parts = [
        f"torch_alloc {before['torch_alloc']}→{after['torch_alloc']} MB "
        f"(Δ{after['torch_alloc'] - before['torch_alloc']:+d})",
        f"torch_reserved {before['torch_reserved']}→{after['torch_reserved']} MB "
        f"(Δ{after['torch_reserved'] - before['torch_reserved']:+d})",
        f"peak_alloc {peak_alloc // 1_000_000} MB",
    ]
    if "nvml_used" in before:
        parts.append(
            f"nvml_used {before['nvml_used']}→{after['nvml_used']} MB "
            f"(Δ{after['nvml_used'] - before['nvml_used']:+d})"
        )
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Hook save / restore
# ---------------------------------------------------------------------------

_HOOK_ATTRS = [
    "_forward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]


def _pop_all_hooks(module):
    saved = []
    for sub in module.modules():
        snap = {}
        for attr in _HOOK_ATTRS:
            if hasattr(sub, attr):
                snap[attr] = getattr(sub, attr)
                setattr(sub, attr, OrderedDict())
        saved.append((sub, snap))
    return saved


def _restore_all_hooks(saved):
    for sub, snap in saved:
        for name, value in snap.items():
            if value is not None:
                setattr(sub, name, value)


def _prepare_compiled_modules_for_capture(modules):
    """Convert ``Module.compile()`` modules to compiled forward bodies.

    ``nn.Module.compile()`` compiles ``Module._call_impl``, which includes
    module-hook dispatch.  FSDP removes those hooks and replaces them with
    ``capture_time_hooks`` while building its explicit CUDA graphs.  Keeping
    the compiled ``_call_impl`` can therefore trigger a guard failure and a
    lazy recompile inside CUDA stream capture.

    Compile the forward body instead, with Inductor CUDA graphs disabled so
    that the FSDP runner remains the sole CUDA-graph owner.  The returned state
    is only for rollback if explicit graph capture fails; after successful
    installation, the stale compiled ``_call_impl`` must remain disabled.
    """
    saved = []
    try:
        for module in modules:
            compiled_call_impl = getattr(module, "_compiled_call_impl", None)
            if compiled_call_impl is None:
                continue

            original_forward = module.forward
            saved.append((module, original_forward, compiled_call_impl))
            module._compiled_call_impl = None

            # Avoid wrapping a forward body that the user already compiled
            # directly.  This branch mainly handles ``module.compile()``.
            if not hasattr(original_forward, "_torchdynamo_orig_callable"):
                module.forward = torch.compile(
                    original_forward, dynamic=False, options={"triton.cudagraphs": False}
                )
    except Exception:
        _restore_compiled_modules_after_capture_failure(saved)
        raise
    return saved


def _restore_compiled_modules_after_capture_failure(saved):
    """Restore module-level compilation when explicit capture fails."""
    for module, original_forward, compiled_call_impl in saved:
        module.forward = original_forward
        module._compiled_call_impl = compiled_call_impl


def _build_input_output_aliases(
    modules: Tuple[torch.nn.Module, ...],
    sample_outputs: Dict[int, Any],
    sample_args: Dict[int, Tuple[Any, ...]],
    sample_kwargs: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Tuple[int, int]], ...]:
    """Match consumer inputs to an unambiguous earlier autograd output.

    :param modules: Captured modules in forward order.
    :type modules: Tuple[torch.nn.Module, ...]
    :param sample_outputs: Recorded outputs keyed by module identity.
    :type sample_outputs: Dict[int, Any]
    :param sample_args: Recorded positional inputs keyed by module identity.
    :type sample_args: Dict[int, Tuple[Any, ...]]
    :param sample_kwargs: Recorded keyword inputs keyed by module identity.
    :type sample_kwargs: Dict[int, Dict[str, Any]]
    :return: Consumer input indices mapped to producer output indices.
    :rtype: Tuple[Dict[int, Tuple[int, int]], ...]
    """
    producer_outputs: Dict[Tuple[Any, ...], List[Tuple[int, int, torch.Tensor]]] = {}
    aliases_by_consumer = []
    for consumer_idx, module in enumerate(modules):
        mid = id(module)
        flat_args, _ = tree_flatten(sample_args[mid])
        flat_kwargs, _ = tree_flatten(list(sample_kwargs[mid].values()))
        aliases = {}
        for input_idx, input_tensor in enumerate(flat_args + flat_kwargs):
            if not isinstance(input_tensor, torch.Tensor) or input_tensor.numel() == 0:
                continue
            candidates = producer_outputs.get(_tensor_storage_key(input_tensor), ())
            producer = None
            exact_matches = [
                candidate for candidate in candidates if input_tensor is candidate[2]
            ]
            if len(exact_matches) == 1:
                producer = exact_matches[0][:2]
            elif not exact_matches:
                direct_matches = [
                    candidate
                    for candidate in candidates
                    if _is_direct_autograd_alias(input_tensor, candidate[2])
                ]
                if len(direct_matches) == 1:
                    producer = direct_matches[0][:2]
            if producer is not None and producer[0] < consumer_idx:
                aliases[input_idx] = producer
        aliases_by_consumer.append(aliases)

        # Address reuse does not identify the autograd edge. Keep every
        # same-storage output from the latest producer, then link only one
        # unambiguous object or direct view.
        flat_outputs, _ = tree_flatten(sample_outputs.get(mid, ()))
        current_outputs = defaultdict(list)
        for output_idx, output in enumerate(flat_outputs):
            if isinstance(output, torch.Tensor) and output.numel() != 0:
                current_outputs[_tensor_storage_key(output)].append(
                    (consumer_idx, output_idx, output)
                )
        producer_outputs.update(current_outputs)
    return tuple(aliases_by_consumer)


class CudaGraphRunner:
    """Orchestrates per-module sample-arg recording and batch graph capture.

    Created once by the root forward pre-hook and stored on
    ``ctx.cuda_graph_runner``.
    """

    def __init__(self, graph_pool: Any, num_warmup_iters: int = 3):
        self._graph_pool = graph_pool
        self._num_warmup = num_warmup_iters
        self._captured = False

        # Per-module state recorded during the first optimized forward.
        self._sample_args: Dict[int, Tuple] = {}
        self._sample_kwargs: Dict[int, Dict[str, Any]] = {}
        self._sample_outputs: Dict[int, Any] = {}
        self._modules_ordered: List[torch.nn.Module] = []
        self._module_indices: Dict[int, int] = {}
        self._original_forwards: Dict[int, Callable] = {}
        self._original_graph_attrs: Dict[int, Dict[str, Any]] = {}
        self._compiled_module_state = []
        self._autocast_states: Dict[int, Tuple[bool, Optional[torch.dtype], bool]] = {}
        self._lifetime_events: List[Tuple[str, int]] = []
        self._backward_lifetime_modules: Set[int] = set()

    # ---- called from hooks ------------------------------------------------

    def record_module(self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]) -> None:
        """Record one module call during the first optimized forward.

        :param module: FSDP module invoked by the trace forward.
        :type module: torch.nn.Module
        :param args: Positional forward arguments.
        :type args: Tuple
        :param kwargs: Keyword forward arguments.
        :type kwargs: Dict[str, Any]
        :raises RuntimeError: If one module instance is invoked more than once.
        """
        if self._captured:
            return
        mid = id(module)
        if mid in self._sample_args:
            raise RuntimeError(
                "M-FSDP CUDA Graph does not support calling the same module "
                "instance more than once per iteration"
            )

        # Normalize Module.compile() before capture setup. te-graph-runtime
        # detects this compiled forward body and warms the capture-equivalent
        # hook specialization before entering torch.cuda.graph.
        self._compiled_module_state.extend(_prepare_compiled_modules_for_capture([module]))
        self._original_forwards[mid] = module.forward
        self._original_graph_attrs[mid] = {
            name: module.__dict__[name]
            for name in _CUDA_GRAPH_RUNTIME_ATTRS
            if name in module.__dict__
        }

        sig = inspect.signature(module.forward)
        has_self = "self" in sig.parameters
        bound = sig.bind(module, *args, **kwargs) if has_self else sig.bind(*args, **kwargs)
        all_kwargs = {
            n: bound.arguments[n] for n in bound.arguments if not (has_self and n == "self")
        }
        self._sample_args[mid] = tuple()  # all via kwargs
        self._sample_kwargs[mid] = all_kwargs
        self._autocast_states[mid] = _cuda_autocast_state()
        module_idx = len(self._modules_ordered)
        self._module_indices[mid] = module_idx
        self._modules_ordered.append(module)
        self._lifetime_events.append(("forward", module_idx))

        n_tensor = sum(1 for v in all_kwargs.values() if isinstance(v, torch.Tensor))
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: recorded module %s (id=%s), %d kwargs (%d tensor)",
                getattr(module, "_fsdp_module_name", module.__class__.__name__),
                id(module),
                len(all_kwargs),
                n_tensor,
            )

    def record_module_backward(self, module: torch.nn.Module) -> None:
        """Record the end of one module activation lifetime.

        :param module: Recorded module whose backward work has completed.
        :type module: torch.nn.Module
        """
        if self._captured:
            return
        module_idx = self._module_indices.get(id(module))
        if module_idx is None:
            return
        if module_idx in self._backward_lifetime_modules:
            return
        self._backward_lifetime_modules.add(module_idx)
        self._lifetime_events.append(("backward", module_idx))

    def record_module_output(self, module: torch.nn.Module, output: Any) -> None:
        """Record an eager output for static graph linking.

        :param module: Recorded FSDP module.
        :type module: torch.nn.Module
        :param output: Output from the eager sample forward.
        :type output: Any
        """
        mid = id(module)
        if self._captured or mid not in self._sample_args or mid in self._sample_outputs:
            return
        self._sample_outputs[mid] = output

    def reset(self) -> None:
        """Destroy captured graphs and restore the original module callables."""
        reset_function_ids = set()
        for module in self._modules_ordered:
            original_attrs = self._original_graph_attrs.get(id(module), {})
            graph_reset = module.__dict__.get("reset")
            if (
                self._captured
                and callable(graph_reset)
                and graph_reset is not original_attrs.get("reset")
                and id(graph_reset) not in reset_function_ids
            ):
                reset_function_ids.add(id(graph_reset))
                graph_reset()
            original_forward = self._original_forwards.get(id(module))
            if original_forward is not None:
                module.forward = original_forward
            for name in _CUDA_GRAPH_RUNTIME_ATTRS:
                if name in original_attrs:
                    setattr(module, name, original_attrs[name])
                else:
                    module.__dict__.pop(name, None)
            module.__dict__.pop("_fsdp_cg_installed", None)
            module.__dict__.pop("_fsdp_cg_activation_recompute", None)
            for param_group in getattr(module, "_fsdp_param_groups", ()):
                for param in param_group.params:
                    param.__dict__.pop("_mfsdp_recorded_te_wgrad", None)
                if not getattr(
                    param_group, "_main_grad_buffer_has_unreduced_data", False
                ):
                    maybe_free_grad_data = getattr(
                        param_group, "_maybe_free_grad_data", None
                    )
                    if callable(maybe_free_grad_data):
                        maybe_free_grad_data()

        if not self._captured:
            _restore_compiled_modules_after_capture_failure(self._compiled_module_state)

        self._sample_args.clear()
        self._sample_kwargs.clear()
        self._sample_outputs.clear()
        self._modules_ordered.clear()
        self._module_indices.clear()
        self._original_forwards.clear()
        self._original_graph_attrs.clear()
        self._compiled_module_state.clear()
        self._autocast_states.clear()
        self._lifetime_events.clear()
        self._backward_lifetime_modules.clear()
        self._captured = False

    def capture_and_install(
        self, root_module: torch.nn.Module, capture_stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """Capture all graphs + install wrappers on recorded modules."""
        if self._captured or not self._modules_ordered:
            return

        modules = tuple(self._modules_ordered)
        n = len(modules)
        autocast_states = {self._autocast_states[id(module)] for module in modules}
        if len(autocast_states) != 1:
            raise RuntimeError("CUDA graph capture requires one recorded CUDA autocast state")
        autocast_enabled, autocast_dtype, _ = next(iter(autocast_states))
        activation_recompute = bool(getattr(root_module, "gradient_checkpointing", False))
        if activation_recompute:
            _validate_activation_recompute_lifetime(self._lifetime_events, n)
            for fsdp_module in modules:
                for module in fsdp_module.modules():
                    if _module_uses_delayed_wgrad(module):
                        raise RuntimeError(
                            "M-FSDP CUDA Graph activation recompute does not yet support "
                            "delayed backward-wgrad"
                        )
        capture_mutable_state = _snapshot_capture_mutable_state(modules)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: capturing %d modules", n)

        # Use the installed runtime only when it supports M-FSDP capture.
        try:
            from te_graph_runtime import make_graphed_callables
            from te_graph_runtime.graph import (
                _MFSDP_CAPTURE_CAPABILITIES as _installed_mfsdp_capabilities,
            )
            from te_graph_runtime.graph import (
                _get_compatible_main_grad_buffer as _installed_static_grad_support,
            )
            from te_graph_runtime.graph import (
                _refresh_module_parameter_surface as _installed_parameter_refresh,
            )

            required_capabilities = {
                "capture_grad_buffer_release",
                "parameter_surface_refresh",
                "registered_buffer_validation",
                "static_grad_binding",
            }
            if activation_recompute:
                required_capabilities.add("activation_recompute")
                required_capabilities.add("activation_recompute_argument_binding")
                required_capabilities.add("activation_recompute_discard_tape")
                required_capabilities.add("fp8_activation_recompute_metadata")
                required_capabilities.add("activation_recompute_preflight")
                required_capabilities.add("static_dgrad_reuse")
                required_capabilities.add("static_fwd_reuse")
            if (
                not all(
                    callable(helper)
                    for helper in (_installed_static_grad_support, _installed_parameter_refresh)
                )
                or not required_capabilities.issubset(_installed_mfsdp_capabilities)
                or "use_main_grad" not in inspect.signature(make_graphed_callables).parameters
                or (
                    activation_recompute
                    and "_activation_recompute"
                    not in inspect.signature(make_graphed_callables).parameters
                )
                or (
                    activation_recompute
                    and "_reuse_graph_input_output_buffers"
                    not in inspect.signature(make_graphed_callables).parameters
                )
            ):
                raise ImportError("Installed te-graph-runtime lacks M-FSDP CUDA graph support")
        except ImportError:
            from .te_graph_runtime import make_graphed_callables

        sample_args_list: List[Tuple] = []
        sample_kwargs_list: List[Dict[str, Any]] = []
        capture_hooks: List[Dict] = []

        input_output_aliases = _build_input_output_aliases(
            modules,
            self._sample_outputs,
            self._sample_args,
            self._sample_kwargs,
        )

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: linked %d static input/output tensors",
                sum(len(aliases) for aliases in input_output_aliases),
            )

        for m in modules:
            mid = id(m)
            capture_sync_groups = _collect_capture_sync_groups(
                m, self._sample_args[mid], self._sample_kwargs[mid]
            )
            # Clone tensor values so warmup gets fresh leaves without
            # residual autograd state from the first forward+backward.
            args = _clone_capture_sample(self._sample_args[mid])
            kw = _clone_capture_sample(self._sample_kwargs[mid])
            sample_args_list.append(args)
            sample_kwargs_list.append(kw)

            capture_hooks.append(
                {
                    "forward_pre_hooks": {
                        0: _make_fwd_pre_hook(m, capture_sync_groups)
                    },
                    "forward_pre_hooks_with_kwargs": {0: True},
                    "forward_hooks": {0: _make_fwd_post_hook(m, capture_sync_groups)},
                    "forward_hooks_with_kwargs": {0: True},
                    "backward_pre_hooks": {
                        0: _make_bwd_pre_hook(
                            m,
                            activation_recompute=activation_recompute,
                            capture_sync_groups=capture_sync_groups,
                        )
                    },
                    "backward_hooks": {0: _make_bwd_post_hook(m, capture_sync_groups)},
                }
            )

        compiled_module_state = list(self._compiled_module_state)
        if compiled_module_state and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            logger.info(
                "CudaGraphRunner: converted %d Module.compile() wrappers to "
                "compiled forward bodies",
                len(compiled_module_state),
            )

        runtime_options = {}
        if activation_recompute:
            runtime_options["_activation_recompute"] = True
            runtime_options["_reuse_graph_input_output_buffers"] = True
        supports_input_output_aliases = (
            "_input_output_aliases" in inspect.signature(make_graphed_callables).parameters
        )
        if any(input_output_aliases):
            if not supports_input_output_aliases:
                from .te_graph_runtime import make_graphed_callables

                supports_input_output_aliases = True
            runtime_options["_input_output_aliases"] = tuple(input_output_aliases)

        # Pop real FSDP hooks so make_graphed_callables passes its assertion.
        # capture_time_hooks handle unshard/reshard during warmup + capture.
        saved_hooks = _pop_all_hooks(root_module)
        flat_outputs = flat_args = flat_kwargs = ()
        output = input_tensor = None
        self._sample_args.clear()
        self._sample_kwargs.clear()
        self._sample_outputs.clear()
        gc.collect()
        self._captured = True

        try:
            torch.cuda.reset_peak_memory_stats()
            _mem_before = _mem_snapshot()

            autocast_kwargs = {
                "enabled": autocast_enabled,
                "cache_enabled": False,
            }
            if autocast_enabled:
                autocast_kwargs["dtype"] = autocast_dtype
            with torch.amp.autocast("cuda", **autocast_kwargs):
                graphed = make_graphed_callables(
                    tuple(modules),
                    sample_args_list,
                    num_warmup_iters=self._num_warmup,
                    sample_kwargs=sample_kwargs_list,
                    pool=self._graph_pool,
                    capture_time_hooks=capture_hooks,
                    capture_stream=capture_stream,
                    use_main_grad=True,
                    **runtime_options,
                )
        except Exception:
            for module in modules:
                try:
                    module.reshard()
                except Exception:
                    logger.exception("Failed to reshard after CUDA graph capture error")
                for param_group in module._fsdp_param_groups:
                    for param in param_group.params:
                        param.grad = None
                    try:
                        param_group.release_grad_buffer()
                    except Exception:
                        logger.exception(
                            "Failed to release a gradient buffer after CUDA graph capture error"
                        )
            self.reset()
            _restore_compiled_modules_after_capture_failure(compiled_module_state)
            raise
        finally:
            _restore_all_hooks(saved_hooks)
            capture_mutable_state.restore()

        _mem_after = _mem_snapshot()
        _peak_alloc = torch.cuda.max_memory_allocated()

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: %d modules captured %s",
                n,
                _fmt_mem_snapshot(_mem_before, _mem_after, _peak_alloc),
            )

        if not isinstance(graphed, tuple):
            graphed = (graphed,)

        # make_graphed_callables already replaced module.forward with
        # the graphed version that handles kwargs natively.
        for module in modules:
            module._cuda_graph_preflight = _make_module_topology_preflight(
                _capture_module_topology(module),
                module.__dict__.get("_cuda_graph_preflight"),
            )
            module._fsdp_cg_installed = True
            module._fsdp_cg_activation_recompute = activation_recompute
        self._compiled_module_state = []

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: installed CUDA graphs on %d modules", n)


# ---------------------------------------------------------------------------
# capture_time_hooks (unshard / reshard outside graph, not replayed)
# ---------------------------------------------------------------------------


def _clone_capture_sample(value):
    """Clone all tensor leaves while preserving the input PyTree structure."""

    def clone_tensor(leaf):
        if not isinstance(leaf, torch.Tensor):
            return leaf
        return leaf.detach().clone().requires_grad_(leaf.requires_grad)

    return tree_map(clone_tensor, value)


def _module_uses_delayed_wgrad(module):
    """Return whether a module requests a separate delayed-wgrad graph."""
    for owner in (module, getattr(module, "config", None)):
        if owner is None:
            continue
        delayed = getattr(owner, "delay_wgrad_compute", False)
        if bool(delayed() if callable(delayed) else delayed):
            return True
    delayed = getattr(getattr(module, "wgrad_store", None), "delay_wgrad_compute", False)
    return bool(delayed() if callable(delayed) else delayed)


def _collect_capture_sync_groups(module, sample_args, sample_kwargs):
    """Collect CP groups used by a module or its captured input metadata."""
    groups = []

    def add_group(group):
        if isinstance(group, (tuple, list)):
            for child in group:
                add_group(child)
            return
        if group is not None and all(group is not existing for existing in groups):
            groups.append(group)

    def visit_input(value):
        add_group(getattr(value, "cp_group", None))
        if isinstance(value, dict):
            for child in value.values():
                visit_input(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit_input(child)

    visit_input(sample_args)
    visit_input(sample_kwargs)
    for submodule in module.modules():
        add_group(getattr(submodule, "cp_group", None))
        pg_collection = getattr(submodule, "pg_collection", None)
        add_group(getattr(pg_collection, "cp", None))
        add_group(getattr(pg_collection, "hcp", None))
    return tuple(groups)


def _capture_group_barrier(capture_sync_groups):
    """Synchronize ranks that capture a graph containing a collective."""
    for group in capture_sync_groups:
        torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])


def _make_fwd_pre_hook(module, capture_sync_groups=()):
    """Build the capture-time forward unshard hook.

    :param module: FSDP module to unshard.
    :type module: torch.nn.Module
    :return: Forward pre-hook callable.
    :rtype: Callable
    """

    def hook(mod, args, kwargs):
        _capture_group_barrier(capture_sync_groups)
        module.unshard()

    return hook


def _make_fwd_post_hook(module, capture_sync_groups=()):
    def hook(mod, args, kwargs, output):
        module.reshard()
        _capture_group_barrier(capture_sync_groups)

    return hook


def _make_bwd_pre_hook(
    module, activation_recompute=False, capture_sync_groups=()
):
    """Build the capture-time backward unshard hook.

    :param module: FSDP module to unshard.
    :type module: torch.nn.Module
    :return: Backward pre-hook callable.
    :rtype: Callable
    """

    def hook(mod, grad_output):
        _capture_group_barrier(capture_sync_groups)
        module.unshard(bwd_pass=True)
        if activation_recompute:
            module.unshard(async_op=False, bwd_pass=False)
        for param_group in module._fsdp_param_groups:
            overwrite_main_grad = (
                param_group.sharding_strategy in ("optim_grads_params", "optim_grads")
                or not getattr(
                    param_group, "_main_grad_buffer_has_unreduced_data", False
                )
            )
            for param in param_group.params:
                param.overwrite_main_grad = overwrite_main_grad
            has_fused_wgrad = any(
                getattr(param, "_mfsdp_recorded_te_wgrad", False) for param in param_group.params
            )
            if has_fused_wgrad and param_group.main_grad_buffer is not None:
                param_group._init_dist_grads()
                param_group.main_grad_buffer.fetch_buffer()
                for param in param_group.params:
                    if getattr(param, "_mfsdp_recorded_te_wgrad", False):
                        param.main_grad = param.get_main_grad()
    return hook


def _make_bwd_post_hook(module, capture_sync_groups=()):
    def hook(mod, grad_input, grad_output):
        module.reshard()
        # Clear capture-only views before the next module reuses their slots.
        for param_group in module._fsdp_param_groups:
            for param in param_group.params:
                param.grad = None
            param_group.release_grad_buffer()
        _capture_group_barrier(capture_sync_groups)

    return hook
