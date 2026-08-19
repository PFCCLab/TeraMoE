"""`teramoe` — the TeraMoE fused MoE operator library.

Bundles the TeraMoE forks of DeepEP and DeepGEMM under one namespace:

    import teramoe.deep_ep
    import teramoe.deep_gemm

These are deliberately *not* drop-in replacements for the upstream operators
shipped in `paddlefleet_ops`; the API semantics differ.  Both can be imported in
the same process:

    import paddlefleet_ops           # standard deep_ep / deep_gemm
    import teramoe.deep_gemm         # TeraMoE deep_gemm

Each side binds its own uniquely-named C++ extension
(`teramoe_deep_gemm_cpp` / `teramoe_deep_ep_cpp` here, `deep_gemm_cpp` /
`deep_ep_cpp` there), which is what keeps their JIT include paths separate.
"""

import paddle

from .version import __version__ as __version__

# Both forks are written against the torch API and rely on paddle's torch
# compatibility proxy.  Scope matching is by module prefix, so naming the
# namespaced packages here covers all of their submodules and leaves
# `paddlefleet_ops`' own `scope={"deep_ep", "deep_gemm"}` untouched.
paddle.enable_compat(
    scope={"teramoe.deep_ep", "teramoe.deep_gemm"},
    silent=True,
)

__all__ = ["deep_ep", "deep_gemm", "__version__"]
