"""The TeraMoE fused MoE operator library."""

import paddle

from .version import __version__

# Only deep_ep needs compat. The deep_gemm has been natively adapted to paddle
paddle.enable_compat(scope={"teramoe.deep_ep"}, silent=True)

from . import deep_ep, deep_gemm
from .autograd import forward_autograd
from .fused_a2a import configure_buffer

__all__ = ["deep_ep", "deep_gemm", "forward_autograd", "configure_buffer", "__version__"]
