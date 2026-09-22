# TeraMoE

TeraMoE 是一个适用于跨节点 EP 并行的通信-计算 overlap 的 MoE 训练算子库

## 亮点
> [!IMPORTANT]
> 
> - 适用于跨机大 EP（EP32/EP64）场景，通过计算-通信 Overlap 实现性能提升，
> - Overlap 有更强的抗路由不均衡能力
> - 与现行 DeepEP+DeepGEMM 方案前反向逐位对齐，收敛风险小

## 快速开始

### Requirements

- SM100 GPUs
- Python 3.10 and above
- CUDA toolchain with SM100 support
- RDMA-capable network for cross-node communication
- NVSHMEM installed

### 安装

已将 DeepEP 和 DeepGEMM 的安装流程打包在一个 setup.py 里，无需手动配置各种路径，一键即可安装

```bash
git clone https://github.com/PFCCLab/TeraMoE.git
cd TeraMoE
git submodule update --init --recursive

python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps dist/teramoe-*.whl
```

## 用户接口

TeraMoE 提供了与 fleet MoELayer 兼容的自动反向接口，参数传入方式基本与 fleet 对齐，API 内部已经封装好了 DeepEP 建连、Overlap 策略、显存管理等逻辑，可以以极少的代码量接入 fleet

```python
import teramoe

# init
teramoe.configure_buffer(48)

# forward
hidden_states = teramoe.forward_autograd(
    hidden_states,
    topk_weights,
    topk_indices,
    grouped_gemm_experts,
    num_experts,
    moe_group,
    combine_overlap_handle,
    chunk_size=4096,
    num_calc_sms=100,
    combine_overlap_ratio=0.3,
)
```

## 单测

在`tests/`下面有 TeraMoE 串起来的前反向单测，用于测试整个系统的正确性和性能，需要分布式环境：

```bash
cd tests
mpirun python run.py 0,1 test_fleet_moe.py  # --fp8 --long-run 100
```

在`third_party/DeepEP/tests_overlap/`下面有 DeepEP 的分布式单测，目前功能与`tests/`重合较高，仅供 DeepEP 分支维护用

在`third_party/DeepGEMM/tests_overlap/`下面有 DeepGEMM 的单卡单测，测试了 DeepGEMM 各算子在各类配置下的情况，是性能调优的重要工具


## Acknowledgement

TeraMoE is developed based on [DeepEP](https://github.com/deepseek-ai/DeepEP) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), and is inspired by [UniEP](https://arxiv.org/abs/2604.19241) and [SonicMoE](https://github.com/Dao-AILab/sonic-moe). We sincerely thank the authors and contributors of these projects for their work.

## License

This code repository is released under [the MIT License](LICENSE), except for code that references NVSHMEM (including `csrc/kernels/ibgda_device.cuh` and `third-party/nvshmem.patch`), which is subject to the [NVSHMEM SLA](https://docs.nvidia.com/nvshmem/api/sla.html). Code derived from [DeepEP](https://github.com/deepseek-ai/DeepEP) in `csrc/kernels` is subject to the license in the DeepEP repository.
