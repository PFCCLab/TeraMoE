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
python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps dist/teramoe-*.whl
```

## Acknowledgement

TeraMoE is developed based on [DeepEP](https://github.com/deepseek-ai/DeepEP) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), and is inspired by [UniEP](https://arxiv.org/abs/2604.19241) and [SonicMoE](https://github.com/Dao-AILab/sonic-moe). We sincerely thank the authors and contributors of these projects for their work.

## License

This code repository is released under [the MIT License](LICENSE), except for code that references NVSHMEM (including `csrc/kernels/ibgda_device.cuh` and `third-party/nvshmem.patch`), which is subject to the [NVSHMEM SLA](https://docs.nvidia.com/nvshmem/api/sla.html). Code derived from [DeepEP](https://github.com/deepseek-ai/DeepEP) in `csrc/kernels` is subject to the license in the DeepEP repository.
