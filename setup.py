import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode features are disabled\n'
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/moe_extension.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu', 'csrc/teramoe/teramoe_orchestrator.cu']
    include_dirs = [os.path.join(_repo_root, 'csrc')]
    _third_party_root = os.path.join(_repo_root, 'third-party')

    _cutlass_root = os.path.join(_third_party_root, 'cutlass')
    if os.path.isdir(os.path.join(_cutlass_root, 'include')):
        include_dirs.append(os.path.join(_cutlass_root, 'include'))
        include_dirs.append(os.path.join(_cutlass_root, 'tools', 'util', 'include'))

    _deepgemm_root = os.path.join(_third_party_root, 'DeepGEMM')
    _deepgemm_include = os.path.join(_deepgemm_root, 'deep_gemm', 'include')
    _deepgemm_cutlass_include = os.path.join(_deepgemm_root, 'third-party', 'cutlass', 'include')
    if os.path.isdir(_deepgemm_include):
        include_dirs.append(_deepgemm_include)
    if os.path.isdir(_deepgemm_cutlass_include):
        include_dirs.append(_deepgemm_cutlass_include)

    _quack_root = os.path.join(_third_party_root, 'quack')
    if os.path.isdir(os.path.join(_quack_root, 'quack')):
        include_dirs.append(_quack_root)
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags
    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu', 'csrc/teramoe/teramoe_notify.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        # CCCL (libcudacxx) headers needed by nvshmem_tensor.h for cuda/std/tuple
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        cccl_include = os.path.join(cuda_home, 'include', 'cccl')
        if os.path.isdir(cccl_include):
            include_dirs.append(cccl_include)
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0;10.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    mk_compute_kernel = int(os.getenv('MK_COMPUTE_KERNEL', '1'))
    assert mk_compute_kernel in (1, 2), 'MK_COMPUTE_KERNEL must be 1 or 2 (WMMA path removed)'
    cxx_flags.append(f'-DMK_COMPUTE_KERNEL={mk_compute_kernel}')
    nvcc_flags.append(f'-DMK_COMPUTE_KERNEL={mk_compute_kernel}')

    if int(os.getenv('ENABLE_FAST_DEBUG', 0)):
        cxx_flags.append('-DENABLE_FAST_DEBUG')
        nvcc_flags.append('-DENABLE_FAST_DEBUG')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='teramoe',
                     version='0.0.1' + revision,
                     packages=setuptools.find_packages(include=['teramoe']),
                     ext_modules=[
                         CUDAExtension(name='teramoe_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
