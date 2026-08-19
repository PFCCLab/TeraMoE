"""Build helpers for the `teramoe` meta-package.

The two innovation forks live in `third_party/{DeepEP,DeepGEMM}` and are kept as
independent, separately installable repos.  This module builds each of them with
its own `setup.py` into a scratch directory and then re-lays the resulting
artifacts out under the `teramoe` namespace, so that a single wheel provides:

    teramoe/deep_ep            (from third_party/DeepEP)
    teramoe/deep_gemm          (from third_party/DeepGEMM)
    teramoe_deep_ep_cpp.so     C++ extension, uniquely named
    teramoe_deep_gemm_cpp/     C++ extension, uniquely named

The unique extension module names are what let `teramoe.deep_gemm` coexist with
`paddlefleet_ops.deep_gemm` in one process: each Python package binds its own
`.so`, so neither clobbers the other's JIT include path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PKG_ROOT / "src"
NS_DIR = SRC_DIR / "teramoe"
STAGE_DIR = PKG_ROOT / "build" / "_stage"


class Lib:
    """One third-party fork and the artifacts it contributes to `teramoe`."""

    def __init__(
        self,
        name: str,
        source_rel_path: str,
        # (path inside the pip-install target) -> (path inside src/)
        artifacts: dict[str, str],
        extra_env: dict[str, str] | None = None,
        pre_build=None,
    ):
        self.name = name
        self.source_dir = PKG_ROOT / source_rel_path
        self.stage_dir = STAGE_DIR / name
        self.artifacts = artifacts
        self.extra_env = extra_env or {}
        self.pre_build = pre_build

    def build(self) -> None:
        if self.pre_build is not None:
            self.pre_build(self.source_dir)

        # Stale build trees from a previous (differently configured) run are the
        # number one source of "why is my .so not being rebuilt" confusion.
        for stale in ("build", "dist"):
            shutil.rmtree(self.source_dir / stale, ignore_errors=True)
        for egg in self.source_dir.glob("*.egg-info"):
            shutil.rmtree(egg, ignore_errors=True)
        shutil.rmtree(self.stage_dir, ignore_errors=True)

        env = os.environ.copy()
        env.update(self.extra_env)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            ".",
            "--target",
            str(self.stage_dir),
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
        ]
        print(f"[teramoe] building {self.name}: {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=self.source_dir, env=env)

    def install(self) -> None:
        for src_rel, dst_rel in self.artifacts.items():
            src = self.stage_dir / src_rel
            dst = SRC_DIR / dst_rel
            if not src.exists():
                raise FileNotFoundError(
                    f"{self.name}: expected artifact {src} not produced by its build"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=False)
            else:
                shutil.copy2(src, dst)
            print(f"[teramoe] staged {src_rel} -> src/{dst_rel}", flush=True)


def _link_cutlass_includes(source_dir: Path) -> None:
    """DeepGEMM's JIT resolves `<cutlass/...>` and `<cute/...>` out of
    `deep_gemm/include`, so the headers have to be reachable there before the
    build copies them into the wheel (see third_party/DeepGEMM/DEVELOP.md)."""
    include_dir = source_dir / "deep_gemm" / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    cutlass_include = source_dir / "third-party" / "cutlass" / "include"
    for name in ("cutlass", "cute"):
        link = include_dir / name
        target = cutlass_include / name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(target)


def get_libs() -> list[Lib]:
    return [
        Lib(
            name="DeepGEMM",
            source_rel_path="third_party/DeepGEMM",
            artifacts={
                "deep_gemm": "teramoe/deep_gemm",
                "teramoe_deep_gemm_cpp": "teramoe_deep_gemm_cpp",
            },
            extra_env={"DG_USE_LOCAL_VERSION": "1"},
            pre_build=_link_cutlass_includes,
        ),
        Lib(
            name="DeepEP",
            source_rel_path="third_party/DeepEP",
            artifacts={
                "deep_ep": "teramoe/deep_ep",
                # The `.py` stub paddle emits alongside is dead code (it points
                # at a `_pd_.so` that is never produced); the extension module
                # itself takes import priority, so only ship the `.so`.
                "teramoe_deep_ep_cpp.so": "teramoe_deep_ep_cpp.so",
            },
        ),
    ]


def prepare_ecosystem(libs: list[Lib] | None = None) -> None:
    libs = libs if libs is not None else get_libs()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    NS_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(libs)) as pool:
        for future in [pool.submit(lib.build) for lib in libs]:
            future.result()
    for lib in libs:
        lib.install()
