import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py as _build_py

from build_utils import PKG_ROOT, SRC_DIR, prepare_ecosystem

BASE_VERSION = "0.0.1"


def get_version() -> str:
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PKG_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return f"{BASE_VERSION}+{rev}"
    except Exception:
        return BASE_VERSION


def write_version_file(version: str) -> None:
    (SRC_DIR / "teramoe" / "version.py").write_text(
        f'__version__ = "{version}"\n'
    )


class BuildPy(_build_py):
    def run(self):
        prepare_ecosystem()
        write_version_file(self.distribution.get_version())
        super().run()
        # `src/` holds JIT headers, `.so` files and a bare top-level extension
        # module, none of which map cleanly onto `package_data`.  Mirroring the
        # staged tree wholesale is simpler and keeps the wheel byte-identical to
        # what the two forks installed on their own.
        shutil.copytree(SRC_DIR, Path(self.build_lib), dirs_exist_ok=True)


class BinaryDistribution(Distribution):
    """Forces a platform-specific (non-pure) wheel: we ship compiled `.so`s."""

    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    version = get_version()
    write_version_file(version)
    setup(
        name="teramoe",
        version=version,
        description=(
            "TeraMoE: A cross-node expert-parallel MoE training library for "
            "computation-communication overlap."
        ),
        package_dir={"": "src"},
        packages=find_packages("src"),
        distclass=BinaryDistribution,
        cmdclass={"build_py": BuildPy},
        python_requires=">=3.10",
    )
