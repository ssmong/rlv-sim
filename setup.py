from __future__ import annotations

import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

QOCO = Path("src/native/vendor/qoco")
qoco_sources = [
    *QOCO.glob("src/*.c"),
    *QOCO.glob("algebra/builtin/*.c"),
    *QOCO.glob("lib/amd/*.c"),
    QOCO / "lib/qdldl/src/qdldl.c",
]
qoco_sources = [
    path
    for path in qoco_sources
    if path.name not in {"timer_linux.c", "timer_macos.c", "cone.cu"}
    and (path.name != "timer_windows.c" or sys.platform == "win32")
]
if sys.platform != "win32":
    timer_source = "src/timer_macos.c" if sys.platform == "darwin" else "src/timer_linux.c"
    qoco_sources.append(QOCO / timer_source)

setup(
    ext_modules=[
        Pybind11Extension(
            "_rlv_sim_native",
            [
                "src/native/frames.cpp",
                "src/native/plant.cpp",
                "src/native/scvx.cpp",
                *(str(path) for path in qoco_sources),
            ],
            include_dirs=[
                str(QOCO / "include"),
                str(QOCO / "algebra"),
                str(QOCO / "algebra/builtin"),
                str(QOCO / "lib/amd"),
                str(QOCO / "lib/qdldl/include"),
            ],
            define_macros=[("NTIMER", "1"), ("QOCO_ALGEBRA_BACKEND_BUILTIN", "1")],
            cxx_std=20,
            extra_compile_args=["/utf-8"] if sys.platform == "win32" else ["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
