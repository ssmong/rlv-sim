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

qoco_include_dirs = [
    str(QOCO / "include"),
    str(QOCO / "algebra"),
    str(QOCO / "algebra/builtin"),
    str(QOCO / "lib/amd"),
    str(QOCO / "lib/qdldl/include"),
]
qoco_macros = [("NTIMER", "1"), ("QOCO_ALGEBRA_BACKEND_BUILTIN", "1")]
compile_args = ["/utf-8"] if sys.platform == "win32" else ["-O3"]

setup(
    # Build C sources separately so the C++ standard flag stays on C++ sources.
    libraries=[
        (
            "qoco",
            {
                "sources": [str(path) for path in qoco_sources],
                "include_dirs": qoco_include_dirs,
                "macros": qoco_macros,
                "cflags": compile_args,
            },
        )
    ],
    ext_modules=[
        Pybind11Extension(
            "_rlv_sim_native",
            [
                "src/native/frames.cpp",
                "src/native/plant.cpp",
                "src/native/scvx.cpp",
            ],
            include_dirs=qoco_include_dirs.copy(),
            define_macros=qoco_macros,
            cxx_std=17,
            extra_compile_args=compile_args.copy(),
        )
    ],
    cmdclass={"build_ext": build_ext},
)
