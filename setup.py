"""
setup.py — builds the bitnet_cuda C++/CUDA extension.

Build command:
    python setup.py build_ext --inplace

Or equivalently with pip (editable install):
    pip install -e .

The compiled .pyd/.so lands next to this file and can be imported directly:
    import bitnet_cuda
    C = bitnet_cuda.bitnet_forward(A, B_packed, M, K, N)

Architecture flag
-----------------
  -gencode arch=compute_86,code=sm_86  →  RTX 3050 (GA107, Ampere, sm_86).
  The compute_86 stage compiles to PTX first, then sm_86 assembles it to
  SASS for the exact chip.  Add more -gencode lines if you later run this
  on a different GPU (e.g. sm_75 for Turing, sm_89 for Ada Lovelace).
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_FLAGS = [
    # ── Target GPU ──────────────────────────────────────────────────────────
    "-gencode", "arch=compute_86,code=sm_86",   # RTX 3050 (Ampere / GA107)

    # ── Optimisation ────────────────────────────────────────────────────────
    "-O3",                   # maximum compiler optimisation
    "--use_fast_math",       # enables fmaf, reciprocal approximations, etc.

    # ── fp16 support ────────────────────────────────────────────────────────
    "-DCUDA_HAS_FP16=1",

    # ── Debug / profiling aids (safe to remove in production) ───────────────
    "-lineinfo",             # embeds source-line info for Nsight / ncu reports
]

CXX_FLAGS = [
    "/O2",       # MSVC: optimise (Windows); on Linux this would be "-O2"
]

setup(
    name="bitnet_cuda",
    ext_modules=[
        CUDAExtension(
            name="bitnet_cuda",
            sources=["bitnet_forward.cu"],
            extra_compile_args={
                "cxx":  CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
