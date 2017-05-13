import numpy as np  # noqa
import tempfile
import os
import ctypes

from loopy.tools import (empty_aligned,
        build_ispc_shared_lib, cptr_from_numpy)


ISPC_TARGET = "avx2-i32x8"
# ISPC_TARGET = "avx2-i64x4"  # use this for double


def build_ispc(code):
    with open("tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()

    tmpdir = tempfile.mkdtemp()

    build_ispc_shared_lib(
            tmpdir,
            [("my.ispc", code)],
            [("tasksys.cpp", tasksys_source)],
            cxx_options=["-g", "-fopenmp", "-DISPC_USE_OMP"],
            ispc_options=([
                "-g", "--no-omit-frame-pointer",
                "--target=" + ISPC_TARGET,
                "--opt=force-aligned-memory",
                "--opt=disable-loop-unroll",
                #"--math-lib=fast",
                #"--opt=fast-math",
                #"--opt=disable-fma",
                ]
                #+ (["--addressing=64"] if INDEX_DTYPE == np.int64 else [])
                ),
            #ispc_bin="/home/andreask/pack/ispc-v1.9.0-linux/ispc",
            quiet=False,
            )

    shared_obj = os.path.join(tmpdir, "shared.so")
    return shared_obj


ISPC_CODE = """
task void stream_scale_tasks_inner(uniform float *uniform a, uniform float const *uniform b, uniform float const scalar, uniform int32 const n)
{
  /* bulk slab for 'i_outer' */
  if (-262145 + -262144 * ((uniform int32) taskIndex0) + n >= 0)
    for (uniform int32 i_inner_outer = 0; i_inner_outer <= 32767; ++i_inner_outer)
      a[262144 * ((uniform int32) taskIndex0) + 8 * i_inner_outer + (varying int32) programIndex] = scalar * b[262144 * ((uniform int32) taskIndex0) + 8 * i_inner_outer + (varying int32) programIndex];
  /* final slab for 'i_outer' */
  if (262144 + 262144 * ((uniform int32) taskIndex0) + -1 * n >= 0)
    for (uniform int32 i_inner_outer = 0; i_inner_outer <= -1 + -1 * (varying int32) programIndex + -32768 * ((uniform int32) taskIndex0) + ((7 + n + 7 * (varying int32) programIndex) / 8); ++i_inner_outer)
      a[262144 * ((uniform int32) taskIndex0) + 8 * i_inner_outer + (varying int32) programIndex] = scalar * b[262144 * ((uniform int32) taskIndex0) + 8 * i_inner_outer + (varying int32) programIndex];
}

export void scale(uniform float *uniform a, uniform float const *uniform b, uniform float const scalar, uniform int32 const n)
{
  assert(programCount == (8));
  launch[((262143 + n) / 262144)] stream_scale_tasks_inner(a, b, scalar, n);
}
"""


def main():
    shared_obj = build_ispc(ISPC_CODE)
    lib = ctypes.cdll.LoadLibrary(shared_obj)

    n = 2**20
    alignment = 4096  # a page
    a = empty_aligned(n, dtype=np.float32, n=alignment)
    b = empty_aligned(n, dtype=np.float32, n=alignment)

    b.fill(np.pi)

    lib.scale(cptr_from_numpy(a), cptr_from_numpy(b), ctypes.c_float(15), ctypes.c_int(n))

if __name__ == "__main__":
    main()
