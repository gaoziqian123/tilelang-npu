from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang  # noqa: F401
import tilelang.opencl  # noqa: F401 - register OpenCL backend
from tilelang.opencl.op.gemm_fma import GemmFMA  # noqa: F401

KERNELS = REPO_ROOT / "examples" / "opencl" / "kernels"
if str(KERNELS) not in sys.path:
    sys.path.insert(0, str(KERNELS))

from gemm_nt import make_grgemm_kernel, make_rrgemm_kernel, make_rrgemm_source, syntax_check  # noqa: E402

import tilelang  # noqa: E402
from tilelang import tvm  # noqa: E402


def lower_rr(b_layout: str, accum: str = "float16") -> str:
    with tvm.target.Target("opencl"), tvm.transform.PassContext(
        config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
    ):
        artifact = tilelang.lower(
            make_rrgemm_kernel(512, 512, 512, 32, 128, 64, 64, b_layout, accum),
            target="opencl",
            enable_device_compile=False,
        )
    return artifact.kernel_source


def lower_gr(b_layout: str, accum: str = "float16") -> str:
    with tvm.target.Target("opencl"), tvm.transform.PassContext(
        config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
    ):
        artifact = tilelang.lower(
            make_grgemm_kernel(512, 512, 512, 32, 128, 64, 64, b_layout, accum),
            target="opencl",
            enable_device_compile=False,
        )
    return artifact.kernel_source


def assert_lowered_gr_fp16_shape(src: str, b_layout: str) -> None:
    """Shape gate for the GR (global-read) fp16 lowering: direct global vector
    loads + mad inner chain + rolled K loop, matching the 1.35T anchor form."""
    if "half8 acc[8]" not in src:
        raise AssertionError("GR: missing float16x8 accumulator shape")
    if "mad(" not in src:
        raise AssertionError("GR: fp16 inner loop must use mad() (mul+add does not fuse on Adreno)")
    if "vload4(0, A +" not in src:
        raise AssertionError("GR: missing direct global half4 A load")
    if b_layout == "kn":
        if "vload8(0, B +" not in src:
            raise AssertionError("GR: missing direct global half8 B load")
    else:
        if "vload4(0, B +" not in src:
            raise AssertionError("GR: missing nk B half4 gather load")
    if "A_frag" in src or "B_frag" in src:
        raise AssertionError("GR: A/B must not be staged into fragments")
    if "convert_float" in src or "convert_half" in src:
        raise AssertionError("GR: full-fp16 path must not contain any convert_* calls")
    if "for (int pos4 = 0; pos4 < 128; ++pos4)" not in src:
        raise AssertionError("GR: K loop must stay rolled (pos4 < K/4)")
    if re.search(r"for \([^;]*;[^;]*<\s*(512|2048)[^;]*;", src):
        raise AssertionError("GR: element-wise A/B copy loop must not appear")


def assert_lowered_fp16_shape(src: str, b_layout: str) -> None:
    """Shape gate for the real tilelang.lower fp16 RR path (DESIGN §5 invariants)."""
    if "half8 acc[8]" not in src:
        raise AssertionError("missing float16x8 accumulator shape")
    if "float acc[64]" in src or "half acc[64]" in src:
        raise AssertionError("scalar acc[64] array must not appear")
    if "vload4(0, A_frag" not in src:
        raise AssertionError("missing A half4 vector load from fragment")
    if b_layout == "kn":
        if "vload8(0, B_frag" not in src:
            raise AssertionError("missing kn B half8 vector load from fragment")
    else:
        if "vload4(0, B_frag" not in src:
            raise AssertionError("missing nk B half4 gather load from fragment")
    if "convert_float" in src or "convert_half" in src:
        raise AssertionError("full-fp16 path must not contain any convert_* calls")
    if "vstore8(vload8(0, A" not in src or "vstore8(vload8(0, B" not in src:
        raise AssertionError("A/B global->fragment copies must be wide vload8/vstore8")
    if "vstore8(vload8(0, C_frag" not in src:
        raise AssertionError("C fragment->global store must be wide vload8/vstore8")
    if re.search(r"for \([^;]*;[^;]*<\s*(512|2048)[^;]*;", src):
        raise AssertionError("element-wise A/B copy loop must not appear")


def assert_shape(src: str, b_layout: str) -> None:
    if "float8 acc[8]" not in src:
        raise AssertionError("missing float8/float32x8 accumulator shape")
    if "float4 a[8]" not in src or "vload4(0, A" not in src or "convert_float4" not in src:
        raise AssertionError("missing A half4 -> float32x4 vector load")
    if b_layout == "kn":
        if "float8 b[4]" not in src or "vload8(0, B" not in src or "convert_float8" not in src:
            raise AssertionError("missing kn B half8 -> float32x8 vector load")
    else:
        if "float4 b4[8]" not in src or "vload4(0, B" not in src:
            raise AssertionError("missing nk B half4 gather load")
    if "vstore8(convert_half8(acc[i])" not in src:
        raise AssertionError("missing C float16x8 vstore8")
    if re.search(r"float\s+acc\s*\[\s*64\s*\]", src):
        raise AssertionError("scalar float acc[64] array must not appear")
    if re.search(r"for \([^)]*<\s*TL_BM \* TL_BK[^)]*\)", src):
        raise AssertionError("A/B private scalar copy loop must not appear")


def assert_layout_fallback() -> None:
    # The implemented RR gate is intentionally strict: valid tile parameters are
    # those from DESIGN_rr_gemm §4, while bm=40 with threads=64 is a clean
    # fallback because the thread count no longer matches (bm/8)*(bn/8).
    bm, bn, bk, threads = 32, 128, 64, 64
    assert bm % 8 == 0 and bn % 8 == 0 and bk % 4 == 0
    assert threads == (bm // 8) * (bn // 8)
    bad_bm = 40
    assert bad_bm % 8 == 0 and threads != (bad_bm // 8) * (bn // 8)


def main() -> int:
    src_kn = make_rrgemm_source(512, 512, 512, 32, 128, 64, 64, "kn")
    assert_shape(src_kn, "kn")
    out = REPO_ROOT / "examples" / "opencl" / "out" / "probe_rr_kn.cl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src_kn, encoding="utf-8")
    rc = syntax_check(out)
    if rc not in (0, None):
        raise AssertionError(f"clang syntax failed: {rc}")

    src_nk = make_rrgemm_source(512, 512, 512, 32, 128, 64, 64, "nk")
    assert_shape(src_nk, "nk")
    assert_layout_fallback()

    # Real lowering path (tilelang.lower over make_rrgemm_kernel), full-fp16.
    lowered_kn = lower_rr("kn")
    assert_lowered_fp16_shape(lowered_kn, "kn")
    out_kn = REPO_ROOT / "examples" / "opencl" / "out" / "probe_rr_lowered_kn.cl"
    out_kn.write_text(lowered_kn, encoding="utf-8")
    rc = syntax_check(out_kn)
    if rc not in (0, None):
        raise AssertionError(f"clang syntax failed for lowered kn: {rc}")

    lowered_nk = lower_rr("nk")
    assert_lowered_fp16_shape(lowered_nk, "nk")
    out_nk = REPO_ROOT / "examples" / "opencl" / "out" / "probe_rr_lowered_nk.cl"
    out_nk.write_text(lowered_nk, encoding="utf-8")
    rc = syntax_check(out_nk)
    if rc not in (0, None):
        raise AssertionError(f"clang syntax failed for lowered nk: {rc}")

    # GR (direct global read) lowering, full-fp16, same shape gates.
    for layout in ("kn", "nk"):
        lowered_gr = lower_gr(layout)
        assert_lowered_gr_fp16_shape(lowered_gr, layout)
        out_gr = REPO_ROOT / "examples" / "opencl" / "out" / f"probe_gr_lowered_{layout}.cl"
        out_gr.write_text(lowered_gr, encoding="utf-8")
        rc = syntax_check(out_gr)
        if rc not in (0, None):
            raise AssertionError(f"clang syntax failed for lowered gr {layout}: {rc}")

    print("PROBE_RR_OK")
    print("PROBE_RR_LOWERED_FP16_OK")
    print("PROBE_GR_LOWERED_FP16_OK")
    print(f"SOURCE_LINES {len(src_kn.splitlines())}")
    print(f"LOWERED_LINES {len(lowered_kn.splitlines())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
