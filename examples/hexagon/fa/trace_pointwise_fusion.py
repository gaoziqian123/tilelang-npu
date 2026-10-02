"""Host-only real FA pass/Hexagon object evidence; no device execution."""
import argparse
from pathlib import Path
import subprocess

import tilelang
import tvm
from tvm import tirx
from fa_persistent import make_fa


@tvm.instrument.pass_instrument
class Capture:
    def __init__(self, out):
        self.out = out
        self.before = None
        self.after = None

    def run_before_pass(self, mod, info):
        if info.name == "tl.FusePointwiseStages":
            self.before = mod
            (self.out / "before.tir").write_text(mod.script())

    def run_after_pass(self, mod, info):
        if info.name == "tl.FusePointwiseStages":
            self.after = mod
            (self.out / "after.tir").write_text(mod.script())


def loops(mod):
    found = []
    for f in mod.functions.values():
        if isinstance(f, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(f.body, lambda n: found.append(n) if isinstance(n, tirx.For) else None)
    return len(found)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pass-off", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    capture = Capture(args.out)
    with tvm.transform.PassContext(config={"tl.disable_fuse_pointwise_stages": args.pass_off}, instruments=[capture]):
        artifact = tilelang.engine.lower(make_fa(), target="hexagon", enable_host_codegen=False,
                                        enable_device_compile=False)
    if capture.before is not None:
        print(f"FA loops before={loops(capture.before)} after={loops(capture.after)}", flush=True)
    src = args.out / "fa.cpp"
    src.write_text(artifact.kernel_source)
    tools = Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin")
    root = Path(tilelang.__file__).resolve().parents[1]
    command = [str(tools / "hexagon-clang++"), "-mv79", "-mhvx", "-mhvx-length=128B", "-mhmx",
               "-O2", "-std=c++17", "-I", str(root / "src"), "-c", str(src), "-o", str(args.out / "fa.o")]
    print("COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    asm = subprocess.check_output([str(tools / "hexagon-llvm-objdump"), "-dr", str(args.out / "fa.o")], text=True)
    (args.out / "fa.asm").write_text(asm)
    print(f"OBJECT_OK vmem={asm.count('vmem(')} vmpy={asm.count('vmpy(')}")
