import hashlib
import os
from pathlib import Path
import subprocess
import sys


def test_fresh_process_codegen(tmp_path):
    root = Path(__file__).resolve().parents[3]
    example = root / "examples/hexagon/gemm/gemm_worker_pipeline.py"
    hashes = []
    for i in range(3):
        output = tmp_path / f"fresh{i}.c"
        subprocess.run([sys.executable, str(example), "--output", str(output)],
                       env={**os.environ, "PYTHONPATH": str(root)}, check=True,
                       capture_output=True, text=True, timeout=60)
        source = output.read_bytes()
        assert source.count(b"#include <tl_templates/hexagon/workergroup_abi.h>") == 1
        hashes.append(hashlib.sha256(source).hexdigest())
    print("fresh-process SHA256:", *hashes)
    assert len(set(hashes)) == 1
