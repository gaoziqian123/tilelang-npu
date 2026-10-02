"""Static dynamic-count audit for the fixed native q32/k128 FA baseline.

Counts team barrier rendezvous, not summed worker arrivals or elapsed cycles.
Explicit generated sites plus AllReduce's two barriers per butterfly step.
"""
from pathlib import Path
import re

source = Path(__file__).resolve().parents[4] / 'backend/npu/attn/skel/src/tl_fa_native.c'
text = source.read_text()
assert 'q32_k128_w4_kernel' in text
assert len(re.findall(r'AllReduce<', text)) == 2
blocks = 16 * sum(q // 4 + 1 for q in range(32))
reduction = blocks * 2 * 32 * 2 * 2
readout = blocks * 2 * (4 + 8)
print(f'BASELINE_STATIC kv_blocks={blocks} reduction_barriers={reduction} '
      f'fragment_readout_barriers={readout} subtotal={reduction+readout}')
print('EXCLUDES copy/prologue/epilogue/runtime-job barriers; not a timing measurement')
