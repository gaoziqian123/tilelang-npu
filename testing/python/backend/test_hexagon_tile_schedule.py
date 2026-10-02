import pytest
from collections import Counter
import importlib.util
from pathlib import Path
import tilelang

spec = importlib.util.spec_from_file_location('tile_schedule', Path(__file__).resolve().parents[3] / 'examples/hexagon/fa/fa_tile_schedule.py')
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)

@pytest.mark.parametrize('bm,bn',[(128,256),(256,256),(128,512),(32,32),(512,1024)])
@pytest.mark.parametrize('workers',range(1,7))
def test_ownership_mask(bm,bn,workers):
    rows=[]
    for qb in range(1024//bm):
        for tx in range(workers):
            for it in range(max(0,(bm//2-tx+workers-1)//workers)):
                for r in range(2):
                    qi=qb*bm+schedule.pair_row(it,tx,workers)+r
                    rows.append(qi)
                    visible=[]
                    for kb in range(schedule.key_blocks(qb,bm,bn)):
                        visible.extend(kb*bn+j for j in range(bn) if kb*bn+j<=qi)
                    assert visible==list(range(qi+1))
    assert Counter(rows)==Counter(range(1024))

@pytest.mark.parametrize('bm,bn,workers', [(128,256,4),(256,256,6),(128,512,3)])
def test_real_resident_dsl_lowering(bm,bn,workers):
    src = tilelang.engine.lower(schedule.make_fa(bm,bn,workers), target='hexagon',
        enable_host_codegen=False, enable_device_compile=False).kernel_source
    assert 'hmx_mma_deep' in src
    assert f'tl_hex_num_workers() == {workers}' in src
    assert 'gemm_pack_ah_rows' in src

def test_mask_off_by_one_negative():
    # A strict inequality loses the diagonal; local qidx loses the prefix.
    qi=257
    assert [k for k in range(1024) if k<qi]!=list(range(qi+1))
    assert [k for k in range(1024) if k<=qi%256]!=list(range(qi+1))
