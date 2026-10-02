import numpy as np
import pytest
from tilelang import language as T
from tilelang.hexagon.physical_contract import contract, pack, unpack, copy_spans


def test_all_half_bits():
    layout=T.Layout((256,256),lambda i,j:(i//32,j//32,i%32//2,j%32*2+i%2))
    spec=contract(layout)
    bits=np.arange(65536,dtype=np.uint16).reshape(256,256)
    got=unpack(pack(bits.view(np.float16),spec),spec)
    assert np.array_equal(got.view(np.uint16),bits)


def test_padding_and_alias():
    spec=contract(T.Layout((3,4),lambda i,j:i*8+j))
    raw=pack(np.ones((3,4),np.float16),spec)
    assert np.all(raw.reshape(-1,2)[4:8]==0)
    with pytest.raises(ValueError,match="aliases"):
        contract(T.Layout((3,4),lambda i,j:j))


def test_strided_panel():
    src=contract(T.Layout((4,8),lambda i,j:i*8+j))
    dst=contract(T.Layout((2,4),lambda i,j:i*4+j))
    assert copy_spans(src,dst,(1,2),(0,0),(2,4)) == [dict(
        source_offset=20,destination_offset=0,width_bytes=8,rows=2,
        source_stride=16,destination_stride=8)]
    with pytest.raises(ValueError,match="bounds"):
        copy_spans(src,dst,(3,2),(0,0),(2,4))
