import pytest
import tilelang
from tvm import IRModule, tirx as tir
from tilelang.hexagon.vector_lowering import VectorLowering


@pytest.mark.parametrize('offset',[1,64])
def test_unaligned_or_outside(offset):
    a=tir.decl_buffer((64,),'float32',name='a')
    b=tir.decl_buffer((32,),'float32',name='b')
    value=tir.BufferLoad(a,[tir.Ramp(offset,1,32)])
    f=tir.PrimFunc([a.data,b.data],tir.BufferStore(b,value,[tir.Ramp(0,1,32)]))
    f=f.with_attr('tl.vector_required',1)
    with pytest.raises(ValueError,match='128B'):
        VectorLowering()(IRModule({'f':f}))


def test_unknown_dynamic_alignment_rejected():
    a=tir.decl_buffer((128,),'float32',name='a')
    b=tir.decl_buffer((32,),'float32',name='b')
    offset=tir.Var('offset','int32')
    value=tir.BufferLoad(a,[tir.Ramp(offset,1,32)])
    f=tir.PrimFunc([a.data,b.data,offset],tir.BufferStore(b,value,[tir.Ramp(0,1,32)]))
    f=f.with_attr('tl.vector_required',1)
    with pytest.raises(ValueError,match='unproven address'):
        VectorLowering()(IRModule({'f':f}))
