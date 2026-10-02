"""Native DMA IR. Binding registered DDR/VTCM capabilities is a runtime duty."""
from tvm import tirx as tir
from tvm.ir import Op


def submit(ticket, dst, src, width, rows, src_stride, dst_stride):
    if type(ticket) is not int or not 0<=ticket<1024:
        raise ValueError('DMA ticket id must be static in [0,1024)')
    for value in (width,rows,src_stride,dst_stride):
        if type(value) is not int or not 0<value<2**32:
            raise ValueError('DMA geometry must be positive uint32 constants')
    if width>src_stride or width>dst_stride:
        raise ValueError('DMA stride smaller than width')
    return tir.call_intrin('int32',Op.get('tl.hexagon.dma_submit'),ticket,dst,src,
                           width,rows,src_stride,dst_stride)


def wait(ticket):
    if type(ticket) is not int or not 0<=ticket<1024:
        raise ValueError('DMA ticket id must be static in [0,1024)')
    return tir.call_intrin('int32',Op.get('tl.hexagon.dma_wait'),ticket)
