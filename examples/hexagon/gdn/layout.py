"""Checked registry slab contract, shared by emitter and runtime-wrapper build."""
import math
import numpy as np


def layout(nt, hk, hv, d=128):
    if min(nt, hk, hv, d) <= 0:
        raise ValueError('positive dimensions required')
    nc = (nt+63)//64
    tp = nc*64
    specs = {
        'K':((hk,tp,d),'float16'),'Q':((hk,tp,d),'float16'),'V':((hv,tp,d),'float16'),
        'g':((hv,tp),'float32'),'beta':((hv,tp),'float32'),
        'S0':((hv,d,d),'float32'),'S1':((hv,d,d),'float32'),'O':((hv,tp,d),'float16'),
        'G':((hk,nc,64,64),'float16'),'KM':((64,d),'float16'),
        'U':((hv,nc,64,d),'float16'),'W':((hv,nc,64,d),'float16'),
        'P':((hv,nc,64),'float32'),'E':((hv,nc,64),'float32'),
        'L':((6,64,64),'float32'),'UF':((6,64,d),'float32'),'WF':((6,64,d),'float32'),
        'X':((6,64),'float32'),'S':((d,d),'float32'),'ST':((d,d),'float32'),
        'SH':((d,d),'float16'),'QR':((64,d),'float16'),'KR':((64,d),'float16'),
        'WS':((64,d),'float16'),'QS':((64,d),'float16'),'QK':((64,64),'float16'),
        'PR':((64,d),'float16'),'R':((64,d),'float16'),'RT':((d,64),'float16'),
        'PT':((64,64),'float16'),'EX':((64,64),'float32'),'KE':((64,),'float32'),
        'KF':((64,d),'float32'),'RF':((64,d),'float32'),'LOG':((34,),'uint64')}
    offsets = {}
    size = 0
    for name, (shape, dtype) in specs.items():
        size = (size+127)//128*128
        offsets[name] = size
        size += math.prod(shape)*np.dtype(dtype).itemsize
    size = (size+127)//128*128
    validate(specs, offsets, size)
    return specs, offsets, size


def validate(specs, offsets, size):
    intervals = sorted((offsets[n], offsets[n]+math.prod(s)*np.dtype(t).itemsize, n)
                       for n, (s, t) in specs.items())
    end = 0
    for lo, hi, name in intervals:
        if lo % 128 or lo < end or hi <= lo or hi > size:
            raise ValueError('invalid/overlapping slab region: '+name)
        end = hi
    return intervals


def checked_noalias(func, nt, hk, hv, d=128):
    specs, offsets, size = layout(nt, hk, hv, d)
    validate(specs, offsets, size)
    for b in func.buffer_map.values():
        shape, dtype = specs[b.name]
        if str(b.dtype) != dtype or math.prod(int(x) for x in b.shape) != math.prod(shape):
            raise ValueError('parameter extent/type differs from registry: '+b.name)
        compact = [math.prod(int(x) for x in b.shape[i+1:]) for i in range(len(b.shape))]
        if (b.strides and [int(x) for x in b.strides] != compact) or int(b.elem_offset) != 0:
            raise ValueError('noncompact registry parameter: '+b.name)
    if len(func.buffer_map) != len(func.params):
        raise ValueError('unaccounted registry parameter')
    return func.with_attr('tir.noalias', True)
