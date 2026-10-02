"""Static causal query-block specialization of the stage-aware FA pipeline.

Each invocation handles one 128-row query block and its visible 256-key panels.
This is an experimental lowering entry, not a complete packed/DMA RPC adapter.
"""
import argparse
from pathlib import Path
import tilelang
from tilelang import language as T
from tilelang.hexagon.language import vector as VX


def native(name, *values):
    return VX.leaf(name, *values, mode='native_target_v1')


def half_splat(value):
    return VX.bitcast(VX.splat_bits(T.reinterpret(value, 'uint16'), 'float16x64'), 'uint8x128')


def half_extract(value):
    return T.reinterpret(VX.extract_bits(VX.bitcast(value, 'float16x64'), 0).astype('uint16'), 'float16')


def half_exp(value):
    log2e = VX.bitcast(VX.splat_bits(T.uint32(0x3dc5), 'float16x64'), 'uint8x128')
    return VX.leaf('exp2_16_nonpositive', native('mul16', value, log2e),
                   mode='ggml_qf16_clamp24_v1')


def exp_chunk(row, offset, maximum):
    x = VX.bitcast(VX.load(row, offset), 'uint8x128')
    return VX.bitcast(half_exp(native('sub16', x, maximum)), 'float16x64')


def scale_chunk(row, offset, scale):
    return VX.bitcast(native('mul16', VX.bitcast(VX.load(row, offset), 'uint8x128'), scale), 'float16x64')


def row_chunks(row, maximum):
    """Reference order: four half chunks, even/odd widen after P rounding."""
    chunks = [VX.bitcast(VX.load(row, c*64), 'uint8x128') for c in range(4)]
    if maximum:
        acc = VX.bitcast(VX.splat_bits(T.uint32(0xfc00), 'float16x64'), 'uint8x128')
        for chunk in chunks:
            acc = native('max16', acc, chunk)
    else:
        acc = VX.bitcast(VX.splat_bits(T.uint32(0), 'float32x32'), 'uint8x128')
        for chunk in chunks:
            for part in (0, 1):
                acc = native('add32', acc, VX.leaf('widen_evenodd', chunk,
                             mode='strict_exact_v1', immediate=part))
    return acc


def packed(rows, cols, transpose=False):
    if transpose:
        return T.Layout((rows, cols), lambda r,c:(r//32,c//32,c%32//2,(r%32)*2+c%2))
    return T.Layout((rows, cols), lambda r,c:(r//32,c//32,r%32//2,(c%32)*2+r%2))

def rowmajor(rows, cols):
    return T.Layout((rows, cols), lambda r,c:(r,c))


def make_fa(query_block: int, intrinsic_rows: bool = True, state_workers: int = 1,
            direct_normalize_output: bool = False):
    if not 0 <= query_block < 8:
        raise ValueError('query_block must be in [0,8)')
    if state_workers not in (1, 2) or (state_workers == 2 and not intrinsic_rows):
        raise ValueError('state_workers must be 1 or 2; two workers require intrinsic rows')
    if direct_normalize_output and state_workers != 1:
        raise ValueError('direct normalization is an isolated single-worker candidate')
    panels = ((query_block + 1) * 128 + 255) // 256
    state_scope = 'shared' if state_workers == 2 else 'local'
    # Persistent row ownership is compiler-owned and cyclic across state workers.
    row_loop = T.Parallel if state_workers == 2 else T.serial
    state_loop = T.Parallel if state_workers == 2 else T.vectorized

    @T.macro
    def exp_rows(src, mx, sm):
        row = T.alloc_local((256,), 'float16')
        for r in row_loop(128):
            T.copy(src[r:r+1, :], row)
            row_maximum = mx[r]
            m = half_splat(row_maximum)
            T.evaluate(VX.store(row, exp_chunk(row, 0, m), 0))
            T.evaluate(VX.store(row, exp_chunk(row, 64, m), 64))
            T.evaluate(VX.store(row, exp_chunk(row, 128, m), 128))
            T.evaluate(VX.store(row, exp_chunk(row, 192, m), 192))
            # Consume HALF-rounded P before its only VTCM writeback. This
            # removes the following VTCM->private reread, not a rounding point.
            acc = row_chunks(row, False)
            a = native('add32', acc, VX.leaf('rotate', acc, immediate=4))
            b = native('add32', a, VX.leaf('rotate', a, immediate=8))
            c = native('add32', b, VX.leaf('rotate', b, immediate=16))
            d = native('add32', c, VX.leaf('rotate', c, immediate=32))
            e = native('add32', d, VX.leaf('rotate', d, immediate=64))
            h = VX.leaf('narrow_evenodd', e, e, mode='rne_gradual_quiet_payload_v1')
            sm[r] = half_extract(h)
            T.copy(row, src[r:r+1, :])

    @T.macro
    def normalize_rows(src, denom, dst):
        row = T.alloc_local((256,), 'float16')
        for r in row_loop(128):
            # Reference reciprocal variant: FP32 NR2, round reciprocal to half,
            # then half multiply. No persistent FP32 state or scalar division.
            row_denominator = denom[r]
            d = VX.leaf('widen_evenodd', half_splat(row_denominator), mode='strict_exact_v1', immediate=0)
            inv = VX.leaf('rcp32_nr2', d, mode='nr2_positive_bounded_v1')
            h = VX.leaf('narrow_evenodd', inv, inv, mode='rne_gradual_quiet_payload_v1')
            T.copy(src[r:r+1, :], row)
            T.evaluate(VX.store(row, scale_chunk(row, 0, h), 0))
            T.evaluate(VX.store(row, scale_chunk(row, 64, h), 64))
            T.evaluate(VX.store(row, scale_chunk(row, 128, h), 128))
            T.evaluate(VX.store(row, scale_chunk(row, 192, h), 192))
            T.copy(row, dst[r:r+1, :])

    @T.macro
    def intrinsic_rows_reduce(src, dst, maximum: bool):
        # Static rank-one staging is required by the physical vector I/O API.
        # Keep the reference chunk accumulation and ascending rotation tree;
        # a generic reduce_sum is not permission to reassociate this sum.
        row = T.alloc_local((256,), 'float16')
        for r in row_loop(128):
            T.copy(src[r:r+1, :], row)
            acc = row_chunks(row, maximum)
            if maximum:
                a = native('max16', acc, VX.leaf('rotate', acc, immediate=2))
                b = native('max16', a, VX.leaf('rotate', a, immediate=4))
                c = native('max16', b, VX.leaf('rotate', b, immediate=8))
                d = native('max16', c, VX.leaf('rotate', c, immediate=16))
                e = native('max16', d, VX.leaf('rotate', d, immediate=32))
                f = native('max16', e, VX.leaf('rotate', e, immediate=64))
                dst[r] = T.reinterpret(VX.extract_bits(VX.bitcast(f, 'float16x64'), 0).astype('uint16'), 'float16')
            else:
                a = native('add32', acc, VX.leaf('rotate', acc, immediate=4))
                b = native('add32', a, VX.leaf('rotate', a, immediate=8))
                c = native('add32', b, VX.leaf('rotate', b, immediate=16))
                d = native('add32', c, VX.leaf('rotate', c, immediate=32))
                e = native('add32', d, VX.leaf('rotate', d, immediate=64))
                h = VX.leaf('narrow_evenodd', e, e, mode='rne_gradual_quiet_payload_v1')
                dst[r] = T.reinterpret(VX.extract_bits(VX.bitcast(h, 'float16x64'), 0).astype('uint16'), 'float16')

    @T.macro
    def rows_reduce(src, dst, maximum: bool):
        if intrinsic_rows:
            intrinsic_rows_reduce(src, dst, maximum)
        else:
            # Control path retained until worker ownership accepts vector I/O.
            rows = T.alloc_local((2,256), 'float32')
            reduced = T.alloc_local((2,), 'float32')
            for pair in T.serial(64):
                T.copy(src[pair*2:pair*2+2, :], rows)
                if maximum:
                    T.reduce_max(rows, reduced, dim=1)
                else:
                    T.reduce_sum(rows, reduced, dim=1)
                for r in T.serial(2):
                    dst[pair*2+r] = T.cast(reduced[r], 'float16')

    @T.prim_func
    def fa(Q: T.Tensor((128, 256), 'float16'),
           K: T.Tensor((1024, 256), 'float16'),
           V: T.Tensor((256, 1024), 'float16'),
           O: T.Tensor((128, 256), 'float16')):
        with T.Kernel(1, threads=2+state_workers):
            q = T.alloc_shared((128, 256), 'float16')
            k = T.alloc_shared((256, 256), 'float16')
            v = T.alloc_shared((256, 256), 'float16')
            score = T.alloc_shared((128, 256), 'float16')
            prob = T.alloc_shared((128, 256), 'float16')
            partial = T.alloc_shared((128, 256), 'float16')
            sf = T.alloc_shared((128, 256), 'float16')
            pf = T.alloc_shared((128, 256), 'float16')
            out = T.alloc_shared((128, 256), 'float16')
            oh = T.alloc_shared((128, 256), 'float16')
            # Small owner-private row state: scalar row broadcasts read ordinary
            # private memory, never VTCM. Row-wide updates remain vectorized.
            maximum = T.alloc_buffer((128,), 'float16', scope=state_scope)
            denom = T.alloc_buffer((128,), 'float16', scope=state_scope)
            alpha = T.alloc_buffer((128,), 'float16', scope=state_scope)
            mx = T.alloc_buffer((128,), 'float16', scope=state_scope)
            sm = T.alloc_buffer((128,), 'float16', scope=state_scope)
            T.annotate_layout({Q:packed(128,256), K:packed(1024,256,True),
                               V:packed(256,1024,True), q:packed(128,256),
                               k:packed(256,256,True), v:packed(256,256,True),
                               prob:packed(128,256), sf:rowmajor(128,256),
                               pf:rowmajor(128,256), out:rowmajor(128,256),
                               oh:rowmajor(128,256), O:rowmajor(128,256)})
            T.clear(sf)
            T.clear(pf)
            T.clear(out)
            T.fill(maximum, -T.infinity('float16'))
            T.clear(denom)
            T.clear(alpha)
            T.clear(mx)
            T.clear(sm)
            for kb in T.Pipelined(panels, num_stages=2,
                                 annotations={'tl.workergroup_schedule': 'async'}):
                with T.pipeline_stage('input', engine='hvx', workers=1):
                    T.copy(Q, q, annotations={'hexagon.dma_direct':1})
                    T.copy(K[kb*256:(kb+1)*256, :], k, annotations={'hexagon.dma_direct':1})
                    T.copy(V[:, kb*256:(kb+1)*256], v, annotations={'hexagon.dma_direct':1})
                with T.pipeline_stage('qk', engine='hmx', workers=1, physical_owner='matrix'):
                    T.gemm(q, k, score, transpose_B=True, clear_accum=True)
                with T.pipeline_stage('softmax', engine='hvx', workers=state_workers, physical_owner='state'):
                    T.transform(score, sf)
                    for i in row_loop(128):
                        for j in T.vectorized(256):
                            sf[i, j] = T.Select(kb*256+j <= query_block*128+i,
                                               sf[i, j]*T.float16(0.0625), -T.infinity('float16'))
                    rows_reduce(sf, mx, True)
                    for i in row_loop(128):
                        # Keep the online max in SSA through alpha calculation.
                        # A load/store/reload of mx through nested bitcasts can
                        # otherwise reuse the pre-store panel max in codegen.
                        old_scalar = maximum[i]
                        panel_scalar = mx[i]
                        old_max = half_splat(old_scalar)
                        new_max = native('max16', half_splat(panel_scalar), old_max)
                        new_alpha = half_exp(native('sub16', old_max, new_max))
                        mx[i] = half_extract(new_max)
                        alpha[i] = half_extract(new_alpha)
                        maximum[i] = half_extract(new_max)
                    exp_rows(sf, mx, sm)
                    for i in state_loop(128):
                        denom[i] = denom[i]*alpha[i]+sm[i]
                    T.transform(sf, prob)
                with T.pipeline_stage('pv', engine='hmx', workers=1, physical_owner='matrix'):
                    T.gemm(prob, v, partial, transpose_B=True, clear_accum=True)
                with T.pipeline_stage('update', engine='hvx', workers=state_workers, physical_owner='state'):
                    T.transform(partial, pf)
                    for i in row_loop(128):
                        for j in T.vectorized(256):
                            out[i, j] = out[i, j]*alpha[i]+pf[i, j]
                    if kb == panels - 1:
                        if direct_normalize_output:
                            # Both buffers are row-major. Keep the same half
                            # multiply but materialize only its terminal result.
                            normalize_rows(out, denom, oh)
                        else:
                            normalize_rows(out, denom, out)
                            T.transform(out, oh)
                        T.copy(oh, O, annotations={'hexagon.dma_direct':1})
    return (fa.with_attr('global_symbol', f'fa_worker_qb{query_block}')
            .with_attr('tl.workergroup_max_workers', 2+state_workers)
            .with_attr('tl.workergroup_max_events', 64)
            .with_attr('tl.workergroup_max_vtcm_bytes', 4*1024*1024-768))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-block', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--direct-normalize-output', action='store_true',
                        help='experimental single-worker terminal copy fusion; default off')
    parser.add_argument('--intrinsic-rows', action=argparse.BooleanOptionalAction, default=True,
                        help='explicit fixed-tree candidate; fails closed if worker vector I/O is unsupported')
    args = parser.parse_args()
    result = tilelang.engine.lower(make_fa(args.query_block, args.intrinsic_rows, args.state_workers,
                                         args.direct_normalize_output), target='hexagon',
                                  enable_host_codegen=False, enable_device_compile=False)
    args.output.write_text(result.kernel_source)
