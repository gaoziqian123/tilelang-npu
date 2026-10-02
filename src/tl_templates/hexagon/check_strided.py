"""Host lane emulator for the strided helpers; no device/compiler dependency."""


def shuff(v):
    return [v[(j & 1) * 32 + (j >> 1)] for j in range(64)]


def deal(v):
    return v[::2] + v[1::2]


def transpose(t):
    t = [v[:] for v in t]
    for i in range(4):
        for v in range(16):
            for _ in range(i + 1):
                t[v] = deal(t[v])
        for a in range(16):
            if not a & (1 << i):
                b = a | (1 << i)
                e = [x for pair in zip(t[a][::2], t[b][::2]) for x in pair]
                o = [x for pair in zip(t[a][1::2], t[b][1::2]) for x in pair]
                t[a], t[b] = e, o
        for v in range(16):
            for _ in range(i + 1):
                t[v] = shuff(t[v])
    for v in range(16):
        r = t[v][32:] + t[v][:32]
        e = [x for x in r[::2] for _ in range(2)]
        o = [x for x in r[1::2] for _ in range(2)]
        sw = [e[l] if l & 1 else o[l] for l in range(64)]
        t[v] = [sw[l] if ((l >> 5) & 1) != (l & 1) else t[v][l]
                for l in range(64)]
    return t


def main():
    cases = 0
    for tiles in (1, 2, 3, 80):
        # Preserve all 960 small cases; exercise large batches with compact,
        # odd, padded and GEMM-parent strides, including split-block rows.
        strides = (32, 33, 47, 64, 129, 2560) if tiles == 80 else (32, 33, 47, 64, 129)
        offsets = (0, 1, 32, 63) if tiles == 80 else range(64)
        for stride in strides:
            for offset in offsets:
                src = [-1] * (offset + tiles * 32 * stride + 64)
                dst = src[:]
                for r in range(tiles * 32):
                    for c in range(32):
                        src[offset + r * stride + c] = r * 32 + c
                for t in range(tiles):
                    rows = [src[offset + (t * 32 + r) * stride:
                                offset + (t * 32 + r) * stride + 32]
                            for r in range(32)]
                    pairs = [rows[r] + rows[r + 1] for r in range(0, 32, 2)]
                    ah = sum([shuff(v) for v in pairs], [])
                    wh = sum([shuff(v) for v in transpose(pairs)], [])
                    for r in range(32):
                        for c in range(32):
                            assert ah[(r // 2) * 64 + 2 * c + r % 2] == rows[r][c]
                            assert wh[(r // 2) * 64 + 2 * c + r % 2] == rows[c][r]
                    for rp in range(16):
                        v = deal(ah[rp * 64:(rp + 1) * 64])
                        for half in range(2):
                            start = offset + (t * 32 + 2 * rp + half) * stride
                            # Emulate vlalign + the two aligned predicate stores.
                            low = v[half * 32:(half + 1) * 32] + [-2] * 32
                            off = start % 64
                            rotated = low[-off:] + low[:-off] if off else low
                            base = start - off
                            for lane in range(off, min(off + 32, 64)):
                                dst[base + lane] = rotated[lane]
                            if off > 32:
                                for lane in range(off - 32):
                                    dst[base + 64 + lane] = rotated[lane]
                assert dst == src, (tiles, stride, offset)
                cases += 1
    print(f"PASS: {cases} cases; AH/WH 32x32 lane mappings, transpose, strided unpack and sentinels")


if __name__ == "__main__":
    main()
