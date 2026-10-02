"""Independent FP64 token recurrence and independently evaluated chunk WY.

Logical inputs are head-major, Q/K [Hk,T,D], V [Hv,T,D], g/beta
[Hv,T], S0 [Hv,D,D]. Q scale is applied exactly once at output.
No device intermediate is accepted by either reference.
"""
import argparse
import numpy as np


def head_index(h, hk, hv, mapping):
    if mapping == 'mod':
        return h % hk
    if mapping == 'group' and hv % hk == 0:
        return h // (hv // hk)
    raise ValueError('mapping must be mod, or divisible group')


def recurrent(q, k, v, g, beta, s0, mapping='mod'):
    q, k, v, g, beta, state = [np.asarray(x, np.float64).copy()
                              for x in (q, k, v, g, beta, s0)]
    hv, nt, d = v.shape
    out = np.empty_like(v)
    for h in range(hv):
        kh = head_index(h, len(k), hv, mapping)
        for t in range(nt):
            state[h] *= np.exp(g[h, t])
            r = beta[h, t] * (v[h, t] - k[kh, t] @ state[h])
            state[h] += np.outer(k[kh, t], r)
            out[h, t] = (q[kh, t] @ state[h]) / np.sqrt(d)
    return out, state


def chunk_wy(q, k, v, g, beta, s0, mapping='mod', chunk=64):
    q, k, v, g, beta, state = [np.asarray(x, np.float64).copy()
                              for x in (q, k, v, g, beta, s0)]
    hv, nt, d = v.shape
    out = np.empty_like(v)
    for h in range(hv):
        kh = head_index(h, len(k), hv, mapping)
        for start in range(0, nt, chunk):
            stop = min(start + chunk, nt)
            kk, qq, vv = k[kh, start:stop], q[kh, start:stop], v[h, start:stop]
            gg = np.cumsum(g[h, start:stop])
            bb = beta[h, start:stop]
            lower = np.tril(bb[:, None] * np.exp(gg[:, None] - gg[None, :])
                            * (kk @ kk.T), -1)
            u = bb[:, None] * vv
            w = (bb * np.exp(gg))[:, None] * kk
            # Ordered forward substitution, not an inverse from tested data.
            for i in range(stop-start):
                for j in range(i):
                    u[i] -= lower[i, j] * u[j]
                    w[i] -= lower[i, j] * w[j]
            r = u - w @ state[h]
            p = np.tril((qq @ kk.T) * np.exp(gg[:, None] - gg[None, :]))
            out[h, start:stop] = (np.exp(gg)[:, None] * (qq @ state[h]) + p @ r) / np.sqrt(d)
            state[h] = np.exp(gg[-1]) * state[h] + (kk * np.exp(gg[-1]-gg)[:, None]).T @ r
    return out, state


def inputs(nt, hk, hv, d=128, mode=0, seed=20260928):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(hk, nt, d)) / np.sqrt(d)
    k = rng.normal(size=(hk, nt, d))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.normal(size=(hv, nt, d))
    g = -rng.uniform(.001, (.02, .2, 1., .05)[mode], size=(hv, nt))
    beta = rng.uniform(.01, 1., size=(hv, nt))
    s0 = rng.normal(0, .1, size=(hv, d, d))
    if mode == 3:
        beta[:, ::7] = 0
        g[:, ::11] = 0
    return (q.astype(np.float16), k.astype(np.float16), v.astype(np.float16),
            g.astype(np.float32), beta.astype(np.float32), s0.astype(np.float32))


def metrics(got, ref):
    x, y = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    cosine = float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))) if finite else float('nan')
    rms = np.sqrt(np.mean(y*y))
    rel = float(np.max(np.abs(x-y)/(np.abs(y)+.02*rms))) if finite else float('inf')
    return finite, cosine, rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--standard', action='store_true')
    args = p.parse_args()
    shapes = [(65, 2, 4)] + ([(1024,16,32)] if args.standard else [])
    for nt,hk,hv in shapes:
        for mapping in ('mod','group'):
            for mode in range(4):
                data = inputs(nt,hk,hv,mode=mode)
                oracle = recurrent(*data,mapping=mapping)
                candidate = chunk_wy(*data,mapping=mapping)
                for name, got, ref in zip(('O','S1'),candidate,oracle):
                    finite,cos,rel = metrics(got,ref)
                    print(f'WY T={nt} Hk={hk} Hv={hv} mapping={mapping} mode={mode} {name} finite={finite} cos={cos:.12f} rms_rel={rel:.3g}',flush=True)
                    np.testing.assert_allclose(got,ref,rtol=1e-10,atol=1e-12)
                    assert finite and cos >= .999
    print('HOST_FP64_WY_PASS (not device or mixed-precision validation)')


if __name__ == '__main__':
    main()
