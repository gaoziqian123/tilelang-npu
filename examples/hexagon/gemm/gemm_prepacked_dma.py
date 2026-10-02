"""External pointers MUST contain packed bytes described by the manifest.

Recommended standard async path (same DSL, no generated-C patching):
  --M 1024 --N 12288 --K 2560 --BM 64 --BN 256 --BK 1280
  --schedule async --reuse-b --merge-edges --load 1 --transform-output
Round-sync defaults remain available as the baseline. This recommendation is
not a performance guarantee for a newly built runtime/device combination.

reuse-b keeps both K panels resident across the entire M traversal: it requires
K/BK == 2 (the pipeline depth), not a larger per-output accumulator. Keep BK
unchanged when comparing numerical results; BK=K changes partial-sum rounding.
"""
import argparse
import json
from pathlib import Path
import tilelang
from tilelang import language as T
from tilelang.hexagon.physical_contract import compact_contract, c_packer


def layout(rows, cols, transpose=False):
    if transpose:
        return T.Layout((rows, cols), lambda r, c:
            (r//32, c//32, c%32//2, (r%32)*2+c%2))
    return T.Layout((rows, cols), lambda r, c:
        (r//32, c//32, r%32//2, (c%32)*2+r%2))


def kernel(M=32, N=64, K=64, BM=32, BN=32, BK=32, transform_output=False,
           schedule="round_sync", reuse_b=False, load=3, merge_edges=False):
    if any(x <= 0 or x % 32 for x in (M,N,K,BM,BN,BK)) or M%BM or N%BN or K%BK:
        raise ValueError("prepacked example requires full aligned tiles")
    if reuse_b and K//BK != 2:
        raise ValueError("reuse-b requires two K panels resident in the two pipeline slots")
    la, lb, sa, sb = layout(M,K), layout(N,K,True), layout(BM,BK), layout(BN,BK,True)
    # Contiguous physical chunks of the producer's layout, not cyclic columns.
    # The fragment map also governs initialization and the final logical copy.
    physical = lambda r,c: ((r//32)*(BN//32)+c//32)*1024+(r%32//2)*64+(c%32)*2+r%2
    chunk = BM*BN//2
    accumulator_layout = T.Fragment((BM,BN),
        forward_thread_fn=lambda r,c: physical(r,c)//chunk,
        forward_index_fn=lambda r,c: physical(r,c)%chunk)
    if (BM//2) % 32:
        # An odd number of tile rows cannot be split into two whole tile-row
        # owners. Split each tile at a 16-row boundary instead: each worker
        # retains whole row pairs and every output vector has a single owner.
        # Compact the worker-local half-tiles; never read another worker's c.
        accumulator_layout = T.Fragment((BM,BN),
            forward_thread_fn=lambda r,c: r%32//16,
            forward_index_fn=lambda r,c:
                ((r//32)*(BN//32)+c//32)*512+(r%16//2)*64+(c%32)*2+r%2)
    @T.prim_func
    def prepacked(A: T.Tensor((M,K), "float16"), B: T.Tensor((N,K), "float16"),
                  C: T.Tensor((M,N), "float16")):
        T.func_attr({"tl.workergroup_max_workers":6, "tl.workergroup_max_events":256,
                     "tl.workergroup_max_vtcm_bytes":8388608})
        with T.Kernel(1, threads=load+3):
            a=T.alloc_shared((BM,BK), "float16")
            b=T.alloc_shared((BN,BK), "float16")
            p=T.alloc_shared((BM,BN), "float16")
            c=T.alloc_fragment((BM,BN), "float16")
            if transform_output:
                output_stage=T.alloc_shared((BM,BN), "float16")
                T.annotate_layout({output_stage:T.Layout((BM,BN),lambda i,j:(i,j)),
                                   C:T.Layout((M,N),lambda i,j:(i,j))})
            T.annotate_layout({A:la, B:lb, a:sa, b:sb, c:accumulator_layout})
            for n in T.serial(N//BN):
                for m in T.serial(M//BM):
                    T.clear(c)
                    for k in T.Pipelined(K//BK, num_stages=2,
                            annotations={"tl.workergroup_schedule":schedule,
                                         "tl.workergroup_merge_edges":int(merge_edges)}):
                        with T.pipeline_stage("input", engine="hvx", workers=load):
                            T.copy(A[m*BM:(m+1)*BM,k*BK:(k+1)*BK],a,
                                   annotations={"hexagon.dma_direct":1})
                            if reuse_b:
                                if m == 0:
                                    T.copy(B[n*BN:(n+1)*BN,k*BK:(k+1)*BK],b,
                                           annotations={"hexagon.dma_direct":1,
                                                        "tl.worker_resident_loop":m})
                            else:
                                T.copy(B[n*BN:(n+1)*BN,k*BK:(k+1)*BK],b,
                                       annotations={"hexagon.dma_direct":1})
                        with T.pipeline_stage("multiply", engine="hmx", workers=1):
                            T.gemm(a,b,p,transpose_B=True,clear_accum=True)
                        with T.pipeline_stage("sum", engine="hvx", workers=2):
                            for i,j in T.Parallel(BM,BN):
                                c[i,j]+=p[i,j]
                    if transform_output:
                        T.transform(c,output_stage)
                        T.copy(output_stage,C[m*BM:(m+1)*BM,n*BN:(n+1)*BN],
                               annotations={"hexagon.dma_direct":1})
                    else:
                        T.copy(c,C[m*BM:(m+1)*BM,n*BN:(n+1)*BN])
    return prepacked


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name,value in (("M",32),("N",64),("K",64),("BM",32),("BN",32),("BK",32)):
        parser.add_argument("--"+name,type=int,default=value)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--transform-output",action="store_true")
    parser.add_argument("--schedule",choices=["round_sync","async"],default="round_sync")
    parser.add_argument("--reuse-b",action="store_true")
    parser.add_argument("--merge-edges",action="store_true")
    parser.add_argument("--load",type=int,default=3)
    parser.add_argument("--host-packer-output",type=Path,required=True)
    args=vars(parser.parse_args()); output=args.pop("output"); host_output=args.pop("host_packer_output")
    lowered=tilelang.engine.lower(kernel(**args),target="hexagon",
                                 enable_host_codegen=False,enable_device_compile=False)
    output.write_text(lowered.kernel_source)
    manifest=dict(version=1,owner="prepacked_owner",shape=args,
                  outputs={"C":{"logical_dtype":"float16", "logical_shape":[args["M"],args["N"]],
                                "bytes":args["M"]*args["N"]*2,"layout":"row-major"}},
                  accumulator_dtype="float16",
                  required_runtime_symbol="tl_hex_dma_copy_2d_wait",
                   completion="synchronous-wait-then-group-barrier",inputs={})
    # Logical DMA payload, not bus transactions or measured device counters.
    # The resident initializer runs only at m == 0 for each n and K slot.
    manifest["dma_payload_bytes"] = dict(
        A=2*args["M"]*args["K"]*(args["N"]//args["BN"]),
        B=2*args["N"]*args["K"]*(1 if args["reuse_b"] else args["M"]//args["BM"]),
        C=2*args["M"]*args["N"] if args["transform_output"] else None)
    # Serialize compiler IR attributes, never reverse-parse generated C.
    for func in lowered.device_mod.functions.values():
        attrs = func.attrs
        if attrs is None or "tl.workergroup_groups" not in attrs:
            continue
        def number_map(spec, keys):
            return {key:int(spec[key]) for key in keys}
        groups = [{**number_map(g,("first_worker","worker_count")),
                   "engine":str(g["engine"])} for g in attrs["tl.workergroup_groups"]]
        edges = []
        if args["schedule"] == "async":
            for edge in attrs["tl.workergroup_edges"]:
                item = number_map(edge,("producer_group","consumer_group","slot_desc","depth",
                                        "ready_event_base","free_event_base"))
                item["protected_slots"] = [int(x) for x in edge["protected_slots"]]
                edges.append(item)
        allocations = [number_map(a,("byte_offset","bytes_per_slot","depth","alignment"))
                       for a in attrs["tl.workergroup_physical_allocations"]]
        abi_slots = []
        for slot in attrs["tl.workergroup_abi_slots"]:
            matches = [a for a in attrs["tl.workergroup_physical_allocations"]
                       if a["data"].same_as(slot["buffer"].data)]
            if len(matches) != 1:
                raise ValueError("ABI slot lacks a unique physical allocation")
            abi_slots.append(number_map(matches[0],("byte_offset","bytes_per_slot","depth","alignment")))
        manifest["plan"] = dict(abi_version=3, groups=groups, edges=edges,
            slots=abi_slots, slot_count=len(abi_slots), edge_count=len(edges),
            allocations=allocations, team_size=sum(g["worker_count"] for g in groups),
            event_count=max((e["free_event_base"]+e["depth"] for e in edges),default=0),
            iteration_count=int(attrs["tl.workergroup_iteration_count"]),
            round_count=int(attrs["tl.workergroup_round_count"]),
            initial_alignment=int(attrs["tl.workergroup_initial_alignment"]),
            vtcm_bytes=int(attrs["tl.workergroup_vtcm_bytes"]),
            ddr_bytes=int(attrs.get("tl.workergroup_ddr_bytes",0)))
    packers=[]
    for name,rows in (("A",args["M"]),("B",args["N"])):
        spec=compact_contract(layout(rows,args["K"],name=="B"))
        manifest["inputs"][name]=spec
        packers.append(c_packer(spec,"prepacked_pack_"+name))
    host_output.write_text("#pragma once\n"+"\n".join(packers))
    output.with_suffix(".json").write_text(json.dumps(manifest,sort_keys=True))
