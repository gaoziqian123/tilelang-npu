"""Opt-in lowering of standard, static FP vector expressions to HVX chunks.

No arithmetic reassociation, contraction, implicit approximation or gather.
The mode is an explicit function contract, not inferred from a kernel name.
"""
import struct
from tvm import DataType, arith, tirx as tir
from .language import vector as V


class _Lower:
    def __init__(self, func):
        self.mode = str(func.attrs.get("tl.hvx_arithmetic_mode", "strict"))
        self.tree = str(func.attrs.get("tl.hvx_reduce_order", "unspecified"))
        self.math = {key:str(func.attrs.get('tl.hvx_'+key+'_mode','strict')) for key in ('exp','exp2','div')}
        self.analyzer = arith.Analyzer()
        self.bindings = {}

    def reduce(self, x):
        if self.tree != "chunks_asc_tree_asc_seed_post_v1" or self.mode != "native_target_v1":
            raise ValueError("VectorLowering: reduction requires explicit tree and native mode")
        if len(x.axis)!=1 or len(x.source)!=1 or int(x.value_index)!=0 or str(x.dtype)!="float32":
            raise ValueError("VectorLowering: only single-axis FP32 reductions supported")
        axis=x.axis[0]
        if not isinstance(axis.dom.min,tir.IntImm) or int(axis.dom.min)!=0 or not isinstance(axis.dom.extent,tir.IntImm):
            raise ValueError("VectorLowering: reduction extent must be static, zero based")
        lanes=int(axis.dom.extent)
        if lanes<=0 or lanes%32 or not isinstance(x.condition,tir.IntImm) or not int(x.condition):
            raise ValueError("VectorLowering: reduction requires complete unmasked chunks")
        comb=x.combiner
        if len(comb.result)!=1 or len(comb.identity_element)!=1:
            raise ValueError("VectorLowering: tuple reducer unsupported")
        result=comb.result[0]
        if not isinstance(result,(tir.Add,tir.Max)) or not result.a.same_as(comb.lhs[0]) or not result.b.same_as(comb.rhs[0]):
            raise ValueError("VectorLowering: unsupported reducer expression")
        maximum=isinstance(result,tir.Max)
        identity=comb.identity_element[0]
        if not isinstance(identity,tir.FloatImm) or float(identity.value)!=(float('-inf') if maximum else 0.0):
            raise ValueError("VectorLowering: reducer identity mismatch")
        # Only a contiguous source (optionally exact half->float cast) is legal.
        source=x.source[0]
        cast=isinstance(source,tir.Cast)
        read=source.value if cast else source
        if not isinstance(read,tir.BufferLoad) or len(read.indices)!=1:
            raise ValueError("VectorLowering: reduction source must be contiguous buffer")
        idx=read.indices[0]
        base=self.analyzer.simplify(tir.stmt_functor.substitute(idx,{axis.var:tir.const(0,axis.var.dtype)}))
        step=self.analyzer.simplify(tir.stmt_functor.substitute(idx,{axis.var:tir.const(1,axis.var.dtype)})-base)
        if not isinstance(step,tir.IntImm) or int(step)!=1:
            raise ValueError("VectorLowering: reduction stride must be one")
        # Prove the entire symbolic index, not merely its first two samples.
        if not self.analyzer.can_prove_equal(idx,base+axis.var):
            raise ValueError("VectorLowering: nonaffine reduction index")
        vector=tir.BufferLoad(read.buffer,[tir.Ramp(base,1,lanes)])
        if cast:
            if str(source.dtype)!="float32" or str(read.dtype)!="float16":
                raise ValueError("VectorLowering: reduction cast must widen half")
            vector=tir.Cast(f"float32x{lanes}",vector)
        chunks=self.expr(vector)
        op="max32" if maximum else "add32"
        acc=V.splat_bits(tir.const(0xff800000 if maximum else 0,"uint32"),"float32x32")
        for chunk in chunks:
            acc=V.map_chunks(op,(acc,),(chunk,),mode=self.mode)[0]
        acc=V.map_chunks("max32_asc" if maximum else "sum32_asc",(acc,),mode=self.mode)[0]
        if len(x.init)>1:
            raise ValueError("VectorLowering: multiple reduction seeds unsupported")
        if len(x.init)==1:
            seed=x.init[0]
            if not isinstance(seed,tir.FloatImm):
                raise ValueError("VectorLowering: reduction seed must be constant FP32")
            bits=int.from_bytes(struct.pack('<f',float(seed.value)),'little')
            acc=V.map_chunks(op,(acc,),(V.splat_bits(tir.const(bits,"uint32"),"float32x32"),),mode=self.mode)[0]
        return acc

    def region(self, buffer, indices, lanes):
        if len(indices) != 1 or not isinstance(indices[0], tir.Ramp):
            raise ValueError("VectorLowering: contiguous Ramp required")
        ramp = self.analyzer.simplify(indices[0])
        if not isinstance(ramp.base, tir.IntImm) or not isinstance(ramp.stride, tir.IntImm) or int(ramp.stride) != 1:
            raise ValueError("VectorLowering: dynamic/unproven address or strided load rejected")
        if int(ramp.lanes) != lanes:
            raise ValueError("VectorLowering: region lane mismatch")
        return int(ramp.base)

    def expr(self, x):
        d = DataType(x.dtype)
        if d.bits not in (16, 32) or not str(x.dtype).startswith("float") or d.lanes*d.bits % 1024:
            raise ValueError("VectorLowering: FP16/FP32 whole physical chunks required")
        width = 1024 // d.bits
        dtype = f"float{d.bits}x{width}"
        if isinstance(x, tir.BufferLoad):
            offset = self.region(x.buffer, x.indices, d.lanes)
            return V.load_chunks(x.buffer, d.lanes, offset)
        if isinstance(x, tir.Broadcast):
            if isinstance(x.value,tir.Var) and x.value in self.bindings:
                return tuple(self.bindings[x.value] for _ in range(d.lanes//width))
            if isinstance(x.value,tir.Reduce):
                if d.bits!=32: raise ValueError("VectorLowering: reduction output must be FP32")
                value=self.reduce(x.value)
                return tuple(value for _ in range(d.lanes//width))
            if not isinstance(x.value, tir.FloatImm):
                raise ValueError("VectorLowering: only constant floating broadcasts supported")
            packed = struct.pack('<e' if d.bits == 16 else '<f', float(x.value.value))
            bits = int.from_bytes(packed, 'little')
            return tuple(V.splat_bits(tir.const(bits, "uint32"), dtype) for _ in range(d.lanes//width))
        if isinstance(x, tir.Cast):
            src = self.expr(x.value)
            if x.dtype == x.value.dtype:
                return src
            if d.bits == 32 and DataType(x.value.dtype).bits == 16:
                return V.map_chunks("widen", src, mode="strict_exact_v1")
            if d.bits == 16 and DataType(x.value.dtype).bits == 32:
                return V.narrow_chunks(src)
            raise ValueError("VectorLowering: unsupported cast")
        ops = ((tir.Add,"add"),(tir.Sub,"sub"),(tir.Mul,"mul"),(tir.Min,"min"),(tir.Max,"max"))
        if isinstance(x,tir.Div):
            if d.bits!=32 or self.math['div']!='nr2_positive_bounded_v1':
                raise ValueError('VectorLowering: division requires explicit nr2_positive_bounded_v1')
            return V.map_chunks('div32_nr2',self.expr(x.a),self.expr(x.b),mode=self.math['div'])
        if isinstance(x,tir.Call):
            name=str(getattr(x.op,'name',''))
            if name=='tirx.exp' and d.bits==32 and self.math['exp']=='hvx_math32_v1':
                return V.map_chunks('exp32_v1',self.expr(x.args[0]),mode=self.math['exp'])
            if name=='tirx.exp2' and d.bits==16 and self.math['exp2']=='ggml_qf16_clamp24_v1':
                return V.map_chunks('exp2_16_nonpositive',self.expr(x.args[0]),mode=self.math['exp2'])
            raise ValueError('VectorLowering: unsupported call or missing explicit math mode '+name)
        for cls, name in ops:
            if isinstance(x, cls):
                if self.mode != "native_target_v1":
                    raise ValueError("VectorLowering: arithmetic requires explicit native_target_v1; strict unavailable")
                return V.map_chunks(name+str(d.bits), self.expr(x.a), self.expr(x.b), mode=self.mode)
        if isinstance(x, tir.Select):
            relations = ((tir.EQ,"eq"),(tir.NE,"ne"),(tir.LT,"lt"),(tir.LE,"le"),(tir.GT,"gt"),(tir.GE,"ge"))
            for cls, name in relations:
                if isinstance(x.condition, cls):
                    c=x.condition
                    # IEEE != is unordered, all other standard relations ordered.
                    masks=V.compare_chunks(self.expr(c.a),self.expr(c.b),relation=name,
                                            nan_policy="unordered" if name=="ne" else "ordered")
                    a,b=self.expr(x.true_value),self.expr(x.false_value)
                    if len(masks)!=len(a) or len(a)!=len(b) or c.a.dtype!=x.true_value.dtype:
                        raise ValueError("VectorLowering: select predicate lane mapping mismatch")
                    return tuple(V.bitcast(V.leaf("select",m,V.bitcast(u,"uint8x128"),V.bitcast(v,"uint8x128")),dtype)
                                 for m,u,v in zip(masks,a,b))
            raise ValueError("VectorLowering: unsupported select predicate")
        if isinstance(x, tir.Shuffle):
            chunks=[]
            for value in x.vectors:
                if DataType(value.dtype).bits!=d.bits:
                    raise ValueError("VectorLowering: shuffle type mismatch")
                chunks.extend(self.expr(value))
            if any(not isinstance(i,tir.IntImm) for i in x.indices):
                raise ValueError("VectorLowering: static shuffle required")
            out=[]
            indices=[int(i) for i in x.indices]
            for base in range(0,len(indices),width):
                ids=indices[base:base+width]
                if any(i<0 or i>=len(chunks)*width for i in ids):
                    raise ValueError("VectorLowering: shuffle index out of range")
                owners=sorted(set(i//width for i in ids))
                if len(owners)==1:
                    raw=V.permute(V.bitcast(chunks[owners[0]],"uint8x128"),[i%width for i in ids],lane_bytes=d.bits//8)
                elif len(owners)==2:
                    raw=V.permute2(*(V.bitcast(chunks[o],"uint8x128") for o in owners),
                                   [owners.index(i//width)*width+i%width for i in ids],lane_bytes=d.bits//8)
                else:
                    raise ValueError("VectorLowering: shuffle spans more than two physical inputs")
                out.append(V.bitcast(raw,dtype))
            return tuple(out)
        raise ValueError("VectorLowering: unsupported vector expression " + type(x).__name__)

    def rewrite(self, node):
        if isinstance(node,tir.Bind) and isinstance(node.value,tir.Reduce):
            value=self.reduce(node.value)
            var=tir.Var(node.var.name+'_hvx','float32x32')
            self.bindings[node.var]=var
            return tir.Bind(var,value)
        if isinstance(node,tir.For):
            # Enumerate a bounded static domain before physical emission. Analyzer
            # simplifies each affine address; no sampled/assumed bounds.
            lo=self.analyzer.simplify(node.min)
            extent=self.analyzer.simplify(node.extent)
            if not isinstance(lo,tir.IntImm) or not isinstance(extent,tir.IntImm) or not 0<=int(extent)<=4096:
                raise ValueError("VectorLowering: unproven loop domain")
            bodies=[]
            for i in range(int(lo),int(lo)+int(extent)):
                body=tir.stmt_functor.substitute(node.body,{node.loop_var:tir.const(i,node.loop_var.dtype)})
                bodies.append(tir.stmt_functor.ir_transform(body,self.rewrite,None,["tirx.For","tirx.BufferStore","tirx.Bind"]))
            return bodies[0] if len(bodies)==1 else tir.SeqStmt(bodies) if bodies else tir.Evaluate(0)
        if isinstance(node,tir.BufferStore) and DataType(node.value.dtype).lanes>1:
            def alias(value):
                if isinstance(value,tir.BufferLoad) and value.buffer.data.same_as(node.buffer.data):
                    raise ValueError("VectorLowering: in-place vector expression requires alias scheduling proof")
            tir.stmt_functor.post_order_visit(node.value,alias)
            lanes=DataType(node.value.dtype).lanes
            offset=self.region(node.buffer,node.indices,lanes)
            return V.store_chunks(node.buffer,self.expr(node.value),lanes,offset)
        return None


def VectorLowering():
    @tir.transform.prim_func_pass(opt_level=0)
    def lower(func, mod, ctx):
        if not func.attrs or not func.attrs.get("tl.vector_required", False):
            return func
        impl=_Lower(func)
        body=tir.stmt_functor.ir_transform(func.body,impl.rewrite,None,["tirx.For","tirx.BufferStore","tirx.Bind"])
        return func.with_body(body)
    return lower
