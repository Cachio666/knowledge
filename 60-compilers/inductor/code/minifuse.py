"""迷你融合调度器 / 代价模型 / codegen。

★ 本模块是**本仓库的迷你实现**，不是 PyTorch Inductor 的调度器。
  **判据与默认值**逐条抄自固定版本的 torch 源码（见 specs.py，每条带 `# 来源：`），
  但**实现的形态是为教学重写的**：真实调度器工作在 `BaseSchedulerNode` 与 sympy
  索引表达式上，本仓库工作在 `minigraph.Node` 与字符串索引元组上。

★ 本模块实现的三条门槛链（对应 02 号文档）：
  1. 垂直（生产者→消费者）：**索引必须逐字匹配**，否则报 "memory deps did not match"；
  2. 水平（消费者↔消费者）：`shared_data_score >= score_fusion_memory_threshold`；
  3. 通用上限：`max_fusion_size` 节点数上限、`aggressive_fusion` 无公共读即拒、
     `max_fusion_unique_io_buffers` 读缓冲上限。
  另有 reduction ↔ reduction：**不是通用能力**，只有 mix-order 特例放行。
"""

import numpy as np

from minigraph import Dep, Graph, InterpError, ITEMSIZE, Node, eval_graph

# ---------------------------------------------------------------------------
# 默认值：全部来自 torch 源码，逐条见 specs.py
# ---------------------------------------------------------------------------

# 来源：torch/_inductor/config.py（2.6.0 / 2.7.1 / 2.13.0 三个版本一致）
DEFAULTS = {
    "score_fusion_memory_threshold": 10,     # 字节
    "max_fusion_size": 64,
    "aggressive_fusion": False,
    "max_fusion_unique_io_buffers": None,    # None = 该检查关闭
    "mix_order_reduction_max_reads": 10,
}
# 来源：torch/_inductor/scheduler.py 的 MixOrderReduction.can_fuse（2.13.0）
MIX_ORDER_SIZE_THRES = 5 * 2 ** 20
MIX_ORDER_NROW_MIN = 4096
MIX_ORDER_NCOL_MAX = 1024 * 16


class FuseConfig:
    def __init__(self, **kw):
        self.score_fusion_memory_threshold = DEFAULTS["score_fusion_memory_threshold"]
        self.max_fusion_size = DEFAULTS["max_fusion_size"]
        self.aggressive_fusion = DEFAULTS["aggressive_fusion"]
        self.max_fusion_unique_io_buffers = DEFAULTS["max_fusion_unique_io_buffers"]
        self.mix_order_reduction_max_reads = DEFAULTS["mix_order_reduction_max_reads"]
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("未知配置项 %s" % k)
            setattr(self, k, v)

    def as_row(self):
        return [("score_fusion_memory_threshold", self.score_fusion_memory_threshold),
                ("max_fusion_size", self.max_fusion_size),
                ("aggressive_fusion", self.aggressive_fusion),
                ("max_fusion_unique_io_buffers",
                 "None（检查关闭）" if self.max_fusion_unique_io_buffers is None
                 else self.max_fusion_unique_io_buffers),
                ("mix_order_reduction_max_reads",
                 self.mix_order_reduction_max_reads)]


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------


class WhyNoFuse(Exception):
    pass


def buf_node(graph, buf):
    n = graph.by_buf(buf)
    if n is None:
        raise InterpError("缓冲 %s 不在图里" % buf)
    return n


def buf_numbytes(graph, buf):
    return buf_node(graph, buf).numbytes


def is_ancestor(graph, a, b):
    """a 是不是 b 的祖先（b 的数据依赖里能否沿边走到 a）。"""
    seen, stack = set(), [b]
    while stack:
        cur = stack.pop()
        for i in cur.inputs:
            if i is a:
                return True
            if id(i) not in seen:
                seen.add(id(i))
                stack.append(i)
    return False


def common_deps(n1, n2):
    d1 = set(n1.reads) | ({n1.writes} if n1.writes else set())
    d2 = set(n2.reads) | ({n2.writes} if n2.writes else set())
    return sorted(d1 & d2, key=repr)


def shared_data_score(graph, n1, n2):
    """共享依赖的**字节数**之和。

    真实实现：`score = sum(dep_size_hint(dep) for dep in common)`，`count_bytes`
    默认 True，`dep_size_hint` 走 `dep.numbytes_hint()`（见 specs.SCORE_UNIT）。
    """
    return sum(buf_numbytes(graph, d.buf) for d in common_deps(n1, n2))


def index_match(producer, consumer):
    """垂直融合的索引判据：消费者对生产者输出的每一次读都要与生产者的写逐字相等。"""
    if producer.writes is None:
        return False, "生产者没有写"
    bad = [d for d in consumer.readers_of(producer.buf)
           if d.index != producer.writes.index]
    if bad:
        return False, "memory deps did not match（%s != %s）" % (bad[0].index,
                                                               producer.writes.index)
    return True, "索引逐字匹配 %s" % (producer.writes.index,)


def mix_order_can_fuse(graph, n1, n2, cfg):
    """`MixOrderReduction.can_fuse` 的迷你重写（常量与判据顺序照抄 specs）。"""
    if not (n1.kind == "reduction" and n2.kind == "reduction"):
        return False, "不是两个归约"
    if is_ancestor(graph, n1, n2) or is_ancestor(graph, n2, n1):
        return False, "祖先关系（mix-order 要求兄弟关系）"
    # has_mix_reduction_orders： (numel, rnumel) 必须互为镜像
    g1 = (n1.numel, n1.red_numel)
    g2 = (n2.numel, n2.red_numel)
    if g1 == g2 or tuple(g1) != tuple(reversed(g2)):
        return False, "循环序不是镜像 (%s vs %s)" % (g1, g2)
    if not common_deps(n1, n2):
        return False, "没有公共读"
    # 连续节点：本仓库取归约循环在**内**的那一侧（loop_order == "PR"）
    cont = n1 if n1.loop_order == "PR" else (n2 if n2.loop_order == "PR" else None)
    if cont is None:
        return False, "没有连续的归约节点"
    nrow, ncol = cont.numel, cont.red_numel
    if nrow * ncol < MIX_ORDER_SIZE_THRES:
        return False, "nrow*ncol < 5*2**20（%d）" % (nrow * ncol)
    if nrow < MIX_ORDER_NROW_MIN:
        return False, "nrow < 4096（%d）" % nrow
    if ncol > MIX_ORDER_NCOL_MAX:
        return False, "ncol > 1024*16（%d）" % ncol
    reads = {d.buf for d in list(n1.reads) + list(n2.reads)}
    if len(reads) > cfg.mix_order_reduction_max_reads:
        return False, "唯一读缓冲数 %d > mix_order_reduction_max_reads=%d" % (
            len(reads), cfg.mix_order_reduction_max_reads)
    return True, "mix-order 放行（nrow=%d ncol=%d 读缓冲 %d）" % (nrow, ncol, len(reads))


def can_fuse_vertical(graph, group, consumer, cfg):
    for p in group:
        if p.buf not in [d.buf for d in consumer.reads]:
            continue
        if p.kind == "extern" or consumer.kind == "extern":
            return False, "extern 核（默认不搜的那一类）不参与模板融合"
        if p.kind == "reduction" and consumer.kind == "reduction":
            return False, "reduction ↔ reduction 不是通用能力"
        ok, why = index_match(p, consumer)
        if not ok:
            return False, why
        if len(group) + 1 > cfg.max_fusion_size:
            return False, "exceeds max fusion（>%d 节点）" % cfg.max_fusion_size
        return True, "垂直融合：%s" % why
    return False, "没有可垂直融合的生产者"


def can_fuse_horizontal(graph, g1, g2, cfg):
    if any(n.kind == "extern" for n in g1 + g2):
        return False, "extern 核不参与横向融合"
    for n1 in g1:
        for n2 in g2:
            if n1.kind == "reduction" and n2.kind == "reduction":
                ok, why = mix_order_can_fuse(graph, n1, n2, cfg)
                if ok:
                    return True, why
                return False, "reduction ↔ reduction 非 mix-order：%s" % why
    score = 0
    for n1 in g1:
        for n2 in g2:
            score += shared_data_score(graph, n1, n2)
    if score == 0 and (not cfg.aggressive_fusion
                       or any(n.kind == "reduction" for n in g1 + g2)):
        return False, "no shared data（score=0）"
    if score < cfg.score_fusion_memory_threshold:
        return False, "score_fusion_memory_threshold（score=%d < %d）" % (
            score, cfg.score_fusion_memory_threshold)
    if len(g1) + len(g2) > cfg.max_fusion_size:
        return False, "exceeds max fusion（>%d 节点）" % cfg.max_fusion_size
    if cfg.max_fusion_unique_io_buffers is not None:
        bufs = {d.buf for n in g1 + g2 for d in n.reads}
        bufs |= {n.buf for n in g1 + g2 if n.writes is not None}
        if len(bufs) > cfg.max_fusion_unique_io_buffers:
            return False, "fusion_prevent_too_many_reads_and_writes"
    return True, "水平融合：共享 %d 字节" % score


# ---------------------------------------------------------------------------
# 贪心调度
# ---------------------------------------------------------------------------


class Decision:
    def __init__(self, kind, n1, n2, ok, why):
        self.kind, self.n1, self.n2, self.ok, self.why = kind, n1, n2, ok, why


class Plan:
    def __init__(self, graph, groups, decisions):
        self.graph = graph
        self.groups = groups
        self.decisions = decisions

    def rejected(self):
        return [d for d in self.decisions if not d.ok]

    def fuseable_nodes(self):
        return [n for g in self.groups for n in g]


def schedule(graph, cfg=None):
    """按拓扑序贪心：先试垂直，再试水平，都不行就自己开一组。"""
    cfg = cfg or FuseConfig()
    cand = [n for n in graph.nodes if n.kind in ("pointwise", "reduction", "extern")]
    groups, decisions = [], []
    gid = {}
    for n in cand:
        placed = False
        for gi, g in enumerate(groups):
            if id(n) in gid.get(gi, ()):
                continue
            ok, why = can_fuse_vertical(graph, g, n, cfg)
            decisions.append(Decision("vertical", g[-1], n, ok, why))
            if ok:
                g.append(n)
                gid.setdefault(gi, set()).add(id(n))
                placed = True
                break
        if placed:
            continue
        for gi, g in enumerate(groups):
            ok, why = can_fuse_horizontal(graph, g, [n], cfg)
            decisions.append(Decision("horizontal", g[0], n, ok, why))
            if ok:
                g.append(n)
                gid.setdefault(gi, set()).add(id(n))
                placed = True
                break
        if not placed:
            groups.append([n])
            gid[len(groups) - 1] = {id(n)}
    return Plan(graph, groups, decisions)


# ---------------------------------------------------------------------------
# 代价模型：材料化字节 vs 重算
# ---------------------------------------------------------------------------


def materialized_nodes(graph, groups):
    """必须**落到存储**的那些节点：图输出，或者有跨组消费者。

    这是「物化还是重算」这条主线在代价模型与 codegen 上的同一个判据，
    两处共用本函数，避免它们漂移。
    """
    gid = {}
    for gi, g in enumerate(groups):
        for n in g:
            gid.setdefault(id(n), set()).add(gi)
    consumers = {}
    for n in graph.nodes:
        for d in n.reads:
            consumers.setdefault(d.buf, []).append(n)
    out = []
    for n in graph.nodes:
        if n.kind in ("placeholder", "effect") or n.writes is None:
            continue
        mine = gid.get(id(n))
        if mine is None:
            continue
        users = consumers.get(n.buf, [])
        if n in graph.outputs or not users \
                or any(not (gid.get(id(c), set()) & mine) for c in users):
            out.append(n)
    return out


def plan_traffic(graph, groups):
    """按「组内不落存储、跨组才落」算一个执行计划的字节流量。

    读：只统计**不在本组内产生**的缓冲（组内中间结果留在寄存器里，不产生 HBM 流量）。
    写：只统计**被跨组消费或本身是图输出**的缓冲。
    重算：一个节点出现在 k 个组里就执行 k 次，其输入读也计 k 次。
    """
    gid = {}
    for gi, g in enumerate(groups):
        for n in g:
            gid.setdefault(id(n), set()).add(gi)
    consumers = {}
    for n in graph.nodes:
        for d in n.reads:
            consumers.setdefault(d.buf, []).append(n)

    read_bytes = 0
    for gi, g in enumerate(groups):
        produced = {n.buf for n in g}
        for n in g:
            for d in n.reads:
                if d.buf in produced:
                    continue
                read_bytes += buf_numbytes(graph, d.buf)

    materialized = materialized_nodes(graph, groups)
    write_bytes = sum(n.numbytes for n in materialized)

    exec_count = {}
    for g in groups:
        for n in g:
            exec_count[id(n)] = exec_count.get(id(n), 0) + 1
    recomputed = sum(c - 1 for c in exec_count.values() if c > 1)
    return {
        "kernels": len(groups),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total": read_bytes + write_bytes,
        "recomputed_nodes": recomputed,
        "materialized": [n.buf for n in materialized],
    }


# ---------------------------------------------------------------------------
# codegen：从调度计划生成 Python 源码（**生成物真的被 exec**）
# ---------------------------------------------------------------------------

_EXPR = {
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "neg": "(-{a})",
    "relu": "np.maximum({a}, 0.0)",
    "exp": "np.exp({a})",
    "sqrt": "np.sqrt({a})",
    "transpose": "np.ascontiguousarray({a}.T)",
    "sum": "np.sum({a}{axis})",
    "mm": "np.matmul({a}, {b})",
    "add_": "({a} + {a})",
    "copy_": "{a}",
}


def _expr(node, names):
    if node.kind == "placeholder":
        return names[node.buf]
    tmpl = _EXPR[node.op]
    if node.op == "sum":
        ax = node.attrs["axis"]
        return tmpl.format(a=names[node.reads[0].buf],
                           axis="" if ax is None else ", axis=%r" % (ax,))
    a = names[node.reads[0].buf] if node.reads else "None"
    const = node.attrs.get("const")
    if const is not None:
        b = repr(const)
    elif len(node.reads) > 1:
        b = names[node.reads[1].buf]
    else:
        b = a                       # 一元算子：模板里没有 {b}
    return tmpl.format(a=a, b=b)


def emit_python(plan, prefix="kern"):
    """为每个融合组生成一个 Python 函数。返回 (源码, 每个核的元信息)。

    **组内不落存储**：组内生产者直接展开成表达式，只有跨组消费者需要的缓冲
    才由该组 `return` 出去——这正是「物化还是重算」这一次决策在 codegen 上的样子。
    """
    graph, groups = plan.graph, plan.groups
    mats = {id(n) for n in materialized_nodes(graph, groups)}
    src, meta = [], []
    for gi, g in enumerate(groups):
        produced = {n.buf for n in g}
        externals = []
        for n in g:
            for d in n.reads:
                if d.buf not in produced and d.buf not in externals:
                    externals.append(d.buf)
        names = {b: "in%d" % i for i, b in enumerate(externals)}
        for n in g:
            names[n.buf] = "v_%s" % n.buf
        params = ["in%d" % i for i in range(len(externals))] or ["_unused"]
        lines = ["def %s%d(%s):" % (prefix, gi, ", ".join(params))]
        for n in g:
            lines.append("    %s = %s" % (names[n.buf], _expr(n, names)))
        outs = [n for n in g if n.writes is not None and id(n) in mats]
        lines.append("    return [%s]" % (", ".join(names[n.buf] for n in outs)
                                          or ""))
        src.append("\n".join(lines))
        meta.append({"index": gi, "nodes": [n.buf for n in g],
                     "externals": externals, "outputs": [n.buf for n in outs]})
    return "\n\n".join(src), meta


def exec_kernels(source, meta, inputs_by_buf, prefix="kern"):
    """`exec` 生成物并在 numpy 上跑一遍，返回 {缓冲名: 数组}。

    组与组按生成顺序串起来跑——**这是真正执行生成出来的 Python 代码**，
    不是把结果从原图里抄一份。
    """
    ns = {"np": np}
    exec(compile(source, "<generated>", "exec"), ns)      # noqa: S102
    env = dict(inputs_by_buf)
    for m in meta:
        fn = ns["%s%d" % (prefix, m["index"])]
        args = [env[b] for b in m["externals"]]
        outs = fn(*args) if args else fn(None)
        for b, v in zip(m["outputs"], outs):
            env[b] = np.asarray(v)
    return env
