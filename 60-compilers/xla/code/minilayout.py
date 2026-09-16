"""迷你 layout assignment 与 memory space assignment。

★ 本模块是**本仓库自己的实现**，不是 XLA。术语（layout constraint / priority /
  mandatory / kCopy / default & alternate memory / prefetch / eviction）借自 XLA
  以便对照，**判据与数据结构都是我们自己简化的**。XLA 的真实行为见 `specs.py`。

两条主线在这个文件里合流：

1. **layout assignment**：约束对象被**创建、排优先级、沿 HLO 图传播**；冲突时高优先级者胜，
   **在需要处插入 `kCopy`**（`specs.LAYOUT_ASSIGNMENT_MECHANISM`）。
   本迷你版把这条实现成一次**自 root 向操作数的反向传播**：
   每个 op 根据自己的**输出**布局，推出它的**操作数**必须是什么布局。

2. **memory space assignment**：**尽量把值塞进 alternate memory**，
   default memory 只是「假定足够大」的兜底；prefetch 是 **default → alt**，
   eviction 是 alt → default（`specs.MSA_DIRECTION` / `specs.MSA_COPY_DIRECTIONS`）。
   ★ 注意方向：**不是**「把大 buffer 放慢存储以腾出快存储」。

另有一件容易混淆的事也放在这里：GPU 上**同名但不同**的 buffer 着色机制
（`MemorySpaceColor` 只有 kDefault / kCollective / kTempBuffer 三个成员），
它发生在 **buffer assignment** 阶段，与上面第 2 条**不是一回事**。
"""

# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


class Layout:
    """`minor_to_major`：从**最次维**到**最主维**的维度编号序列。

    行主序（row-major）的 2D 布局是 `{1,0}`；rank-1 是 `{0}`。
    """

    __slots__ = ("minor_to_major",)

    def __init__(self, minor_to_major):
        self.minor_to_major = tuple(int(d) for d in minor_to_major)

    def __eq__(self, other):
        return (isinstance(other, Layout)
                and self.minor_to_major == other.minor_to_major)

    def __hash__(self):
        return hash(self.minor_to_major)

    def __repr__(self):
        return "{%s}" % ",".join(str(d) for d in self.minor_to_major)

    @staticmethod
    def row_major(rank):
        return Layout(range(rank - 1, -1, -1))

    def is_row_major(self):
        return self.minor_to_major == tuple(range(len(self.minor_to_major) - 1, -1, -1))


class LayoutConstraint:
    """一条布局约束：布局 + 优先级 + 是否强制。"""

    def __init__(self, op_name, layout, priority, mandatory=True, source="propagated"):
        self.op_name = op_name
        self.layout = layout
        self.priority = priority
        self.mandatory = mandatory
        self.source = source

    def __repr__(self):
        return "%s=%s(p=%d%s)" % (self.op_name, self.layout, self.priority,
                                  ",mandatory" if self.mandatory else "")


# ---------------------------------------------------------------------------
# 操作数布局需求：给定 op 的**输出**布局，推出操作数必须是什么布局
# ---------------------------------------------------------------------------


def _compact(seq):
    """把 remap 后的编号压回 0..n-1，保持相对顺序。"""
    order = {d: i for i, d in enumerate(sorted(seq))}
    return tuple(order[d] for d in seq)


def required_operand_layout(op, out_layout, operand_index=0):
    """返回**约束**（Layout 或 None）。None 表示「该操作数无布局要求」。"""
    o = op.opcode
    if o in ("add", "subtract", "multiply", "divide", "maximum", "minimum"):
        # 逐元素：操作数必须与输出同布局
        return out_layout if operand_index == 0 else _rhs_layout(op, out_layout)
    if o == "broadcast":
        sizes = op.attrs.get("sizes", [])
        if operand_index != 0:
            return None
        # 结果的前 len(sizes) 维是广播出来的，操作数没有它们的布局要求
        rest = [d for d in out_layout.minor_to_major if d >= len(sizes)]
        return Layout(_compact([d - len(sizes) for d in rest]))
    if o == "reduce":
        if operand_index != 0:
            return None
        # 归约维应当**最次**（这样归约沿连续内存走）：归约维排在前面，
        # 其余各维按它们在**输出**里的次→主顺序接在后面。
        # ★ 输出的维度编号是「压缩过」的，要先用 remap 还原成操作数的原编号。
        rank = len(op.operands[0].shape.dims)
        red = sorted(op.attrs["dimensions"])
        remap = [d for d in range(rank) if d not in red]
        keep = [remap[d] for d in out_layout.minor_to_major]
        return Layout(tuple(red) + tuple(keep))
    if o == "transpose":
        if operand_index != 0:
            return None
        perm = list(op.attrs["permutation"])
        # 输出第 j 维来自输入第 perm[j] 维
        return Layout([perm[j] for j in out_layout.minor_to_major])
    if o == "reshape":
        return None            # 本迷你版把 reshape 当作无布局要求
    if o == "dot_general":
        return None            # 见 op_output_layout：dot 的布局是**自己钉死**的
    return None


def _rhs_layout(op, out_layout):
    bd = op.attrs.get("broadcast_dimensions", [])
    if not bd:
        return out_layout
    # rhs 的第 i 维映射到结果的 bd[i] 维：抽出这些维并按它们在 out 中的相对顺序排
    sub = [d for d in out_layout.minor_to_major if d in bd]
    return Layout([bd.index(d) for d in sub])


def op_output_layout(op, operand_shapes):
    """某些 op 的**输出布局是自己钉死的**（硬件/算子约束），与消费者无关。

    本迷你版把 `dot_general` 建模成这样的 op：它的输出被钉在
    **`{n-1,...,1,0}`**（最主维最次）。真实的 XLA 里同类约束确实存在
    —— `gpu_compiler.cc` 里有一句注释说 GemmRewriter 把 matmul 的输出布局
    钉在 `{n-1,...,1,0}`（见 `specs.py` 的 GPU_PIPELINE_SOURCE 同文件）。
    """
    if op.opcode == "dot_general":
        rank = len(op.shape.dims)
        return Layout(range(rank - 1, -1, -1))
    return None


# ---------------------------------------------------------------------------
# layout assignment：反向传播 + 冲突插 kCopy
# ---------------------------------------------------------------------------

PIN_PRIORITY = 100          # 算子自身钉死的约束
ROOT_PRIORITY = 50          # 计算 root 的输出布局
PROPAGATED_PRIORITY = 10    # 沿图传播出去的约束


class Copy:
    """一条被判定的 `kCopy`：把某个值从一种布局搬到另一种布局。"""

    def __init__(self, op_name, frm, to, reason):
        self.op_name = op_name
        self.frm = frm
        self.to = to
        self.reason = reason

    def __repr__(self):
        return "kCopy(%s: %s -> %s)" % (self.op_name, self.frm, self.to)


def assign_layouts(module, root_layout=None):
    """返回 `(layouts, copies, notes)`。

    反向遍历（消费者先于生产者），把每个 op 的操作数需求往下传；
    遇到「op 自身钉死的布局」与「消费者要求的布局」不一致时，**记一条 kCopy**。
    """
    if root_layout is None:
        root_layout = Layout.row_major(module.root().shape.rank)
    requested = {}
    pins = {}
    layouts = {}
    copies = []
    notes = []

    for op in reversed(module.ops):
        pinned = op_output_layout(op, [o.shape for o in op.operands])
        if pinned is not None:
            pins[op.name] = pinned

    for op in reversed(module.ops):
        if op.name in pins:
            out = pins[op.name]
            req = requested.get(op.name)
            if req is not None and req != out:
                copies.append(Copy(op.name, req, out,
                                   "op 自身钉死在 %s，消费者要 %s" % (out, req)))
        else:
            req = requested.get(op.name)
            out = req if req is not None else (
                root_layout if op is module.root()
                else Layout.row_major(op.shape.rank))
        layouts[op.name] = out

        for i, operand in enumerate(op.operands):
            need = required_operand_layout(op, out, i)
            if need is None:
                continue
            prev = requested.get(operand.name)
            if prev is not None and prev != need:
                notes.append(
                    "值 %s 被多个消费者要求不同布局：%s vs %s（本迷你版取后者）"
                    % (operand.name, prev, need))
            requested[operand.name] = need

    return layouts, copies, notes


def check_layouts(module, layouts, copies):
    """独立校验器：逐条边检查「布局一致 或 有一条 kCopy」。

    ★ 这条路径**不看 assign_layouts 的中间状态**，只拿最终的 layouts 表 + copies 列表
      重新走一遍图——所以它能抓住 assign_layouts 自己算错的情形（A 类）。
    """
    copy_at = {c.op_name for c in copies}
    bad = []
    for op in module.ops:
        if op.opcode in ("parameter", "constant"):
            continue
        for i, operand in enumerate(op.operands):
            need = required_operand_layout(op, layouts[op.name], i)
            if need is None:
                continue
            if layouts[operand.name] != need and operand.name not in copy_at:
                bad.append("%s 需要操作数 %s 是 %s，实际 %s，且没有 kCopy"
                           % (op.name, operand.name, need, layouts[operand.name]))
    return bad


# ---------------------------------------------------------------------------
# ★ GPU 上**同名但不同**的 buffer 着色（buffer assignment 阶段）
# ---------------------------------------------------------------------------


class MemorySpaceColor:
    kDefault = 0
    kCollective = 1
    kTempBuffer = 2


COLOR_NAMES = {0: "kDefault", 1: "kCollective", 2: "kTempBuffer"}


def color_value(op):
    """迷你 colorer。对应 `specs.GPU_MSA_COLOR_PURPOSE` 列的四类。

    ★ 这里返回的**不是** MSA 的 default/alternate —— 是**另一套**同样叫
      "memory space" 的东西。两者正交：一个值可以被着色成 kCollective，
      同时被 MSA 放进 **default** memory space。
    """
    if op.attrs.get("collective"):
        return MemorySpaceColor.kCollective
    if op.attrs.get("temp_buffer"):
        return MemorySpaceColor.kTempBuffer
    return MemorySpaceColor.kDefault


# ---------------------------------------------------------------------------
# memory space assignment：**尽量塞进 alternate**
# ---------------------------------------------------------------------------

DEFAULT_LIVE_RANGE_LIMIT = 4     # 超过这个长度的活跃区间不适合作 no-copy 的 alt 分配


class ValueInfo:
    def __init__(self, name, def_index, last_use, nbytes, reads=1):
        self.name = name
        self.def_index = def_index
        self.last_use = last_use
        self.nbytes = nbytes
        self.reads = reads

    @property
    def live_range(self):
        return self.last_use - self.def_index

    def __repr__(self):
        return "%s[def=%d,use=%d,%dB]" % (self.name, self.def_index,
                                          self.last_use, self.nbytes)


class MsaResult:
    def __init__(self):
        self.alt = []          # 放进 alternate 的值名
        self.default = []      # 落在 default 的值名
        self.prefetches = []   # (name, nbytes)   default -> alt
        self.evictions = []    # (name, nbytes)   alt -> default
        self.alt_bytes = 0
        self.default_bytes = 0

    def report(self):
        return ("alt=%d 个(%dB) default=%d 个(%dB) prefetch=%d evict=%d"
                % (len(self.alt), self.alt_bytes, len(self.default),
                   self.default_bytes, len(self.prefetches), len(self.evictions)))


def memory_space_assignment(values, alt_capacity,
                            live_range_limit=DEFAULT_LIVE_RANGE_LIMIT):
    """**贪心地尽量把值放进 alternate memory**。

    三条口径来自 `specs.MSA_DIRECTION` / `specs.MSA_COPY_DIRECTIONS`：
      · 目标是「尽量多塞进 alt」，**不是**「把大的赶去慢存储」；
      · default memory「假定足够大」，所以**没有容量上限**，塞不进 alt 的一律落 default；
      · alt → 需要 prefetch（default → alt）；用不完的 → eviction（alt → default）。

    本迷你版的两条自有规则（**不是 XLA 的规则**）：
      ① 活跃区间短于 `live_range_limit` 的值才做 no-copy 的 alt 分配；
      ② 一次失败**不终止**扫描——后面更小的值仍可以进 alt（这就是「as many as possible」）。
    """
    res = MsaResult()
    used = 0
    for v in values:
        if v.live_range < live_range_limit and used + v.nbytes <= alt_capacity:
            res.alt.append(v.name)
            res.prefetches.append((v.name, v.nbytes))
            used += v.nbytes
        else:
            res.default.append(v.name)
    res.alt_bytes = used
    res.default_bytes = sum(v.nbytes for v in values if v.name in set(res.default))
    # eviction：进了 alt、但活跃区间偏长的那些，最后一次使用后要搬回 default
    by_name = {v.name: v for v in values}
    for name in res.alt:
        v = by_name[name]
        if v.live_range >= live_range_limit - 1 and v.live_range > 0:
            res.evictions.append((name, v.nbytes))
    return res


# ---------------------------------------------------------------------------
# 两条 pipeline 的 pass 列表（本仓库迷你版；GPU 那条**不含** MSA pass）
# ---------------------------------------------------------------------------

# 这两条列表是**照着 `specs.GPU_PIPELINE_ORDER`（gpu_compiler.cc 的 OptimizeHloModule）
# 的结构写的迷你版**。它们描述的是本仓库的迷你 pipeline，**不是 XLA 的 pass 列表**。
CPU_PIPELINE = ["layout_assignment", "fusion", "memory_space_assignment",
                "buffer_assignment"]
GPU_PIPELINE = ["layout_assignment", "layout_normalization", "fusion",
                "host_offloader", "buffer_assignment",
                "gpu_memory_space_coloring"]


def pipeline_passes(which):
    return list(GPU_PIPELINE if which == "gpu" else CPU_PIPELINE)


def host_offload_custom_calls():
    """host 卸载走的是 custom call，target 是 **PascalCase**（`specs.MOVE_TO_HOST_CUSTOM_CALL`）。"""
    return ["MoveToHost", "MoveToDevice"]
