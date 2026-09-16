"""迷你融合：两套 fusion kind 枚举 + 「预算 + 合法性 + 代价模型」三层判据。

★ 本模块是**本仓库自己的实现**，不是 XLA。枚举成员名与拒绝理由字符串借自 XLA
  （来源逐条见 `specs.py`），**但判据的形式与阈值都是我们自己定的**。
  真实 XLA 的行为**不作为任何程序的断言对象**（§0.3 规矩 4）。

两件必须分清楚的事：

1. **`FusionKind` 与 `EmitterFusionKind` 不是一回事。**
   前者是 `HloInstruction` 上的 4 个成员（kLoop / kInput / kOutput / kCustom），
   描述「fusion 的 primary node 是不是 root」；
   后者是 GPU emitter 侧的 9 个成员，描述「这个 fusion 由哪个 emitter 出码」。
   **`EmitterFusionKind` 多出来的那些成员（kTriton / kCuDnn / kSort …）在
   `FusionKind` 里根本不存在。**

2. **multi-output fusion 不是 `FusionKind` 的一个成员。**
   XLA 自己的 TODO 逐字说这套 scheme 对它不适用：root 是 `kTuple`
   （`specs.FUSION_KIND_TODO`）。本迷你版把 multi-output 建模成
   「组内有**多个 root**，且它们由一个 tuple 汇聚」——**一个组的形态，不是一个 kind**。

三层判据的形态对应真实 XLA 里 `ProducerCandidateIsFusible` 的调用次序
（`specs.GPU_FUSION_LAYERED_EVIDENCE`）：
    层 1 预算      —— `FusionFitsInBudget` 类
    层 2 合法性    —— 结构性约束（kind 匹配、至少一个 kFusion、无环 …）
    层 3 代价模型  —— `GpuPerformanceModel` 类，可**否决**一个已经合法的融合
"""

from enum import Enum

from minihlo import eval_op

# ---------------------------------------------------------------------------
# ★ 两套枚举：名字像，但**不是一回事**
# ---------------------------------------------------------------------------


class FusionKind(Enum):
    """照抄 `HloInstruction::FusionKind` 的四个成员（`specs.HLO_FUSION_KIND_ENUM`）。

    语义（`specs.FUSION_KIND_KINPUT` / `_KOUTPUT` / `_KLOOP` / `_KCUSTOM`）：
      kInput  —— primary node **是** root
      kOutput —— primary node **不是** root（要求某个 operand buffer 能与 output buffer alias）
      kLoop   —— primary node 是 root，但规定了特定实现：一次生成 root 的**一个元素**
      kCustom —— 后端特有、不符合以上模式的

    → **区分 kInput 与 kOutput 的是「primary node 是不是 root」，与「多输出」无关。**
    """

    kLoop = "kLoop"
    kInput = "kInput"
    kOutput = "kOutput"
    kCustom = "kCustom"


class EmitterFusionKind(Enum):
    """照抄 GPU emitter 侧的 `EmitterFusionKind` 的**九个**成员
    （`specs.EMITTER_FUSION_KIND_ENUM`）。

    ★ 它不是 `FusionKind` 的扩展：`kTriton` / `kCuDnn` / `kSort` / `kReduction` …
      这些成员在 `FusionKind` 里**并不存在**。
    """

    kLoop = "kLoop"
    kCustomFusion = "kCustomFusion"
    kTriton = "kTriton"
    kReduction = "kReduction"
    kTranspose = "kTranspose"
    kConcatenate = "kConcatenate"
    kScatter = "kScatter"
    kCuDnn = "kCuDnn"
    kSort = "kSort"


# ---------------------------------------------------------------------------
# 融合组与判定
# ---------------------------------------------------------------------------

ELEMENTWISE = ("add", "subtract", "multiply", "divide", "maximum", "minimum")


class Group:
    """一个候选融合组。

    `members`      成员 op 名，按拓扑序
    `kind`         这个组的 `FusionKind`
    `multi_output` 是否是 multi-output 融合（root 由一个 tuple 汇聚）
    `already_fusion` 组内是否**已经**有 op 是 fusion（multi-output fusion 的硬约束）
    `member_kinds` 组内那些「已经是 fusion」的成员各自的 kind（用于 kind 匹配检查）
    """

    def __init__(self, members, kind=FusionKind.kLoop, multi_output=False,
                 already_fusion=False, member_kinds=()):
        self.members = list(members)
        self.kind = kind
        self.multi_output = multi_output
        self.already_fusion = already_fusion
        self.member_kinds = list(member_kinds)

    def roots(self, module):
        """组内**没有被组内其它成员消费**的成员 = 这个组的 root。"""
        ms = set(self.members)
        return [n for n in self.members
                if not any(u.name in ms for u in module.by_name(n).users)]

    def external_operands(self, module):
        ms = set(self.members)
        out = []
        for n in self.members:
            for o in module.by_name(n).operands:
                if o.name not in ms and o.name not in out:
                    out.append(o.name)
        return out

    def __repr__(self):
        return "Group(%s, %s%s)" % ("+".join(self.members), self.kind.value,
                                    ", multi-output" if self.multi_output else "")


class FusionDecision:
    def __init__(self, allowed, layer, reason):
        self.allowed = allowed
        self.layer = layer
        self.reason = reason

    def __repr__(self):
        if self.allowed:
            return "Allow（%s 层）：%s" % (self.layer, self.reason)
        return "Forbid（%s 层）：%s" % (self.layer, self.reason)


# ---------------------------------------------------------------------------
# 层 1：预算
# ---------------------------------------------------------------------------

BUDGET_LIMIT = 96        # = specs.MAX_OPERANDS_AND_OUTPUTS_PER_FUSION


def check_budget(group, module, budget=BUDGET_LIMIT):
    """operands + outputs 的数量上限 —— 对应 `FusionFitsInParameterLimit`
    （`specs.FUSION_FITS_IN_PARAMETER_LIMIT`）。"""
    ext = group.external_operands(module)
    outs = group.roots(module)
    n = len(ext) + len(outs)
    if n > budget:
        return FusionDecision(False, "预算",
                              "operands(%d)+outputs(%d)=%d 超出上限 %d"
                              % (len(ext), len(outs), n, budget))
    return FusionDecision(True, "预算", "operands+outputs=%d <= %d" % (n, budget))


# ---------------------------------------------------------------------------
# 层 2：合法性（结构性约束）
# ---------------------------------------------------------------------------


def check_legality(group, module):
    """结构性约束。两条带引号的理由**逐字来自 XLA**（见 `specs`），其余是本迷你版的。"""
    ms = set(group.members)
    if len(ms) != len(group.members):
        return FusionDecision(False, "合法性", "组成员有重复")

    for n in group.members:
        if module.by_name(n).opcode == "parameter":
            return FusionDecision(False, "合法性", "parameter 不可融合（它不是计算）")

    # 拓扑序：每个成员的操作数要么在组内更早出现，要么在组外
    seen = set()
    for n in group.members:
        for o in module.by_name(n).operands:
            if o.name in ms and o.name not in seen:
                return FusionDecision(False, "合法性",
                                      "%s 的操作数 %s 不在它之前（会成环）" % (n, o.name))
        seen.add(n)

    roots = group.roots(module)
    if len(roots) > 1:
        # 一手约束逐字：multi_output_fusion.h 的
        #   "Sibling fusion (1) requires at least one op to be a kFusion."
        if not group.already_fusion:
            return FusionDecision(
                False, "合法性",
                "sibling fusion requires at least one op to be a kFusion")
        # 一手约束逐字：
        #   "Sibling fusion (1) does not fuse kInput fusions with kLoop fusions,
        #    i.e. the fusion kinds must match."
        if FusionKind.kInput in group.member_kinds and \
                FusionKind.kLoop in group.member_kinds:
            return FusionDecision(
                False, "合法性",
                "sibling fusion 不融合 kInput 与 kLoop，i.e. the fusion kinds must match")

    # 归约 × 归约：不是通用能力（本迷你版直接拒）
    reds = [n for n in group.members if module.by_name(n).opcode == "reduce"]
    if len(reds) > 1:
        return FusionDecision(False, "合法性",
                              "归约↔归约不是通用能力（本迷你版只放行单归约组）")

    # multi-output 的 root 必须由 tuple 汇聚，故至少要两个 root
    if group.multi_output and len(roots) < 2:
        return FusionDecision(False, "合法性",
                              "声明为 multi-output 但只有一个 root（tuple 没有意义）")

    return FusionDecision(True, "合法性", "结构检查通过（%d 个 root）" % len(roots))


# ---------------------------------------------------------------------------
# 层 3：代价模型（可以否决一个**合法**的融合）
# ---------------------------------------------------------------------------

CONVERSIONS_PER_TRANSPOSING_OP = 2   # 物理转置型算子：输入侧 + 输出侧各一条转换
IR_SIZE_LIMIT = 24                   # 融合体内部 op 数上限


def value_bytes(module, name):
    for p in module.params:
        if p.name == name:
            return p.shape.nbytes()
    return module.by_name(name).shape.nbytes()


def physically_transposing(op):
    """是否是「改动物理布局」的 op。

    真实 XLA 的对应物是 `gpu_fusible.h` 的 `IsPhysicallyTransposing`，注释逐字：
    「Whether the op transposes the physical data layout. Fusing such ops may lead to
    uncoalesced data access and may thus not be beneficial.」

    ★ 本迷你版把 `transpose` 与 `dot_general` 都算作这一类——**这是我们自己定的**，
      真实 XLA 的这个函数作用于它自己的 opcode 集合，两者不通用。
    """
    return op.opcode in ("transpose", "dot_general")


def layout_conversions(group, module, layouts, strict=True):
    """融合体内需要的**布局转换**条数。

    融合体只有**一个工作布局**（root 的布局）—— 因为
    「we don't emit layouts inside of fusion」（`specs.TF_COMMIT_358FD36D_MESSAGE`）。
    于是：

      · `strict=True` 时，任何**钉死布局 ≠ 工作布局**的成员，结果侧记 1 条转换；
      · `strict=True` 时，物理转置型算子再记 `CONVERSIONS_PER_TRANSPOSING_OP` 条
        —— 它**横跨在布局边界上**：操作数要按工作布局读进来一次，结果要按工作布局
        写出去一次。这条是本迷你版的规则（真实 XLA 的 `IsPhysicallyTransposing`
        只回答「是否改动物理布局」，不含条数）；
      · `strict=False` 是「布局指派阶段就已知道这台 op 会被融合」的反事实：
        内部值一律按工作布局生成，**不需要任何转换**。
    """
    if not strict:
        return 0
    work = layouts.get(module.root().name)
    n = 0
    for name in group.members:
        op = module.by_name(name)
        if work is not None and layouts.get(name) != work:
            n += 1
        if physically_transposing(op):
            n += CONVERSIONS_PER_TRANSPOSING_OP
    return n


def cost_model(group, module, layouts, strict_layout=True, ir_limit=IR_SIZE_LIMIT):
    """按**访存字节流量**估代价；**融合更慢就否决**。

    这是 XLA 里 `GpuPerformanceModel` 那一层的迷你替身，对应两条逐字的拒绝理由
    （`specs.GPU_FUSION_REJECT_TOO_LARGE_IR` / `specs.GPU_FUSION_REJECT_SLOWER_IF_FUSED`）。

    两项代价分开计：
      · **流量** —— 只有落内存的往返才算。融合省掉的正是中间结果那一趟写 + 一趟读。
        布局转换发生在片内，**不计入流量**。
      · **重算** —— 组内成员若还有**组外**的消费者，融合体内就得把它再算一遍
        （不物化就重算）。这一项计入流量，因为它的操作数还得从内存里再读一次。
      · **IR 规模** —— 成员数 + 转换条数，超过 `ir_limit` 直接拒。
    """
    conv = layout_conversions(group, module, layouts, strict=strict_layout)
    ir_ops = len(group.members) + conv
    if ir_ops > ir_limit:
        return FusionDecision(False, "代价模型",
                              "will generate too large IR（%d > %d）" % (ir_ops, ir_limit))

    # 未融合：每个成员各自「读操作数 + 写结果」
    unfused = 0
    for name in group.members:
        op = module.by_name(name)
        unfused += sum(o.shape.nbytes() for o in op.operands) + op.shape.nbytes()

    # 融合：读外部操作数 + 写 root；组内往返不再落内存
    fused = sum(value_bytes(module, n) for n in group.external_operands(module))
    fused += sum(module.by_name(r).shape.nbytes() for r in group.roots(module))

    # 重算项：组内非 root 的成员，若还有组外的消费者
    roots = set(group.roots(module))
    dup = 0
    for name in group.members:
        if name in roots:
            continue
        op = module.by_name(name)
        if any(u.name not in set(group.members) for u in op.users):
            dup += sum(o.shape.nbytes() for o in op.operands) + op.shape.nbytes()
    fused += dup

    if fused > unfused:
        return FusionDecision(
            False, "代价模型",
            "will execute slower if fused（fused=%d > unfused=%d，conversions=%d，"
            "重算 %dB）" % (fused, unfused, conv, dup))
    return FusionDecision(True, "代价模型",
                          "fused=%d <= unfused=%d（conversions=%d，重算 %dB）"
                          % (fused, unfused, conv, dup))


def perfect_cost_model(group, module, layouts, strict_layout=True, **kw):
    """**完美的代价模型**：无条件放行。

    ★ 它存在的唯一目的是**反证**（§0.4 第 5 条）：把代价模型换成它之后，
      「代价模型否决了这次融合」那条断言必须**不再成立**——
      否则那条断言检验的就不是代价模型。**它不参与任何正常判定。**
    """
    return FusionDecision(True, "代价模型", "完美模型：无条件放行")


# ---------------------------------------------------------------------------
# 三层串起来
# ---------------------------------------------------------------------------


def try_fuse(group, module, layouts, strict_layout=True, cost_fn=cost_model, **kw):
    """**顺序是硬的**：预算 → 合法性 → 代价模型。返回 (decision, trace)。

    `cost_fn` 只用来做反证（换成 `perfect_cost_model`），正常调用不用传。
    """
    trace = []
    for layer_fn in (lambda: check_budget(group, module),
                     lambda: check_legality(group, module),
                     lambda: cost_fn(group, module, layouts,
                                     strict_layout=strict_layout, **kw)):
        d = layer_fn()
        trace.append(d)
        if not d.allowed:
            return d, trace
    return trace[-1], trace


# ---------------------------------------------------------------------------
# 执行：未融合 vs 分组执行（两条**不同**的代码路径）
# ---------------------------------------------------------------------------


class Plan:
    """一个融合方案：把模块的 op 划分成若干组（组按拓扑序排列）。"""

    def __init__(self, groups):
        self.groups = list(groups)

    def group_of(self, name):
        for g in self.groups:
            if name in g.members:
                return g
        return None

    def check_covers(self, module):
        """每个 op 恰好属于一个组。返回 (missing, dup)。"""
        seen = {}
        for g in self.groups:
            for n in g.members:
                seen[n] = seen.get(n, 0) + 1
        missing = [o.name for o in module.ops if o.name not in seen]
        dup = [n for n, c in seen.items() if c > 1]
        return missing, dup


def execute_plan(module, plan, feed):
    """**按组**执行：逐组、组内按拓扑序求值。与 `minihlo.execute` 是两条路径。"""
    env = {p.name: eval_op(p, {}, feed) for p in module.params}
    for g in plan.groups:
        for name in g.members:
            op = module.by_name(name)
            env[name] = eval_op(op, env, feed)
    return env


def ops_after_fusion(module, plan, layouts, strict_layout=True):
    """融合后的**算子数**（把插入的布局转换也算成算子）。

    融合掉一个 k 个成员的组：先 -k，再加回 1 个 fusion 包裹，再补上组内需要的转换。
    """
    n = len(module.ops)
    for g in plan.groups:
        if len(g.members) > 1:
            n += -len(g.members) + 1 + layout_conversions(
                g, module, layouts, strict=strict_layout)
    return n
