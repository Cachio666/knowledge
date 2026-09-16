"""迷你 TIR：计算与调度分离 + 物化代价模型 + 解释器 + 三类断言框架。

★ 本模块是**本仓库的迷你实现**，不是 TVM。它借用 TVM 的术语（`prim_func` / block /
  `compute_at` / `compute_inline` / pipeline / tile 原语）以便与真实系统对照，
  但**没有任何一行是 TVM 的代码**，也**跑不出 TVM 的输出**。
  由它产生的每一个数字都是「本机实测值」，只能支撑关于**机制**的结论，
  不能支撑任何关于真实 TVM 的结论（CONTRIBUTING §2、计划 §0.3）。

★ 「物化」的口径（计划 §0.8 已写死，本模块照此实现）：
  **物化 = 为中间结果分配独立的存储**——在寄存器、共享内存还是全局内存里都算物化；
  **重算 = 根本不分配存储**，把计算直接替换进消费者的表达式。

★ 迷你 Schedule 只有**两态**：`compute_at`（物化，缩到指定循环内）与
  `compute_inline`（不物化）。**没有 `compute_root`**——默认就在根位置。
  这两态的对应关系见 code/specs.py（钉在 v0.26.0）。

被四篇文档共同使用：
  01_compute_schedule.py     计算与调度分离；默认全物化
  02_schedule_primitives.py  原语与物化位置；三态对照
  03_two_tracks.py           两条并列的 TIR 轨道与 tile 原语
  04_scheduling_paths.py     规则式与搜索式两条调度路径
"""

import sys

import numpy as np

ESIZE = 4                      # f32


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class MiniTIRError(Exception):
    """本仓库迷你实现的错误基类。"""


class ScheduleFailure(MiniTIRError):
    """调度原语的前置条件不满足 —— 必须**拒绝**，不得静默产出错误计划。"""


class InfeasibleSchedule(ScheduleFailure):
    """按代价模型判定该调度不可行（例如峰值存储超过可用容量）。"""


class NativeTrackError(MiniTIRError):
    """native 轨（TIRx 式）的错误。"""


class ScopeMismatch(NativeTrackError):
    """tile 原语的协作 scope 与调用上下文不匹配。"""


class PipelineError(MiniTIRError):
    """pipeline / PassContext 相关错误。"""


# ---------------------------------------------------------------------------
# 轴 / 缓冲 / 计算阶段 / prim_func
# ---------------------------------------------------------------------------


class Axis:
    """一条循环轴。"""

    def __init__(self, name, extent):
        self.name = name
        self.extent = int(extent)

    def __repr__(self):
        return "Axis(%s, %d)" % (self.name, self.extent)


class Buffer:
    """一块缓冲。`role` 只有三种：input / intermediate / output。"""

    def __init__(self, name, axes, role="intermediate"):
        self.name = name
        self.axes = list(axes)
        self.role = role

    @property
    def shape(self):
        return tuple(a.extent for a in self.axes)

    @property
    def size(self):
        n = 1
        for a in self.axes:
            n *= a.extent
        return n

    @property
    def nbytes(self):
        return self.size * ESIZE

    def __repr__(self):
        return "Buffer(%s, %s, %s)" % (self.name, self.shape, self.role)


class Stage:
    """一个计算阶段 —— 对应 s_tir 里一个可调度的 block。

    `out[idx] = fn(*ins)`；fn 必须是**形状无关**的逐元素语义，
    这样同一次计算既能整块算、也能按切片算。
    """

    def __init__(self, name, axes, ins, out, fn):
        self.name = name
        self.axes = list(axes)
        self.ins = list(ins)
        self.out = out
        self.fn = fn

    def __repr__(self):
        return "Stage(%s)" % self.name


class PrimFunc:
    """一个计算：阶段的有向无环图 + 入参 + 输出缓冲。计算与调度在这里是分开的。"""

    def __init__(self, name, inputs, stages, s_tir=True):
        self.name = name
        self.inputs = list(inputs)
        self.stages = list(stages)
        self.s_tir = s_tir
        self._check()

    def _check(self):
        names = [st.name for st in self.stages]
        if len(set(names)) != len(names):
            raise MiniTIRError("stage 名字重复：%s" % names)
        if not any(st.out.role == "output" for st in self.stages):
            raise MiniTIRError("prim_func 必须至少有一个输出 stage")

    def stage(self, name):
        for st in self.stages:
            if st.name == name:
                return st
        raise MiniTIRError("没有这个 stage：%s" % name)

    def producer_of(self, buf):
        for st in self.stages:
            if st.out is buf:
                return st
        return None

    def consumers(self, stage):
        return [st for st in self.stages if stage.out in st.ins]

    def producers(self, stage):
        return [st for st in self.stages if st.out in stage.ins]

    def sinks(self):
        produced = {st.out.name for st in self.stages}
        used = set()
        for st in self.stages:
            for b in st.ins:
                used.add(b.name)
        return [st for st in self.stages if st.out.name in produced and
                st.out.name not in used]

    def topo(self):
        """拓扑序（生产者在前）。"""
        order, seen = [], set()

        def visit(st):
            if st.name in seen:
                return
            seen.add(st.name)
            for p in self.producers(st):
                visit(p)
            order.append(st)

        for st in self.stages:
            visit(st)
        return order

    def descendants(self, stage):
        """沿 consumer 边走到底得到的阶段集合（不含 stage 自己）。"""
        out, stack = set(), list(self.consumers(stage))
        while stack:
            st = stack.pop()
            if st.name in out:
                continue
            out.add(st.name)
            stack.extend(self.consumers(st))
        return out


# ---------------------------------------------------------------------------
# 迷你 Schedule —— 只有两态
# ---------------------------------------------------------------------------


class LoopRef:
    """一个循环：某个 stage 的某条轴。"""

    def __init__(self, stage, axis):
        self.stage = stage
        self.axis = axis

    def __repr__(self):
        return "LoopRef(%s.%s)" % (self.stage.name, self.axis.name)


class Schedule:
    """迷你版 s_tir.Schedule —— **只有两态**：compute_at / compute_inline。

    ★ 没有 `compute_root`：本迷你实现不提供把 block「放回根位置」的方法，
      因为默认就在根位置。默认位置来自构造时对每个 stage 的初始化。

    签名照抄 v0.26.0 的 Python 层（见 code/specs.py 的 SCHEDULE_API），
    这样文档里讲参数含义时能一一对上。
    """

    def __init__(self, func):
        if not isinstance(func, PrimFunc):
            raise ScheduleFailure("Schedule 作用在 prim_func 上")
        if not func.s_tir:
            raise ScheduleFailure(
                "这个 prim_func 是 native 轨（s_tir=False）：它没有可调度的 block，"
                "调度原语无从下手")
        self.func = func
        # 默认：每个 stage 都在根位置（**物化**）
        self._place = {st.name: ("root",) for st in func.stages}

    # -- 取引用 ------------------------------------------------------------

    def get_sblock(self, name):
        return self.func.stage(name)

    def get_loops(self, stage):
        if isinstance(stage, str):
            stage = self.func.stage(stage)
        return [LoopRef(stage, a) for a in stage.axes]

    def get_consumers(self, stage):
        if isinstance(stage, str):
            stage = self.func.stage(stage)
        return self.func.consumers(stage)

    # -- 块位置 ------------------------------------------------------------

    def compute_at(self, block, loop, preserve_unit_loops=False, index=-1):
        """物化到 `loop` 内：**仍然分配独立存储**，只是作用域缩到该循环一次迭代。"""
        st = self._as_stage(block)
        if st.out.role == "output":
            raise ScheduleFailure("输出 stage 不能被搬动：%s" % st.name)
        self._require_sink_anchor(loop, st)
        missing = [c.name for c in self.func.consumers(st)
                   if not self._is_under(c, loop)]
        if missing:
            raise ScheduleFailure(
                "compute_at(%s, %s) 违反前置条件：**该 block 的所有 consumer 都必须在"
                "目标 loop 下**，但 %s 不在（来源：s_tir.Schedule.compute_at 的 docstring，"
                "见 specs.COMPUTE_AT_PRECONDITION）" % (st.name, loop, missing))
        self._place[st.name] = ("at", loop.stage.name, loop.axis.name)

    def _is_under(self, stage, loop):
        """`stage` 现在是不是位于 `loop` 的循环体内。

        ★ 判据看的是**当前**的放置状态，不是数据依赖图上的可达性——
          这让调度带上**过程式**的次序依赖：先搬 s5 才能搬 s4，再搬 s3……
          反过来做会被前置条件拒绝。真实 s_tir 的调度同样是过程式的、
          **不幂等**（official 原文见 specs.PROCEDURAL_QUOTE），本迷你实现复现的
          正是这一条。
        """
        if stage.name == loop.stage.name:
            return True
        p = self._place[stage.name]
        return p[0] == "at" and p[1] == loop.stage.name and p[2] == loop.axis.name

    def _require_sink_anchor(self, loop, st):
        """★ 本迷你实现的**支持范围**（真实 s_tir 没有这条限制）。

        解释器只在**汇（输出）stage** 的循环里展开切片迭代；把 block 挂到中间 stage 的
        循环上，本迷你实现无法执行（也不做静默的降级）。这条限制写进文档。
        """
        if loop.stage not in self.func.sinks():
            raise ScheduleFailure(
                "本迷你实现只支持把 block 物化到**汇（输出）stage** 的循环内；"
                "%s 不是汇 stage（真实 s_tir 没有这条限制，这是本仓库的简化）"
                % loop.stage.name)

    def reverse_compute_at(self, block, loop, preserve_unit_loops=False, index=-1):
        """按**生产者**方向定位（TE 时代没有这条原语）。"""
        st = self._as_stage(block)
        if st.out.role == "output":
            raise ScheduleFailure("输出 stage 不能被搬动：%s" % st.name)
        self._require_sink_anchor(loop, st)
        missing = [p.name for p in self.func.producers(st)
                   if p.name != loop.stage.name and
                   p.name not in self.func.descendants(loop.stage)]
        if missing:
            raise ScheduleFailure(
                "reverse_compute_at(%s, %s) 违反前置条件：**该 block 的所有 producer "
                "都必须在目标 loop 下**，但 %s 不在（见 specs.REVERSE_COMPUTE_AT_PRECONDITION）"
                % (st.name, loop, missing))
        self._place[st.name] = ("at", loop.stage.name, loop.axis.name)

    def compute_inline(self, block):
        """不物化：把计算直接替换进消费者（重算）。"""
        st = self._as_stage(block)
        if st.out.role == "output":
            raise ScheduleFailure("输出 stage 不能被 inline：%s" % st.name)
        self._place[st.name] = ("inline",)

    def reverse_compute_inline(self, block):
        """反向 inline（TE 时代没有这条原语）。"""
        st = self._as_stage(block)
        if st.out.role == "output":
            raise ScheduleFailure("输出 stage 不能被 inline：%s" % st.name)
        self._place[st.name] = ("inline",)

    # -- 内部 --------------------------------------------------------------

    def _as_stage(self, block):
        if isinstance(block, str):
            return self.func.stage(block)
        if isinstance(block, Stage):
            return block
        raise ScheduleFailure("不是 block：%r" % (block,))

    def placement(self, stage):
        if isinstance(stage, str):
            name = stage
        else:
            name = stage.name
        return self._place[name]

    def has_compute_root(self):
        """本迷你实现**没有** compute_root（对照 v0.26.0 的 `s_tir.Schedule`）。"""
        return False


# ---------------------------------------------------------------------------
# 代价模型（解析式）
# ---------------------------------------------------------------------------


class Plan:
    """一份调度展开后的**解析式**代价模型。

    口径（写死在文档 §1，断言就断言这些整数）：

      · `n_buffers`  中间 buffer 的个数 = 被物化的中间 stage 数
      · `peak_elems` 峰值**中间**存储占用（元素数）：
                     根位置的全部 + 附着循环内各切片，最坏情况同时活跃
      · `writes` / `reads`：中间 buffer 的元素写入/读取次数
      · `evals`      各 stage 的 fn 被求值的**元素**次数（重算的度量）
      · `times`      每个 stage 每元素被求值的次数（物化 = 1，inline = 消费者数）
    """

    def __init__(self, func, schedule):
        self.func = func
        self.schedule = schedule
        self.topo = func.topo()
        self.sinks = func.sinks()
        self._compute()

    def _compute(self):
        f, sch = self.func, self.schedule
        self.place = {st.name: sch.placement(st) for st in f.stages}
        self.materialized = [st for st in f.stages
                             if self.place[st.name] != ("inline",)]
        self.intermediates = [st for st in self.materialized
                              if st.out.role == "intermediate"]

        # 每个物化 stage 的存储足迹（元素数）
        # ★ 峰值只统计**中间结果**：输出 buffer 是函数的结果，本来就要存在，
        #   把它算进「中间存储」会让默认与 compute_at 的比较失去意义。
        self.footprint = {}
        root_elems = 0
        at_elems = 0
        for st in self.materialized:
            p = self.place[st.name]
            if p == ("root",):
                n = st.out.size
                if st.out.role == "intermediate":
                    root_elems += n
            else:
                n = st.out.size // st.axes[0].extent      # 切片：只覆盖一条轴
                if st.out.role == "intermediate":
                    at_elems += n
            self.footprint[st.name] = n
        self.root_elems = root_elems
        self.at_elems = at_elems
        self.peak_elems = root_elems + at_elems

        self.n_buffers = len(self.intermediates)

        # 流量：物化的中间结果写一次、被每个消费者读一次
        self.writes = sum(st.out.size for st in self.intermediates)
        self.reads = 0
        for st in self.intermediates:
            self.reads += st.out.size * len(f.consumers(st))

        # 求值次数：materialized -> 1；inline -> 消费者次数之和
        memo = {}

        def times(st):
            if st.name in memo:
                return memo[st.name]
            p = self.place[st.name]
            if p != ("inline",):
                memo[st.name] = 1
                return 1
            n = sum(times(c) for c in f.consumers(st))
            memo[st.name] = n
            return n

        self.times = {st.name: times(st) for st in f.stages}
        self.evals = sum(self.times[st.name] * st.out.size for st in f.stages)

    # -- 查询 --------------------------------------------------------------

    def is_inline(self, stage):
        return self.place[stage.name] == ("inline",)

    def attached_to(self, sink, axis_name):
        """附着在某个 sink 的某条轴上的 stage，拓扑序。"""
        out = [st for st in self.topo
               if self.place[st.name][0] == "at"
               and self.place[st.name][1] == sink.name
               and self.place[st.name][2] == axis_name]
        return out

    def slice_axis(self, sink):
        """sink 上被 `at` 占用的轴；没有则返回 None。"""
        for st in self.topo:
            p = self.place[st.name]
            if p[0] == "at" and p[1] == sink.name:
                for a in sink.axes:
                    if a.name == p[2]:
                        return a
        return None

    def summary_rows(self):
        return [
            ("中间 buffer 个数", self.n_buffers),
            ("峰值中间存储（元素）", self.peak_elems),
            ("峰值中间存储（字节）", self.peak_elems * ESIZE),
            ("中间写入（元素）", self.writes),
            ("中间读取（元素）", self.reads),
            ("中间流量（元素）", self.writes + self.reads),
            ("全 stage 求值（元素）", self.evals),
        ]

    def check_capacity(self, cap_bytes):
        """按代价模型判定存储是否够用；不够就抛 InfeasibleSchedule。"""
        if self.peak_elems * ESIZE > cap_bytes:
            raise InfeasibleSchedule(
                "峰值中间存储 %d 字节 > 可用容量 %d 字节"
                % (self.peak_elems * ESIZE, cap_bytes))


# ---------------------------------------------------------------------------
# 解释器 —— 真的执行，并**观察**记号
# ---------------------------------------------------------------------------


class Interp:
    """执行一份 Plan，同时记录**实测**的写入/读取/求值/存储计数。

    实测计数与 Plan 的解析值是两条独立的路径：解析值按公式算，
    实测值在解释器执行时逐次累加。两者必须一致（B 类断言）。
    """

    def __init__(self, plan):
        self.plan = plan
        self.counters = {"writes": 0, "reads": 0, "evals": 0}
        self.peak_elems = 0
        self._allocated = set()
        self._sliced = set()
        self.env = {}

    def run(self, values):
        plan = self.plan
        self.env = dict(values)
        self.counters = {"writes": 0, "reads": 0, "evals": 0}
        self.peak_elems = 0
        self._allocated = set()
        self._sliced = set()

        # 1) 根位置物化的 stage：整块算（**汇 stage 除外**——它们在下一步执行）
        sink_names = {st.name for st in plan.sinks}
        for st in plan.topo:
            if plan.place[st.name] == ("root",) and st.name not in sink_names:
                self.env[st.out.name] = self._produce(st, None)
        self._sample_peak(at_extra=0)

        # 2) 汇（输出）阶段：若它的某条轴上有 at 物化的 stage，就沿该轴切片执行
        out = {}
        for sink in plan.sinks:
            axis = plan.slice_axis(sink)
            if axis is None:
                out[sink.out.name] = self._produce(sink, None)
                continue
            acc = np.zeros(sink.out.shape, dtype=np.float64)
            attached = plan.attached_to(sink, axis.name)
            for idx in range(axis.extent):
                self._sliced = set()
                slice_elems = 0
                for st in attached:
                    self.env[st.out.name] = self._produce(st, idx)
                    self._sliced.add(st.out.name)
                    slice_elems += self.env[st.out.name].size
                self._sample_peak(at_extra=slice_elems)
                acc[idx] = self._produce(sink, idx)
            out[sink.out.name] = acc
        return out

    @property
    def n_buffers(self):
        """**实测**的中间 buffer 个数 = 执行过程中真的分配过的不同 buffer 数。"""
        return len(self._allocated)

    # -- 内部 --------------------------------------------------------------

    def _sample_peak(self, at_extra):
        """峰值口径：根位置的 buffer 全程存活；附着切片在一次迭代内同时活跃。"""
        live = 0
        for st in self.plan.materialized:
            if self.plan.place[st.name] == ("root",) and st.out.role == "intermediate":
                live += st.out.size
        self.peak_elems = max(self.peak_elems, live + at_extra)

    def _produce(self, stage, ctx):
        vals = [self._value(b, ctx) for b in stage.ins]
        res = np.asarray(stage.fn(*vals), dtype=np.float64)
        self.counters["evals"] += res.size
        if stage.out.role == "intermediate":
            self.counters["writes"] += res.size
            self._allocated.add(stage.out.name)
        return res

    def _value(self, buf, ctx):
        st = self.plan.func.producer_of(buf)
        if st is None:                                  # 函数入参
            arr = self.env[buf.name]
            return arr if ctx is None else arr[ctx]
        if self.plan.is_inline(st):                     # 重算：不取存储，现算
            vals = [self._value(b, ctx) for b in st.ins]
            res = np.asarray(st.fn(*vals), dtype=np.float64)
            self.counters["evals"] += res.size
            return res
        arr = self.env[buf.name]                        # 物化：读存储
        if ctx is not None and buf.name in self._sliced:
            part = arr                                 # 附着在循环内的切片已经就是这一片
        else:
            part = arr if ctx is None else arr[ctx]
        if buf.role == "intermediate":
            self.counters["reads"] += part.size
        return part

    def report(self):
        """把实测值整理成与 Plan.summary_rows 同序的表。"""
        c = self.counters
        return [
            ("中间 buffer 个数", self.n_buffers),
            ("峰值中间存储（元素）", self.peak_elems),
            ("峰值中间存储（字节）", self.peak_elems * ESIZE),
            ("中间写入（元素）", c["writes"]),
            ("中间读取（元素）", c["reads"]),
            ("中间流量（元素）", c["writes"] + c["reads"]),
            ("全 stage 求值（元素）", c["evals"]),
        ]


# ---------------------------------------------------------------------------
# native 轨（TIRx 式）：协作 scope 全手写，没有可调度的 block
# ---------------------------------------------------------------------------


class TileCall:
    """一次 tile 原语调用。`op` 是全名，命名空间**编码协作 scope**。"""

    def __init__(self, op, fn, note=""):
        if op not in MINI_TILE_OPS:
            raise NativeTrackError("迷你 registry 里没有这个原语：%s" % op)
        self.op = op
        self.scope = MINI_TILE_OPS[op]
        self.fn = fn
        self.note = note


# 本仓库迷你 registry —— **官方清单的子集**（31 个名字的完整清单未取到，见 specs.UNSOURCED）。
# 名字取自 specs.TILE_OP_GROUPS 里有出处的组内示例名。
MINI_TILE_OPS = {
    "Tx.tile.copy": "thread",              # 裸命名空间 = thread 级
    "Tx.tile.copy_async": "thread",
    "Tx.tile.cta.copy": "cta",
    "Tx.tile.warp.gemm": "warp",
    "Tx.tile.cta.gemm": "cta",
    "Tx.tile.cta.sum": "cta",
    "Tx.tile.warp.sum": "warp",
}


class NativeKernel:
    """TIRx native 式 kernel：grid / block 显式给出，协作 scope 写在原语名里。

    **没有 block 可调度** —— 这正是它与 s_tir 轨的根本区别。
    """

    def __init__(self, name, grid, block, calls):
        self.name = name
        self.grid = tuple(grid)          # CTA 维度
        self.block = tuple(block)        # 线程维度
        self.calls = list(calls)
        self.s_tir = False

    @property
    def n_threads(self):
        n = 1
        for b in self.block:
            n *= b
        return n

    @property
    def n_cta(self):
        n = 1
        for g in self.grid:
            n *= g
        return n

    def scopes_used(self):
        return sorted({c.scope for c in self.calls})


def run_native(kernel, values):
    """执行 native kernel，并数每个原语**被调用多少次**（协作 scope 决定这个次数）。"""
    ctx = {"values": dict(values), "buf": {}, "cta": 0, "thread": 0, "warp": 0}
    counts = {}
    n_warp = max(1, kernel.n_threads // 32)
    for call in kernel.calls:
        require_scope(kernel, call.scope, "原语 %s" % call.op)
        counts[call.op] = counts.get(call.op, 0)
        if call.scope == "cta":
            reps = kernel.n_cta
            for ci in range(kernel.n_cta):
                ctx["cta"] = ci
                call.fn(ctx)
                counts[call.op] += 1
        elif call.scope == "warp":
            reps = kernel.n_cta * n_warp
            for ci in range(kernel.n_cta):
                for wi in range(n_warp):
                    ctx["cta"], ctx["warp"] = ci, wi
                    call.fn(ctx)
                    counts[call.op] += 1
        elif call.scope == "thread":
            reps = kernel.n_cta * kernel.n_threads
            for ci in range(kernel.n_cta):
                for ti in range(kernel.n_threads):
                    ctx["cta"], ctx["thread"] = ci, ti
                    call.fn(ctx)
                    counts[call.op] += 1
        else:
            raise NativeTrackError("未知协作 scope：%s" % call.scope)
        if reps == 0:
            raise ScopeMismatch("原语 %s 的调用次数为 0" % call.op)
    return ctx, counts


def require_scope(kernel, scope, why):
    """协作 scope 的**前置检查**：声明了该 scope 却给不出对应的并行度 → 拒绝。"""
    if scope == "warp" and kernel.n_threads < 32:
        raise ScopeMismatch(
            "%s：warp 级原语需要至少 32 个线程（当前 block=%s，共 %d 个线程）"
            % (why, kernel.block, kernel.n_threads))
    if scope == "cta" and kernel.n_cta < 1:
        raise ScopeMismatch("%s：CTA 级原语需要至少一个 CTA" % why)
    return True


# ---------------------------------------------------------------------------
# pipeline 与 PassContext（迷你模型）
# ---------------------------------------------------------------------------
#
# ★ 这是**本仓库的迷你模型**：用「pass 类别」而不是真实 pass 名来刻画两条轨道。
#   真实 TVM 的 pass 名表（约 40 条 vs 18–19 条）本仓库未逐条核对，见 specs.UNSOURCED。
#   迷你模型要刻画**唯一**一条已核实的事实：默认 pipeline 解析到 s_tir 轨道。
#
#   ★ 迷你模型的分类依据：native 轨的并行映射是**手写**的，所以它不需要
#     「并行映射」这一类 pass —— 这是本仓库的建模，不是官方声明。

PIPELINE_CATEGORIES = {
    "s_tir": ["循环结构变换", "块位置与缓存", "并行映射", "向量化", "后端 lowering", "清理"],
    "tirx": ["向量化", "后端 lowering", "清理"],
}


class PassContext:
    """迷你 PassContext：一个字符串键的字典。"""

    def __init__(self, **config):
        self.config = dict(config)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __repr__(self):
        return "PassContext(%r)" % (self.config,)


def get_tir_pipeline(name="default"):
    """解析 pipeline 名。

    ★ 解析规则取自 v0.26.0 的 `get_tir_pipeline` 首句
      （`if name == "default": name = "s_tir"`，来源见 specs.PIPELINE_DEFAULT_RULE）：
      **默认走的是 s_tir 轨道**，tirx 那条必须显式指定。
    """
    if name == "default":
        name = "s_tir"
    if name not in PIPELINE_CATEGORIES:
        raise PipelineError("未知 pipeline：%s" % name)
    return name


class Executable:
    """迷你 compile 的产物。"""

    def __init__(self, mod, pipeline, categories, ctx):
        self.mod = mod
        self.pipeline = pipeline
        self.categories = list(categories)
        self.ctx = ctx

    @property
    def n_passes(self):
        return len(self.categories)


def compile(mod, target="cuda", *, relax_pipeline="default", tir_pipeline="default", ctx=None):
    """迷你 compile：签名照抄 `tvm.compile`（见 specs.COMPILE_SIGNATURE）。

    ★ 这条检查是**迷你实现**对真实陷阱的建模：`tir_pipeline` 的默认值是 "default"，
      而 "default" 解析到 **s_tir** 轨道；native 式（`s_tir=False`）的 prim_func
      必须**显式**指定 `tir_pipeline="tirx"`，否则落进一条没有可映射循环的 pipeline。
      真实 TVM 里具体报什么，本仓库**不写**（跑不了真实 TVM）。
    """
    ctx = ctx or PassContext()
    name = get_tir_pipeline(tir_pipeline)
    if getattr(mod, "s_tir", True) is False and name == "s_tir":
        raise PipelineError(
            "native 轨（s_tir=False）的模块落进了 s_tir pipeline："
            "`tir_pipeline` 的默认值 \"default\" 解析到 s_tir，"
            "native 轨要显式指定 tir_pipeline=\"tirx\""
            "（来源见 specs.PIPELINE_DEFAULT_RULE）")
    cats = list(PIPELINE_CATEGORIES[name])
    return Executable(mod, name, cats, ctx)


# ---------------------------------------------------------------------------
# 三类断言的记账器（与 60-compilers/mlir 的形态一致）
# ---------------------------------------------------------------------------

A_NAME, B_NAME, C_NAME = "A 算法正确性", "B 代价模型", "C 边界与反例"


class Checks:
    """三类断言的记账器。

    A 算法正确性：与**独立参考实现**逐项比对，打印最大误差 / 差异数。
    B 代价模型  ：按文档公式算出的数值，须与文档表格逐行一致。
    C 边界与反例：文档 §4 声称「会出问题」的输入**真的跑出问题**——
                  断言的是**失败行为成立**，不是成功。
    """

    def __init__(self):
        self.counts = {"A": 0, "B": 0, "C": 0}
        self.failures = []

    def _record(self, cls, ok, msg, detail=""):
        self.counts[cls] += 1
        line = "  [%s] %s: %s" % ("ok  " if ok else "FAIL", cls, msg)
        if detail:
            line += "  | " + detail
        print(line)
        if not ok:
            self.failures.append((cls, msg))
        return ok

    def A(self, ok, msg, detail=""):
        return self._record("A", ok, msg, detail)

    def B(self, ok, msg, detail=""):
        return self._record("B", ok, msg, detail)

    def C(self, ok, msg, detail=""):
        return self._record("C", ok, msg, detail)

    def expect_raises(self, fn, exc, msg, detail=""):
        """跑 fn，断言它抛出 exc。**没抛 = 这条 C 类断言失败。**"""
        try:
            fn()
        except exc as e:
            return self._record("C", True, msg,
                                detail or ("抛出 %s: %s" % (type(e).__name__, e)))
        except Exception as e:                                      # noqa: BLE001
            return self._record("C", False, msg,
                                detail or ("抛了非预期的 %s: %s" % (type(e).__name__, e)))
        return self._record("C", False, msg, detail or "**没有抛异常**——失败行为没发生")

    def expect_ok(self, fn, msg, detail=""):
        """断言 fn 能跑通。给 C 类配一个「换成正确做法就通过」的对照。"""
        try:
            r = fn()
        except Exception as e:                                      # noqa: BLE001
            return self._record("A", False, msg, "抛了 %s: %s" % (type(e).__name__, e))
        return self._record("A", True, msg, detail if detail else str(r))

    def summary(self):
        total = sum(self.counts.values())
        print()
        print("-" * 78)
        if self.failures:
            for cls, msg in self.failures:
                print("  失败项 [%s] %s" % (cls, msg))
            print("断言汇总：通过 %d 条断言，失败 %d 条（共 %d 条）"
                  % (total - len(self.failures), len(self.failures), total))
            sys.exit(1)
        print("通过 %d 条断言（%s %d / %s %d / %s %d）"
              % (total, A_NAME, self.counts["A"], B_NAME, self.counts["B"],
                 C_NAME, self.counts["C"]))
        return 0


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def max_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def table(rows, headers=("项", "值")):
    width = max(len(str(r[0])) for r in rows) + 2
    print("%-*s %s" % (width, headers[0], headers[1]))
    print("-" * (width + 14))
    for name, val in rows:
        print("%-*s %s" % (width, name, val))
