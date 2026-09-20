#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA Graph：capture 的**合法边界**与宿主 API 调用次数的账。

配套文档：`03-memory-and-graphs.md`
运行：`make run`（或 `make 03_graph_capture`）

===============================================================================
★★★ 限定语（**不是硬件**）—— 本程序每一句结论都受它约束
===============================================================================

  > **本程序不执行任何 CUDA。** 本机是 Apple M1 Pro，无 GPU、无 nvcc、无 libcublas。
  > 它做两件事：
  >   (a) 把 hw13 的两个真实文件**按行数出宿主 API 调用次数**（数字可对着源文件复核）；
  >   (b) 用一个**显式的 capture 状态机**，按官方原文判定 capture 期间哪些调用非法。
  > 一个 kernel 在硬件上会不会真的跑起来、capture 出来的图能不能 replay、
  > libcublas 是否支持被 capture —— 本程序**一概没有验证**，也不声称。

  > **「宿主 API 调用次数」不是「时间」。** 本程序数的是**调用**，
  > 不是耗时、不是吞吐。hw13 README 第 74 行只说
  > 「you should still be able to see a small preformance increase」——
  > **没有倍数**，本机也没有 GPU 去测。**收益是「宿主 API 调用次数」，不是时间。**

  > **capture 图的 node 数是 4 还是 8，本机无法判定。** 只有三份互相吻合的
  > **间接证据**指向 4（见 §B.5 与 `specs.UNSOURCED`）。**证据吻合 ≠ 执行结果。**

===============================================================================
★ 三类断言（本程序内部，失败即非零退出；CONTRIBUTING 第 121 行第二种情形）
===============================================================================

  §A 算法正确性 —— 16 条。**两条不同的代码路径**：
      路径 1（被检实现）：`Recorder` 状态机回放转写出来的调用序列并计数；
      路径 2（参照实现）：`count_calls_in_range()` 直接**正则扫源文件的行范围**。
      两者必须给出同一个数（A.1）。另有 12 条锚点的逐字完整性与归属核对（A.4）、
      A.5b 里对源文件第 67 行的**逐字核对**、以及 A.9/A.9b 的 `covered_by` **双向对齐**。
  §B 代价模型   —— 5 条。宿主调用次数 8:1；kernel 发射次数不变；node 数两种口径。
  §C 边界与反例 —— 9 条。声称「会失效」的调用必须**真的失效**。

★★ 本程序的可证伪断言（改哪一处、断言会怎么失败）

  A.1a 把 `Recorder` 在定时循环里的计数改成 7 或 9 ⇒ A.1a 与源文件正则扫描的结果对不上。
  A.1c 把源文件第 167 行（`cudaGraphLaunch`）改成 `cudaEventRecord` ⇒
       A.1c 要求「只有一种 API」的断言失败。
  A.5  把 hw13 第 67 行的 `cudaStreamNonBlocking` 从转写里删掉 ⇒
       A.5/C.2 的第六种变体从 cudaSuccess 变成 cudaErrorStreamCaptureInvalidated。
  B.1  把 `HW13_TIMER_CALLS_PER_ITER` 从 8 改成 7 ⇒ B.1 失败。
  C.3  让 capture 起始于 legacy 流 —— 若把 B3 的判定删掉，C.3 要求「违规」的断言失败。
  C.5  把 `event_query` 改成「Relaxed 模式下合法」⇒ C.5 要求「Relaxed 下仍然违规」失败。
  C.6  让 `CaptureState.end()` 在 invalidated 时返回一个**部分图** ⇒ C.6 要求
       「返回 NULL graph」失败。
  A.5b 改 `specs.HW13_STREAM_FLAGS_TEXT` 一个字 ⇒ 第 67 行的逐字核对失败
       （该常量原本只由 `01` 的 C.13 那张 14 处表核对，A.5b 只用了行号；
        订正轮把「变体表最后两行差的就是这一行」落进同一条断言）。
  A.9  把任一锚点的 `covered_by` 里的一个标签改掉 —— **哪怕改成另一个真实存在的标签**
       ⇒ 该锚点的 `note` 里出现的标签集合与 `covered_by` 不再相等，A.9 立刻失败。

★★ 本程序**不修改磁盘上的任何文件**：所有反例都在内存里的**文本**上构造。
"""

import os
import re
import sys
from collections import Counter

import numpy as np

import specs

# ★ 本程序覆盖的语义锚点（01 用 ast 读它，断言并集 == specs.ANCHORS 全集，两两不交）
COVERED_ANCHORS = ("B3", "B4a", "B4b", "B4c", "B5a", "B5b", "B5c",
                   "B6", "B8", "B9a", "B9b", "B9c")

LEGACY = "NULL"


# ===========================================================================
# 0. 被检实现：镜像 CUDA 宿主 API 的调用记录器 + capture 状态机
# ===========================================================================
class Recorder:
    """镜像 CUDA 宿主 API 的调用记录器 + stream capture 状态机。

    ★ 每一次 `_rec` 都是「宿主线程调用了一个 CUDA API」这一件事的**计数单位**。
      `cudaCheckErrors(...)` 这个**诊断宏**刻意**不计入** —— 它是每个调用后面都跟一个的
      检查，计入会让两边**同时**翻倍（比值不变），但绝对数会失去意义。
      **这个口径必须写明**，否则「8 次/迭代」对不上读者的数法（见 §B.1）。
    """

    def __init__(self):
        self.trace = []                  # [(phase, api, detail)]
        self.phase = "setup"
        self.capturing = {}              # stream -> graph name
        self.origin = None
        self.joined = set()              # 已 join 回 origin 的被捕获流
        self.last_rec_stream = {}        # event -> 最后一次 record 所在的流
        self.nonblocking = {}
        self.violations = []
        self.mode = "global"
        self.invalidated = False
        self.graph_nodes = []
        self.end_status = None
        self.rejoin_seen = False

    # ---------------- 记录 ----------------
    def _rec(self, api, detail=""):
        self.trace.append((self.phase, api, detail))

    def n_calls(self, phase=None):
        return sum(1 for p, _, _ in self.trace if phase is None or p == phase)

    def calls(self, api, phase=None):
        return sum(1 for p, a, _ in self.trace
                   if a == api and (phase is None or p == phase))

    def breakdown(self, phase=None):
        return Counter(a for p, a, _ in self.trace if phase is None or p == phase)

    # ---------------- 宿主侧资源 ----------------
    def stream_create(self, name, nonblocking=False):
        self._rec("cudaStreamCreateWithFlags" if nonblocking else "cudaStreamCreate", name)
        self.nonblocking[name] = nonblocking

    def event_create(self, name):
        self._rec("cudaEventCreate", name)

    def malloc(self, name):
        self._rec("cudaMalloc", name)

    # ---------------- capture 控制 ----------------
    def stream_begin_capture(self, s, mode="global"):
        self._rec("cudaStreamBeginCapture", "%s,%s" % (s, mode))
        self.mode = mode
        self.origin = s
        self.capturing[s] = "graph0"

    def _capturing(self):
        return bool(self.capturing)

    def stream_end_capture(self, s):
        self._rec("cudaStreamEndCapture", s)
        if self.invalidated:
            self.capturing.clear()
            self.end_status = ("cudaErrorStreamCaptureInvalidated", None)     # B6
            return self.end_status
        unjoined = [t for t in self.capturing if t != s and t not in self.joined]
        self.capturing.clear()
        if unjoined:
            self.end_status = ("cudaErrorStreamCaptureUnjoined", None)        # B4c
            return self.end_status
        self.end_status = ("cudaSuccess", "graph0")
        return self.end_status

    def graph_create(self, name):
        self._rec("cudaGraphCreate", name)

    def graph_instantiate(self, inst, graph):
        self._rec("cudaGraphInstantiate", "%s<-%s" % (inst, graph))

    def graph_launch(self, inst, s):
        self._rec("cudaGraphLaunch", "%s@%s" % (inst, s))

    def graph_add_child_graph_node(self, node, graph, ndeps, child):
        self._rec("cudaGraphAddChildGraphNode", "%s child=%s deps=%d" % (node, child, ndeps))

    # ---------------- 设备侧动作 ----------------
    def _violate(self, api, why):
        self.violations.append((api, why))
        self.invalidated = True                                              # B6

    def kernel(self, name, stream):
        self._rec("kernel<<<>>>", "%s@%s" % (name, stream))
        if stream == LEGACY and self._capturing():
            self._violate("kernel<<<>>>@NULL(legacy)",
                          "B3: Stream capture can be used on any CUDA stream except "
                          "cudaStreamLegacy (the NULL stream).")
        if self._capturing():
            self.graph_nodes.append(("kernel", name))

    def memcpy_async(self, name, kind, stream):
        self._rec("cudaMemcpyAsync", "%s %s@%s" % (kind, name, stream))
        if self._capturing():
            self.graph_nodes.append(("memcpy", "%s %s" % (kind, name)))

    def memcpy_sync(self, name, kind):
        self._rec("cudaMemcpy", "%s %s" % (kind, name))
        if self._capturing():
            self._violate("cudaMemcpy",
                          "B5c: also invalid to call synchronous APIs in this case. "
                          "Synchronous APIs, such as cudaMemcpy(), enqueue work to the "
                          "legacy stream and synchronize it before returning.")
            self.graph_nodes.append(("memcpy", "%s %s" % (kind, name)))

    def memset_sync(self, name):
        self._rec("cudaMemset", name)
        if self._capturing():
            self._violate("cudaMemset", "B5c: 同步 API 落到 legacy stream")

    def event_record(self, ev, stream):
        self._rec("cudaEventRecord", "%s@%s" % (ev, stream))
        self.last_rec_stream[ev] = stream
        if self._capturing():
            self.graph_nodes.append(("record", ev))

    def event_synchronize(self, ev):
        self._rec("cudaEventSynchronize", ev)
        if self._capturing():
            self._violate("cudaEventSynchronize",
                          "B5a: invalid to synchronize or query the execution status of a "
                          "stream which is being captured or a captured event")

    def event_query(self, ev, mode="global"):
        """★ B9c：**Relaxed 模式不是豁免**。

        它原文举的例子恰好就是 `cudaEventQuery`：
          「the thread is still prohibited from API calls which necessarily conflict with
            stream capture, for example, attempting cudaEventQuery on an event that was
            last recorded inside a capture sequence」
        ⇒ 本方法**不看 mode**，只要在 capture 期间 query 被捕获的 event 就违规。
        """
        self._rec("cudaEventQuery", ev)
        if self._capturing():
            self._violate("cudaEventQuery",
                          "B9c: Relaxed 下仍然违规 —— 原文举的例子就是 cudaEventQuery")

    def stream_wait_event(self, stream, ev):
        self._rec("cudaStreamWaitEvent", "%s<-%s" % (stream, ev))
        if self._capturing():
            r = self.last_rec_stream.get(ev)
            # B4b 前半：被 wait 的流**自动进入 capture**
            if stream not in self.capturing and self.origin is not None:
                self.capturing[stream] = self.capturing.get(self.origin, "graph0")
            # B4c：join 的表现形式 = 在 origin 流上等一个从别的流 record 出来的 event
            if r is not None and r != stream and stream == self.origin:
                self.joined.add(r)
                self.rejoin_seen = True
            self.graph_nodes.append(("wait", ev))

    def stream_synchronize(self, stream):
        self._rec("cudaStreamSynchronize", stream)
        if self._capturing():
            self._violate("cudaStreamSynchronize@%s" % stream,
                          "B5a: invalid to synchronize or query the execution status of a "
                          "stream which is being captured")

    def device_synchronize(self):
        self._rec("cudaDeviceSynchronize", "")
        if self._capturing():
            self._violate("cudaDeviceSynchronize",
                          "B5a: also invalid to query the execution status of or synchronize "
                          "a broader handle which encompasses an active stream capture, such "
                          "as a device or context handle")

    def legacy_command(self, label="cudaMemcpyToSymbol@NULL"):
        """任何落到 legacy stream 的动作。"""
        self._rec(label, "")
        if self._capturing():
            bad = [s for s in self.capturing if not self.nonblocking.get(s, False)]
            if bad:
                self._violate(label,
                              "B5b: When any stream in the same context is being captured, "
                              "and it was not created with cudaStreamNonBlocking, any "
                              "attempted use of the legacy stream is invalid. 涉事流: %r"
                              % (bad,))

    def cublas_call(self, name, stream):
        self._rec("cublas" + name, "@" + stream)
        if self._capturing():
            self.graph_nodes.append(("library", name))


# ===========================================================================
# 1. 两个真实文件的转写（逐行钉回源文件，见 §A.6）
# ===========================================================================
def hw13_task1_common(r, nonblocking=True):
    r.event_create("event1")                                   # 58
    r.event_create("event2")                                   # 59
    for i in range(2):
        r.stream_create("streams[%d]" % i, nonblocking=nonblocking)   # 66-68
    r.malloc("d_x")                                            # 88
    r.malloc("d_y")                                            # 89
    r.memcpy_sync("d_x", "H2D")                                # 92
    r.memcpy_sync("d_y", "H2D")                                # 93


def hw13_task1_graph(n_iter, capture_extra=None, nonblocking=True):
    """转写 `Solutions/axpy_stream_capture_solution.cu`（capture 一次 + replay）。"""
    r = Recorder()
    hw13_task1_common(r, nonblocking=nonblocking)
    r.graph_create("graph")                                    # 101

    r.phase = "first_pass"
    r.stream_begin_capture("streams[0]", "global")              # 110
    r.kernel("kernel_a", "streams[0]")                          # 113
    r.event_record("event1", "streams[0]")                       # 116
    r.kernel("kernel_b", "streams[0]")                          # 119
    r.stream_wait_event("streams[1]", "event1")                  # 122
    r.kernel("kernel_c", "streams[1]")                          # 125
    r.event_record("event2", "streams[1]")                       # 128
    r.stream_wait_event("streams[0]", "event2")                  # 131  <-- join
    r.kernel("kernel_d", "streams[0]")                          # 134
    if capture_extra is not None:
        capture_extra(r)
    r.stream_end_capture("streams[0]")                          # 137
    if r.end_status[0] != "cudaSuccess":
        return r
    r.graph_instantiate("instance", "graph")                     # 141
    r.graph_launch("instance", "streams[0]")                     # 147
    r.stream_synchronize("streams[0]")                           # 149

    r.phase = "timed_loop"
    for _ in range(n_iter):
        r.graph_launch("instance", "streams[0]")                 # 167
    r.phase = "epilogue"
    return r


def hw13_task1_timer(n_iter):
    """转写 `axpy_stream_capture_timer.cu`（逐 launch，无 graph）。"""
    r = Recorder()
    hw13_task1_common(r, nonblocking=True)
    r.phase = "first_pass"
    r.kernel("kernel_a", "streams[0]")                          # 101
    r.event_record("event1", "streams[0]")                       # 104
    r.kernel("kernel_b", "streams[0]")                          # 107
    r.stream_wait_event("streams[1]", "event1")                  # 110
    r.kernel("kernel_c", "streams[1]")                          # 113
    r.event_record("event2", "streams[1]")                       # 116
    r.stream_wait_event("streams[0]", "event2")                  # 119
    r.kernel("kernel_d", "streams[0]")                          # 122
    r.stream_synchronize("streams[0]")                           # 125

    r.phase = "timed_loop"
    for _ in range(n_iter):
        r.kernel("kernel_a", "streams[0]")                       # 135
        r.event_record("event1", "streams[0]")                    # 138
        r.kernel("kernel_b", "streams[0]")                       # 141
        r.stream_wait_event("streams[1]", "event1")               # 144
        r.kernel("kernel_c", "streams[1]")                       # 147
        r.event_record("event2", "streams[1]")                    # 150
        r.stream_wait_event("streams[0]", "event2")               # 153
        r.kernel("kernel_d", "streams[0]")                       # 156
    r.phase = "epilogue"
    return r


def hw13_task2_cublas(n_iter):
    """转写 `axpy_cublas_with_fixme.cu`（填好 FIXME 之后的形态）。

    ★★ 与 task1 的**三处差别**（都对着源文件核过）：
      ① 循环是 **100** 次（不是 1000）；
      ② 每次迭代有 **2** 个宿主调用（`cudaGraphLaunch` + `cudaStreamSynchronize`），
         而且**没有** `cudaCheckErrors`；
      ③ 源文件里 task2 **根本没有计时**（没有 `std::chrono`）。
      ⇒ 计划 §0.7 G7 与探针把 task2 的循环也叫「定时循环」，**这是错的**：
        task2 是**发射循环**，不是定时循环。
    """
    r = Recorder()
    r.stream_create("stream1", nonblocking=True)                 # 38
    r._rec("cublasCreate", "")                                   # 41
    r._rec("cublasSetStream", "stream1")                         # 42
    r.malloc("d_x"); r.malloc("d_y")                             # 67-68
    r.memcpy_sync("d_x", "H2D")                                  # 71 cublasSetVector
    r.memcpy_sync("d_y", "H2D")                                  # 72
    r.graph_create("graph")                                      # 85
    r._rec("cudaGraphAddKernelNode", "kernelNode1(kernel_a)")     # 102
    r.graph_nodes.append(("kernel", "kernel_a"))
    r.phase = "first_pass"
    r.stream_begin_capture("stream1", "global")                  # 109
    r.cublas_call("Saxpy", "stream1")                            # 113 <-- 捕获库调用
    r.stream_end_capture("stream1")                              # 116
    if r.end_status[0] != "cudaSuccess":
        return r
    r.graph_add_child_graph_node("libraryNode", "graph", 1, "libraryGraph")   # 120
    r._rec("cudaGraphAddKernelNode", "kernelNode2(kernel_c)")     # 135
    r.graph_nodes.append(("kernel", "kernel_c"))
    r.graph_instantiate("instance", "graph")                     # 149
    r.phase = "launch_loop"
    for _ in range(n_iter):
        r.graph_launch("instance", "stream1")                    # 155
        r.stream_synchronize("stream1")                          # 156
    r.phase = "epilogue"
    return r


# ===========================================================================
# 2. ★★ 参照实现：直接**正则扫源文件**（与被检的 `Recorder` 不同的路径）
# ===========================================================================
# ★ 这条路径**完全不经过 Recorder**：它读源文件的行，用正则数出 CUDA 运行时调用
#   与 kernel 发射。两条路径给出同一个数，才说明转写没有漂。
API_RE = re.compile(r"\bcuda[A-Z]\w*\s*\(")
LAUNCH_MARK = "<<<"


def count_calls_in_range(text, lo, hi, drop=("cudaCheckErrors",)):
    """数第 lo..hi 行（1 起，含端点）里的：
      · CUDA 运行时 API 调用（形如 `cudaXxx(`，**排除**诊断宏 `cudaCheckErrors`）；
      · kernel 发射（`<<<`）。
    返回 (api_calls, launches, detail_counter)。
    """
    c = Counter()
    launches = 0
    for i in range(lo - 1, min(hi, len(text.split("\n")))):
        line = text.split("\n")[i]
        line = line.split("//", 1)[0]
        for m in API_RE.finditer(line):
            name = m.group(0).rstrip("( ").strip()
            if name in drop:
                continue
            c[name] += 1
        launches += line.count(LAUNCH_MARK)
    return sum(c.values()), launches, c


# ===========================================================================
# 3. 断言框架
# ===========================================================================
FAILS = []
STATS = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}
SKIPS = []

# ★★ 断言标签（`A.5b`、`C.7` …）：`specs.ANCHORS[*]['covered_by']` 引用的就是它。
#    由程序**在运行时**从断言名里提取，不另立手写清单（那会与真实断言脱节）。
LABEL_RE = re.compile(r"[ABC]\.\d+[a-z]*")
SEEN_LABELS = set()


def label_of(name):
    m = LABEL_RE.match(name)      # ★ 必须出现在**开头**才算标签
    return m.group(0) if m else None


def check(cls, name, cond, detail):
    STATS[cls][0] += 1
    lab = label_of(name)
    if lab:
        SEEN_LABELS.add(lab)
    ok = bool(cond)
    if ok:
        STATS[cls][1] += 1
    else:
        FAILS.append("%s/%s" % (cls, name))
    print("  [%s/%-6s] %s" % (cls, "PASS" if ok else "FAIL", name))
    print("        %s" % detail)
    return ok


def eq(cls, name, got, want, detail):
    return check(cls, name, got == want,
                 "%s\n        实得=%r\n        期望=%r" % (detail, got, want))


def find_root(env, marker, rels):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get(env)] + [os.path.join(here, r) for r in rels]
    tried = []
    for c in cands:
        if not c:
            continue
        c = os.path.expanduser(c)
        tried.append(c)
        if os.path.exists(os.path.join(c, marker)):
            return c, tried
    return None, tried


# ===========================================================================
# 4. 主流程
# ===========================================================================
def main():
    train, tried = find_root(specs.TRAIN_ROOT_ENV, "exercises/hw13",
                             ["../cuda-training-series", "../../cuda-training-series",
                              "../../../cuda-training-series",
                              "../../../../cuda-training-series"])
    print("=" * 78)
    print("03_graph_capture —— CUDA Graph 的收益账与 capture 的合法边界")
    print("=" * 78)
    print("  解释器      : %s" % sys.executable)
    print("  python      : %s" % sys.version.split()[0])
    print("  numpy       : %s（变体表与比值用它复核）" % np.__version__)
    print("  ★ **不是硬件**：本程序不执行任何 CUDA，也不执行任何 kernel。")
    print("  ★ **收益是「宿主 API 调用次数」，不是时间**：本程序不做、也无法做计时。")
    print("  ★ **node 数是 4 还是 8，本机无法判定** —— 只有间接证据。")
    if train is None:
        print()
        print("!! 找不到 %s —— 本程序的全部数字都从它的源文件里数出来。"
              % specs.TRAIN_ROOT_ENV)
        print("   探测过的路径：")
        for t in tried:
            print("       %s" % t)
        print("   怎么修：把外部工件放到本目录的兄弟目录，或显式指定：")
        print("       make run %s=/path/to/cuda-training-series" % specs.TRAIN_ROOT_ENV)
        print("   注：`~` 请写成 `$HOME`（make 不做 `~` 展开）。")
        print("   本程序的断言**没有**跑，退出码非零。")
        return 2
    print("  外部工件    : %s（%s，B 级：厂商训练材料，不是规范）"
          % (train, specs.TRAIN_ROOT_ENV))
    print("  工件版本    : commit %s (%s)，无 release tag"
          % (specs.TRAIN_COMMIT_SHORT, specs.TRAIN_COMMIT_DATE))
    print()

    def tp(rel):
        with open(os.path.join(train, rel), "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    N = 1000
    src_time = tp(specs.HW13_TIMER_PATH)
    src_sol = tp(specs.HW13_SOLUTION_PATH)
    src_cub = tp(specs.HW13_TASK2_PATH)
    # ★ task1 的**起始文件**：A.5b 要核对的第 67 行（建流标志）在这里。
    #   ⇒ 「变体表最后两行差的**就是这一行**」变成同一条断言里的证据
    #     （`HW13_STREAM_FLAGS_TEXT` 本来由 01 的 C.13 核对，这里补的是**联动**）。
    src_fix = tp(specs.HW13_TASK1_PATH)

    # ---------------------------------------------------------------------
    print("-" * 78)
    print("§A 算法正确性（参照实现 = 直接正则扫源文件，**不经过 Recorder**）")
    print("-" * 78)

    T = hw13_task1_timer(N)
    G = hw13_task1_graph(N)

    lo, hi, iters = specs.HW13_TIMER_LOOP
    t_api, t_launch, t_detail = count_calls_in_range(src_time, lo, hi)
    lo2, hi2, iters2 = specs.HW13_GRAPH_LOOP
    g_api, g_launch, g_detail = count_calls_in_range(src_sol, lo2, hi2)

    tk_rec = T.calls("kernel<<<>>>", "timed_loop")          # Recorder 侧：kernel 发射
    gk_rec = G.calls("kernel<<<>>>", "timed_loop")
    t_non_kernel = T.n_calls("timed_loop") - tk_rec

    eq("A", "A.1a ★ 两条路径给出同一个数：`for` 循环体里的**宿主侧动作**数",
       (t_api + t_launch, T.n_calls("timed_loop") / N),
       (8, 8.0),
       "★ **测量条件必须写明**：源文件里的 `for` 循环体**只写了一次**"
       "（`for (int i = 0; i < %d; ++i)`），所以正则扫第 %d–%d 行数出来的是"
       "**「一次迭代」的量**：%d 个 CUDA API 调用 + %d 个 kernel 发射 = **%d**。"
       "路径 2（`Recorder` 回放 %d 次迭代）给 %d 次 ⇒ 每次迭代 %g。"
       "★ 两条路径的**代码完全不同**（一个读磁盘上的行、一个读内存里的转写），"
       "一致的唯一解释是转写没漂。"
       % (iters, lo, hi, t_api, t_launch, t_api + t_launch, N,
          T.n_calls("timed_loop"), T.n_calls("timed_loop") / N))

    eq("A", "A.1b ★ 口径分解：**CUDA API 调用 %d 个/迭代 + kernel 发射 %d 个/迭代**"
       % (t_api, t_launch),
       (t_api, t_launch, t_non_kernel / N, tk_rec / N),
       (4, 4, 4.0, 4.0),
       "源文件第 %d–%d 行：2 个 `cudaEventRecord` + 2 个 `cudaStreamWaitEvent` = **4** 个 "
       "CUDA API 调用；4 个 kernel 发射（`<<<` 不匹配 `cuda[A-Z]`，所以与那 4 个分开数）。"
       "**合计 8** —— 这就是「8 次/迭代」的构成。"
       "★ **口径必须写明**，否则读者会数出 4：`cudaCheckErrors(...)` **不计**"
       "（它是诊断宏，每个调用后面都跟一个，计入会让两边同时翻倍、绝对数失去意义）；"
       "而 **kernel 发射计入**（它是货真价实的宿主侧动作）。"
       "★ **P9 提示**：计划与探针把这个数叫「宿主 API 调用次数」，"
       "但它**含 kernel 发射** —— 严格说该叫「宿主侧动作」。本程序两个名字都给出，"
       "并在正文里统一用后一个。"
       % (lo, hi))

    eq("A", "A.1c 有 graph 的定时循环里**只有一种** API：%r" % (sorted(g_detail),),
       (sorted(g_detail), g_launch), (["cudaGraphLaunch"], 0),
       "★ 源文件第 %d–%d 行的**全部** `cuda*` 调用只有 `cudaGraphLaunch` 一种，"
       "且**一个 `<<<` 都没有**（那 4 个 kernel 在图里）。"
       "★ 若把第 167 行改成别的调用，这一条立刻失败。" % (lo2, hi2))

    eq("A", "A.2 两个版本的 **kernel 执行次数/迭代相同**（4 个：kernel_a/b/c/d）",
       (t_launch, tk_rec / N, g_launch, gk_rec / N,
        G.calls("cudaGraphLaunch", "timed_loop") * 4 / N),
       (4, 4.0, 0, 0.0, 4.0),
       "★ 无 graph 版：源文件里有 4 个 `<<<`。"
       "有 graph 版：源文件里 **0** 个 `<<<`（`Recorder` 侧也是 0）—— "
       "那 4 个 kernel **在图里**，每次 `cudaGraphLaunch` 把它们一起发射。"
       "⇒ **省下来的是宿主侧动作，不是 kernel 执行。**"
       "★ 把「少了 7 个宿主动作」误读成「少了 7 倍工作量」，是本主题最常见的错。")

    # ---- A.3 转写与源文件的行号钉合 ----
    pin_bad = []
    for ln, txt in specs.HW13_SOLUTION_PINS:
        got = src_sol.split("\n")[ln - 1].strip()
        if txt not in got:
            pin_bad.append((ln, got[:60], txt[:60]))
    eq("A", "A.3a solution 文件 %d 处「行号 + 逐字内容」全部命中"
       % len(specs.HW13_SOLUTION_PINS), pin_bad, [],
       "★ capture 区的两个边界（第 110 行 BeginCapture、第 137 行 EndCapture）"
       "与定时循环里的唯一一条 launch（第 167 行）都由逐字内容钉住。")
    pin_bad2 = []
    for ln, txt, _ in specs.HW13_TASK2_OPS:
        got = src_cub.split("\n")[ln - 1].strip()
        if txt not in got:
            pin_bad2.append((ln, got[:60], txt[:60]))
    eq("A", "A.3b cublas（task2）文件 %d 处「行号 + 逐字内容」全部命中"
       % len(specs.HW13_TASK2_OPS), pin_bad2, [],
       "★ 其中第 %d 行 `cublasSaxpy(...)` 是**被 capture 的那个库调用**；"
       "第 %d 行是发射循环的 `for`。"
       % (specs.HW13_TASK2_OPS[3][0], specs.HW13_TASK2_LOOP[0]))

    # ---- A.4 锚点引文的逐字完整性（本程序覆盖的 12 条）----
    q_bad = []
    for aid in COVERED_ANCHORS:
        a = specs.ANCHORS[aid]
        for frag in a["must_contain"]:
            if a["quote"].count(frag) != 1:
                q_bad.append((aid, frag[:40], a["quote"].count(frag)))
    eq("A", "A.4 本程序覆盖的 %d 条锚点（%s），`must_contain` 都在 `quote` 里恰好 1 次"
       % (len(COVERED_ANCHORS), "/".join(COVERED_ANCHORS)), q_bad, [],
       "★ 「凡引用外部原文，把原文与断言写在同一处」。"
       "★ 特别注意 B5b 的 `not` 与 B9c 的 `still prohibited` —— "
       "这两处的**否定词**是整篇结论的承重墙，抄漏一个结论就反过来。")

    # ---- A.5 ★ 七种变体的判定表（B5b 的载体）----
    variant_got = []
    for desc, key, nb, want in specs.CAPTURE_VARIANTS:
        r = hw13_task1_graph(N, capture_extra=make_extra(key), nonblocking=nb)
        variant_got.append((desc, r.end_status[0]))
    eq("A", "A.5 ★ capture 的 %d 种变体：实际返回码与规格表逐行一致"
       % len(specs.CAPTURE_VARIANTS),
       [x[1] for x in variant_got], [x[3] for x in specs.CAPTURE_VARIANTS],
       "判定依据逐条来自锚点：B5a（同步/查询非法）、B5c（同步 API 非法）、"
       "B5b（legacy 流的使用取决于建流标志）、B6（违规使图作废）。"
       "\n        " + "\n        ".join("%-46s -> %s" % (d, s) for d, s in variant_got))

    eq("A", "A.5b ★★ 最后两行**只差一个建流标志**，返回码却不同；那个标志在源文件第 %d 行"
       "**逐字**命中" % specs.HW13_STREAM_FLAGS_LINE,
       ([(d, s) for d, s in variant_got if d.endswith("NonBlocking 的）")],
        [(d, s) for d, s in variant_got if d.endswith("blocking 的）")],
        specs.HW13_STREAM_FLAGS_TEXT in
        src_fix.split("\n")[specs.HW13_STREAM_FLAGS_LINE - 1]),
       ([("插 legacy 流动作（该流是 NonBlocking 的）", "cudaSuccess")],
        [("插 legacy 流动作（该流是 blocking 的）",
          "cudaErrorStreamCaptureInvalidated")],
        True),
       "★★ **这是本篇的反直觉量化结论。** 来源（逐字）B5b："
       "「When any stream in the same context is being captured, **and it was not "
       "created with cudaStreamNonBlocking**, any attempted use of the legacy stream is "
       "invalid.」⇒ 建流时带了 `cudaStreamNonBlocking` ⇒ **合法**；"
       "没带 ⇒ `cudaErrorStreamCaptureInvalidated`。"
       "⇒ hw13 第 %d 行的 `cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking)` "
       "**不是随手写的**，它是这段 capture 合法性的前提。\n"
       "        ★ 第三项是**订正轮补的联动**：此前 `A.5b` 只把第 %d 行的**行号**带进打印，"
       "那个常量的**逐字内容**在 `01_sync_contract.py` 的 C.13（14 处「行号 + 逐字内容」表）"
       "里被核对。本处补的是让两件事在**同一条断言**里对齐：变体表最后两行差的"
       "**就是这一行**（`specs.ANCHORS['A3']['covered_by']` 里 `A.5b` 指的就是这里）。"
       "★ 措辞更正：实施第一版曾把它写成「该常量没有任何断言消费」——**那是错的**，"
       "C.13 一直在核对它；`make mutate` 的漏网清单里它也确实**不在**漏网之列。"
       % (specs.HW13_STREAM_FLAGS_LINE, specs.HW13_STREAM_FLAGS_LINE))

    # ---- A.6 task1 的两次 capture 只发生一次 ----
    eq("A", "A.6 solution 的 capture 区只对**第一趟**生效（`graphCreated` 标志）",
       (G.calls("cudaStreamBeginCapture", "first_pass"),
        G.calls("cudaStreamBeginCapture", "timed_loop"),
        G.calls("cudaGraphInstantiate", "first_pass"),
        G.calls("cudaGraphInstantiate", "timed_loop")), (1, 0, 1, 0),
       "★ 源文件第 107–145 行的结构是 `for(i<100){ if (graphCreated == false) { 捕获 + "
       "instantiate；graphCreated = true; } cudaGraphLaunch(...); }` —— "
       "capture 与 instantiate 都在 `if` 里，只在第一趟执行。"
       "★ README 第 19 行把它写成方法：`bool graphCreated=false;` will be our method to "
       "set up the graph on the first pass only。")

    # ---- A.7 task2 的库调用被 capture ----
    C = hw13_task2_cublas(specs.HW13_TASK2_LOOP[2])
    lib = [n for k, n in C.graph_nodes if k == "library"]
    eq("A", "A.7 task2：库调用（cuBLAS Saxpy）被 capture 进 child graph",
       (C.end_status[0], lib), ("cudaSuccess", ["Saxpy"]),
       "★ README 第 81 行（逐字）：「We are creating 2 kernel nodes and a child graph "
       "derived from a cuBLAS axpy function call.」⇒ 顶层 node 数应为 3。"
       "★ **本机无法验证** libcublas 是否真的支持被 capture（无 GPU、无 libcublas）；"
       "本程序只复现「按文档，这个调用在 capture 区里是合法的」这一步。")

    # ---- A.8 ★ 把 loop 的**行范围与迭代数**从源文件机械推导出来，钉住 specs 的常量 ----
    def derive_loop(text, header_prefix, iters_default=None):
        lines = text.split("\n")
        lo = None
        for i, l in enumerate(lines, 1):
            if header_prefix in l:
                lo = i
                break
        if lo is None:
            return None
        it = None
        m = re.search(r"i\s*<\s*(\d+)", lines[lo - 1])
        if m:
            it = int(m.group(1))
        d, hi = 0, None
        for i in range(lo - 1, len(lines)):
            d += lines[i].count("{") - lines[i].count("}")
            if d == 0 and i > lo - 1:
                hi = i + 1
                break
        return (lo, hi, it)

    d_timer = derive_loop(src_time, "for (int i = 0; i < 1000; ++i){")
    d_graph = derive_loop(src_sol, "for (int i = 0; i < 1000; ++i){")
    d_task2 = derive_loop(src_cub, "for (int i = 0; i < 100; ++i){")
    eq("A", "A.8 ★ 三个循环的**行范围与迭代数**由源文件机械推导，与 specs 的常量逐项相同",
       (d_timer, d_graph, d_task2),
       (specs.HW13_TIMER_LOOP, specs.HW13_GRAPH_LOOP, specs.HW13_TASK2_LOOP),
       "推导方式：在源文件里找 `for (int i = 0; i < ...; ++i){` 那一行（起点）、"
       "用**手写括号配平**找它的闭合行（终点）、从 `i < N` 里取迭代数。"
       "★ 这一条把 `HW13_TIMER_LOOP` / `HW13_GRAPH_LOOP` / `HW13_TASK2_LOOP` "
       "**三个常量的九个格子全部钉住** —— 改任何一个数字（含行号）都会失败。\n"
       "        实得 = timer %r / graph %r / task2 %r" % (d_timer, d_graph, d_task2))

    eq("A", "A.8b 每迭代的动作数常量与实测一致（timer %d / graph %d / kernel %d / task2 %d）"
       % (specs.HW13_TIMER_CALLS_PER_ITER, specs.HW13_GRAPH_CALLS_PER_ITER,
          specs.HW13_KERNELS_PER_ITER, specs.HW13_TASK2_CALLS_PER_ITER),
       (t_api + t_launch == specs.HW13_TIMER_CALLS_PER_ITER,
        g_api + g_launch == specs.HW13_GRAPH_CALLS_PER_ITER,
        t_launch == specs.HW13_KERNELS_PER_ITER,
        C.n_calls("launch_loop") // specs.HW13_TASK2_LOOP[2]
        == specs.HW13_TASK2_CALLS_PER_ITER),
       (True, True, True, True),
       "四个常量各自对着**从源文件数出来的数**：timer 的 `API+launch` = %d+%d、"
       "graph 的 = %d+%d、timer 的 kernel 发射 = %d、task2 的每迭代 = %d。"
       "★ 改其中任何一个常量，这一条失败。"
       % (t_api, t_launch, g_api, g_launch, t_launch,
          specs.HW13_TASK2_CALLS_PER_ITER))

    eq("A", "A.8c solution 的定时循环里**只有** `cudaGraphLaunch`（`HW13_SOLUTION_ONLY_LAUNCH_IN_LOOP`）",
       (specs.HW13_SOLUTION_ONLY_LAUNCH_IN_LOOP, sorted(g_detail)),
       (True, ["cudaGraphLaunch"]),
       "★ 这一条把那个布尔常量与**实际扫出来的结果**绑在一起："
       "把它改成 False，或让源文件里多出别的调用，都会失败。")

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    print("§B 代价模型（推导值；**数源码行得来，可对着源文件复核**）")
    print("-" * 78)

    th = T.n_calls("timed_loop")
    gh = G.n_calls("timed_loop")
    tk = T.calls("kernel<<<>>>", "timed_loop")
    eq("B", "B.1 定时循环内的宿主 API 调用：无 graph %d/%d 次 = **%g 次/迭代**；"
       "有 graph %g 次/迭代" % (th, N, th / N, gh / N),
       (th, gh, th / N, gh / N), (8000, 1000, 8.0, 1.0),
       "★ 口径（**必须写明**）：只数 **CUDA runtime API 调用**，不数 `cudaCheckErrors` "
       "展开出的 `cudaGetLastError()` —— 它是诊断宏，每个调用后面都跟一个。"
       "★ 这个数**不是时间**：它不告诉你快了多少。README 第 74 行只说 small preformance "
       "increase，**没有倍数**。")

    eq("B", "B.2 比值 = **8 : 1**（宿主侧动作），而 **kernel 执行次数相同**（4 : 4）",
       (th / gh, t_launch, tk_rec / N, gk_rec / N),
       (8.0, 4, 4.0, 0.0),
       "★ 无 graph 版：每迭代 4 次 kernel 发射（源文件里 4 个 `<<<`）+ 4 个 CUDA API 调用 = 8。"
       "有 graph 版：每迭代 1 次 `cudaGraphLaunch`、**0** 次 `<<<`。"
       "⇒ 两个版本的 **kernel 执行次数完全相同**（每迭代 4 个），"
       "**省下来的只是宿主侧动作**。"
       "★ 把「8:1」读成「快了 8 倍」，是这一篇最要防的错。")

    saved_np = np.array([th], dtype=np.int64) - np.array([gh], dtype=np.int64)
    eq("B", "B.3 1000 次迭代省下的宿主动作数 = **%d 次**（int64 数组差）" % saved_np[0],
       int(saved_np[0]), 7000,
       "8×1000 − 1×1000 = 7000。★ 这是**调用次数**，不是时间也不是 CPU 时间 —— "
       "一次 `cudaGraphLaunch` 的 CPU 开销比一次 `<<<>>>` 大还是小，本机测不了。")

    eq("B", "B.4 task2 的发射循环是 **%d** 次（不是 task1 的 %d），每次 %d 个宿主调用"
       % (specs.HW13_TASK2_LOOP[2], N, specs.HW13_TASK2_CALLS_PER_ITER),
       (C.n_calls("launch_loop") / specs.HW13_TASK2_LOOP[2],
        sorted(C.breakdown("launch_loop"))),
       (2.0, ["cudaGraphLaunch", "cudaStreamSynchronize"]),
       "★★ **实施时发现的规格偏差**：计划 §0.7 G7 与探针把 task2 的循环也叫「定时循环」，"
       "但源文件第 %d 行是 `for (int i = 0; i < 100; ++i)`，"
       "而且 task2 里**没有 `std::chrono`**（没有计时）。"
       "⇒ task2 是**发射循环**：每趟 launch 后**立刻同步**（第 156 行），"
       "所以它的宿主调用结构（2 次/迭代、带同步）与 task1 的定时循环"
       "（1 次/迭代、**无**同步）**根本不同类**，不能放在一起比较。"
       % specs.HW13_TASK2_LOOP[0])

    raw_nodes = G.graph_nodes
    collapsed = [1 for k, _ in raw_nodes if k not in ("record", "wait")]
    eq("B", "B.5 capture 期产生的设备侧动作 = %d 个；两种 node 口径给 **%d** 与 **%d**"
       % (len(raw_nodes), len(raw_nodes), len(collapsed)),
       (len(raw_nodes), len(collapsed)),
       (specs.HW13_TASK1_RAWNODES, specs.HW13_TASK1_COLLAPSED_NODES),
       "口径 A（record/wait 各算一个节点）：§3.2.8.7.1.1 Node Types 里确实列了 "
       "waiting on an event / recording an event ⇒ %d 个。\n"
       "        口径 B（record/wait 折叠成依赖边、不产生节点）：依据是 B4b 的**后半句**"
       "「the next item in the stream will have **additional dependencies** on the nodes "
       "in the captured event」—— 只加依赖、没加节点 —— 以及 B8（只有 external event "
       "node 才显式成节点）⇒ %d 个。\n"
       "        ★★ **本机无法判定运行时给哪个。** 三份**间接证据**都指向口径 B："
       "① B4b 的措辞；② B8 的 external event node 专项；"
       "③ 仓库里 NVIDIA 自己画的 `%s` 右侧是 A/B/C/D **四个**节点（**人眼读过**）。"
       "⇒ 这一条只说「证据吻合」、**不说「已确定」**。"
       % (len(raw_nodes), len(collapsed), specs.HW13_DIAGRAM_PNG))

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    print("§C 边界与反例（**C 类成立时程序应当失败**）")
    print("-" * 78)

    # ★ 用 numpy 把变体表做成一张「码表」：合法 = 1、失效 = -1，
    #   然后逐行与 specs.CAPTURE_VARIANTS 的期望做**逐格**比较（不是逐项循环比较）。
    codes = np.array([1 if s == "cudaSuccess" else -1 for _, s in variant_got],
                     dtype=np.int8)
    want_codes = np.array([1 if v[3] == "cudaSuccess" else -1
                           for v in specs.CAPTURE_VARIANTS], dtype=np.int8)
    eq("C", "C.0 ★ 变体表的 numpy 码表（合法=1 / 失效=-1）**逐格**与规格相同",
       (codes.tolist(), int(codes.sum())), (want_codes.tolist(), -3),
       "★ 用 int8 数组而不是逐项 `==`：一处不符在码表上就是一个 -1/1 的错位，"
       "`(codes == want_codes).all()` 与逐格打印一起给出。")

    inval = [d for d, s in variant_got if s == "cudaErrorStreamCaptureInvalidated"]
    okv = [d for d, s in variant_got if s == "cudaSuccess"]
    eq("C", "C.1 %d 种变体里有且只有 **2** 种合法（原样 / NonBlocking 流上的 legacy 动作）"
       % len(specs.CAPTURE_VARIANTS), (len(okv), len(inval)), (2, 5),
       "合法的两种：%r；失效的 %d 种：%r。"
       "★ 失效的都有一个共同点：**它们在 capture 区里做了「需要真实执行」的事**"
       "（同步 / 查询 / 落 legacy 流）。这正是 B5a 原文的理由："
       "「because they do not represent items scheduled for execution」——"
       "被捕获的流那时**不是**一条能同步的执行队列。"
       "★ 这句话**不能**改写成「因为会破坏图的完整性」之类（那是 P4 编机制从句）。"
       % (okv, len(inval), [x[:26] for x in inval]))

    eq("C", "C.2 ★ 把转写里的 `cudaStreamNonBlocking` 去掉 ⇒ 变体 6 从合法变失效",
       (hw13_task1_graph(N, capture_extra=make_extra("legacy_command"), nonblocking=False)
        .end_status[0],),
       ("cudaErrorStreamCaptureInvalidated",),
       "★ 这一条是 A.5b 的**反向测试**：同一个插入动作、只改一个建流标志，"
       "返回码就变了。⇒ A.5b 的结论钉的**就是那个标志**，不是别的。")

    # C.3 capture 起始于 legacy 流
    def cap_on_legacy():
        r = Recorder()
        r.stream_create("s", nonblocking=True)
        r.stream_begin_capture(LEGACY, "global")
        r.kernel("k", LEGACY)
        return r.stream_end_capture(LEGACY)
    eq("C", "C.3 ★ 让 capture **起始于 legacy 流** ⇒ 违规（B3）",
       cap_on_legacy()[0], "cudaErrorStreamCaptureInvalidated",
       "出处 B3：「Stream capture can be used on any CUDA stream except "
       "cudaStreamLegacy (the “NULL stream”).」"
       "★ 与 B5b 是**两条不同的规则**：B3 管「能不能**起始**」，"
       "B5b 管「期间能不能**用**」。把它们合成一条会同时讲错两件事。")

    # C.4 不 join 回 origin
    def no_join():
        r = Recorder()
        for i in range(2):
            r.stream_create("streams[%d]" % i, nonblocking=True)
        r.stream_begin_capture("streams[0]", "global")
        r.kernel("a", "streams[0]")
        r.event_record("e1", "streams[0]")
        r.stream_wait_event("streams[1]", "e1")      # 把 streams[1] 拉进 capture
        r.kernel("c", "streams[1]")
        return r.stream_end_capture("streams[0]")     # ★ 没有从 streams[1] 等回 origin
    eq("C", "C.4 ★ 被拉进 capture 的流**没有 join 回 origin** ⇒ capture 整体失败（B4c）",
       no_join()[0], "cudaErrorStreamCaptureUnjoined",
       "出处 B4c：「Failure to rejoin to the origin stream will result in failure of the "
       "overall capture operation.」"
       "★ hw13 第 %d 行 `cudaStreamWaitEvent(streams[0], event2)` 就是**那个 join** —— "
       "它看起来像一条普通的 event 边，实际是 capture 合法的**必需动作**。"
       "把这一行删掉，capture 就整体失败。" % 132)

    # C.5 Relaxed 模式不是豁免
    def query_relaxed():
        r = Recorder()
        r.stream_create("s", nonblocking=True)
        r.stream_begin_capture("s", "relaxed")
        r.event_record("e1", "s")
        r.event_query("e1", mode="relaxed")
        r.kernel("k", "s")
        st = r.stream_end_capture("s")
        return st[0], len(r.violations)
    eq("C", "C.5 ★ `cudaEventQuery` 在 **Relaxed** 模式下**仍然违规**（B9c）",
       query_relaxed(), ("cudaErrorStreamCaptureInvalidated", 1),
       "出处 B9c（逐字）：「The local thread is **not** prohibited from potentially unsafe "
       "API calls. Note that the thread is **still prohibited** from API calls which "
       "necessarily conflict with stream capture, **for example, attempting "
       "cudaEventQuery on an event that was last recorded inside a capture sequence**.」"
       "★★ 它举的例子恰好就是 `cudaEventQuery`。⇒ **Relaxed 不是豁免**。"
       "把 capture mode 当成「安全总开关」是这一条最常见的误读。")

    # C.6 违规使**所有**关联图作废
    def two_graphs():
        r = Recorder()
        r.stream_create("s", nonblocking=True)
        r.stream_begin_capture("s", "global")
        r.capturing["other"] = "graph1"          # 同一个 context 里还有另一张图
        r.device_synchronize()
        return r.stream_end_capture("s")
    eq("C", "C.6 ★ 一次违规 ⇒ 返回 **NULL graph**，不是「部分图」（B6）",
       two_graphs(), ("cudaErrorStreamCaptureInvalidated", None),
       "出处 B6：「When an invalid operation is attempted during stream capture, "
       "**any associated capture graphs** are invalidated.」——「any associated」是复数。"
       "★ 所以 `CaptureState.end()` 在 invalidated 时返回 `(错误码, None)`；"
       "若改成返回一个「已经捕获到的部分图」，这一条立刻失败。")

    # C.7 B4b 前半：被 wait 的流自动进入 capture
    def auto_capture():
        r = Recorder()
        for i in range(2):
            r.stream_create("s%d" % i, nonblocking=True)
        r.stream_begin_capture("s0", "global")
        r.event_record("e", "s0")
        r.stream_wait_event("s1", "e")
        return "s1" in r.capturing
    eq("C", "C.7 ★ 被 wait 的流**自动进入** capture（B4b 前半句）", auto_capture(), True,
       "出处 B4b：「When a captured event is waited on by a stream, **it places the stream "
       "in capture mode if it is not already**.」"
       "★ 这一条与 B4c 是一对：**自动进入** ⇒ 也就**必须手动 join 回来**，"
       "否则整体失败（C.4）。两半句合起来才是 B4b 的完整含义。")

    # C.8 task1 vs task2 的循环结构不能混为一谈
    eq("C", "C.8 ★ task1 的定时循环**无同步**、task2 的发射循环**每趟都同步**",
       (T.calls("cudaStreamSynchronize", "timed_loop"),
        G.calls("cudaStreamSynchronize", "timed_loop"),
        C.calls("cudaStreamSynchronize", "launch_loop") / specs.HW13_TASK2_LOOP[2]),
       (0, 0, 1.0),
       "★ 这一条是 B.4 那个规格偏差的**可运行证据**：如果 task2 真的是「定时循环」，"
       "它就不该在循环体里同步；而它同步了（第 %d 行）—— "
       "因为 task2 里**根本没有计时**。"
       "⇒ 「8:1」这个结论**只对 task1 成立**，把它推广到 task2 是 P10"
       "（单一实践升格为通用规则）。" % (specs.HW13_TASK2_LOOP[0] + 3))

    # ---- A.9 ★★ 锚点的 `covered_by`：`note` 是**索引**，不是**唯一载体** ----
    #   ★ 订正轮裁定：`note` 保留，但必须机器可查。做法是 `covered_by` +
    #     「`note` 里出现的标签集合 == `covered_by`」这条**双向**约束。
    #     ⇒ 改 `covered_by` 里任何一个标签，这一条立刻失败。
    cb_bad = []
    per_anchor_local = {}
    cb_foreign = []
    for aid in COVERED_ANCHORS:
        a = specs.ANCHORS[aid]
        cbs = a.get("covered_by")
        if not isinstance(cbs, tuple) or not cbs or not all(
                isinstance(x, str) and x for x in cbs):
            cb_bad.append((aid, "不是非空字符串元组", repr(cbs)))
            continue
        note_tags = set(LABEL_RE.findall(a.get("note", "")))
        if note_tags != set(cbs):
            cb_bad.append((aid, "note 标签集 != covered_by",
                           (sorted(note_tags - set(cbs)), sorted(set(cbs) - note_tags))))
        per_anchor_local[aid] = [lab for lab in cbs if lab in SEEN_LABELS]
        cb_foreign += [(aid, lab) for lab in cbs if lab not in SEEN_LABELS]
    eq("A", "A.9 ★★ 本程序 %d 条锚点的 `covered_by` 非空，且与 `note` 里的标签集合"
       "**恰好相等**（双向）" % len(COVERED_ANCHORS), cb_bad, [],
       "格式：`(锚点, 说明, (note 多出来的, covered_by 多出来的))`。\n"
       "        ★★ 「`note` 是索引、不是唯一载体」的机械执行 —— "
       "**改 `covered_by` 里任何一个标签，这一条立刻失败**。\n"
       "        ★ 覆盖边界（如实说出）：程序没有「标签 ↔ 锚点」的语义知识，"
       "把标签换成**另一个真实存在且也恰好在 note 里**的标签是抓不到的；"
       "机器只保证 `note` 与 `covered_by` 两处**一致**，语义由 `note` 的措辞负责。")
    no_local = sorted(aid for aid in COVERED_ANCHORS if not per_anchor_local.get(aid))
    eq("A", "A.9b ★ 每条锚点至少有一个**本程序**的断言标签（%d 条共 %d 个本程序标签，"
       "%d 个跨程序标签）" % (len(COVERED_ANCHORS),
                             sum(len(v) for v in per_anchor_local.values()), len(cb_foreign)),
       no_local, [],
       "★ 「本程序标签」= 运行到这一行为止**真的出现过**的标签（`SEEN_LABELS`）——"
       "所以这条同时证明「`covered_by` 指向的是真实跑过的断言」，不是编的。\n"
       "        ★ 跨程序的那些：%r —— 它们属于 01 或 02，存在性由 01 的 A.11h 统一校验"
       "（01 用 `ast` 静态提取三个程序的标签池）。"
       % (sorted(cb_foreign),))

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    total = sum(v[0] for v in STATS.values())
    passed = sum(v[1] for v in STATS.values())
    print("汇总: A %d/%d · B %d/%d · C %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (STATS['A'][1], STATS['A'][0], STATS['B'][1], STATS['B'][0],
             STATS['C'][1], STATS['C'][0], total, passed, len(FAILS)))
    if SKIPS:
        print("跳过 %d 项（**不计入通过数**）：" % len(SKIPS))
        for s in SKIPS:
            print("  [SKIP] %s" % s)
    for f in FAILS:
        print("  FAILED: %s" % f)
    print()
    print("★ 限定语（本程序每一句结论都受它约束）：")
    print("  · **不是硬件** —— 不执行任何 CUDA，不执行任何 kernel。")
    print("  · **收益是「宿主 API 调用次数」，不是时间** —— README 第 74 行只说 small")
    print("    preformance increase，**没有倍数**；本机也没有 GPU 去测。")
    print("  · **node 数是 4 还是 8，本机无法判定** —— 只有三份间接证据吻合口径 B。")
    print("  · `cuda-learning` 的目标环境是 **Tesla T4 / sm_75（Turing）**：")
    print("    **没有** bf16 / TF32 / cp.async / TMA。引用它的代码时必须同时说明。")
    print("=" * 78)
    return 1 if FAILS else 0


def make_extra(key):
    """按 specs.CAPTURE_VARIANTS 的键构造「在 capture 区内插入一个动作」的函数。"""
    if key is None:
        return None
    if key == "stream_synchronize":
        return lambda r: r.stream_synchronize("streams[0]")
    if key == "device_synchronize":
        return lambda r: r.device_synchronize()
    if key == "event_synchronize":
        return lambda r: r.event_synchronize("event1")
    if key == "memcpy_sync":
        return lambda r: r.memcpy_sync("h_y", "D2H")
    if key == "legacy_command":
        return lambda r: r.legacy_command()
    raise ValueError("unknown variant key %r" % key)


if __name__ == "__main__":
    sys.exit(main())
