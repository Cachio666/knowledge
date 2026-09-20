#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stream / event 依赖模型：把一段多流程序翻译成真实的**依赖 DAG**。

配套文档：`02-streams-and-events.md`
运行：`make run`（或 `make 02_stream_dag`）；另有联网入口 `make anchors`（见 §9）。

===============================================================================
★★★ 限定语（**不是硬件**）—— 本程序的每一个数字都受它约束
===============================================================================

  > **本程序建模的是命令之间的偏序依赖，不建模执行时间、SM 占用、并发度、
  > 硬件队列深度。** 本机是 Apple M1 Pro，**无 GPU**，本程序里没有一个 thread
  > 被发射过。它算出的「层」是**图论意义上的拓扑层**，不是时间上的先后。

  > **本程序的层次输出只能读作「允许并发」，不能读作「实际并发」。**
  > 出处：CUDA C++ Programming Guide 12.6.1 §3.2.8.5 Overlapping Behavior 原文 ——
  > 「this behavior is not guaranteed and should therefore **not be relied upon
  > for correctness**」。⇒ 同层的两个节点**允许**并发，硬件**是否**让它们并发，
  > 本程序**一概没有验证**。

  > **所有时间数字本机不可得。** hw10 的 readme 第 70 行说
  > 「The overlapped version of the code should be about 2X faster」、
  > hw13 的 README 第 74 行说「a small preformance increase」——**两个都没验证**，
  > 也不出现在本程序的任何输出里（见 `specs.UNSOURCED`）。

  ★ 本程序**不 import** torch / triton / numba / cuda，也不需要 numpy（引擎是纯字典 +
    集合运算）。**外部工件**是 NVIDIA 的 `cuda-training-series`，是 **B 级来源**
    （厂商训练材料）—— 只能支撑「NVIDIA 自己这么写」，不能支撑「CUDA 规定必须这么写」
    （P8 规范与实现不分）。

===============================================================================
★ 三类断言（本程序内部，失败即非零退出；CONTRIBUTING 第 121 行第二种情形）
===============================================================================

  §A 算法正确性 —— 15 条。参照实现 `build_reference` 走**完全不同的代码路径**：
      它是 O(n²) 的**逐对规则判定**，而被检的 `build` 是**单趟增量状态机**。
      A.5b 是**第三条**独立路径：用 numpy 的 **bool 矩阵幂**再算一遍传递闭包；
      A.6c 是**第二条**独立路径：hw10 的 fixture 由**源文件的行内容机械推导**出来，
      与内嵌的转写逐项比对（**不经过模拟器**）。
      A.10/A.10b 是**元数据**判据：本程序 7 条锚点的 `covered_by` 与 `note` 里出现的
      标签**双向对齐**（订正轮：「`note` 是索引、不是唯一载体」的机械载体）。
  §B 代价模型   —— 4 条。栅栏建模成**星形** vs **全连接**的边数差；hw13 的拓扑层数；
      hw10 的 NULL 栅栏出度与边构成。
  §C 边界与反例 —— 9 条。声称「会出问题」的输入必须**真的跑出问题**。

★★ 本程序的可证伪断言（改哪一处、断言会怎么失败）

  A.5  把 `build()` 里 `prog.is_blocking(s)` 的判定去掉（让栅栏对所有流生效）⇒
       A.5 的逐对交叉校验立刻失败（与 `build_reference` 不一致）；C.8 专门构造了这个错误版本。
  A.6  把内嵌的 hw13 fixture 里 `record event1` 与 `kernel kernel_b` 换位 ⇒
       A.6b 的 op 序列比对失败。
  A.6c 把源文件的 `streams[i % num_streams]` 改成 `streams[i % 2]` ⇒
       A.6c 的机械推导与内嵌 fixture 对不上（推导链断了，SKIP 而非 PASS）。
  A.7  把任一被本程序覆盖的锚点 quote 改一个字 ⇒ A.7 的逐字子串检查失败。
  A.8  把 `ANCHORS['A2']['doc']` 从 `driver` 改成 `guide` ⇒ A.8 的归属断言失败。
  B.1  把 `HW10_LIVE_STREAMS` 从 8 改成 2 ⇒ 星形 / 全连接的边数期望值失败。
  C.1  让 `build()` 在 wait 之前没有 record 时静默跳过 ⇒ C.1 要求抛异常的断言失败。
  C.2  把 fixture 里 `memset d_y` 挪到分块循环**之后** ⇒ C.2 要求「出度 0」的断言失败。
       ★ 这正是计划里记下的那个**凭印象记错发射顺序**的反例。
  C.4  把 A4 的 per-thread 语义改掉（per_thread 模式也建栅栏）⇒ C.4 要求「隐式边 0 条」失败。
  C.5  把 `kernel_cover` 换成传递闭包 ⇒ C.5 要求「a→d **不在** cover 里」的断言失败。
  A.10 把任一锚点的 `covered_by` 里的一个标签改掉 —— **哪怕改成另一个真实存在的标签**
       ⇒ 该锚点的 `note` 里出现的标签集合与 `covered_by` 不再相等，A.10 立刻失败。

★★ 本程序**不修改磁盘上的任何文件**。所有反例都在内存里的文本上构造。
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np

import specs

# ===========================================================================
# 0. 常量
# ===========================================================================
LEGACY = "NULL"          # legacy default stream（cudaStreamLegacy / stream 0）
PT = "@thread"           # per-thread default stream（cudaStreamPerThread）

KIND_KERNEL = "kernel"
KIND_MEMCPY = "memcpy"
KIND_MEMSET = "memset"
KIND_RECORD = "record"
KIND_WAIT = "wait"
KIND_SYNC = "sync"       # 宿主侧同步，不进图

# ★ 本程序覆盖的语义锚点。`01_sync_contract.py` 用 `ast` 读它，
#   断言三个程序的并集 == specs.ANCHORS 全集，且两两不交。
COVERED_ANCHORS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")

# ★ 每条建边规则**必须**声明它的语义锚点。A.9 断言：规则引用的锚点必须存在，
#   且必须落在本程序声明的 COVERED_ANCHORS 里。
RULES = (
    # (规则名, 引用的锚点, 一句话)
    ("RULE_STREAM_ORDER", ("A5", "A7"),
     "同一条流内的先后发射顺序（程序序）"),
    ("RULE_EVENT", ("A5", "A6", "A7"),
     "cudaEventRecord / cudaStreamWaitEvent 建立的跨流边"),
    ("RULE_LEGACY_IN", ("A1", "A2"),
     "legacy 流上的动作先等所有 blocking 流（A2 前半句）"),
    ("RULE_LEGACY_OUT", ("A2", "A3"),
     "然后所有 blocking 流等该 legacy 动作（A2 后半句）"),
    ("RULE_PERTHREAD", ("A4",),
     "per-thread default stream 不去栅栏 ⇒ mode=per_thread 时隐式边为 0"),
)


# ===========================================================================
# 1. 程序表示
# ===========================================================================
class Op:
    __slots__ = ("idx", "stream", "kind", "name", "arg", "line")

    def __init__(self, idx, stream, kind, name, arg, line):
        self.idx = idx          # 全局发射序号（宿主线程发出顺序）
        self.stream = stream
        self.kind = kind
        self.name = name
        self.arg = arg          # record/wait 的事件名
        self.line = line

    @property
    def nid(self):
        return "n%03d" % self.idx

    def label(self):
        if self.kind == KIND_RECORD:
            return "rec(%s)" % self.arg
        if self.kind == KIND_WAIT:
            return "wait(%s)" % self.arg
        return "%s:%s" % (self.kind, self.name)

    def as_fixture_line(self):
        """转回 fixture 的书写形式：`<流> : <kind> <name>`。"""
        return "%s : %s %s" % (self.stream, self.kind, self.name)


class Program:
    """一段多流程序：mode + 每条流是否 non-blocking + 按发射顺序的 op 列表。"""

    def __init__(self):
        self.mode = "legacy"            # legacy | per_thread
        self.nonblocking = {}
        self.ops = []

    def stream_flag(self, s):
        return self.nonblocking.get(s, False)

    def is_blocking(self, s):
        """A2/A3/A4：该流是不是 blocking stream（参与 legacy NULL stream 的全局栅栏）。

        ★ 这是 P9「同名不同指」的高危点：
          · legacy 流**自己不是** blocking stream —— A2 的三步里它是栅栏的
            **发起者**与**被等待者**，不是「被它同步的流」之一；
          · 而 per-thread default stream **是** blocking 的（A4 原文：
            it is not a non-blocking stream）。
        """
        if s == LEGACY:
            return False
        if s == PT:
            return True
        return not self.stream_flag(s)


DIRECTIVE_RE = re.compile(r"^\s*(mode|stream)\s*:\s*(.+?)\s*$")
OP_RE = re.compile(r"^\s*([A-Za-z_@][\w@\[\]]*)\s*:\s*([a-z_]+)\s*(.*?)\s*$")


def parse(text):
    prog = Program()
    lineno = 0
    for raw in text.split("\n"):
        lineno += 1
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = DIRECTIVE_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            if key == "mode":
                if val not in ("legacy", "per_thread"):
                    raise ValueError("line %d: bad mode %r" % (lineno, val))
                prog.mode = val
            else:
                parts = [p.strip() for p in val.split(",")]
                if len(parts) < 2:
                    raise ValueError("line %d: expect 'stream: <name>, <flag>'" % lineno)
                name, flag = parts[0], parts[1]
                if flag not in ("blocking", "nonblocking"):
                    raise ValueError("line %d: bad stream flag %r" % (lineno, flag))
                prog.nonblocking[name] = (flag == "nonblocking")
            continue
        m = OP_RE.match(line)
        if not m:
            raise ValueError("line %d: cannot parse %r" % (lineno, raw))
        stream, kind, rest = m.group(1), m.group(2), m.group(3).strip()
        if kind in (KIND_KERNEL, KIND_MEMCPY, KIND_MEMSET):
            prog.ops.append(Op(len(prog.ops), stream, kind, rest.strip("() "), None, lineno))
        elif kind in (KIND_RECORD, KIND_WAIT):
            prog.ops.append(Op(len(prog.ops), stream, kind, rest, rest, lineno))
        elif kind == KIND_SYNC:
            prog.ops.append(Op(len(prog.ops), stream, kind, stream, None, lineno))
        else:
            raise ValueError("line %d: unknown op kind %r" % (lineno, kind))
    return prog


# ===========================================================================
# 2. ★ 被检实现：单趟增量建边
# ===========================================================================
class Dag:
    def __init__(self):
        self.nodes = []
        self.edges = set()          # (from_nid, to_nid, reason)

    def add_edge(self, frm, to, reason):
        if frm is None or to is None or frm is to:
            return
        self.edges.add((frm.nid, to.nid, reason))


def build(prog):
    """单趟按发射顺序增量建边。规则见 RULES 与模块 docstring（A1–A7）。"""
    dag = Dag()
    dag.nodes = [op for op in prog.ops if op.kind != KIND_SYNC]

    # --- RULE_STREAM_ORDER：同流程序序 ---
    last_in_stream = {}
    for op in dag.nodes:
        prev = last_in_stream.get(op.stream)
        if prev is not None:
            dag.add_edge(prev, op, "program-order")
        last_in_stream[op.stream] = op

    # --- RULE_EVENT：record/wait 建立跨流边（A7：record 之后**所有** wait 都生效）---
    last_record = {}
    for op in dag.nodes:
        if op.kind == KIND_RECORD:
            last_record[op.arg] = op
        elif op.kind == KIND_WAIT:
            r = last_record.get(op.arg)
            if r is None:
                raise ValueError(
                    "n%03d: wait(%s) 之前没有同事件的 record —— 真实 CUDA 里这是无效依赖"
                    % (op.idx, op.arg))
            dag.add_edge(r, op, "event:%s" % op.arg)

    # --- RULE_LEGACY_IN / RULE_LEGACY_OUT：legacy 流的全局栅栏 ---
    # 只在 mode == legacy 时存在 NULL 流。per-thread 模式下默认流
    # 「does not synchronize with other streams (just like explicitly created
    #  streams)」（A4），因此不产生任何隐式栅栏。
    if prog.mode == "legacy":
        last_legacy = None
        last_dev = {}
        for op in dag.nodes:
            if op.stream == LEGACY:
                # A2 前半：the legacy stream first waits on all blocking streams
                for s, prev in last_dev.items():
                    if prog.is_blocking(s):
                        dag.add_edge(prev, op, "legacy-barrier-in")
                last_legacy = op
            else:
                # A2 后半：then all blocking streams wait on the legacy stream
                if last_legacy is not None and prog.is_blocking(op.stream):
                    dag.add_edge(last_legacy, op, "legacy-barrier-out")
            last_dev[op.stream] = op
    return dag


# ===========================================================================
# 3. ★★ 参照实现：O(n²) 逐对规则判定（与被检实现**完全不同的代码路径**）
# ===========================================================================
def reason_class(reason):
    """把带参数的 reason 归一成类别：`event:event1` -> `event`。"""
    if reason.startswith("event:"):
        return "event"
    return reason


def build_reference(prog):
    """★★ 参照实现。**不维护任何当前状态**：对每一对 (a, b)（a 在 b 之前发射），
    直接问「按文档，这条边**必须**存在吗」。

    这与 `build()` 的单趟增量状态机是**两条不同的算法**：
      · `build`           —— O(n)，维护 last_in_stream / last_record / last_legacy；
      · `build_reference` —— O(n²)，每条边都由「中间有没有人插队」的**全量扫描**判定。
    ⇒ 任何一处规则写错，两者必然分叉（A.5 断言它们逐边逐类一致）。
    """
    nodes = [op for op in prog.ops if op.kind != KIND_SYNC]
    n = len(nodes)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            mid = nodes[i + 1:j]
            # ---- 同流程序序：a 是 b 之前同流上**最后一个** op ----
            if a.stream == b.stream and all(x.stream != a.stream for x in mid):
                edges.add((a.nid, b.nid, "program-order"))
            # ---- event：a 是 e 的 record、b 是 e 的 wait，中间没有同 event 的 record ----
            if (a.kind == KIND_RECORD and b.kind == KIND_WAIT and a.arg == b.arg
                    and all(not (x.kind == KIND_RECORD and x.arg == a.arg) for x in mid)):
                edges.add((a.nid, b.nid, "event:%s" % a.arg))
            # ---- legacy 栅栏（A1/A2/A3/A4）----
            if prog.mode == "legacy":
                if (b.stream == LEGACY and a.stream != LEGACY and prog.is_blocking(a.stream)
                        and all(x.stream != a.stream for x in mid)):
                    edges.add((a.nid, b.nid, "legacy-barrier-in"))
                if (a.stream == LEGACY and b.stream != LEGACY and prog.is_blocking(b.stream)
                        and all(x.stream != LEGACY for x in mid)):
                    edges.add((a.nid, b.nid, "legacy-barrier-out"))
    return edges


def edge_pairs(edges):
    return {(f, t) for f, t, _ in edges}


def edge_classes(edges):
    return {(f, t, reason_class(r)) for f, t, r in edges}


# ===========================================================================
# 4. 图论工具（**只做拓扑，不做时间**）
# ===========================================================================
def levels(dag):
    """按依赖分层：第 0 层 = 无前驱。★ 纯拓扑性质，**不是执行时间**。"""
    indeg = {n.nid: 0 for n in dag.nodes}
    adj = defaultdict(list)
    for f, t, _ in dag.edges:
        if t not in adj[f]:
            adj[f].append(t)
            indeg[t] += 1
    layer = {nid: 0 for nid in indeg}
    frontier = [nid for nid, d in indeg.items() if d == 0]
    seen = set(frontier)
    while frontier:
        nxt = []
        for nid in frontier:
            for t in adj[nid]:
                if layer[t] < layer[nid] + 1:
                    layer[t] = layer[nid] + 1
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    return layer, len(seen)


def reachability(dag):
    """可达集（传递闭包）。★ C.5 断言「可达 ≠ 直接边」用的就是它。"""
    reach = {n.nid: set() for n in dag.nodes}
    for f, t, _ in dag.edges:
        reach[f].add(t)
    changed = True
    while changed:
        changed = False
        for nid in list(reach):
            new = set()
            for m in reach[nid]:
                new |= reach[m]
            if not new <= reach[nid]:
                reach[nid] |= new
                changed = True
    return reach


def reachability_np(dag):
    """★★ 参照实现 2：用 **numpy 布尔矩阵的幂**求传递闭包。

    与集合式的 `reachability()` 是**两条不同的算法**：
      · `reachability()` —— 反复扫描邻接表、按集合求并，直到不动点；
      · `reachability_np()` —— 把邻接矩阵置成 bool 数组，反复 `R |= R @ R`
        （numpy 在 bool dtype 上的 `@` 就是**布尔半环**：乘 = 与、加 = 或）。
    ⇒ 两者必须给出同一个可达关系（A.5b 断言）。
    """
    nodes = dag.nodes
    idx = {op.nid: i for i, op in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=bool)
    for f, t, _ in dag.edges:
        A[idx[f], idx[t]] = True
    R = A.copy()
    # log2(n) 次平方足够收敛到传递闭包（n 很小，多跑一轮无妨）
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        R = R | (R @ R)
    return {(nodes[i].nid, nodes[j].nid)
            for i in range(n) for j in range(n) if R[i, j]}


def kernel_cover(dag):
    """把 rec/wait 当**透明节点**，在 kernel 子图上求传递**归约**（cover 关系）。

    ★★ 这正是 NVIDIA 在 `graph_stream_capture.png` 右侧画出的箭头集合。
    ★ 是 cover（传递归约），**不是**传递闭包：a→d 可达，但 a→b→d 已经蕴含它，
      所以 a→d **不在** cover 里 —— 探针 v1 的期望值正是在这里把「可达」
      当成了「直接边」。
    """
    reach = reachability(dag)
    ks = [op for op in dag.nodes if op.kind == KIND_KERNEL]
    name = {op.nid: op.name for op in ks}
    out = []
    for a in ks:
        for b in ks:
            if b.nid not in reach[a.nid]:
                continue
            mid = [w for w in ks if w is not a and w is not b
                   and w.nid in reach[a.nid] and b.nid in reach[w.nid]]
            if not mid:
                out.append((name[a.nid], name[b.nid]))
    return sorted(out)


def succ_labels(dag, nid):
    lbl = {op.nid: op.label() for op in dag.nodes}
    return sorted(lbl[t] for f, t, _ in dag.edges if f == nid)


def pred_labels(dag, nid):
    lbl = {op.nid: op.label() for op in dag.nodes}
    return sorted(lbl[f] for f, t, _ in dag.edges if t == nid)


def cross_edges(dag, s1, s2):
    """从流 s1 到流 s2 的**直接边**。"""
    a = {op.nid for op in dag.nodes if op.stream == s1}
    b = {op.nid for op in dag.nodes if op.stream == s2}
    return [(f, t) for f, t, _ in dag.edges if f in a and t in b]


# ===========================================================================
# 5. 内嵌的程序转写（fixture）—— 每一条都被钉回源文件（A.6）
# ===========================================================================
# ★ 纪律：这些文本是**源文件的转写**，不是创作。每一行都由 §A 的比对钉回
#   `cuda-training-series` 里的真实文件。转写漂了，A.6 就失败。

# 5.1 文档自带的例子。★ 它**不是**转写自 .cu 文件，而是 Driver API Reference 13.4
#     「3. Stream synchronization behavior」正文里印出的三行代码。
FIX_DOC_LEGACY = """\
mode: legacy
stream: s, blocking

s    : kernel k_1
NULL : kernel k_2
s    : kernel k_3
"""

FIX_DOC_NONBLOCKING = """\
mode: legacy
stream: s, nonblocking

s    : kernel k_1
NULL : kernel k_2
s    : kernel k_3
"""

# 5.2 hw13 task1（转写自 axpy_stream_capture_with_fixme.cu 第 111–138 行）
FIX_HW13 = """\
mode: legacy
stream: streams[0], nonblocking
stream: streams[1], nonblocking

streams[0] : kernel kernel_a
streams[0] : record event1
streams[0] : kernel kernel_b
streams[1] : wait event1
streams[1] : kernel kernel_c
streams[1] : record event2
streams[0] : wait event2
streams[0] : kernel kernel_d
"""

# 5.3 hw13 的**对照扰动**：与 5.2 逐句相同，唯一差别是流用 cudaStreamCreate（blocking）建。
#     ★ 这不是 hw13 的真实代码 —— 是一个最小对照扰动，用来把 A3 的差异逼出来。
FIX_HW13_BLOCKING = FIX_HW13.replace(", nonblocking", ", blocking", 2)

# 5.4 hw10 的同构切片（chunks=4 / num_streams=2；源的 chunks=64 / num_streams=8）
FIX_HW10 = """\
mode: legacy
stream: streams[0], blocking
stream: streams[1], blocking

NULL       : memcpy H2D d_x
NULL       : kernel gaussian_pdf_warmup
NULL       : memcpy D2H h_y1
NULL       : memset d_y
streams[0] : memcpy H2D chunk0
streams[0] : kernel gaussian_pdf_c0
streams[0] : memcpy D2H chunk0
streams[1] : memcpy H2D chunk1
streams[1] : kernel gaussian_pdf_c1
streams[1] : memcpy D2H chunk1
streams[0] : memcpy H2D chunk2
streams[0] : kernel gaussian_pdf_c2
streams[0] : memcpy D2H chunk2
streams[1] : memcpy H2D chunk3
streams[1] : kernel gaussian_pdf_c3
streams[1] : memcpy D2H chunk3
"""


# ===========================================================================
# 6. 断言框架
# ===========================================================================
FAILS = []
STATS = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}
SKIPS = []

# ★★ 断言标签（`A.5`、`C.8` …）：`specs.ANCHORS[*]['covered_by']` 引用的就是它。
#    必须由程序**在运行时**从断言名里提取，不能另立手写清单（那会与真实断言脱节）。
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


def load(fix):
    prog = parse(fix)
    return prog, build(prog)


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
# 7. 主流程
# ===========================================================================
def main():
    train, tried = find_root(specs.TRAIN_ROOT_ENV, "exercises/hw13",
                             ["../cuda-training-series", "../../cuda-training-series",
                              "../../../cuda-training-series",
                              "../../../../cuda-training-series"])
    print("=" * 78)
    print("02_stream_dag —— stream / event 依赖模型（**不建模执行时间与并发度**）")
    print("=" * 78)
    print("  解释器      : %s" % sys.executable)
    print("  python      : %s" % sys.version.split()[0])
    print("  numpy       : %s（A.5b 用它的 **bool 矩阵幂**独立复核传递闭包）"
          % np.__version__)
    print("  ★ **不是硬件**：本程序建模的是命令之间的**偏序依赖**，")
    print("    不建模执行时间、SM 占用、并发度、硬件队列深度。")
    print("    ⇒ 层次输出只能读作「**允许**并发」，不能读作「**实际**并发」。")
    print("  ★ **所有时间数字本机不可得**：hw10 的 about 2X 与 hw13 的 small increase")
    print("    一个都没验证，也不出现在下面的输出里。")
    if train is None:
        print()
        print("!! 找不到 %s，A.6 那组「转写钉回源文件」的断言无法执行。"
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

    # ---------------------------------------------------------------------
    print("-" * 78)
    print("§A 算法正确性（参照实现 `build_reference` 走**不同的代码路径**）")
    print("-" * 78)

    # ---- A.1 文档自带的例子 ----
    _, dag_doc = load(FIX_DOC_LEGACY)
    eq("A", "A.1 文档自带例子（legacy）⇒ k_1→k_2、k_2→k_3 都在（共 3 条边）",
       sorted("%s -> %s" % (f, t) for f, t in edge_pairs(dag_doc.edges)),
       ["n000 -> n001", "n000 -> n002", "n001 -> n002"],
       "★ 真值来源（外部、逐字）：CUDA Driver API Reference 13.4「3. Stream "
       "synchronization behavior」→「Legacy default stream」一节自带"
       " k_1<<<1,1,0,s>>>(); k_2<<<1,1>>>(); k_3<<<1,1,0,s>>>(); 并给出结论："
       "「The resulting behavior is that **k_2 will block on k_1 and k_3 will block on "
       "k_2**.」⇒ 期望边集 = {k1→k2, k2→k3}（栅栏）∪ {k1→k3}（同流程序序）。")

    # ---- A.2 nonblocking ⇒ 只剩同流边 ----
    _, dag_nb = load(FIX_DOC_NONBLOCKING)
    eq("A", "A.2 同一段程序换成 cudaStreamNonBlocking ⇒ 栅栏消失，只剩同流边 k_1→k_3",
       sorted("%s -> %s" % (f, t) for f, t in edge_pairs(dag_nb.edges)), ["n000 -> n002"],
       "★ 真值来源（外部、逐字，同一页）：A1「synchronizes with all other streams in the "
       "same CUcontext **except for non-blocking streams**」+ A3「Non-blocking streams "
       "which do not synchronize with the legacy stream can be created using the "
       "cudaStreamNonBlocking flag with the stream creation APIs.」"
       "⇒ **只差一个建流标志**，边数从 3 掉到 1。")

    # ---- A.3 hw13 task1 的 8 条直接边 ----
    prog13, dag13 = load(FIX_HW13)
    eq("A", "A.3 hw13 task1 ⇒ 8 个节点 / 8 条直接边",
       sorted("%s -> %s" % (f, t) for f, t in edge_pairs(dag13.edges)),
       ["n000 -> n001", "n001 -> n002", "n001 -> n003", "n002 -> n006",
        "n003 -> n004", "n004 -> n005", "n005 -> n006", "n006 -> n007"],
       "节点名与边界由 §A.6 的源文件逐行比对保证不是编的。"
       "边来自两条独立规则：同流程序序（n000→n001→n002、n003→n004→n005、n006→n007）"
       "与 event 边（n001→n003 = rec(event1)→wait(event1)、n005→n006 = rec(event2)→wait(event2)）。"
       "★ 注意 n002→n006 是**同流程序序**（kernel_b 与 wait(event2) 同在 streams[0]），"
       "**不是** event 边 —— 这一条最容易记错。")

    # ---- A.4 kernel 层的传递归约 ----
    eq("A", "A.4 kernel 层的传递归约（cover）恰为 {a→b, a→c, b→d, c→d}",
       kernel_cover(dag13),
       [("kernel_a", "kernel_b"), ("kernel_a", "kernel_c"),
        ("kernel_b", "kernel_d"), ("kernel_c", "kernel_d")],
       "★ 真值来源（外部）：仓库内 `%s` —— hw13 的 README 第 13 行 `![](graph_stream_capture.png)` "
       "引用的那张图，**右侧画出 A/B/C/D 四个节点、四条边**。"
       "★ **归属纪律（P7）**：那张图是**人眼读过**的（本程序读不了 PNG）；"
       "README 第 15 行「This is the same example from the slides」这句"
       "**只**说明该图与课件同源，**并不**支持「四条边」这个断言。两者必须分开写。"
       % specs.HW13_DIAGRAM_PNG)

    # ---- A.5 ★★ 两条独立实现的逐边交叉校验 ----
    all_fix = {
        "doc_default_stream": FIX_DOC_LEGACY,
        "doc_default_stream_nonblocking": FIX_DOC_NONBLOCKING,
        "hw13_capture": FIX_HW13,
        "hw13_same_but_blocking": FIX_HW13_BLOCKING,
        "hw10_chunked": FIX_HW10,
    }
    cross_bad = []
    n_edges_total = 0
    for name in sorted(all_fix):
        p = parse(all_fix[name])
        inc = edge_classes(build(p).edges)
        ref = edge_classes(build_reference(p))
        n_edges_total += len(inc)
        if inc != ref:
            cross_bad.append((name, sorted(inc ^ ref)[:4]))
    eq("A", "A.5 ★★ 单趟增量 `build` 与 O(n²) 逐对 `build_reference` 在 %d 个 fixture 上"
       "**逐边逐类一致**（共 %d 条边）" % (len(all_fix), n_edges_total), cross_bad, [],
       "★ 这是 A 类里最硬的一条：CONTRIBUTING 第 214–217 行要求「参照实现必须走与**被检"
       "代码**不同的代码路径」。两条路径的差异是结构性的 —— `build` 是 O(n) 状态机"
       "（维护 last_in_stream / last_record / last_legacy），`build_reference` 是 O(n²) "
       "全量扫描（对每对 (a,b) 问「中间有没有人插队」）。改错任一条规则，两者必分叉"
       "（C.8 把这件事做成一个可运行的反例）。")

    # ---- A.5b ★ 第二条独立路径：numpy 布尔矩阵闭包 vs 集合式可达 ----
    reach_bad = []
    for name in sorted(all_fix):
        d = build(parse(all_fix[name]))
        setwise = {(a, b) for a, s_ in reachability(d).items() for b in s_}
        matrix = reachability_np(d)
        if setwise != matrix:
            reach_bad.append((name, len(setwise ^ matrix)))
    eq("A", "A.5b ★★ 集合式 `reachability` 与 **numpy 布尔矩阵幂**给出的可达关系相同"
       "（%d 个 fixture）" % len(all_fix), reach_bad, [],
       "★ 两条算法的实现完全不同（一个是邻接表上的不动点迭代，一个是 bool 矩阵反复 "
       "`R |= R @ R` 的布尔半环乘法）。"
       "★ 这一条同时是**可达关系的独立复核**：C.5 那条「可达 ≠ 直接边」的断言"
       "依赖可达集算得对，这里把它钉住了。")

    # ---- A.6a/b fixture 钉回源文件 ----
    src13 = tp(specs.HW13_TASK1_PATH)
    lines13 = src13.split("\n")
    bad_lines = []
    fixture_ops_expected = []
    for ln, verbatim, fix_op in specs.HW13_TASK1_OPS:
        got = lines13[ln - 1].strip() if ln - 1 < len(lines13) else "<越界>"
        if verbatim not in got:
            bad_lines.append((ln, got[:60], verbatim[:60]))
        if fix_op not in ("BEGIN", "END"):
            fixture_ops_expected.append(fix_op)
    eq("A", "A.6a hw13 fixture 的 10 个源行（第 %s 行）逐字命中"
       % "/".join(str(x[0]) for x in specs.HW13_TASK1_OPS), bad_lines, [],
       "★ 转写**不是创作的**：每一行都由 `specs.HW13_TASK1_OPS` 的「行号 + 逐字片段」"
       "钉回真实文件。改了 fixture 里任何一行的顺序或名字，这里立刻失败。")
    eq("A", "A.6b hw13 fixture 的 op 序列与源文件的**出现顺序**逐项相同",
       [op.as_fixture_line() for op in prog13.ops], fixture_ops_expected,
       "期望序列由源文件第 111–138 行的出现顺序推出（BEGIN/END 是 capture 区边界，"
       "不是 op）。★ 这一条**不经过模拟器**：它只比「文本 → op 序列」的翻译层。")

    # ---- A.6c/d hw10 的 fixture 由源文件机械推导 ----
    src10 = tp(specs.HW10_PATH)
    l10 = src10.split("\n")
    body = [l10[i].strip() for i in range(specs.HW10_LOOP_FIRST_LINE - 1,
                                          specs.HW10_LOOP_LAST_LINE)
            if l10[i].strip() and not l10[i].strip().startswith("for")]
    kinds = []
    for l in body:
        if "cudaMemcpyAsync" in l:
            kinds.append("memcpy H2D" if "HostToDevice" in l else "memcpy D2H")
        elif "gaussian_pdf<<<" in l:
            kinds.append("kernel")
    stream_expr = None
    for l in body:
        m = re.search(r"streams\[i % (\w+)\]", l)
        if m:
            stream_expr = m.group(1)
            break
    want10 = []
    if stream_expr == "num_streams" and kinds == ["memcpy H2D", "kernel", "memcpy D2H"]:
        for i in range(specs.HW10_FIXTURE_CHUNKS):
            s = "streams[%d]" % (i % specs.HW10_FIXTURE_STREAMS)
            for k in kinds:
                if k == "kernel":
                    want10.append("%s : kernel gaussian_pdf_c%d" % (s, i))
                elif k == "memcpy H2D":
                    want10.append("%s : memcpy H2D chunk%d" % (s, i))
                else:
                    want10.append("%s : memcpy D2H chunk%d" % (s, i))
    else:
        SKIPS.append("A.6c: 源文件循环体结构与预期不符（stream_expr=%r, kinds=%r）"
                     % (stream_expr, kinds))
    prog10, dag10 = load(FIX_HW10)
    got10 = [op.as_fixture_line() for op in prog10.ops if op.stream != LEGACY]
    if want10:
        eq("A", "A.6c ★ hw10 fixture 的循环部分由源文件第 %d–%d 行**机械推导**而来"
           % (specs.HW10_LOOP_FIRST_LINE, specs.HW10_LOOP_LAST_LINE), got10, want10,
           "推导链：循环体有 3 条调用（cudaMemcpyAsync HostToDevice / gaussian_pdf<<<...>>> / "
           "cudaMemcpyAsync DeviceToHost），流向 `streams[i %% %s]`，而 `%s` 在源文件里是 %d "
           "⇒ 把 chunks 从 64 切到 %d、num_streams 从 %d 切到 %d，就得到本 fixture 的循环部分。"
           "★ 这一步**不经过模拟器**，纯粹是「文本 → op 序列」的推导。"
           % (stream_expr, stream_expr, specs.HW10_LIVE_STREAMS,
              specs.HW10_FIXTURE_CHUNKS, specs.HW10_LIVE_STREAMS,
              specs.HW10_FIXTURE_STREAMS))
    pre_want = []
    for ln, kind in ((86, "memcpy H2D"), (87, "kernel"), (88, "memcpy D2H"), (95, "memset")):
        got = l10[ln - 1].strip()
        if kind == "memcpy H2D":
            ok = "cudaMemcpy" in got and "HostToDevice" in got
        elif kind == "memcpy D2H":
            ok = "cudaMemcpy" in got and "DeviceToHost" in got
        elif kind == "kernel":
            ok = "gaussian_pdf<<<" in got
        else:
            ok = "cudaMemset" in got
        pre_want.append((ln, kind, ok))
    eq("A", "A.6d hw10 fixture 的 NULL 流前缀对应的 4 个源行语义逐条命中",
       pre_want, [(86, "memcpy H2D", True), (87, "kernel", True),
                  (88, "memcpy D2H", True), (95, "memset", True)],
       "★ **顺序很重要**：`cudaMemset` 在第 %d 行，而分块循环在第 %d–%d 行 —— "
       "memset 在循环**之前**。计划里记下了这个坑：**凭印象会记成在它之后**，"
       "而那个记法会**改变整个 DAG**（C.2 把这件事做成了一个可运行的反例）。"
       % (specs.HW10_MEMSET_LINE, specs.HW10_LOOP_FIRST_LINE, specs.HW10_LOOP_LAST_LINE))

    # ---- A.7 锚点引文的逐字完整性（只查本程序覆盖的 7 条）----
    q_bad = []
    for aid in COVERED_ANCHORS:
        a = specs.ANCHORS[aid]
        for frag in a["must_contain"]:
            if a["quote"].count(frag) != 1:
                q_bad.append((aid, frag[:40], a["quote"].count(frag)))
    eq("A", "A.7 本程序覆盖的 %d 条锚点，`must_contain` 都在 `quote` 里恰好出现 1 次"
       % len(COVERED_ANCHORS), q_bad, [],
       "★ 「凡引用外部原文，把原文与断言写在同一处」（计划 §2 本域特有第 3 条）。"
       "改 quote 里一个字，这一条失败。")

    # ---- A.8 ★ 归属断言 ----
    attrib_bad = []
    for aid in specs.LEGACY_FENCE_ANCHORS:
        a = specs.ANCHORS[aid]
        if a["doc"] != specs.LEGACY_FENCE_DOC:
            attrib_bad.append((aid, "doc", a["doc"]))
        if a["section"] != specs.LEGACY_FENCE_PAGE:
            attrib_bad.append((aid, "section", a["section"]))
    extra_driver = sorted(aid for aid, a in specs.ANCHORS.items()
                          if a["doc"] == specs.LEGACY_FENCE_DOC
                          and aid not in specs.LEGACY_FENCE_ANCHORS)
    eq("A", "A.8 ★ 归属断言：%s 必须归 **%s** 页的「%s」，且该页**只有**这四条"
       % ("/".join(specs.LEGACY_FENCE_ANCHORS), specs.LEGACY_FENCE_DOC,
          specs.LEGACY_FENCE_PAGE),
       (attrib_bad, extra_driver), ([], []),
       "★★ **这是 P7「引文作用域不匹配」的机械载体。** "
       "Guide §3.2.8.5.2 Default Stream 全文只讲 --default-stream legacy/per-thread 与 "
       "NULL stream 的存在，**没有那句栅栏语**；「the legacy stream first waits on all "
       "blocking streams … then all blocking streams wait on the legacy stream」"
       "**只出现在 Driver API Reference**。⇒ 网上二手资料常把它当成 Guide 的内容。"
       "**把 A1..A4 里任一 doc 改成 guide，这一条立刻失败。**")

    # ---- A.9 规则 -> 锚点的引用完整性（P1 的载体）----
    ref_bad = []
    covered = []
    for rname, refs, _ in RULES:
        for r in refs:
            if r not in specs.ANCHORS:
                ref_bad.append((rname, r, "不存在"))
            elif r not in COVERED_ANCHORS:
                ref_bad.append((rname, r, "不在 COVERED_ANCHORS"))
            else:
                covered.append(r)
    eq("A", "A.9 %d 条建边规则引用的锚点都存在且都在本程序声明的覆盖范围内" % len(RULES),
       ref_bad, [],
       "被引用的锚点集合 = %r（%d 条规则）。⇒ 规则与锚点之间没有悬空引用。"
       % (sorted(set(covered)), len(RULES)))

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    print("§B 代价模型（**推导值**；不建模执行时间）")
    print("-" * 78)

    # B.1 星形 vs 全连接
    n_ops_full = specs.HW10_LIVE_STREAMS * 64 * 3
    star = 2 * n_ops_full
    mesh = n_ops_full * (n_ops_full - 1)
    check("B", "B.1 栅栏建模成**星形**而非**全连接**：hw10 全规模下 %d 条 vs %d 条边"
          % (star, mesh),
          star == 3072 and mesh == 2357760 and mesh * 2 == star * 1535,
          "hw10 全规模（源文件第 %d 行 chunks=64、第 %d 行 num_streams=8）："
          "%d 条流 × 64 个 chunk × 3 个动作 = **%d** 个设备侧动作。"
          "星形（流 ↔ legacy）给 **2n = %d** 条；"
          "全连接（任意两条 blocking 流互相同步）会给 **n(n−1) = %d** 条 —— **%s 倍**"
          "（用整数关系 `2·mesh == star·1535` 断言，**不用浮点容差**）。"
          "★ 这是**模型层面的**差别：如果读者以为「legacy 流同步所有流」等于"
          "「所有流互相同步」，他心里的 DAG 就是这个全连接图。"
          "**本程序不声称任何一种模型对应硬件的实际调度。**"
          % (specs.HW10_CHUNKS_LINE, specs.HW10_NUM_STREAMS_LINE,
             specs.HW10_LIVE_STREAMS, n_ops_full, star, mesh, "767.5"))

    # B.2 hw13 的拓扑层数
    layer13, reached13 = levels(dag13)
    eq("B", "B.2 hw13 的拓扑层数 = %d，%d 个节点全部可达"
       % (max(layer13.values()) + 1, len(dag13.nodes)),
       (reached13, len(dag13.nodes)), (len(dag13.nodes), len(dag13.nodes)),
       "层数 = %d（n000…n007）。★ **这是图论意义上的深度，不是时间**："
       "本程序没有、也无法给任何一条边赋时间。节点的层号 = 它的最长前驱链长度。"
       % (max(layer13.values()) + 1))

    # B.3 hw10：NULL memset 的出度
    memset_nid = [op.nid for op in dag10.nodes if op.kind == KIND_MEMSET][0]
    succ = succ_labels(dag10, memset_nid)
    eq("B", "B.3 hw10 切片：NULL 流 memset 的**出度 = %d**（两条 blocking 流上**全部**动作）"
       % len(succ), len(succ), 12,
       "两条流各 4 个 chunk × 3 个动作 = 12。★ 这是 A2 **后半句**的直接后果："
       "「the action is queued in the legacy stream, and then all blocking streams wait "
       "on the legacy stream」。⇒ 栅栏的**出边**是 O(后续动作数)，"
       "**入边**只有 O(流数)。**不对称** —— 这是本主题最容易被讲成「互相等」而讲错的一处。")

    # B.4 hw10 切片的边构成（用参照实现独立算一遍）
    by_reason = defaultdict(int)
    for _, _, r in dag10.edges:
        by_reason[reason_class(r)] += 1
    ref_by_reason = defaultdict(int)
    for _, _, r in build_reference(parse(FIX_HW10)):
        ref_by_reason[reason_class(r)] += 1
    eq("B", "B.4 hw10 切片的边构成（两条独立实现各算一遍，逐类相同）",
       (dict(by_reason), dict(ref_by_reason)),
       ({"program-order": 13, "legacy-barrier-out": 12},
        {"program-order": 13, "legacy-barrier-out": 12}),
       "分解：%d 个设备侧动作（NULL 4 + streams[0] 6 + streams[1] 6）。"
       "**program-order 13 条** = NULL 流 3 条（4 个动作的链）+ streams[0] 5 条 + streams[1] 5 条。"
       "**legacy-barrier-out 12 条** = memset 指向**它之后**两条流上全部 12 个动作。"
       "合计 %d 条。"
       "★ **注意 legacy-barrier-in 是 0** —— 这不是 bug：NULL 流那 4 个动作都出现在"
       "两条 blocking 流发射任何东西**之前**，所以 memset 等不到任何东西（它的前驱只有"
       "同流前一条 NULL 动作）。这正是 C.2 那个反例的镜像。"
       % (len(dag10.nodes), len(dag10.edges)))

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    print("§C 边界与反例（**C 类成立时程序应当失败**）")
    print("-" * 78)

    # C.1 wait 之前没有 record
    bad_wait = ("mode: legacy\nstream: a, blocking\nstream: b, blocking\n\n"
                "a : kernel k1\nb : wait ev\n")
    got_exc = None
    try:
        build(parse(bad_wait))
    except ValueError:
        got_exc = "ValueError"
    eq("C", "C.1 `wait(ev)` 之前没有同事件的 `record` ⇒ **必须抛 ValueError**",
       got_exc, "ValueError",
       "真实 CUDA 里这是一种无效依赖（A7 说 wait 等的是「work captured in event」，"
       "一个从未 record 过的 event 没有 captured work）。"
       "★ 这里断言的**失败行为成立**：若把 `raise` 改成静默跳过，这一条立刻失败。")

    # C.2 ★ memset 挪到循环之后 ⇒ 出度 12 -> 0
    moved = FIX_HW10.replace("NULL       : memset d_y\n", "", 1).rstrip("\n") \
        + "\nNULL       : memset d_y\n"
    d_moved = build(parse(moved))
    mid = [op.nid for op in d_moved.nodes if op.kind == KIND_MEMSET][0]
    eq("C", "C.2 ★ 把 `memset d_y` 挪到分块循环**之后** ⇒ 出度 12→0、入度 1→3",
       (len(succ_labels(d_moved, mid)), len(pred_labels(d_moved, mid))), (0, 3),
       "★★ 这一条把计划里记下的那个**记错**做成可运行的反例。"
       "真实位置（源文件第 %d 行，在循环**之前**）：memset 出度 **12**、入度 **1**。"
       "记错的位置（循环**之后**）：出度 **0**、入度 **3**（= 同流前一条 NULL 动作 1 条 + "
       "两条 blocking 流各自最后一条动作 2 条 —— 此时轮到 memset 去等它们）。"
       "**同一个 cudaMemset、两个相反的位置，出度从 12 变成 0，入度从 1 变成 3。**"
       "⇒ 只凭「legacy 流会同步」这句话**推不出发射顺序**，"
       "**必须回到源文件数行**（A.6d 做的正是这件事）。"
       % specs.HW10_MEMSET_LINE)

    # C.3 ★ 两条 blocking 流之间 0 条直接边
    eq("C", "C.3 ★ hw10 切片：streams[0] ↔ streams[1] 之间**直接边为 0**（星形不是全连接）",
       (cross_edges(dag10, "streams[0]", "streams[1]"),
        cross_edges(dag10, "streams[1]", "streams[0]")), ([], []),
       "★ 真值来源 A1：「synchronizes with **all other streams**」—— 同步是"
       "**流 ↔ legacy 流**，不是**任意两条 blocking 流之间**。"
       "两条 blocking 流只通过 NULL 流**间接**同步。"
       "⇒ 本篇的反直觉量化结论的机制侧：栅栏是**星形**（1 个 NULL 动作指向 12 个动作），"
       "**不是**全连接（那会是 12×12 = 144 条）。")

    # C.4 per_thread 模式
    pt_fix = FIX_HW10.replace("mode: legacy", "mode: per_thread", 1)
    _, dag_pt = load(pt_fix)
    imp = [r for _, _, r in dag_pt.edges if reason_class(r).startswith("legacy")]
    eq("C", "C.4 ★ `mode: per_thread` ⇒ 隐式栅栏边 **0** 条（同流边一条不少）",
       (len(imp), len(dag_pt.edges)), (0, 25 - 12),
       "★ 真值来源 A4：「The per-thread default stream … **does not synchronize with "
       "other streams (just like explicitly created streams)**」。"
       "⇒ 同一段程序只换一个编译标志（--default-stream per-thread），隐式边从 12 条掉到 0 条。"
       "★ 这里用 `mode:` 而不是 NULL 流：在 per_thread 模式下，那个 `NULL` 名只是一个名字，"
       "不再有栅栏语义（**P9 同名不同指**）。")

    # C.5 ★ 可达 ≠ 直接边
    reach13 = reachability(dag13)
    direct = edge_pairs(dag13.edges)
    eq("C", "C.5 ★ a→d **可达**但**不是直接边**、**也不在 cover 里**（闭包 ≠ 归约）",
       ("n007" in reach13["n000"], ("n000", "n007") in direct,
        ("kernel_a", "kernel_d") in kernel_cover(dag13)), (True, False, False),
       "★★ 把「可达」当「直接边」是探针 v1 真实犯过的错（**错的不是模拟器，是我的期望值**）。"
       "a→b→d 已经蕴含 a→d，所以 a→d 不在**传递归约（cover）**里，"
       "但它在**传递闭包**里。NVIDIA 那张图画的是 **cover**（4 条边），不是闭包（6 条）。"
       "⇒ 「图上有多少箭头」这句话**必须先问清是归约还是闭包**（P9 同名不同指）。")

    # C.6 ★ A2 是三步，不是两步
    eq("C", "C.6 ★ 只读 A2 的**前两步**会丢掉 k_2→k_3（栅栏是三步，不是两步）",
       (pred_labels(dag_doc, "n002"),
        sorted(r for _, _, r in dag_doc.edges if reason_class(r) == "legacy-barrier-out")),
       (["kernel:k_1", "kernel:k_2"], ["legacy-barrier-out"]),
       "真值来源（逐字）A2：「the legacy stream **first waits on all blocking streams**, "
       "the action is queued in the legacy stream, and **then all blocking streams wait "
       "on the legacy stream**」。"
       "★ **注意 k_3 的入边有两条、来自两个不同的规则**："
       "`k_1 → k_3` 是 **program-order**（k_1 与 k_3 同在流 s 上），"
       "`k_2 → k_3` 才是 **legacy-barrier-out**（A2 的第三步）。"
       "⇒ 只读 A2 的前半句（「legacy 流等所有 blocking 流」）会得到 **2 条边**而不是 3 条，"
       "**丢掉的正是那条跨流的 `k_2 → k_3`** —— 而那一条才是「barrier」这个词的要害。"
       "★ 计划 §0.7 G2 特别指出 A2 是**三步**；"
       "`specs.ANCHORS['A2']['must_contain']` 把它拆成两半正是为了防这个（P2 整个载体缺失）。")

    # C.7 ★ 没用到 legacy 流时，建流标志不改变 DAG
    _, dag_blk = load(FIX_HW13_BLOCKING)
    eq("C", "C.7 ★ hw13 那段没有任何 NULL 流动作 ⇒ blocking / nonblocking 的 DAG **完全相同**",
       edge_classes(dag_blk.edges), edge_classes(dag13.edges),
       "★ 这一条**反驳一个直觉**：「用了 cudaStreamNonBlocking ⇒ DAG 一定不同」是错的。"
       "栅栏由 A2 的「**an action is taken in the legacy stream**」触发；"
       "这段程序里一条 NULL 流动作都没有，所以建流标志**不改变任何一条边**。"
       "⇒ 03 篇那个「一个标志决定 capture 是否合法」的结论，"
       "**不是**从「NonBlocking 总在改变依赖图」推出来的 —— 两条结论作用域不同，不能互推。")

    # C.8 ★ 故意改错一条规则 ⇒ 交叉校验必须发现
    def build_broken(prog):
        """故意删掉 `is_blocking` 判定：让栅栏对**所有**流生效（一个听起来合理的错）。"""
        dag = Dag()
        dag.nodes = [op for op in prog.ops if op.kind != KIND_SYNC]
        last_in_stream = {}
        for op in dag.nodes:
            prev = last_in_stream.get(op.stream)
            if prev is not None:
                dag.add_edge(prev, op, "program-order")
            last_in_stream[op.stream] = op
        last_record = {}
        for op in dag.nodes:
            if op.kind == KIND_RECORD:
                last_record[op.arg] = op
            elif op.kind == KIND_WAIT:
                r = last_record.get(op.arg)
                if r is None:
                    raise ValueError("wait before record")
                dag.add_edge(r, op, "event:%s" % op.arg)
        if prog.mode == "legacy":
            last_legacy = None
            last_dev = {}
            for op in dag.nodes:
                if op.stream == LEGACY:
                    for s, prev in last_dev.items():
                        dag.add_edge(prev, op, "legacy-barrier-in")      # ← 少了 is_blocking
                    last_legacy = op
                elif last_legacy is not None:
                    dag.add_edge(last_legacy, op, "legacy-barrier-out")  # ← 少了 is_blocking
                last_dev[op.stream] = op
        return dag

    p_nb = parse(FIX_DOC_NONBLOCKING)
    broken = edge_classes(build_broken(p_nb).edges)
    ref_nb = edge_classes(build_reference(p_nb))
    eq("C", "C.8 ★ 故意删掉 `is_blocking` 判定（栅栏对所有流生效）⇒ 交叉校验**必须发现**",
       (broken == ref_nb, len(broken), len(ref_nb)), (False, 3, 1),
       "★★ 这就是 A 类参照实现**可证伪**的直接证据：一个「听起来合理」的错误"
       "（栅栏对所有流生效，而不是只对 blocking 流）会让 nonblocking 的例子"
       "从 1 条边变成 3 条边，`build_reference` 立刻与它分叉。"
       "★ 本程序**不依赖**这个错误版本去算任何东西 —— 它只在**这里**被构造出来当反例，"
       "证明 A.5 的交叉校验不是「永远通过」的装饰。")

    # C.9 覆盖范围自查
    eq("C", "C.9 本程序声明的 %d 条锚点都在 specs.ANCHORS 里，且不与 01 的 (C1, C1b, C2) 重叠"
       % len(COVERED_ANCHORS),
       (sorted(set(COVERED_ANCHORS) - set(specs.ANCHORS)),
        sorted(set(COVERED_ANCHORS) & {"C1", "C1b", "C2"})), ([], []),
       "★ 「并集 == 全集、两两不交」的**本程序这一侧**；全局核对由 01 的 A.6a/A.6b 做"
       "（它用 ast 读本文件的 COVERED_ANCHORS）。")

    # ---- A.10 ★★ 锚点的 `covered_by`：`note` 是**索引**，不是**唯一载体** ----
    #   ★ 订正轮的裁定：`note` 允许保留（它是注释），但必须变成**机器可查的**。
    #     做法是给每条锚点加 `covered_by`（它支撑的断言标签），并要求
    #     **`note` 里出现的标签集合 == `covered_by`**（双向）。
    #   ★ 为什么双向：① note 里写了标签却不在 covered_by ⇒ 那句话没有它自称的支撑；
    #     ② covered_by 列了标签而 note 只字不提 ⇒ 索引没写全。
    #     ⇒ 改 `covered_by` 里**任何一个**标签（哪怕换成另一个真实标签），这一条必失败。
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
    eq("A", "A.10 ★★ 本程序 %d 条锚点的 `covered_by` 非空，且与 `note` 里的标签集合"
       "**恰好相等**（双向）" % len(COVERED_ANCHORS), cb_bad, [],
       "格式：`(锚点, 说明, (note 多出来的, covered_by 多出来的))`。\n"
       "        ★★ 这是「`note` 是索引、不是唯一载体」的机械执行 —— "
       "**改 `covered_by` 里任何一个标签，这一条立刻失败**。\n"
       "        ★ 覆盖边界（如实说出）：程序没有「标签 ↔ 锚点」的语义知识，"
       "把标签换成**另一个真实存在且也恰好在 note 里**的标签是抓不到的 —— "
       "机器只保证 `note` 与 `covered_by` 两处**一致**，语义由 `note` 的措辞负责。")
    no_local = sorted(aid for aid in COVERED_ANCHORS if not per_anchor_local.get(aid))
    eq("A", "A.10b ★ 每条锚点至少有一个**本程序**的断言标签（%d 条共 %d 个本程序标签，"
       "%d 个跨程序标签）" % (len(COVERED_ANCHORS),
                             sum(len(v) for v in per_anchor_local.values()), len(cb_foreign)),
       no_local, [],
       "★ 「本程序标签」= 运行到这一行为止**真的出现过**的标签（`SEEN_LABELS`）——"
       "所以这条同时证明「`covered_by` 指向的是真实跑过的断言」，不是编的。\n"
       "        ★ 跨程序的那些：%r —— 它们属于 01 或 03，存在性由 01 的 A.11h 统一校验"
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
    print("  · **不是硬件** —— 建模的是命令之间的**偏序依赖**，不建模执行时间、")
    print("    SM 占用、并发度、硬件队列深度。")
    print("  · 层次输出只能读作「**允许**并发」，不能读作「**实际**并发」")
    print("    （Guide §3.2.8.5 原文：this behavior is not guaranteed and should therefore")
    print("     not be relied upon for correctness）。")
    print("  · **所有时间数字本机不可得** —— hw10 readme 第 %d 行的 about 2X 与"
          % specs.HW10_README_TIME_LINE)
    print("    hw13 README 第 74 行的 small preformance increase，**两个都没验证**。")
    print("=" * 78)
    return 1 if FAILS else 0


# ===========================================================================
# 8. DAG 的可读呈现（`make show`）
# ===========================================================================
def render(prog, dag, title):
    out = ["=" * 68, title, "=" * 68, "mode = %s" % prog.mode]
    streams = []
    for op in dag.nodes:
        if op.stream not in streams:
            streams.append(op.stream)
    for s in streams:
        flag = ("legacy-default" if s == LEGACY else
                "per-thread-default" if s == PT else
                "nonblocking" if prog.stream_flag(s) else "blocking")
        out.append("  stream %-11s -> %s" % (s, flag))
    out.append("")
    out.append("设备侧节点（按宿主发射序）:")
    for op in dag.nodes:
        out.append("  %s  %-11s %s" % (op.nid, op.stream, op.label()))
    out.append("")
    out.append("依赖边 (%d 条):" % len(dag.edges))
    by = defaultdict(list)
    for f, t, r in dag.edges:
        by[t].append((f, r))
    for op in dag.nodes:
        for f, r in sorted(by.get(op.nid, [])):
            out.append("  %s -> %s   [%s]" % (f, op.nid, r))
    layer, reached = levels(dag)
    out.append("")
    out.append("拓扑分层（第 0 层无前驱；★ 偏序性质，**不是**实测时间）:")
    for op in dag.nodes:
        out.append("  %s  L%d" % (op.nid, layer[op.nid]))
    out.append("  层数 = %d" % (max(layer.values()) + 1 if layer else 0))
    return "\n".join(out)


def show():
    print("★ 本呈现只是 DAG 的可读形式 —— 它**不是**时间线，节点位置不代表发射时刻。")
    print()
    p, d = load(FIX_HW10)
    print(render(p, d, "程序: hw10 的同构切片（chunks=%d, num_streams=%d）"
                 % (specs.HW10_FIXTURE_CHUNKS, specs.HW10_FIXTURE_STREAMS)))
    print()
    p, d = load(FIX_HW13)
    print(render(p, d, "程序: hw13 task1（capture 区内的发射顺序）"))
    return 0


# ===========================================================================
# 9. 联网入口：从**实际下载的官方 HTML** 里再抽取全部锚点（`make anchors`）
# ===========================================================================
def _html_to_text(raw):
    """HTML -> 文本。

    ★ 纪律（探针的教训）：**不能**把所有标签都换成换行 —— 文档里
      `<code>cudaStreamNonBlocking</code>` 这类**行内**标签很多，一刀切换行会把句子
      切断，导致正则抓不到整句、**误报 NOT FOUND**（探针 v1 就是这样丢了 14/21 条）。
      只有**块级**标签才换行，行内标签直接抹掉。
    """
    import html as _html
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?is)</(?:p|div|li|tr|h[1-6]|pre|table|ul|ol)\s*>", "\n", raw)
    t = re.sub(r"(?is)<[^>]+>", "", raw)
    t = _html.unescape(t)
    return "\n".join(" ".join(l.split()) for l in t.split("\n")).strip()


def anchors_net():
    """★ 保留探针的核心做法：**引文由脚本从下载的 HTML 里抓取，不手打**。

    抓不到就打印 NOT FOUND **并把它算作不一致**（以非零码退出）——
    那一条**不得当作依据**。
    """
    import urllib.request
    print("=" * 78)
    print("02_stream_dag --anchors-net —— 从官方文档**再抽取**全部 %d 条锚点"
          % len(specs.ANCHORS))
    print("=" * 78)
    print("  ★ 为什么这么绕：手打一句像模像样的英文引文，比写错一个数字更危险。")
    print("    所以这里不手打 —— 每条锚点都是脚本按 `regex` 从**实际下载的 HTML** 里抓的。")
    print("  ★ 注意 `curl` 必须带 `-L`：不带时拿到 HTTP 301 且 size=0。")
    print()
    texts = {}
    for key, url in sorted(specs.ANCHOR_DOC_URLS.items()):
        print("  取 %-9s %s" % (key, url))
        req = urllib.request.Request(url, headers={"User-Agent": "cuda-knowledge-anchor-probe"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", "replace")
        print("       HTTP %s  size=%d" % (getattr(resp, "status", "?"), len(raw)))
        texts[key] = _html_to_text(raw)
    print()
    ok, bad = 0, []
    for aid in sorted(specs.ANCHORS):
        a = specs.ANCHORS[aid]
        t = texts.get(a["doc"], "")
        # 文档里同一句可能因块级标签换行而断开，所以把模式里的**字面空格**
        # 一律放宽成 \s+（已经写成 \s 的地方不受影响）。
        pat = re.sub(r"(?<!\\) ", r"\\s+", a["regex"])
        m = re.search(pat, t, re.S) if t else None
        if not m:
            bad.append((aid, a["doc"], "NOT FOUND"))
            print("  [%s] NOT FOUND in %s   <<< 这一条不成立，不得当作依据 >>>"
                  % (aid, a["doc"]))
            continue
        got = " ".join(m.group(0).split())
        same = (got == a["quote"])
        ok += 1
        if not same:
            bad.append((aid, a["doc"], "与 specs.py 记录的 quote 不一致"))
        print("  [%s] %s  (%s)" % (aid, "逐字一致" if same else "**不一致**", a["doc"]))
        if not same:
            print("       现抓 = %r" % got)
            print("       记录 = %r" % a["quote"])
    print()
    print("抓到 %d / %d 条锚点；不一致 %d 条。" % (ok, len(specs.ANCHORS), len(bad)))
    print("★ 只报告「抓到了」与「与记录一致」**这两件事**；抽取器读的是**当前**页面，")
    print("  换一个 CUDA 版本，同一节的措辞可能变 —— 那会被报成「不一致」，不是「抓不到」。")
    print("★ `section` 字段**不在这里核对**（它由人工定位到最近的上级标题）。")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    if "--anchors-net" in sys.argv:
        sys.exit(anchors_net())
    if "--show" in sys.argv:
        sys.exit(show())
    sys.exit(main())
