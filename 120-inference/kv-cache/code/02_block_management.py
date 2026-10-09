#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_block_management.py —— **系统侧的块管理器**：块池 / 块表 / 引用计数 / copy-on-write。

配套文档：`02-block-management.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题不写「paged KV 的 kernel 侧」**（页表延迟 / 碎片 / 布局 / 块大小的双向权衡 ——
  那是 `40-kernels/attention/05:130-179,283-290` 的正文，本主题**只回链**）。
  本程序只写 **系统侧的块管理器**：**块池 / 块表 / 引用计数 / copy-on-write / 分配-回收**。

★★★ **本程序是【本域最强的一条 A 类】**（照 `§1.3.2b` / `CONTRIBUTING:235`）：
  · **被测** = `CowManager`：多序列**共享**块，只在**写一个 `ref > 1` 的块时**才复制（copy-on-write）；
  · **参照** = `NaiveManager`：**朴素深拷贝** —— 每条序列自己一份，**完全不共享**
    （★ 它与 COW 路径**没有任何共享代码**）。
  ★★★ **判据**：**任意随机 `create / fork / append / remove` 序列之后，两个实现的【每序列内容**
    **必须逐字相同】**；★★ **并且断言两者的【块数不同】**（COW 更少）—— 否则「共享」根本没生效。
  ★★ **守卫**（`D.4`/`D.5`）：把引用计数检查改错（写共享块时不复制、或 fork 时忘 `ref += 1`）
    ⇒ **内容立刻不一致** ⇒ 断言**必须失败**。

★★ **覆盖边界**：本程序守的是「**块管理器的不变量**」（引用计数守恒 / CoW 的正确性 / 泄漏与双释放的后果）。
  **不守**任何**真实引擎**的块分配器行为、也不碰 kernel 侧（本机**无 GPU / 无引擎**）。
  ★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **COW == 朴素深拷贝**（`COW_OP_ROUNDS` 轮随机的 content 逐字相同）；
  · `A.2` ★★★ **块数不同**（COW 更少）且**真的发生过 CoW 复制**；
  · `A.3` ★★ **引用计数守恒**（`Σref == Σ` 每序列块数）；
  · `B.1/B.2` ★ 块池的**开销账**（块表条目数 × 每条的字节；共享省下的活块）；
  · `C.1` ★★★ **引用计数泄漏** ⇒ 池被耗光（不变量被破坏 + 分配失败）；
  · `C.2` ★★★ **双重释放** ⇒ 引用计数转负、块被回收复用 ⇒ **内容真的被改坏**；
  · `D.4/D.5` ★★★ **守卫实验**：把 CoW 改坏 / 把 fork 的 `ref += 1` 改坏 ⇒ 断言必须失败；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **失败标记自检** + `UNSOURCED`。
"""
import ast
import collections
import inspect
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（kv-cache）取 `-- FAIL: kv-cache`；`mutate.py` 用 AST 读出（不写死）。
ASSERT_FAIL_MARK = "-- FAIL: kv-cache"

OTHER_THEME_MARKS = ("-- FAIL: batching-and-scheduling", "-- FAIL: decoding-strategies",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

VOCAB = 100                          # 随机 token 的取值范围（**本主题算例**，不是外部事实）
BLOCK_TABLE_ENTRY_BYTES = 4          # ★ **模型假设**：块表**每条**的字节数（物理号 + 粗粒度元数据）
LEAK_POOL_CAP = 32                   # ★ **模型假设**：`C.1`（泄漏）专用的小池 —— 小到会被耗光
GIB = 1073741824                     # 2^30


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D）。★ 失败路径**不复述结论**。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name,
                                detail_fail if detail_fail is not None else
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立」的措辞是"
                                "**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 被测：**块池 + 引用计数 + copy-on-write**
# ===========================================================================
class CowManager:
    """★★★ **被测实现**：多序列**共享**块；写一个 `ref > 1` 的块时**先复制再写**。

    ★★ 不变量（`A.3` 判它）：**`Σ_块 ref == Σ_live序列 该序列的块数`**。
    ★★★ `corrupt` 只用于 `D.4`/`D.5`/`C.1`/`C.2` 的失败实验：
      · `"no_cow"`      —— 写共享块时**不复制**（直接改别人的块）；
      · `"no_inc_ref"`  —— `fork` 时**忘** `ref += 1`；
      · `"leak"`        —— `remove` 时**不减引用计数**（块永不回收）；
      · `"double_free"` —— `remove` 时**多减一次**（引用计数转负）。
    """

    def __init__(self, block_tokens, pool_cap, corrupt=None):
        self.B = block_tokens
        self.cap = pool_cap
        self.corrupt = corrupt
        self.blocks = []      # 每项 = [data(list[int]), ref(count)]
        self.free = []        # 已回收（ref == 0）的块下标
        self.seqs = {}        # sid -> [块下标]
        self.lens = {}        # sid -> token 个数
        self.n_created = 0    # 一共**创建过**多少块（★ 与朴素实现比较用）
        self.n_cow = 0        # 一共发生过多少次 CoW 复制
        self.alloc_fail = False

    def _alloc(self):
        if self.free:
            i = self.free.pop()
            self.blocks[i][1] = 0
            return i
        if len(self.blocks) >= self.cap:
            self.alloc_fail = True
            return None
        self.blocks.append([[0] * self.B, 0])
        self.n_created += 1
        return len(self.blocks) - 1

    def create(self, sid, tokens=()):
        self.seqs[sid] = []
        self.lens[sid] = 0
        self.append(sid, tokens)

    def fork(self, sid, child):
        self.seqs[child] = list(self.seqs[sid])
        for b in self.seqs[child]:
            if self.corrupt != "no_inc_ref":
                self.blocks[b][1] += 1
        self.lens[child] = self.lens[sid]

    def append(self, sid, tokens):
        blk = self.seqs[sid]
        for tok in tokens:
            off = self.lens[sid] % self.B
            if off == 0:
                b = self._alloc()
                if b is None:
                    return False
                self.blocks[b][1] += 1        # ★ 新块挂到本条序列上
                blk.append(b)
            else:
                last = blk[-1]
                if self.blocks[last][1] > 1 and self.corrupt != "no_cow":
                    b = self._alloc()         # ★★ copy-on-write：先复制，再写
                    if b is None:
                        return False
                    self.blocks[b][0] = list(self.blocks[last][0])
                    self.blocks[b][1] += 1
                    self.blocks[last][1] -= 1
                    blk[-1] = b
                    self.n_cow += 1
            self.blocks[blk[-1]][0][off] = tok
            self.lens[sid] += 1
        return True

    def remove(self, sid):
        for b in self.seqs[sid]:
            if self.corrupt == "leak":
                continue                       # ★ 坏法：不减引用计数
            dec = 2 if self.corrupt == "double_free" else 1
            self.blocks[b][1] -= dec
            if self.blocks[b][1] <= 0:
                self.free.append(b)
        del self.seqs[sid]
        del self.lens[sid]

    def content(self, sid):
        out = []
        for b in self.seqs[sid]:
            out.extend(self.blocks[b][0])
        return tuple(out[:self.lens[sid]])

    def n_live(self):
        return len(self.blocks) - len(self.free)

    def ref_sum(self):
        return sum(b[1] for b in self.blocks)

    def expected_ref_sum(self):
        return sum(len(v) for v in self.seqs.values())

    def min_ref(self):
        return min((b[1] for b in self.blocks), default=0)


# ===========================================================================
# 2. 参照：**朴素深拷贝**（每条序列自己一份，**完全不共享** —— 无共享代码）
# ===========================================================================
class NaiveManager:
    """★★★ **参照实现**：`fork` 时**深拷贝**父序列的**每一个块**；`append` **就地写自己的块**。

    ★★ 它**没有引用计数、没有 CoW、没有回收** —— 与 `CowManager` **不共享任何一行**（`D.8` 核）。
    ★★ 它是「**不共享**」这条基线：若 COW 的实现有 bug，内容会与它**不一致**。
    """

    def __init__(self, block_tokens):
        self.B = block_tokens
        self.blocks = []
        self.seqs = {}
        self.lens = {}
        self.n_created = 0

    def _alloc(self):
        self.blocks.append([0] * self.B)
        self.n_created += 1
        return len(self.blocks) - 1

    def _copy(self, b):
        i = self._alloc()
        self.blocks[i] = list(self.blocks[b])
        return i

    def create(self, sid, tokens=()):
        self.seqs[sid] = []
        self.lens[sid] = 0
        self.append(sid, tokens)

    def fork(self, sid, child):
        self.seqs[child] = [self._copy(b) for b in self.seqs[sid]]
        self.lens[child] = self.lens[sid]

    def append(self, sid, tokens):
        blk = self.seqs[sid]
        for tok in tokens:
            off = self.lens[sid] % self.B
            if off == 0:
                blk.append(self._alloc())
            self.blocks[blk[-1]][off] = tok
            self.lens[sid] += 1
        return True

    def remove(self, sid):
        del self.seqs[sid]
        del self.lens[sid]

    def n_live(self):
        return len(self.blocks)          # ★ 朴素实现**从不回收** ⇒ 活块 = 创建过的全部

    def content(self, sid):
        out = []
        for b in self.seqs[sid]:
            out.extend(self.blocks[b])
        return tuple(out[:self.lens[sid]])


# ===========================================================================
# 3. 随机操作序列（两实现**共用同一条**）
# ===========================================================================
def gen_ops(seed, rounds, block_tokens):
    """生成 `ops`：`(kind, ...)` 的列表，四类 `create / fork / append / remove`（★ 单遍生成）。"""
    rng = np.random.default_rng(seed)
    ops = []
    live = []
    nxt = 0
    for _ in range(rounds):
        r = float(rng.random())
        if not live or r < 0.30:
            sid = nxt
            nxt += 1
            k = int(rng.integers(0, 2 * block_tokens + 1))
            ops.append(("create", sid, tuple(int(x) for x in rng.integers(1, VOCAB, size=k))))
            live.append(sid)
        elif r < 0.58:
            sid = live[int(rng.integers(0, len(live)))]
            k = int(rng.integers(0, 2 * block_tokens + 1))
            ops.append(("append", sid, tuple(int(x) for x in rng.integers(1, VOCAB, size=k))))
        elif r < 0.85:
            sid = live[int(rng.integers(0, len(live)))]
            child = nxt
            nxt += 1
            ops.append(("fork", sid, child))
            live.append(child)
        elif len(live) > 1:
            sid = live.pop(int(rng.integers(0, len(live))))
            ops.append(("remove", sid))
        else:
            ops.append(("append", live[0], ()))
    return ops


def apply_ops(mgr, ops):
    for op in ops:
        if op[0] == "create":
            mgr.create(op[1], op[2])
        elif op[0] == "append":
            mgr.append(op[1], op[2])
        elif op[0] == "fork":
            mgr.fork(op[1], op[2])
        else:
            mgr.remove(op[1])
    return mgr


# ===========================================================================
# 4. A 类 —— ★★★ **CoW vs 朴素深拷贝**（本域最强的一条）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **块管理器**：引用计数 + copy-on-write（被测）vs 朴素深拷贝（参照）")
    print("=" * 78)
    print("  ★★★ 被测 = `CowManager`（共享块，写 `ref>1` 的块时才复制）；")
    print("      参照 = `NaiveManager`（**每条序列自己一份，完全不共享**）。")
    print("  ★★ 两条路径**无共享代码**（`D.8` 在源码层核）。")

    ops = gen_ops(specs.COW_SEED, specs.COW_OP_ROUNDS, specs.COW_BLOCK_TOKENS)
    cow = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, specs.COW_POOL_CAP), ops)
    nai = apply_ops(NaiveManager(specs.COW_BLOCK_TOKENS), ops)
    live = sorted(cow.seqs.keys())

    # ── A.1 ★★★ 300 轮之后：每序列内容逐字相同 ─────────────────────────
    print()
    print("  ── A.1 ★★★ **任意随机操作后，两实现的每序列内容逐字相同** ──")
    print("     ★ 操作序列：%d 步（`create / fork / append / remove`）· 存活序列 %d 条 · 块 = %d token"
          % (len(ops), len(live), specs.COW_BLOCK_TOKENS))
    same = True
    for sid in live:
        a, b = cow.content(sid), nai.content(sid)
        same = same and (a == b)
    print("     ★ %d 条存活序列的内容**全部逐字相同** = %s" % (len(live), same))
    print("     ★★ 样例（前 3 条）：%s"
          % "; ".join("seq%d %s" % (s, cow.content(s)[:6]) for s in live[:3]))
    check("A.1", same and len(live) >= 2 and len(ops) == specs.COW_OP_ROUNDS,
          "★★★ **判据成立**：**%d 步随机操作**之后，`CowManager`（共享 + CoW）与 `NaiveManager`"
          "（深拷贝、不共享）在 **%d 条存活序列**上**逐字相同** —— ★★ 一面靠**引用计数**决定何时复制，"
          "一面**根本不共享**，两条独立路径会合" % (len(ops), len(live)),
          "**两实现的内容不一致** —— ★ 先查 CoW 的触发条件（`ref > 1` 才复制）与 `fork` 的 `ref += 1`")

    # ── A.2 ★★★ 块数不同（CoW 更少）且真的复制过 ───────────────────────
    print()
    print("  ── A.2 ★★★ **块数不同**：共享真的生效了吗 ──")
    print("     一共**创建过**的块：CoW %d · 朴素 %d" % (cow.n_created, nai.n_created))
    print("     期末**活块**：CoW %d · 朴素 %d（池容量 %d）"
          % (cow.n_live(), nai.n_live(), specs.COW_POOL_CAP))
    print("     发生了什么：**fork %d 次** · **CoW 复制 %d 次** · 分配失败 = %s"
          % (sum(1 for o in ops if o[0] == "fork"), cow.n_cow, cow.alloc_fail))
    check("A.2", cow.n_created < nai.n_created and cow.n_cow > 0
          and sum(1 for o in ops if o[0] == "fork") > 0,
          "★★★ **判据成立**：CoW **创建过 %d 块** < 朴素 **%d 块** —— ★★ 差的正是 `fork` 时**没复制**的那些"
          "（★ 本次 `fork` %d 次、`CoW` 复制 %d 次 ⇒ **共享确实生效**，不是恒真）"
          % (cow.n_created, nai.n_created, sum(1 for o in ops if o[0] == "fork"), cow.n_cow),
          "**块数居然相同** —— ★ 那说明「共享」根本没发生（A.1 的相等就退化成恒真了）")

    # ── A.3 ★★ 引用计数守恒（不变量）────────────────────────────────────
    print()
    print("  ── A.3 ★★ **引用计数守恒**：`Σ_块 ref == Σ_live 每序列块数` ──")
    print("     `Σ ref` = %d · 期望 = %d · 最小 ref = %d"
          % (cow.ref_sum(), cow.expected_ref_sum(), cow.min_ref()))
    check("A.3", cow.ref_sum() == cow.expected_ref_sum() and cow.min_ref() >= 1,
          "★★ **判据成立**：`Σ_块 ref == %d == Σ_live 每序列块数`，且**每个活块的 ref ≥ 1**"
          " —— ★★ 这是块管理器**唯一**的不变量（`C.1`/`C.2` 两条反例都在破坏它）"
          % cow.ref_sum(),
          "**引用计数不守恒** —— ★ 先查 `fork` 的 `ref += 1`、`remove` 的 `ref -= 1`、CoW 的**旧块减 1**")


# ===========================================================================
# 5. B 类：块池的**开销账**
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **块池的开销账**：块表条目数 × 每条的字节")
    print("=" * 78)
    print("  ★★ **块数**这条链在 `01-capacity-planning` 里算过（`可用显存 → 块数`）——")
    print("     本节**只给它的字节代价**（★ **「这点字节值不值得担心」的结论在 "
          "`40-kernels/attention/05:130-179`，本主题只回链**）。")

    t = specs.LLAMA2_7B_KV_BYTES_PER_TOKEN
    avail = 63 * GIB

    # ── B.1 ★ 条目数 ⇒ 块表字节 ──────────────────────────────────────────
    print()
    print("  ── B.1 ★ **块表条目数 × %d B/条 ⇒ 块表字节**（可用 63 GiB）──"
          % BLOCK_TABLE_ENTRY_BYTES)
    print("     %-10s %-14s %-16s %-14s" % ("块token", "块数(条目)", "块表字节", "每 token 表的字节"))
    rows = {}
    for bt in specs.KV_BLOCK_TOKEN_GRID:
        blocks = avail // (bt * t)
        table = blocks * BLOCK_TABLE_ENTRY_BYTES
        rows[bt] = (blocks, table)
        print("     %-10d %-14d %-16d %-14.4f" % (bt, blocks, table, table / (avail // t)))
    print("     ★★ **读法**：块越小 ⇒ 条目越多 ⇒ 块表越大（★ 近似 **`1/块大小`**）；")
    print("        ★★ **它不是 kernel 侧的页表账** —— 那笔账（页表延迟 / 碎片 / 布局 / 块大小权衡）在")
    print("           `40-kernels/attention/05:130-179,283-290`，**本主题只回链**。")
    check("B.1", rows[16][0] == 8064 and rows[16][1] == 32256
          and rows[256][0] == 504 and rows[1][0] == 129024
          and rows[16][1] < rows[1][1] and rows[256][1] < rows[16][1],
          "★ **判据成立**：块 = `16` ⇒ **8064 条 × 4 B = 32256 B**；块 = `1` ⇒ **129024 条**"
          "（**16 倍**）；块 = `256` ⇒ **504 条** —— ★★ 块表字节**随块变小而线性变大**（∝ `1/块大小`）",
          "**块表的条目数被动过** —— ★ 先查「块数」是不是 `可用显存 // (块 × 每token字节)`")

    # ── B.2 ★ 共享省下的活块 ⇒ 同一显存能多开几条 ────────────────────────
    print()
    print("  ── B.2 ★ **共享省下的活块**：同一操作序列下，CoW 的活块 ≤ 朴素的活块 ──")
    ops = gen_ops(specs.COW_SEED, specs.COW_OP_ROUNDS, specs.COW_BLOCK_TOKENS)
    cow = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, specs.COW_POOL_CAP), ops)
    nai = apply_ops(NaiveManager(specs.COW_BLOCK_TOKENS), ops)
    print("     期末活块：CoW %d · 朴素 %d ⇒ 省下 %d 块（+%d%%）"
          % (cow.n_live(), nai.n_live(), nai.n_live() - cow.n_live(),
             100 * (nai.n_live() - cow.n_live()) // max(nai.n_live(), 1)))
    print("     ★★ **读法**：省下的块**就是能多开的序列**（同一个池装得下更多）——")
    print("        ★ 这条把 `01` 的「容量」与本节的「共享」接起来：**共享提高的是并发，不是吞吐**。")
    check("B.2", cow.n_live() <= nai.n_live() and cow.n_live() > 0,
          "★ **判据成立**：期末活块 CoW **%d** ≤ 朴素 **%d**（★ 省下 %d 块 ⟹ 同一个池能多装这么多）"
          % (cow.n_live(), nai.n_live(), nai.n_live() - cow.n_live()),
          "**共享没有省下活块** —— ★ 先查 `fork` 是否真的共享了块（而不是复制）")


# ===========================================================================
# 6. C 类：**引用计数泄漏 / 双重释放**的后果
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**引用计数泄漏** 与 **双重释放**")
    print("=" * 78)

    # ── C.1 ★★★ 泄漏：不变量被破坏 + 池被耗光 ──────────────────────────
    print()
    print("  ── C.1 ★★★ **引用计数泄漏**（`remove` 时**不减**引用计数）──")
    ops = gen_ops(specs.COW_SEED, specs.COW_OP_ROUNDS, specs.COW_BLOCK_TOKENS)
    leak = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, LEAK_POOL_CAP, corrupt="leak"), ops)
    print("     泄漏版：`Σ ref` = %d · 期望 = %d ⇒ **守恒被破坏 = %s**"
          % (leak.ref_sum(), leak.expected_ref_sum(), leak.ref_sum() != leak.expected_ref_sum()))
    print("             剩余空块 = %d · **分配失败 = %s**" % (len(leak.free), leak.alloc_fail))
    check("C.1", leak.ref_sum() != leak.expected_ref_sum() and leak.alloc_fail
          and leak.ref_sum() > leak.expected_ref_sum(),
          "★★★ **判据成立**：泄漏后 **`Σ ref` = %d ≠ 期望 %d**（★ 不变量当场破），且**空块耗尽 ⇒ "
          "分配失败** —— ★★ 后果是**池假满**：块明明没人用，却再也分不出去（`A.3` 正是被它破坏）"
          % (leak.ref_sum(), leak.expected_ref_sum()),
          "**泄漏没有跑出后果** —— ★ 先查 `remove` 是否真的不减引用计数、池容量是否够小")

    # ── C.2 ★★★ 双重释放：引用计数转负 + 块被复用 ⇒ 内容被改坏 ─────────
    print()
    print("  ── C.2 ★★★ **双重释放**（`remove` 时**多减一次**）──")
    # (a) 引用计数**转负**：单序列、块 ref 本来是 1，多减一次 ⇒ 变 -1
    neg_ops = [("create", 0, (1, 2, 3, 4)), ("remove", 0)]
    neg_good = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8), neg_ops)
    neg_bad = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8, corrupt="double_free"), neg_ops)
    # (b) **块被回收复用 ⇒ 内容被改坏**：两条序列共享一块，一条走了之后另一条还在
    dbl_ops = [("create", 0, (1, 2, 3, 4)), ("fork", 0, 1), ("remove", 0), ("create", 2, (9, 9, 9, 9))]
    good = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8), dbl_ops)
    bad = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8, corrupt="double_free"), dbl_ops)
    print("     (a) 单序列 create(0,1234) → remove(0)：正确版最小 ref %d · 双释放版最小 ref %d"
          % (neg_good.min_ref(), neg_bad.min_ref()))
    print("     (b) create(0,1234) → fork(0→1) → remove(0) → create(2,9999)：")
    print("         正确版 seq1 内容 %s · 双释放版 seq1 内容 %s（块被回收后复用覆写）"
          % (good.content(1), bad.content(1)))
    print("     ★★★ **读法**：ref 只有 1 时多减一次 ⇒ **转负**（物理上不可能）；")
    print("        共享时（ref=2）多减一次 ⇒ **直接减到 0 ⇒ 被回收**，下一条 `create(2)` 复用并覆写它。")
    check("C.2", neg_bad.min_ref() < 0 and neg_good.min_ref() == 0
          and good.content(1) == (1, 2, 3, 4) and bad.content(1) == (9, 9, 9, 9),
          "★★★ **判据成立**：双释放后 **最小 ref = %d < 0**（`(a)`：ref 只有 1 也减两次）；"
          "★ 且共享块被**误回收**后复用 ⇒ **seq1 的内容从 `(1,2,3,4)` 变成 `%s`** ——"
          " ★★ 正确版里它**纹丝不动**" % (neg_bad.min_ref(), bad.content(1)),
          "**双重释放没有跑出后果** —— ★ 先查 `remove` 的减量、以及空块是否真的被复用")


# ===========================================================================
# 7. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无引擎**；")
    print("      ★★ 但**本程序的 A 类是真的**：**映射到的两条独立路径**（CoW vs 朴素深拷贝）。")
    before = len(FAILS)

    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    hits = [n.value for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "ASSERT_FAIL_MARK"]
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1 and uniq == {ASSERT_FAIL_MARK}
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: kv-cache" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: kv-cache`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies",
                                        "-- FAIL: engines", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.KV_FETCH_DAY == "2026-10-09"
          and specs.COW_BLOCK_TOKENS == 4 and specs.COW_OP_ROUNDS == 300
          and specs.COW_SEED == 20261009 and specs.COW_POOL_CAP == 4096
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：抓取日 `%s` · 块 = `%d` token · 轮数 `%d` · 种子 `%d` · 池容量 `%d`；"
          "★ `UNSOURCED == ()` 一并钉住"
          % (specs.KV_FETCH_DAY, 4, 300, 20261009, 4096))

    # ★★★ 口径守卫（★ 跑在任何计算之前）：池容量必须**够小**才能让 `C.1` 的「池被耗光」跑出来，
    #   又必须**够大**让 `A.1` 的正常运行不撞池上限（否则会把「断言捕获」污染成「分配失败」）。
    ok_cap = specs.COW_POOL_CAP >= 8 and specs.COW_BLOCK_TOKENS >= 1 \
        and specs.COW_OP_ROUNDS >= 50 and VOCAB > 1 and LEAK_POOL_CAP < specs.COW_POOL_CAP
    print("     ★★ 池容量 `%d`（≥ 8 且足以在正常运行时不被撞满）= %s；泄漏专用小池 `%d` < 池容量 = %s"
          % (specs.COW_POOL_CAP, ok_cap, LEAK_POOL_CAP, LEAK_POOL_CAP < specs.COW_POOL_CAP))
    check("D.3b", ok_cap and specs.COW_BLOCK_TOKENS == 4 and VOCAB == 100,
          "★★★ **口径守卫**：块 > 0、池容量 ≥ 8、轮数 ≥ 50（★ 否则 `A.1` 的随机序列会在**正常运行**里"
          "撞上池上限，把「断言捕获」污染成「分配失败」）；★ 泄漏专用小池**必须 < 池容量**",
          "**块 / 池容量 / 轮数被动过** —— ★ 先查 `COW_*`、`LEAK_POOL_CAP` 与 `VOCAB`")
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性
# ===========================================================================
def _class_names(cls):
    src = textwrap.dedent(inspect.getsource(cls))
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)          # ★ 也收**定义**的名字（方法名不会以 Name 节点出现）
    return names


def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★★ 守卫实验 ①：把 CoW 改坏 ─────────────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：写共享块时**不复制**（把 CoW 去掉）──")
    guard_ops = [("create", 0, (1, 2, 3, 4, 5)), ("fork", 0, 1),
                 ("append", 1, (9,)), ("append", 0, (7,))]
    g_good = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8), guard_ops)
    g_bad = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8, corrupt="no_cow"), guard_ops)
    nai = apply_ops(NaiveManager(specs.COW_BLOCK_TOKENS), guard_ops)
    print("     构造序列：create(0, 12345) → fork(0→1) → append(1, 9) → append(0, 7)"
          "（B = %d ⇒ 共享一个不满块）" % specs.COW_BLOCK_TOKENS)
    print("     子 seq1 内容：正确 %s · 坏法 %s · 参照(朴素) %s"
          % (g_good.content(1), g_bad.content(1), nai.content(1)))
    print("     ★★★ **读法**：坏法让父、子**直接共写同一个物理块** ⇒ 后写的把先写的**覆盖掉** ⇒")
    print("        子序列读到的是父写进去的 `7` 而不是自己写的 `9`。")
    print("     ★★ **一条容易漏的细节**：只写「共享块的多余槽位」是**看不见**的（`content` 按长度截断）——")
    print("        必须让**两条序列都往同一个索引写**，才能把差异暴露出来。")
    check("D.4", g_good.content(1) == nai.content(1) == (1, 2, 3, 4, 5, 9)
          and g_bad.content(1) != nai.content(1),
          "★★★ **守卫成立**：去掉 CoW 后，子序列内容**从 %s 变成 %s**（参照 = %s）⇒ "
          "★★ **`A.1` 的「逐字相同」对「写共享块要先复制」这条规则是敏感的**"
          % (g_good.content(1), g_bad.content(1), nai.content(1)),
          "**守卫失效** —— ★ 去掉 CoW 后内容居然不变（那 `A.1` 就抓不到它 ⇒ 需要加判据）")

    # ── D.5 ★★★ 守卫实验 ②：fork 时忘 ref += 1 ─────────────────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：`fork` 时**忘** `ref += 1` ──")
    guard2 = [("create", 0, (1, 2, 3, 4)), ("fork", 0, 1), ("remove", 0), ("create", 2, (9, 9, 9, 9))]
    h_good = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8), guard2)
    h_bad = apply_ops(CowManager(specs.COW_BLOCK_TOKENS, 8, corrupt="no_inc_ref"), guard2)
    print("     构造序列：create(0, 1234) → fork(0→1) → remove(0) → create(2, 9999)")
    print("     seq1 内容：正确 %s（链上还有一块）· 坏法 %s（块被误回收后复用）"
          % (h_good.content(1), h_bad.content(1)))
    check("D.5", h_good.content(1) == (1, 2, 3, 4)
          and h_bad.content(1) != h_good.content(1)
          and h_bad.content(1) == (9, 9, 9, 9),
          "★★★ **守卫成立**：`fork` 忘加引用后，`remove(0)` 把块的 ref 减到 0 ⇒ **误回收** ⇒ 被 `create(2)` "
          "复用覆写 ⇒ **seq1 内容从 %s 变成 %s** —— ★★ `A.1` 对 `fork` 的 `ref += 1` 敏感"
          % (h_good.content(1), h_bad.content(1)),
          "**守卫失效** —— ★ 忘加引用后内容不变（那 `A.1` 抓不到它）")

    # ── D.6 ★★ 引文逐字（块粒度）+ 消歧 ─────────────────────────────────
    check("D.6", specs.VLLM_DEFAULT_BLOCK_SIZE == 16 and specs.SGLANG_PAGE_TOKENS == 1
          and specs.VLLM_BLOCK_SIZE_SOURCE_QUOTE == "DEFAULT_BLOCK_SIZE: ClassVar[int] = 16"
          and specs.PAGE_BLOCK_DISAMBIGUATION.startswith("block / page 是同名不同指"),
          "★★ **块粒度两条外部依据逐字钉住**：vLLM 源码常量 `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` · "
          "SGLang 的 `page = 1 token`；★★ **两者同名不同指（差 16 倍）**；\n"
          "        ★★ 载体 = 源码 `vllm/config/cache.py` + LMSYS 博客，抓取日 %s" % specs.KV_FETCH_DAY,
          "**块粒度常量被动过** —— ★ 先查 vLLM 的 `16` 与 SGLang 的 `1`")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.COW_BLOCK_TOKENS == 4 and specs.COW_OP_ROUNDS == 300
          and specs.COW_SEED == 20261009 and specs.COW_POOL_CAP == 4096
          and specs.KV_BLOCK_TOKEN_GRID == (1, 16, 64, 256)
          and specs.LLAMA2_7B_KV_BYTES_PER_TOKEN == 524288
          and specs.VLLM_DEFAULT_BLOCK_SIZE == 16 and specs.SGLANG_PAGE_TOKENS == 1
          and specs.KV_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == ()
          and BLOCK_TABLE_ENTRY_BYTES == 4 and VOCAB == 100 and LEAK_POOL_CAP == 32
          and GIB == 1073741824,
          "★★★ **四个标量 + 一个元组逐元素 + 三个程序内常量**钉住（块 · 轮数 · 种子 · 池容量 · "
          "块网格 · 每 token 字节 · 块表每条字节 · 词表 · 泄漏专用小池 · 抓取日）；★★ `UNSOURCED == ()` 一并钉住",
          "**某个常量被动过**")

    # ── D.8 ★★★ 独立性（源码级）：两实现互不引用 ────────────────────────
    cow_names = _class_names(CowManager)
    nai_names = _class_names(NaiveManager)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：两实现互不引用 ──")
    print("     `CowManager` 里出现 `NaiveManager` = %s ；`NaiveManager` 里出现 `CowManager` = %s"
          % ("NaiveManager" in cow_names, "CowManager" in nai_names))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", "NaiveManager" not in cow_names and "CowManager" not in nai_names
          and "content" in cow_names and "content" in nai_names,
          "★★ **源码级线索成立**：**参照（朴素深拷贝）与被测（CoW）互不引用**（★ **两侧都审**）——"
          " ★★ 于是「两实现内容相同」**不是恒真**，而是**两条独立路径的会合**",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 边界 ─────────────────────────────────────────────────────
    B = specs.COW_BLOCK_TOKENS
    m = CowManager(B, 8)
    m.create(0, tuple(range(1, B + 1)))
    m.create(1, ())
    m.fork(0, 2)
    check("D.9", m.content(0) == tuple(range(1, B + 1)) and m.content(1) == ()
          and m.content(2) == tuple(range(1, B + 1))
          and len(m.seqs[0]) == 1 and m.content(1) != m.content(0),
          "★★ **边界**：**恰好填满一块**（`%d` 个 token ⇒ `1` 块，不多开第二块）；"
          "**空序列**内容 = `()`；`fork` 出来的子序列内容与父相同；"
          "★ 空序列与满序列**内容不同**（★ 下标读取没有越界，因为按 `len` 截断）" % B,
          "**边界值被动过**")


# ===========================================================================
# 9. 汇总 / 主程序
# ===========================================================================
def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类（本域最强的一条）**：引用计数 + CoW vs **朴素深拷贝**"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：本主题的块管理器是 **`[E]` 级（本库实现 / 假设）**；块粒度取自 vLLM / SGLang（`[B]`）。")
    print("★ 本程序只核对「**块管理器的不变量**」——**不是**任何真实引擎的块分配器，**也不碰** kernel 侧。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_block_management.py —— 系统侧的块管理器：块池 / 引用计数 / copy-on-write")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **系统侧（块管理）**；★★ 本程序**不碰** kernel 侧（页表 / 碎片 / 布局 —— 回链 `40-kernels/05`）。")
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无引擎。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
