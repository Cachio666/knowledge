#!/usr/bin/env python3
r"""01-reference-independence —— 参考实现与被测实现是否真的独立。

对应文档：`../01-reference-first.md`

★ 本程序的可证伪断言：
  A 类  无缺陷（对照组）时，**两个**参考实现报告的最大误差都必须是 **0.0**；
        参考实现报告的核对数（补位元素数）必须恰好等于 `(-N) % 4`。
        **把 `N` 改掉、把补位值改掉，A 类立即给出不同的数。**
  B 类  缺陷位于**独立部分**（被测实现自己的分组逻辑）时，**两个**参考实现**都能**捕获。
        若把 `R_shared` 的独立部分写成与 `F` 共用同一段代码，B 类会退化成「都捕获不到」。
  C 类  ★★ 缺陷位于**共享代码**时：
        ① 共享代码版参考实现的**捕获数必须为 0**；
        ② 且它报出的**最大误差必须是 0.0**——即**连最紧的那条容差都会通过**，
           不是「没抓到」，而是「显示一切正常」；
        ③ 独立版参考实现在**同一次运行、同一份输入**上捕获到 **3.0**。
        三个断言都要在输出里出现。**把 `S` 的补位值从 1.0 改成 0.0，C 类立即失败。**
  D 类  `specs.py` §1 的每条常量逐条被断言钉住；含一条 `ast` 的跨程序机械检查——
        三个程序的常量覆盖表的**并集 == `specs.py` 的全部模块级常量**、且**两两不交**。

跑法：`make 01_reference_independence`。纯 python3 + numpy，不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则本程序的结论不可比）：
  · 本程序里的「被测实现」与「两个参考实现」都是**本库构造的仿真对象**，
    不是任何真实框架的实现，也**没有**把它说成是某个项目的真实故障
    （见 `specs.UNSOURCED` 第 4 项）。
  · 外部事实（vLLM IR 的 native 三合一、`register_impl` 的 schema 校验、
    dispatch 一致性的 `rtol=0/atol=0`）来自 `specs.py` §1，全部标
    **[B]（某一实现的官方规格）**⇒ 只支撑「vLLM 的做法是……」。
  · **本程序不含任何 kernel 实现代码**（域 README 的硬边界）：`S`/`F`/`R_*` 是
    **数组变换**，用来把「共享子表达式」这件事变成可计算的对象，与 GPU 无关。
"""

import ast
import inspect
import pathlib
import sys
import textwrap

import numpy as np

import specs as SP

FAILED = []


def check(name, cond, detail=""):
    print(f"    [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def sub(t):
    print()
    print(f"  ── {t}")


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ===========================================================================
# 0. 被检查的对象：一个被测实现 + 两个参考实现
# ===========================================================================
#
# ★ 结构（文档 §1 的符号）：
#       F        = S ∘ f         被测实现
#       R_shared = S ∘ r         参考实现（共享 S，但自己的 r 另写一份）
#       R_indep  = r'            参考实现（完全不碰 S）
#
# ★ **这个结构本身就是本篇要讲的那个结构**：`R_shared` 与 `F` 共享 `S`，
#   而它们**不共享** `f` / `r`。所以「共享了多少」决定了「哪一类缺陷测不出来」。

PAD_MODULUS = 4
N_ELEMENTS = 13
SEED = 20260920

# 共享代码 `S` 的**调用计数器**——它是「独立性」的机械判据（文档 §1 第 5 步）。
SHARED_CALLS = {"n": 0}


def S(x, bug_shared=False):
    """共享代码：把序列补到 `PAD_MODULUS` 的倍数。

    缺陷（`bug_shared=True`）= **用 1.0 而不是 0.0 填充**。
    ★ 这个缺陷是**静默的**：它只改变结果的值，不改变控制流，不抛异常、不越界。
      这是 `specs.SHARED_DEFECT_IS_SILENT_IS_PREREQUISITE` 要求的形态。
    """
    SHARED_CALLS["n"] += 1
    n = len(x)
    pad = (-n) % PAD_MODULUS
    if pad == 0:
        return x.copy()
    return np.concatenate([x, np.full(pad, 1.0 if bug_shared else 0.0)])


def S_short_pad(x):
    """**规格初稿里那个被实跑推翻的缺陷**：补位的**个数**算错（少补一个）。

    ★ 它不是静默的——`len(x)` 不再是 4 的倍数，`reshape(-1, 4)` 会抛 `ValueError`。
      本程序 §C.5 用它来演示**假捕获**：崩溃证明的是「有地方不对」，
      **不是**「这个参考实现能测出这个缺陷」。
    """
    n = len(x)
    pad = max((-n) % PAD_MODULUS - 1, 0)
    if pad == 0:
        return x.copy()
    return np.concatenate([x, np.zeros(pad)])


def F(x, bug_shared=False, bug_own=False):
    """被测实现：补位（走 `S`），然后按 `PAD_MODULUS` 个一组求和。

    `bug_own=True` 是**位于独立部分**的缺陷：分组时丢掉每组的最后一个元素。
    ★ 同样静默：补位之后每组都是满的，切片永远合法，不会抛异常。
    """
    y = S(x, bug_shared)
    groups = y.reshape(-1, PAD_MODULUS)
    if bug_own:
        groups = groups[:, :PAD_MODULUS - 1]
    return groups.sum(axis=1)


def R_shared(x, bug_shared=False):
    """共享代码版参考实现：**走同一个 `S`**，分组用另一套等价写法。

    ★ 它的「独立部分」永远是**正确的**——它只可能通过 `S` 继承缺陷。
      这不是偷懒：它正是「共享代码的参考实现」这一类东西的**忠实模型**。
    """
    y = S(x, bug_shared)
    return np.array([np.add.reduce(y[i:i + PAD_MODULUS]) for i in range(0, len(y), PAD_MODULUS)])


def R_indep(x):
    """独立版参考实现：**不碰 `S`**，只对真实存在的元素求和。

    ★ 它连「补位」这个操作都没有——所以它对「补位值错」这件事**天然免疫**。
    """
    n = len(x)
    return np.array([x[i:i + PAD_MODULUS].sum() for i in range(0, n, PAD_MODULUS)])


def make_input(n=N_ELEMENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=n).astype(np.float64)


def maxdiff(a, b):
    """两个结果的最大绝对差。形状不同 ⇒ 直接判为「不可比」，返回 `inf`。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def captured(diff):
    """捕获判据：最大误差**严格大于 0**。

    ★ 用严格大于 0，而不是「大于容差」——因为本程序要展示的正是
      **「误差为 0 ⇒ 任何容差都放行」**（§C.3）。
    """
    return int(diff > 0.0)


x = make_input()
PAD_FILLED = (-N_ELEMENTS) % PAD_MODULUS

head("§0  被检查的对象")
print(f"  输入：n = {N_ELEMENTS}，元素 = 0–9 的随机整数（固定种子 SEED = {SEED}）")
print(f"        x = {x.astype(int).tolist()}")
print(f"  补位个数 = (-{N_ELEMENTS}) % {PAD_MODULUS} = {PAD_FILLED}")
print(f"  三个实现：F = S∘f（被测）· R_shared = S∘r（共享 S）· R_indep = r'（不碰 S）")
print(f"  ★ 注意 R_shared 与 F **共享 S 但不共享分组逻辑**——共享的范围决定了漏报的范围。")


# ===========================================================================
# A. 对照组：无缺陷时，两个参考实现都必须报 0.0
# ===========================================================================
head("§A  对照组（无缺陷）：两个参考实现都必须报 0.0")

f0 = F(x)
a_shared = maxdiff(f0, R_shared(x))
a_indep = maxdiff(f0, R_indep(x))

sub("A.1  两个参考实现各自的最大误差")
print(f"    F(x)          = {f0.tolist()}")
print(f"    R_shared(x)   = {R_shared(x).tolist()}")
print(f"    R_indep(x)    = {R_indep(x).tolist()}")
check("A.1 F vs R_shared 的最大误差为 0.0", a_shared == 0.0, f"{a_shared!r}")
check("A.1 F vs R_indep  的最大误差为 0.0", a_indep == 0.0, f"{a_indep!r}")

sub("A.2  参考实现报告出的中间量：补位个数必须恰好是 (-N) % 4")
n_filled = len(S(x)) - len(x)
check("A.2 ★ 补位个数 == (-N) % PAD_MODULUS（这个 3 是算出来的，不是调出来的）",
      n_filled == PAD_FILLED == 3, f"{n_filled} vs {PAD_FILLED}")
check("A.2 ★ 补位值**全为 0**（对照组里 `S` 没有缺陷）",
      bool(np.all(S(x)[len(x):] == 0.0)))
check("A.2 改 N 就改这个数 ⇒ 它是**可反驳的**（把它硬编码成 3 会在别的 N 上失败）",
      len(S(make_input(15))) - 15 == (-15) % PAD_MODULUS == 1)

sub("A.3  三个实现的输出长度一致（否则 maxdiff 无意义）")
check("A.3 三者长度相同", len(f0) == len(R_shared(x)) == len(R_indep(x)) == 4,
      f"{len(f0)} / {len(R_shared(x))} / {len(R_indep(x))}")


# ===========================================================================
# B. 缺陷位于**独立部分**时，两个参考实现都能捕获
# ===========================================================================
head("§B  缺陷在独立部分：两个参考实现**都能**捕获")

fB = F(x, bug_shared=False, bug_own=True)
b_shared = maxdiff(fB, R_shared(x))
b_indep = maxdiff(fB, R_indep(x))

sub("B.1  缺陷：被测实现自己的分组丢掉每组最后一个元素")
print(f"    F(bug_own=True)              = {fB.tolist()}")
print(f"    R_shared（独立部分是正确的）= {R_shared(x).tolist()}")
print(f"    R_indep                      = {R_indep(x).tolist()}")
check("B.1 F vs R_shared 捕获（maxdiff > 0）", captured(b_shared) == 1, f"maxdiff = {b_shared}")
check("B.1 F vs R_indep  捕获（maxdiff > 0）", captured(b_indep) == 1, f"maxdiff = {b_indep}")
print("    ★ 两个都捕获，是因为**共享的只有 S**——分组逻辑 `f` 与 `r` 是两份不同的代码。")
print("      ⇒ 定理的作用域要写准：漏报的是**共享的那一部分**，不是「共享代码版参考实现」整体。")

sub("B.2  ★ 反事实：若把 R_shared 的分组逻辑改成与 F 共用同一段代码，两种缺陷都测不出来")
print("    这不是假设——它就是 §C 的形态（把共享范围从 S 扩大到 S∘f）。")
print("    ⇒ 「参考实现独立不独立」是一个**连续量**：共享的范围越大，测得出的东西越少。")


# ===========================================================================
# C. ★★ 缺陷位于**共享代码**时：捕获数为 0，且报出的最大误差是 0.0
# ===========================================================================
head("§C  ★★ 缺陷在共享代码：捕获数 = 0，且显示一切正常")

fC = F(x, bug_shared=True)
c_shared = maxdiff(fC, R_shared(x, bug_shared=True))      # ← 两边都走同一个 S（都带 bug）
c_indep = maxdiff(fC, R_indep(x))                          # ← R_indep 不碰 S

sub("C.1  同一次运行、同一份输入，两个参考实现给出的最大误差")
print(f"    F(bug_shared=True)            = {fC.tolist()}")
print(f"    R_shared(bug_shared=True)     = {R_shared(x, bug_shared=True).tolist()}")
print(f"    R_indep(x)                    = {R_indep(x).tolist()}")
print(f"    ── F vs R_shared  maxdiff = {c_shared}")
print(f"    ── F vs R_indep   maxdiff = {c_indep}")

sub("C.2  断言①：共享代码版参考实现的**捕获数必须为 0**")
cap_shared = captured(c_shared)
check("C.2 ★★ F vs R_shared 的捕获数 == 0",
      cap_shared == SP.SHARED_REFERENCE_CAPTURE_COUNT_ON_SHARED_DEFECT == 0,
      f"捕获数 = {cap_shared}")

sub("C.3  断言②：它报出的最大误差必须是 **0.0**，即「显示一切正常」")
check("C.3 ★★ maxdiff 恰好是 0.0（不是「小」，是**恒等**）", c_shared == 0.0, f"{c_shared!r}")
tightest = min(SP.VLLM_DEFAULT_TOLERANCES[d][1] for d in ("float16", "float32", "float64"))
check("C.3 ★★ 这个 0.0 **能通过本库收录的最紧那条容差** ⇒ 它不是「差点没抓到」",
      c_shared <= tightest,
      f"0.0 ≤ rtol({tightest}) —— 任何容差都会放行")
print("    ★★ 措辞的精确性（正文不得写成「共享代码的参考实现一定测不出任何 bug」）：")
print("      零捕获成立的前提是 ① 比对的是 **S 之后未经独立处理的输出**；② 缺陷**静默**。")
print("      本程序 §B 已经给出前提①的反例：缺陷在非共享部分时，它照样捕获。")

sub("C.4  断言③：独立版参考实现在同一次运行上捕获到 3.0")
cap_indep = captured(c_indep)
check("C.4 ★★ F vs R_indep 的捕获数 == 1",
      cap_indep == 1, f"捕获数 = {cap_indep}")
check("C.4 ★★ maxdiff 恰好 == (-N) % 4 == 3.0（补了 3 个 1.0，独立版看不见它们）",
      c_indep == float(PAD_FILLED) == SP.INDEPENDENT_REFERENCE_CAPTURE_COUNT_ON_SHARED_DEFECT == 3.0,
      f"{c_indep}")
check("C.4 ★ 同一个缺陷、同一份输入：**共享版的捕获数 0，独立版的捕获数 1**",
      (cap_shared, cap_indep) == (0, 1))

sub("C.5  ★ 假捕获：换成「会崩溃」的缺陷，得到的不是捕获而是异常")
false_capture = 0
try:
    y = S_short_pad(x)
    _ = y.reshape(-1, PAD_MODULUS)      # ← 15 不是 4 的倍数 ⇒ ValueError
except ValueError as e:
    false_capture = 1
    print(f"    异常：ValueError: {e}")
check("C.5 ★ 规格初稿的那个缺陷**真的会崩**（确认它属于「假捕获」那一类）",
      false_capture == 1)
check("C.5 ★ 崩溃必须被记成 `false_capture`，**不得并进捕获数**"
      "（否则变异测试会自己骗自己）",
      SP.CRASH_COUNTS_AS_FALSE_CAPTURE is True)

sub("C.6  ★★ 三个 0.0：同一个数，两种含义——这是判读错误的常见来源")
same = maxdiff(f0, f0)
check("C.6 ① 同一个实现与自己比：maxdiff = 0.0", same == 0.0)
check("C.6 ② vLLM 要求 dispatch 路径与直接调用**恰好一致**（rtol=0, atol=0）",
      SP.VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE == (0.0, 0.0))
print("    ⇒ ①与②的 0 意味着**正确**（两边本来就是同一个东西）；")
print(f"      ③ §C.2 的 0 = {c_shared} 意味着**测不出来**。")
print("    ★★ 所以看到 `maxdiff == 0.0` 时的第一个问题是「**两边是不是同一个东西**」，")
print("      不是「通过」。这也正是 vLLM 把这两处拆成两个不同设计的原因。")

sub("C.7  ★ 零捕获不是「参考实现写得差」——它是代数恒等")
print("    F = S∘f、R = S∘r ⇒ 比对 S 直接产出的量时，两者之差 = S 的错误 − S 的错误 ≡ 0。")
check("C.7 ★ 定理的代数形式被本程序断言",
      SP.INDEPENDENCE_SHARED_SUBEXPRESSION_DIFF_IS_ZERO is True)
check("C.7 ★ 定理的前提被本程序断言（缺了它，上面的说法不成立）",
      SP.SHARED_DEFECT_IS_SILENT_IS_PREREQUISITE is True)


# ===========================================================================
# D. 常量自洽 + 跨程序覆盖检查
# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")

sub("D.1  机械独立性判据①：参考实现**有没有真的经过共享代码**")
SHARED_CALLS["n"] = 0
R_indep(x)
indep_calls = SHARED_CALLS["n"]
SHARED_CALLS["n"] = 0
R_shared(x)
shared_calls = SHARED_CALLS["n"]
check("D.1 ★★ `R_indep` 调用 `S` 的次数 == 0（「独立」在这里是一个**计数**）",
      indep_calls == 0, f"{indep_calls}")
check("D.1 ★ `R_shared` 调用 `S` 的次数 == 1（它**确实**共享了）",
      shared_calls == 1, f"{shared_calls}")
check("D.1 ★★ 上面两条一起 ⇒ 「我们的参考实现是独立的」这句话**有判据**，不是自我声明",
      SP.INDEPENDENCE_IS_MECHANICALLY_CHECKABLE is True)

sub("D.2  机械独立性判据②：参考实现**不得比被测实现更复杂**")
# ★ §4 踩坑 ③：参考实现比被测实现还复杂时，出错无法判断是哪边的错。
#   判据用**语句节点数**做一个粗糙但可计算的代理——本程序明写它是代理，不是「复杂度」的正确定义。
def stmt_count(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return sum(1 for nd in ast.walk(tree) if isinstance(nd, ast.stmt))


n_F, n_ri, n_rs = stmt_count(F), stmt_count(R_indep), stmt_count(R_shared)
print(f"    语句节点数：F = {n_F} · R_indep = {n_ri} · R_shared = {n_rs}")
check("D.2 ★ 独立版参考实现的语句数 **不多于** 被测实现（否则出错时无法归因）",
      n_ri < n_F, f"{n_ri} < {n_F}")
print("    ⚠ 这是**代理指标**，不是「复杂度」的正确定义——本程序明写它的局限，不假装它是定理。")

sub("D.3  参考实现的选取规则（三条优先级）")
check("D.3 第一档是 CPU fp64，第二档 CPU fp32（重点是**与被测路径机制不同**）",
      SP.REFERENCE_SELECTION_ORDER[:2] == ("cpu_fp64", "cpu_fp32"),
      f"{SP.REFERENCE_SELECTION_ORDER[:2]}")
check("D.3 第三档是框架内置 op、第四档才是手写朴素版（最弱的一档在最后）",
      SP.REFERENCE_SELECTION_ORDER[2:] == ("existing_framework_op", "naive_numpy"),
      f"{SP.REFERENCE_SELECTION_ORDER[2:]}")
check("D.3 ★ 四条互不相同（重复会退化成 3 档，改变优先级语义）",
      len(set(SP.REFERENCE_SELECTION_ORDER)) == 4)

sub("D.4  vLLM 做法：native 实现的三合一（**[B] 级 · 只支撑「vLLM 这么做」**）")
check("D.4 native 实现恰好承担三个角色",
      SP.VLLM_IR_NATIVE_IMPL_ROLES
      == ("semantic_definition", "default_implementation", "test_baseline"),
      f"{SP.VLLM_IR_NATIVE_IMPL_ROLES}")
check("D.4 保留 provider 名是 'native'", SP.VLLM_IR_NATIVE_PROVIDER_NAME == "native")
check("D.4 native 恒 supported ⇒ 永远有兜底", SP.VLLM_IR_NATIVE_ALWAYS_SUPPORTED is True)
print("    ★★ 三合一解决的是**漂移**（三份东西会互相不一致），**不解决**「两者共有部分的错误」——")
print("      它与 §C 的定理**不矛盾**，但作用域完全不同（P7：引文与断言作用域不匹配）。")

sub("D.5  vLLM 做法：schema 校验与热路径约束")
check("D.5 `register_impl` 的 impl 必须与 native 的 schema 逐个参数一致",
      SP.VLLM_IR_IMPL_MUST_MATCH_NATIVE_SCHEMA is True)
check("D.5 不一致时抛 ValueError（**当场报错**，不是静默注册）",
      SP.VLLM_IR_SCHEMA_MISMATCH_RAISES == "ValueError")
check("D.5 关键字专用参数被禁止（因为支持性检查跑在热路径上）",
      SP.VLLM_IR_SUPPORTS_ARGS_KEYWORD_ONLY_DISALLOWED is True)

sub("D.6  ★★ dispatch 一致性的容差是 (0.0, 0.0)——两个 0 含义相反")
check("D.6 容差是 (rtol, atol) = (0.0, 0.0)", SP.VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE == (0.0, 0.0))
check("D.6 ★ 它要求的是**逐位相同**，不是「接近」",
      SP.VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE[0] == SP.VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE[1] == 0.0)
check("D.6 ★ 与 §C.2 的 0 是**同一个数值、不同含义**——本程序把两者并排断言",
      c_shared == SP.VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE[0] == 0.0)

sub("D.7  vLLM 做法：数值比对之外的检查（`opcheck`）")
check("D.7 语义自检的 API 名是 opcheck", SP.VLLM_IR_OP_SEMANTIC_SELFCHECK_API == "opcheck")
print("    ★ 容差比对**查不出**「参数被就地改了」与「输出与输入别名」——验证矩阵里")
print("      必须有一维不属于数值比对（02 篇的 §1 把它列为「对抗输入」维的一部分）。")

sub("D.8  `UNSOURCED` 逐条比对（防止「既保留数字又记 UNSOURCED」）")
UNSOURCED_EXPECTED = (
    "spec_d_tolerance_table_source_page",
    "spec_d_int8_tolerance_row",
    "vllm_bf16_atol_two_readings_disagree",
    "real_world_shared_reference_escaped_bug",
    "debug_layer_priors_and_costs_measured",
    "cold_start_times_belong_to_tuning_topic",
)
check("D.8 UNSOURCED 元组与逐条复核的结果完全一致",
      tuple(SP.UNSOURCED) == UNSOURCED_EXPECTED, f"{SP.UNSOURCED}")
check("D.8 ★ 第 4 项存在 ⇒ §C 的仿真是**本库构造的**（正文不得暗示它是真实故障）",
      "real_world_shared_reference_escaped_bug" in SP.UNSOURCED)
check("D.8 UNSOURCED 里没有任何一项出现在正文常量中（二者只能取其一）",
      not (set(SP.UNSOURCED) & {n for n in dir(SP) if n.isupper() and n != "UNSOURCED"}))

sub("D.9  ★ 全模块常量覆盖的**机械检查**（本域 C6 的机器化）")
COVERED_01 = (
    # §1 参考实现的独立性（E 定理 + B 级 vLLM 的做法）
    "INDEPENDENCE_SHARED_SUBEXPRESSION_DIFF_IS_ZERO",
    "SHARED_REFERENCE_CAPTURE_COUNT_ON_SHARED_DEFECT",
    "INDEPENDENT_REFERENCE_CAPTURE_COUNT_ON_SHARED_DEFECT",
    "SHARED_DEFECT_IS_SILENT_IS_PREREQUISITE",
    "CRASH_COUNTS_AS_FALSE_CAPTURE",
    "REFERENCE_SELECTION_ORDER",
    "INDEPENDENCE_IS_MECHANICALLY_CHECKABLE",
    "VLLM_IR_NATIVE_IMPL_ROLES",
    "VLLM_IR_NATIVE_PROVIDER_NAME",
    "VLLM_IR_NATIVE_ALWAYS_SUPPORTED",
    "VLLM_IR_IMPL_MUST_MATCH_NATIVE_SCHEMA",
    "VLLM_IR_SCHEMA_MISMATCH_RAISES",
    "VLLM_IR_SUPPORTS_ARGS_KEYWORD_ONLY_DISALLOWED",
    "VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE",
    "VLLM_IR_OP_SEMANTIC_SELFCHECK_API",
    # 待办清单
    "UNSOURCED",
)
print(f"    01 覆盖 {len(COVERED_01)} 条")
for _n in COVERED_01:
    print(f"      {_n:56s} 有断言")


def _covered_names_in(filename, varname):
    """用 `ast` 读同目录另一个程序里的 `COVERED_*` 元组（**不用正则、不用 grep**）。"""
    src = pathlib.Path(__file__).resolve().parent / filename
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == varname:
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{filename} 里找不到 {varname}")


ALL_CONSTANTS = frozenset(n for n in dir(SP) if n.isupper() and not n.startswith("_"))
c02 = _covered_names_in("02_verification_matrix.py", "COVERED_02")
c03 = _covered_names_in("03_debug_order.py", "COVERED_03")
union = frozenset(COVERED_01) | frozenset(c02) | frozenset(c03)
print(f"    specs.py 的模块级常量共 {len(ALL_CONSTANTS)} 条")
print(f"    三个程序的覆盖：01={len(COVERED_01)} 02={len(c02)} 03={len(c03)} 并集={len(union)}")
print(f"    未被任何程序断言 = {sorted(ALL_CONSTANTS - union)}")
print(f"    断言了但 specs 里没有 = {sorted(union - ALL_CONSTANTS)}")
check("D.9 ★ 三程序覆盖的**并集**恰好等于 specs.py 的全部模块级常量（无漏网）",
      union == ALL_CONSTANTS, f"漏 {sorted(ALL_CONSTANTS - union)}")
check("D.9 三个程序之间**没有**重复覆盖（每条常量恰好一个归属）",
      len(COVERED_01) + len(c02) + len(c03) == len(union),
      f"{len(COVERED_01)}+{len(c02)}+{len(c03)} vs {len(union)}")
check("D.9 COVERED_01 里没有拼错的常量名",
      set(COVERED_01) <= ALL_CONSTANTS, f"{sorted(set(COVERED_01) - ALL_CONSTANTS)}")
check("D.9 `ALL_CONSTANTS` 非空（否则上面两条会退化成恒真）",
      len(ALL_CONSTANTS) > 40, f"{len(ALL_CONSTANTS)}")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 3 组 · B 类 2 组 · C 类 7 组 · D 类 9 组）")
print()
print("  ★ 本程序的对象是**本库构造的三个数组实现**，不是任何真实框架的实现；")
print("    外部事实全部来自 specs.py §1（B 级：某一实现的官方规格，只支撑「vLLM 这么做」）；")
print("    §C 的「零捕获」是**本库构造的仿真**——检索未找到任何真实案例")
print("    （specs.UNSOURCED 第 4 项），正文不得暗示它是某个项目的故障。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}（未 import torch）")
