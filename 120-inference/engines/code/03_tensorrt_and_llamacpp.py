#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_tensorrt_and_llamacpp.py —— **TensorRT-LLM 的编译式路线**（`in-flight batching` 的 `[A]` 级术语
+ **该实现当前**的顺序约束）与 **llama.cpp 的边界**（本文不展开的原因）。

配套文档：`03-tensorrt-llm-and-llamacpp.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题【没有 A 类】**：本机**无 GPU、无任何引擎**（无 TensorRT-LLM / llama.cpp）
  ⇒ TensorRT-LLM 的**【行为与性能】无处可验**。★ 本程序跑在默认 `python3` 下、只用标准库
  ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**（**程序内断言**）。
  ★★ **如实点破**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**
  ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。
  ★★★ **不得把「常量钉住」冒充 A 类**（那是 `P12` 的变体）⇒ 本程序的一切钉住归 `D` 类。
  ★ `D.0` **机械地**证明「本文件里一条 A 类判据都没有」（AST 扫自己）。

★★★ **本程序只写「各家的选择与差异」** —— ★ **机制一律回链**（`P37`）：**分页** → `kv-cache/02`；
  **前缀树** → `kv-cache/03`；**请求调度器** → `batching-and-scheduling/01`；
  **PagedAttention 的算式** → `130-quantization/kv-cache-quantization/01:68-84`；
  ★★ **TensorRT-LLM 的 per-iteration 时间预算** → `45-kernel-lifecycle/requirement-discovery/03`。

★★★ **llama.cpp 的处置（★ 如实报告）**：★★★ **研究原始报告的 13 条里【没有】llama.cpp 的一手来源**
  ⇒ **不得**凭印象写它的实现细节（那是「把 B 级当 A 级」的现场）。★ 本程序把这一条
  **变成可被断言的边界**（`C.3` 用一个 `[E]` 记录 + 「不得自造引文」的机械检查），
  ★ 而不是编一段社区转述。★★ **「数不到」≠「不存在」**。

★★★ **本程序的禁写清单（计划 `§0.7 C.5`）**：
  · `F1`（**不得写那类「没有第二个独立主体」的断言**）—— ★ 本程序一律写「**TensorRT-LLM 的做法是……**」；
  · ★★ **不得把「context 段必须排在 generation 段之前」说成 continuous batching 的【通用约束】**
    —— 它是**该实现当前版本**的约束（`§0.7 C` 事实 8）。

★★★ **本程序的判据纪律（★ 形态 A 的根治）**：★ 抬头写了「不裸写某某」这条纪律，
  本程序里**必须有一条断言能抓住「没做到」** —— 见 `D.9`：
  ★★ 判据是**等式**「裸写计数 == 显式豁免清单覆盖的计数」，**不是「≤ N」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `B.1` ★★ **编译式路线的组合账**：引擎份数 = 三个配置轴的**乘积**；
  · `B.2` ★★ **顺序约束的结构后果**：**紧凑（无 padding）张量 + 顺序要求 ⇒ 布局被唯一确定**；
  · `C.1` ★★★ **把该实现的顺序约束当成 continuous batching 的【通用约束】**必须被拒；
  · `C.2` ★★★ **把 `in-flight batching` 当成 ORCA 的原始术语**必须被拒（NVIDIA 是 `[A]`，ORCA 是 `[C]`）；
  · `C.3` ★★★ **llama.cpp 的一手来源【未取到】** ⇒ **不得自造引文**（机械检查 `specs` 里没有它的引文常量）；
  · `D.0` ★★★ **机械证明「本文件没有 A 类判据」**（AST 扫 `check("A…")` == 0）；
  · `D.5` ★★ 六条引文**逐字**钉住；`D.7` ★★★ **逐元素**钉住；`D.9` ★★★ **术语纪律**。
"""
import ast
import collections
import sys

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（engines）取 `-- FAIL: engines`；`mutate.py` 用 AST 读出（**不写死**）。
ASSERT_FAIL_MARK = "-- FAIL: engines"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: decoding-strategies", "-- FAIL: serving-infra")

# ★★ **术语纪律**（本域 README §同名不同指）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
TERM_KV_CACHE = "KV 缓存"
TERM_SCHEDULER = "请求调度器"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ 形态 A：「纪律宣称了而断言没盯住」，本域已栽过两次）：
#   本程序源码里**裸写那个被禁的裸词**（`F1` 明文禁的那类断言用词）的【计数】
#   必须**恰好等于**下面这份**显式豁免清单**覆盖的计数 ——
#   ★★★ 判据是【计数 == 豁免覆盖数】，**不是「≤ N」**（「≤ N」挡不住**新增**一处裸写）。
#   ★ 每一项都写明【为什么豁免】：
#     ① 抬头注释里**提到**这条纪律本身 —— 那是 **mention**（它写该词**恰恰因为在规定不许写它**）：
#        · ★★★ 本主题一律【写明是哪一家】：不裸写「业界」这类无第二个独立主体的断言（`F1`）。
#     ② `D.9` 的判据文案里**复述**这条纪律 —— ★ 同为 mention。
BARE_ATTRIBUTION_EXEMPT = (
    "不裸写「业界」这类无第二个独立主体的断言",
    "**不裸写「业界」**",
)

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★★ **本程序的算例**（`[E]`，不是外部事实）：一个 batch 的构成 ——
#   `3` 条 context 段（长度 `4/6/5`）+ `2` 条 generation 段（各 `1` token）。★ 越短越便于手算。
BATCH_CONTEXT_LENS = (4, 6, 5)
BATCH_N_GEN = 2


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
    """
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
# 1. 被测的「编译式路线校验器」—— ★★ **它【不是】A 类**（本机没有 TensorRT-LLM 可测）
#    ★★★ 它是**本库的一段校验器**：把「编译式路线的份数」「顺序约束的后果」变成**可数**的量，
#      用来判「把某实现的选择当成通用约束」这类错法。★ 一切钉住归 `D` 类。
# ===========================================================================
def n_engine_builds(axes):
    """★ **编译式路线的份数** = 三个配置轴的**乘积**（★ 本主题的模型假设）。

    ★★ 假设：每个 `(max_batch_size, max_input_len, max_output_len)` 组合**各编一份** ——
      ★ **这是假设，不是任何引擎的实测**（本机无引擎）。
    """
    out = 1
    for a in axes:
        out *= a
    return out


def packed_total(context_lens, n_gen):
    """★ **紧凑（无 padding）输入张量的总长**：`Σ context 段长度 + (gen 段数 × 1 token)`。

    ★★ 依据：`inflight batching` **要求张量 packed**（逐字见 `specs.TRTLLM_PACKED_QUOTE`）。
    """
    return sum(context_lens) + n_gen


def first_gen_offset(context_lens):
    """★★★ **第一条 generation 段的下标** = `Σ context 段长度`。

    ★★★ 依据：`context 段必须排在 generation 段之前`（逐字见 `specs.TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE`）
      ＋ **紧凑张量**（无 padding）⇒ ★ **布局被【唯一确定】**（本题的 B.2）。
    """
    return sum(context_lens)


def layout_is_ordered(types):
    """★ 顺序要求是否满足：所有 `"c"`（context）必须排在所有 `"g"`（generation）**之前**。"""
    seen_g = False
    for t in types:
        if t == "g":
            seen_g = True
        elif t == "c" and seen_g:
            return False
    return True


def constraint_is_universal(qualifier_present):
    """★★★ **「顺序约束是 continuous batching 的【通用约束】」成立吗？** —— 只有当文档**没有**限定语时才成立。

    ★★★ 取数到的原文**带限定语**（「In the current implementation …」＋「may or may not be relaxed」）
      ⇒ 本函数返回 `False` ⇒ ★★ **把它当通用约束是错的**（`C.1` 判它）。
    """
    return not qualifier_present


# ===========================================================================
# 2. A 类 —— ★★★ 【本主题没有 A 类】
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ 【本主题没有 A 类】—— 本机无 GPU / 无任何引擎")
    print("=" * 78)
    print("  ★★★ 「TensorRT-LLM 的行为与性能」**无处可验**：本机既没有 GPU 也没有该引擎。")
    print("      ⇒ 本程序**跑在默认 `python3` 下** ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**")
    print("        （**程序内断言**）；★ 而「**把不可测的那一档写明**」是【另一条】纪律（不是 §2 的形态）。")
    print("      ★★ **如实点破**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**")
    print("        ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。")
    print("  ★★★ **不得把「常量钉住」冒充 A 类**（`P12` 的变体）⇒ ★★ 钉住一律归 `D` 类。")
    print("  ★★ 本程序只守一件事：**编译式路线的结构账、以及三条禁写真的会被拒**。")
    print()
    n_a = _count_a_checks_in_source()
    print("  ── D.0 ★★★ **机械证明**：AST 扫本文件，`check(\"A...\")` 的调用数 = **%d** ──" % n_a)
    check("D.0", n_a == 0,
          "★★★ **判据成立**：本文件里**一条 A 类判据都没有**（AST 逐个数 `check()` 的"
          "第一实参，字面量以 `A.` 开头者 = **0**）",
          "★★ 本文件里出现了 A 类判据 —— ★ 那与「本主题没有 A 类」**自相矛盾**")


def _count_a_checks_in_source():
    """★ 数本文件里 `check("A...", ...)` 的调用数（**读自己的源码**，不是写死的数）。"""
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "check" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                    and a0.value.startswith("A."):
                n += 1
    return n


# ===========================================================================
# 3. B 类：**编译式路线的结构性账**
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `[A]`/`[B]` 级的账：**编译式的份数** · **顺序约束的结构后果**")
    print("=" * 78)
    print("  ★★ 载体 = NVIDIA 官方文档 `advanced/gpt-attention.html`（**`[A]` 级**，厂商第一方），"
          "抓取日 **%s**。" % specs.ENGINES_FETCH_DAY)
    print("     逐字（术语）：`%s`" % specs.TRTLLM_INFLIGHT_SYNONYM_QUOTE)

    # ── B.1 ★★ 编译式路线的份数 = 三轴乘积 ───────────────────────────────
    print()
    print("  ── B.1 ★★ **编译式路线的份数** = 三个配置轴的**乘积** ──")
    axes = specs.TRTLLM_CONFIG_AXES
    prod = n_engine_builds(axes)
    print("     配置轴取值数：`max_batch_size` = %d · `max_input_len` = %d · `max_output_len` = %d"
          % (axes[0], axes[1], axes[2]))
    print("     ★★ **份数 = %d × %d × %d = %d**（★ 本主题的模型假设：每个组合**各编一份**）"
          % (axes[0], axes[1], axes[2], prod))
    print("     ★★★ **读法**：**编译式路线的代价是【乘性】的** —— 每加一档配置，份数就乘一次；")
    print("        这与**运行期**决定形态的引擎（只受显存约束）是**两种不同的代价结构**。")
    print("     ★★ 注：本机**无引擎** ⇒ 这是**结构账**，**不是**任何引擎的实测编译次数。")
    check("B.1", prod == 18 and axes == (3, 3, 2)
          and n_engine_builds((axes[0] + 1, axes[1], axes[2])) > prod,
          "★★ **判据成立**：份数 = **%d**（`%d×%d×%d`）；★ 把任一轴加一档 ⇒ 份数 **%d → %d**（乘性增长）"
          % (prod, axes[0], axes[1], axes[2], prod, n_engine_builds((axes[0] + 1, axes[1], axes[2]))),
          "**配置轴或乘积被动过** —— ★ 先查 `TRTLLM_CONFIG_AXES` 与 `n_engine_builds`")

    # ── B.2 ★★ 顺序约束的结构后果 ───────────────────────────────────────
    print()
    print("  ── B.2 ★★ **顺序约束的结构后果**（紧凑张量 + 顺序 ⇒ 布局被唯一确定）──")
    total = packed_total(BATCH_CONTEXT_LENS, BATCH_N_GEN)
    off = first_gen_offset(BATCH_CONTEXT_LENS)
    types = ["c"] * len(BATCH_CONTEXT_LENS) + ["g"] * BATCH_N_GEN
    print("     batch 构成：context 段 `%s`（共 %d token）+ generation 段 `%d` 条（各 1 token）"
          % (BATCH_CONTEXT_LENS, sum(BATCH_CONTEXT_LENS), BATCH_N_GEN))
    print("     ★★ **紧凑张量**总长 = %d + %d = **%d**" % (sum(BATCH_CONTEXT_LENS), BATCH_N_GEN, total))
    print("     ★★★ **第一条 generation 段的下标 = %d**（= `Σ context 长度`）⇒ **布局被唯一确定**"
          % off)
    print("        ★ 依据：`%s`" % specs.TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE)
    print("     ★★ 顺序判定：`%s` ⇒ **%s**（应 `True`）；把 c/g 对调 ⇒ **%s**（应 `False`）"
          % (types, layout_is_ordered(types), layout_is_ordered(list(reversed(types)))))
    print("     ★★ 读法：**顺序要求 + 无 padding ⇒ 没有「自由摆放」的余地** ——")
    print("        布局不是实现细节，而是**输入张量必须满足的前置条件**。")
    check("B.2", total == 17 and off == 15 and layout_is_ordered(types)
          and not layout_is_ordered(list(reversed(types)))
          and layout_is_ordered(["c", "c", "g"]) and not layout_is_ordered(["g", "c"]),
          "★★ **判据成立**：紧凑张量总长 **%d**、第一条 generation 段下标 **%d**（`Σ context`）；"
          "★ 顺序判定对「c 在 g 之后」**敏感**" % (total, off),
          "**顺序判定的结构被动过** —— ★ 先查 `packed_total` / `first_gen_offset` / `layout_is_ordered`")


# ===========================================================================
# 4. C 类：**三条禁写的真现场**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**把实现的选择当通用约束** · **术语误归** · **未取到的一手来源**")
    print("=" * 78)
    print("  ★★★ 三条都对应计划 `§0.7 C.5` 的**明文禁写** 与 **llama.cpp 的边界**。")

    # ── C.1 ★★★ 把该实现的顺序约束当【通用约束】────────────────────────
    print()
    print("  ── C.1 ★★★ **把该实现的顺序约束当成 continuous batching 的【通用约束】**──")
    print("     逐字（约束）：`%s`" % specs.TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE)
    print("     逐字（限定）：`%s`" % specs.TRTLLM_ORDER_RELAX_QUOTE)
    print("     限定语记录：`%s`" % specs.TRTLLM_INFLIGHT_ORDER_QUALIFIER)
    print("     判定：「通用约束」⇒ **%s**（应 `False`，★ 因为限定语在）"
          % constraint_is_universal(bool(specs.TRTLLM_INFLIGHT_ORDER_QUALIFIER)))
    print("     ★★★ **读法**：文档自述该约束**可能在未来版本放宽** ⇒ 它是**该实现当前版本**的，")
    print("        ★★ **不得**说成 continuous batching 的通用约束（`§0.7 C` 事实 8）。")
    check("C.1", not constraint_is_universal(bool(specs.TRTLLM_INFLIGHT_ORDER_QUALIFIER))
          and "current implementation" in specs.TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE
          and "future version" in specs.TRTLLM_ORDER_RELAX_QUOTE
          and specs.TRTLLM_INFLIGHT_ORDER_QUALIFIER
          == ("该顺序约束（context 段必须排在 generation 段之前）是 TensorRT-LLM 当前实现的约束，"
              "文档自述可能在未来版本放宽，不得说成 continuous batching 的通用约束"),
          "★★★ **判据成立**：把该顺序约束当**通用约束**会被拒 —— 原文写的是「**In the current "
          "implementation**」，且自述「**may or may not be relaxed in a future version**」",
          "**顺序约束或其限定语被动过** —— ★ 先查 `TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE` / "
          "`TRTLLM_ORDER_RELAX_QUOTE` / `TRTLLM_INFLIGHT_ORDER_QUALIFIER`")

    # ── C.2 ★★★ 把 in-flight batching 当成 ORCA 的原始术语 ──────────────
    print()
    print("  ── C.2 ★★★ **把 `in-flight batching` 当成 ORCA 的原始术语**──")
    print("     逐字（NVIDIA）：`%s`" % specs.TRTLLM_INFLIGHT_SYNONYM_QUOTE)
    print("     逐字（ORCA）：`%s`" % specs.ORCA_ITERATION_LEVEL_QUOTE)
    print("     ★★★ **读法**：`in-flight batching` 是 **NVIDIA 对 TensorRT-LLM 自己**的术语")
    print("        （`[A]` 级，且它**自己**把三者声明为同义）；而**原始术语** `iteration-level scheduling`")
    print("        的出处是 **ORCA（OSDI '22，`[C]`）** ⇒ ★★ **两者不可混**（`F3` 的邻形）。")
    check("C.2", "also known as continuous batching or iteration-level batching"
          in specs.TRTLLM_INFLIGHT_SYNONYM_QUOTE
          and "granularity of iteration" in specs.ORCA_ITERATION_LEVEL_QUOTE
          and specs.TRTLLM_INFLIGHT_SYNONYM_QUOTE != specs.ORCA_ITERATION_LEVEL_QUOTE,
          "★★★ **判据成立**：`in-flight batching`（NVIDIA，`[A]`）与 `iteration-level scheduling`"
          "（ORCA，`[C]`）是**两条不同出处的术语** —— ★ NVIDIA 自己声明三者同义，"
          "但**原始术语**归 ORCA",
          "**两条术语之一被动过** —— ★ 先查 `TRTLLM_INFLIGHT_SYNONYM_QUOTE` / `ORCA_ITERATION_LEVEL_QUOTE`")

    # ── C.3 ★★★ llama.cpp：一手来源【未取到】⇒ 不得自造引文 ─────────────
    print()
    print("  ── C.3 ★★★ **llama.cpp：一手来源【未取到】** ⇒ **不得自造引文** ──")
    llamacpp_consts = sorted(n for n in dir(specs) if "LLAMACPP" in n)
    print("     `specs` 里与 llama.cpp 有关的常量：`%s`（★ 只有**边界记录**，**没有引文常量**）"
          % llamacpp_consts)
    print("     ★★★ 依据：**研究原始报告的 13 条里【没有】llama.cpp 的一手来源** ⇒")
    print("       ★★ 本库**不得**凭印象写它的实现细节（那是「把 B 级当 A 级」的现场）。")
    print("     ★★ 记录：`LLAMACPP_FIRST_SOURCE_OBTAINED` = **%s**（★ 应 `False`）"
          % specs.LLAMACPP_FIRST_SOURCE_OBTAINED)
    print("     ★★ **一条纪律**：**「数不到」≠「不存在」** —— 它只说明**本次取数没取到**，")
    print("       所以正文写的是「**未取到一手来源，不展开**」，而不是「llama.cpp 没有某性质」。")
    check("C.3", specs.LLAMACPP_FIRST_SOURCE_OBTAINED is False
          and llamacpp_consts == ["LLAMACPP_FIRST_SOURCE_OBTAINED"],
          "★★★ **判据成立**：llama.cpp 的一手来源**未取到**（记录 = `%s`），且 `specs` 里**没有**"
          "任何 llama.cpp 的引文常量 ⇒ **不得自造引文**" % specs.LLAMACPP_FIRST_SOURCE_OBTAINED,
          "**llama.cpp 的边界记录被动过，或 `specs` 里出现了自造的 llama.cpp 引文常量** —— "
          "★ 先查 `LLAMACPP_FIRST_SOURCE_OBTAINED`")


# ===========================================================================
# 5. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何引擎。")
    print("  ★★ 本程序**没有 A 类**（`D.0` 机械证明），**一切钉住归 `D` 类**。")
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

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: engines" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: engines`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.ENGINES_FETCH_DAY == "2026-10-09"
          and specs.TRTLLM_CONFIG_AXES == (3, 3, 2)
          and specs.LLAMACPP_FIRST_SOURCE_OBTAINED is False
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ 配置轴 `%s`；★ llama.cpp 的一手来源**未取到** = `%s`；"
          "★ `UNSOURCED == ()` 一并钉住"
          % (specs.ENGINES_FETCH_DAY, specs.TRTLLM_CONFIG_AXES,
             specs.LLAMACPP_FIRST_SOURCE_OBTAINED))

    # ★★★ 口径守卫（★ 跑在任何计算之前）：配置轴必须**全 ≥ 1**（否则乘积为 0 ⇒ `B.1` 退化）；
    #   batch 算例必须**非空**（否则 `B.2` 的偏移无意义）。
    ok_axes = all(a >= 1 for a in specs.TRTLLM_CONFIG_AXES) and len(specs.TRTLLM_CONFIG_AXES) == 3
    ok_batch = len(BATCH_CONTEXT_LENS) >= 1 and min(BATCH_CONTEXT_LENS) >= 1 and BATCH_N_GEN >= 1
    print("     ★★ 配置轴全 ≥ 1 且三轴 = %s；batch 算例非空 = %s" % (ok_axes, ok_batch))
    check("D.3b", ok_axes and ok_batch,
          "★★★ **口径守卫**：配置轴 `%s` 三轴且全 ≥ 1（★ 否则乘积为 0 ⇒ `B.1` 退化）；"
          "batch 算例非空（context %s · gen %d）" % (specs.TRTLLM_CONFIG_AXES, BATCH_CONTEXT_LENS, BATCH_N_GEN),
          "**配置轴或 batch 算例被动过** —— ★ 先查 `TRTLLM_CONFIG_AXES` / `BATCH_*`")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：引文逐字 · 逐元素 · 术语纪律
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · **逐元素**钉住 · 术语纪律（★ 纪律与断言成对）")
    print("=" * 78)

    # ── D.5 ★★ 六条引文逐字（TensorRT-LLM 四 + ORCA 一 + 顺序约束）───────
    check("D.5", specs.TRTLLM_INFLIGHT_BATCHING_QUOTE
          == ("TensorRT-LLM supports in-flight batching of requests (also known as continuous "
              "batching or iteration-level batching) for higher serving throughput.")
          and specs.TRTLLM_INFLIGHT_SYNONYM_QUOTE
          == "in-flight batching of requests (also known as continuous batching or iteration-level batching)"
          and specs.TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE
          == ("In the current implementation, the sequences that are going through the context "
              "phase must be before the sequences in the generation phase in the input tensor.")
          and specs.TRTLLM_ORDER_RELAX_QUOTE
          == "The constraint may or may not be relaxed in a future version"
          and specs.TRTLLM_PACKED_QUOTE
          == "the support for inflight batching requires the input tensors to be packed (no padding)"
          and "granularity of iteration" in specs.ORCA_ITERATION_LEVEL_QUOTE,
          "★★ **六条引文逐字钉住**（`[A]` NVIDIA 官方 + `[C]` ORCA）：in-flight batching 的**完整句**与"
          "**同义子句** · **顺序约束** · **可能放宽**的限定 · **packed 要求** · ORCA 的原始术语\n"
          "        ★★★ **顺序约束是【该实现当前版本】的**（`F` 的邻形纪律）⇒ 与限定语**同一条**断言\n"
          "        ★★ 载体抓取日 %s" % specs.ENGINES_FETCH_DAY,
          "**六条引文之一被动过** —— ★ 先查 `TRTLLM_*` / `ORCA_*`")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.TRTLLM_CONFIG_AXES == (3, 3, 2)
          and specs.LLAMACPP_FIRST_SOURCE_OBTAINED is False
          and specs.ENGINES_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：配置轴三档 `(3, 3, 2)` · llama.cpp 边界记录 `False` · 抓取日；"
          "★ `UNSOURCED == ()` 一并钉住\n"
          "        —— ★ 只钉「整体」会让**元素级**变异漏网（本库实测过：一次报出 58 条漏网）",
          "**某个常量被动过** —— ★ 先查 `TRTLLM_CONFIG_AXES` 的元素")

    # ── D.9 ★★★ 术语纪律（★ 形态 A 的根治：计数 == 豁免清单）──────────────
    src = open(__file__, encoding="utf-8").read()
    # ★★ 被禁裸词（**刻意用拼接**而不是字面量）：★ 若这里写成字面量，**计数代码自己**也会被算进去，
    #   于是「计数 == 豁免覆盖数」这条**等式**永远差 1。★ 拼接后计数行里**不再出现该词的字面量**。
    WORD = "业" + "界"
    n_bare = src.count(WORD)
    n_exempt = sum(src.count(s) * s.count(WORD) for s in BARE_ATTRIBUTION_EXEMPT)
    print()
    print("  ── D.9 ★★★ **术语纪律**：一律写「%s」/「%s」全称；限定语逐字出现 ──"
          % (TERM_KV_CACHE, TERM_SCHEDULER))
    print("     ★★ 源码里**裸写那个被禁裸词** = **%d** 处；豁免清单覆盖 = **%d**（清单 %d 项，逐项理由见模块头）"
          % (n_bare, n_exempt, len(BARE_ATTRIBUTION_EXEMPT)))
    check("D.9", TERM_KV_CACHE == "KV 缓存" and TERM_SCHEDULER == "请求调度器"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_KV_CACHE in src
          and BARE_ATTRIBUTION_EXEMPT
          == ("不裸写「" + WORD + "」这类无第二个独立主体的断言", "**不裸写「" + WORD + "」**")
          and n_bare == n_exempt
          and all(s in src for s in BARE_ATTRIBUTION_EXEMPT),
          "★★ **判据成立**：两条术语常量 `%r` / `%r`（★ **不裸写** `cache` / `scheduler`）；"
          "★ 限定语 `%r` **逐字出现在本程序源码里**；"
          " ★★ **被禁裸词的裸写计数 `%d` 恰好等于豁免清单覆盖的 `%d`**"
          % (TERM_KV_CACHE, TERM_SCHEDULER, LIMIT_QUOTE, n_bare, n_exempt),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写**")


# ===========================================================================
# 7. 汇总 / 主程序
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
    print("★★★ **A = %d/%d** —— ★ **本主题【没有 A 类】**：本机无 GPU / 无任何引擎（`D.0` 机械证明）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("    ★★ 「常量钉住」**不算 A 类**（那是 `P12` 的变体）⇒ 本程序把它们全归了 `D` 类。")
    print("★ 依据在 `specs.py` 的 `# 来源：`（NVIDIA 官方文档 `[A]`，抓取日 %s；ORCA 论文 `[C]`）。"
          % specs.ENGINES_FETCH_DAY)
    print("★ 本程序只核对「编译式路线的结构账」与「三条禁写真的会被拒」。")
    print("★★ **llama.cpp 的一手来源未取到** ⇒ 本程序只把它写成**边界**（`C.3`），**不自造引文**。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_tensorrt_and_llamacpp.py —— TensorRT-LLM 的编译式路线 + llama.cpp 的边界")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何引擎。")
    print("  ★★★ **本主题没有 A 类**（`D.0` 机械证明）；★★ **机制一律回链**（分页 / 前缀树 / 调度器）。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
