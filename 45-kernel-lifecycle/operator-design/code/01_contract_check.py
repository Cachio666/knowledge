#!/usr/bin/env python3
r"""01-contract-check —— 契约自洽性 / 可测性 / cache key 覆盖度。

对应文档：`../01-interface-contract.md`

★ 本程序的可证伪断言：
  A 类  一份**完整**契约的检查报告必须是**空**的；缺 `determinism` 的契约必须被报出
        「缺必填字段」；把 `boundary_behavior` 的取值域声明成 `unspecified` 的契约必须被
        报出「不可测」。**改错这三条中的任何一条，A 类立即失败。**
  B 类  枚举**所有改变构建结果的字段**，逐个比对 cache key 的覆盖；正确契约的缺失项
        必须是**空集**，而漏一项的那份必须**恰好**报出那一项。
  C 类  ★ 构造两个**只在 `optional_params` 上不同**的调用，显示它们
        ① 哈希到**同一个 cache key**、② 返回**同一个 kernel 对象**、③ 而正确的构建
        本该产出**两个不同的 kernel**、④ **全程没有抛异常**——
        即：**静默**用错了 kernel。这是 §4 坑 ①/③ 的直接证据。

跑法：`make 01_contract_check`。纯 python3 + 标准库，不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则本程序的结论不可比）：
  · 本程序里被检查的**「契约」与「构建器」都是本库构造的仿真对象**，不是任何真实框架
    的实现，也**没有**把它说成是某个真实故障（见 specs.UNSOURCED 第 5 项）。
  · 外部事实（vLLM 的 `supported` / `supports_args` 两层检查、compile key 的规则）来自
    `specs.py` §1，全部标 **[B]（某一实现的官方规格）**⇒ 只支撑「vLLM 的做法是……」。
"""

import ast
import pathlib
import sys

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
# 0. 被检查的对象：契约声明的形状
# ===========================================================================
#
# ★ 这**不是**从来源抄来的东西，是本篇要定义的那个工件本身（文档 §1 的字段清单）。
#   字段的**取值域**是声明式的；检查器只看声明，不看实现。

# 字段名 → 它在文档 §1 里的中文名（正文用中文，程序用英文键——两处必须一一对应）
FIELD_NAMES_ZH = {
    "shape_rule": "形状规则",
    "dtype": "dtype",
    "layout_stride": "layout 与 stride",
    "optional_params": "可选参数",
    "output_precision": "输出精度",
    "determinism": "确定性",
    "boundary_behavior": "边界行为",
}
# 来源：本域规格 §Chunk2「文档 01 的 §1」逐项列出的七个字段（**规格即工件定义**）
# 口径：七个字段**每一个都是必填**——缺任何一个，检查器都要报出来（A 类）。

# 字段名 → 它的取值域。`None` 表示「声明为 unspecified」⇒ **不可测**。
GOOD_DOMAINS = {
    "shape_rule": ("rank_2", "rank_2_or_3"),          # 是"规则"不是具体形状
    "dtype": ("fp16", "bf16", "fp32"),
    "layout_stride": ("row_major_contiguous", "col_major_contiguous", "strided"),
    "optional_params": ("absent", "present"),
    "output_precision": ("same_as_input", "fp32_always"),
    "determinism": ("bitwise_reproducible", "run_to_run_stable"),
    "boundary_behavior": ("raise", "clamp", "pad_and_mask"),
}

# ★ 哪些字段会**改变「构建出哪个 kernel」**。
#   这是本程序的**核心分类**——B 类与 C 类都建立在它之上。
#   判据（文档 §2）：一个字段的值不同 ⇒ 构建器产出的**不是一个 kernel** ⇒ 它必须进 key。
AFFECTS_BUILD = {
    "shape_rule": False,     # 它是**模式**（schema），不是每次调用的取值 ⇒ 不进 key
    "dtype": True,
    "layout_stride": True,
    "optional_params": True,  # ★ 「可选参数**是否出现**」改变构建结果（规格逐字要求）
    "output_precision": True,
    "determinism": True,      # 确定性 kernel 与用原子加的实现不是同一个 kernel
    "boundary_behavior": True,
}
# 口径：★ `shape_rule` 判 `False` 是**刻意**的——它声明的是「形状怎么算」这条规则，
#       每次调用都相同；变的只是**具体形状值**（M/N/K），而那是**调用参数**不是模式。
#       ⚠ 具体形状值（M/N/K）**确实**影响 tile 选择 ⇒ 它们**确实**要进 key。
#         本程序把它们建模为 `call_shape`（一个调用级输入，见 §B.3），
#         与「契约字段」分开——这个区分本身是本篇的一个论点。

# 契约「应该」覆盖的字段 = 所有 affects_build 的字段（正确契约的 in_key 与它相等）
EXPECTED_KEY_FIELDS = frozenset(f for f, v in AFFECTS_BUILD.items() if v)

# ★ 可选参数里有哪些会让**累加顺序不固定**（⇒ 与 `bitwise_reproducible` 冲突）。
#   来源：**本库推论 [E]**，由 specs 的 A 级事实推出——
#   PTX ISA 明写 `wmma.mma` 的 `accumulation order` **未规定**（`PTX_WMMA_UNSPECIFIED_ASPECTS`），
#   所以**任何让累加顺序依赖调度/并发到达次序的实现，都不可能逐位可复现**。
#   ⇒ 这不是「原子加慢」这类性能判断，而是**规范层可推出**的冲突。
OPTIONAL_PARAMS_WITH_UNSPECIFIED_ORDER = ("use_atomics",)
# 口径：★ 本库只把这一条**声明为冲突**，并**不**声称「原子加一定不可复现」——
#       声明的是「这个契约把二者标为互斥」，理由是上面的规范事实。


def good_contract():
    """一份**完整且自洽**的契约：七个字段齐全、取值域都可测、关键字段全覆盖。"""
    return {
        "fields": {f: {"domain": GOOD_DOMAINS[f], "required": True} for f in FIELD_NAMES_ZH},
        # 契约声明自己要进 cache key 的字段（**正确契约 = 覆盖全部 affects_build 的字段**）
        "cache_key_fields": sorted(EXPECTED_KEY_FIELDS),
        # 契约声明的互斥对（正确契约里为空）
        "incompatible_pairs": (),
    }


def contract_missing_determinism():
    """缺「确定性」字段的契约——A 类必须报出它。"""
    c = good_contract()
    del c["fields"]["determinism"]
    c["cache_key_fields"] = [f for f in c["cache_key_fields"] if f != "determinism"]
    return c


def contract_with_unspecified_boundary():
    """把「边界行为」声明成 `unspecified` 的契约——A 类必须报出「不可测」。"""
    c = good_contract()
    c["fields"]["boundary_behavior"] = {"domain": None, "required": True}
    return c


def contract_missing_key_field():
    """cache key 漏掉 `optional_params` 的契约——B 类必须恰好报出它。"""
    c = good_contract()
    c["cache_key_fields"] = [f for f in c["cache_key_fields"] if f != "optional_params"]
    return c


def contract_with_conflict():
    """声明了互斥对、且两个互斥项**同时被选中**的契约——A 类必须报出冲突。"""
    c = good_contract()
    c["incompatible_pairs"] = (
        ("determinism", "bitwise_reproducible", "optional_params", "present"),
    )
    return c


# ── 检查器（三个纯函数，各自只回答一个问题）─────────────────────────────────

def report_missing_fields(contract):
    """必填字段里**声明缺席**的那些。"""
    declared = set(contract["fields"])
    return sorted(f for f in FIELD_NAMES_ZH if f in EXPECTED_ALL_REQUIRED and f not in declared)


EXPECTED_ALL_REQUIRED = frozenset(FIELD_NAMES_ZH)
# 口径：七个字段**全部**必填（文档 §1 逐项说明「这一项为什么必须写」）。
#       ⇒ 缺任何一个都要报出来，不只 `determinism`。


def report_untestable_fields(contract):
    """被声明了、但**没人能写出验证它的测试**的字段。

    判据（**机械的**）：字段的取值域声明成 `unspecified`（`domain is None`）
    ⇒ 任何测试都无法「失败」——因为契约根本没有承诺任何可观察的行为。
    ★ 这正是 §4 坑 ② 的形态：**契约里写「未定义行为」，然后下游依赖它。**
    """
    return sorted(f for f, d in contract["fields"].items() if d["domain"] is None)


def report_conflicts(contract):
    """被声明为互斥、却在同一份契约里**同时被选中**的字段对。"""
    out = []
    for fa, va, fb, vb in contract["incompatible_pairs"]:
        da, db = contract["fields"].get(fa), contract["fields"].get(fb)
        if da is None or db is None:
            continue
        # 「同时被选中」= 双方的取值域里各自都含那个值
        if va in (da["domain"] or ()) and vb in (db["domain"] or ()):
            out.append((fa, va, fb, vb))
    return out


def report_key_gaps(contract):
    """cache key 的**缺失项**：会改变构建结果、却没进 key 的字段。"""
    in_key = set(contract["cache_key_fields"])
    return sorted(f for f in EXPECTED_KEY_FIELDS if f not in in_key)


def check_contract(contract):
    """一份契约的完整检查报告。**空报告 = 自洽且可测**。"""
    return {
        "missing": report_missing_fields(contract),
        "untestable": report_untestable_fields(contract),
        "conflicts": report_conflicts(contract),
        "key_gaps": report_key_gaps(contract),
    }


def is_empty_report(r):
    return not (r["missing"] or r["untestable"] or r["conflicts"] or r["key_gaps"])


# ===========================================================================
head("§A  算法正确性：三份契约的检查报告必须与手判一致")
# ===========================================================================

sub("A.1  完整契约 ⇒ 报告为空")
r_good = check_contract(good_contract())
print(f"    字段数 = {len(good_contract()['fields'])}（应为 7）")
print(f"    报告 = {r_good}")
check("A.1 完整契约的报告是**空**的（四类发现全为空）", is_empty_report(r_good), f"{r_good}")
check("A.1 契约确实声明了 7 个字段", len(good_contract()["fields"]) == 7,
      f"{len(good_contract()['fields'])}")
check("A.1 手判：应进 key 的字段恰好 6 个（7 个字段减去 `shape_rule`）",
      len(EXPECTED_KEY_FIELDS) == 6, f"{sorted(EXPECTED_KEY_FIELDS)}")

sub("A.2  缺「确定性」字段的契约 ⇒ 必须报出它")
r_miss = check_contract(contract_missing_determinism())
print(f"    missing = {r_miss['missing']}")
check("A.2 报告里**恰好**是 ['determinism']（多报或少报都算错）",
      r_miss["missing"] == ["determinism"], f"{r_miss['missing']}")
check("A.2 缺字段 ⇒ 报告不为空", not is_empty_report(r_miss))
print("    ★ 为什么这一项必须写：`determinism` 缺省时，「这个算子的结果能不能逐位复现」")
print("      就是一个**没有答案**的问题。下游会各自假设一个默认值，而两个默认值不相等。")

sub("A.3  把「边界行为」声明成 `unspecified` ⇒ 必须报出「不可测」")
r_uns = check_contract(contract_with_unspecified_boundary())
print(f"    untestable = {r_uns['untestable']}")
check("A.3 报告里恰好是 ['boundary_behavior']",
      r_uns["untestable"] == ["boundary_behavior"], f"{r_uns['untestable']}")
check("A.3 它**没有**报成 missing（字段写了，只是不可测——两件事不能混）",
      r_uns["missing"] == [], f"{r_uns['missing']}")
print("    ★ 可测性的判据是**机械的**：域为 `unspecified` ⇒ **不存在能失败的测试**。")
print("      这不是「测试写得好不好」的问题——契约**根本没有承诺**任何可观察行为。")

sub("A.4  互斥声明同时被选中 ⇒ 必须报出冲突，且冲突的来源是**规范事实**")
r_cf = check_contract(contract_with_conflict())
print(f"    conflicts = {r_cf['conflicts']}")
check("A.4 报告里恰好一对冲突",
      len(r_cf["conflicts"]) == 1 and r_cf["conflicts"][0][0] == "determinism",
      f"{r_cf['conflicts']}")
check("A.4 冲突的两个值分别是 bitwise_reproducible 与 present",
      r_cf["conflicts"][0][1] == "bitwise_reproducible"
      and r_cf["conflicts"][0][3] == "present", f"{r_cf['conflicts'][0]}")
check("A.4 ★ 冲突的**依据**是 A 级规范事实（累加顺序未规定），不是性能判断",
      "accumulation order" in SP.PTX_WMMA_UNSPECIFIED_ASPECTS,
      f"{SP.PTX_WMMA_UNSPECIFIED_ASPECTS}")
print("    ★ 这条把 01 篇与 02 篇接上了：`bitwise_reproducible` 与"
      "「累加顺序可能变的实现」互斥，")
print("      理由不是「原子加慢」，而是 **PTX 明写 accumulation order 未规定**。")


# ===========================================================================
head("§B  模型与公式：cache key 的覆盖度")
# ===========================================================================

sub("B.1  枚举所有**改变构建结果**的字段，逐个比对 key 的覆盖")
print(f"    被枚举的字段（全 7 个）：")
for f in FIELD_NAMES_ZH:
    print(f"      {f:18s} affects_build={str(AFFECTS_BUILD[f]):5s}   {FIELD_NAMES_ZH[f]}")
gaps_good = report_key_gaps(good_contract())
print(f"    正确契约的缺失项 = {gaps_good}")
check("B.1 正确契约的缺失项是**空集**", gaps_good == [], f"{gaps_good}")

sub("B.2  漏掉 `optional_params` 的契约 ⇒ 必须**恰好**报出那一项")
gaps_bad = report_key_gaps(contract_missing_key_field())
print(f"    缺失项 = {gaps_bad}")
check("B.2 缺失项恰好是 ['optional_params']", gaps_bad == ["optional_params"], f"{gaps_bad}")

sub("B.3  调用级输入：**具体形状值**也要进 key，但它不是契约字段")
CALL_LEVEL_KEY_INPUTS = ("call_shape_MNK",)
# 口径：★ 这一条是 §B.1 那个区分的落点：`shape_rule`（模式）**不进** key，
#       而**具体形状值** `call_shape_MNK`（tile 选择直接依赖它）**必须进**。
#       ⇒ 「契约字段」与「调用级输入」是两类东西，**漏掉任何一类都会命中错误的 kernel**。
MISSING_CALL_LEVEL_INPUTS_IF_ABSENT_FROM_KEY = (
    "call_shape_MNK", "stream_device_context",
)
# 口径：★ 第二个是**故意放在这里的对照项**：`stream_device_context` **不改变**
#       「构建出哪个 kernel」（同一份代码在哪个 stream 上跑是运行期的事）
#       ⇒ **它不该进 key**。把它放进 key 就是反方向的错误（缓存永不命中）。
print(f"    契约字段里进 key 的：{sorted(EXPECTED_KEY_FIELDS)}")
print(f"    调用级输入里进 key 的：{('call_shape_MNK',)}")
print(f"    调用级输入里**不该**进 key 的：{('stream_device_context',)}")
check("B.3 `call_shape_MNK` 与 `shape_rule` 是两回事（一个进 key、一个不进）",
      "call_shape_MNK" in CALL_LEVEL_KEY_INPUTS and AFFECTS_BUILD["shape_rule"] is False,
      f"{CALL_LEVEL_KEY_INPUTS}")
check("B.3 `stream_device_context` **不在**该进 key 的清单里（反方向对照）",
      "stream_device_context" not in CALL_LEVEL_KEY_INPUTS, f"{CALL_LEVEL_KEY_INPUTS}")
check("B.3 B 级的规则是**双向**的：漏加与多加都是错的",
      SP.VLLM_KERNELCONFIG_HASH_INCLUDES_COMPILE_AFFECTING_FIELDS
      and SP.VLLM_KERNELCONFIG_HASH_EXCLUDES_NON_COMPILE_FIELDS)


# ===========================================================================
head("§C  边界与反例：只在**未进 key 的字段**上不同的两个调用，命中同一个条目")
# ===========================================================================
#
# ★ 场景（**本库构造的仿真**，见 specs.UNSOURCED 第 5 项）：一个契约的 cache key
#   **漏掉了 `optional_params`**。两个调用除它之外**完全相同**。

sub("C.1  被仿真的构建器：它把**每一个** affects_build 字段都编进 kernel 的身份")


def build_kernel(call):
    """决定「构建出哪个 kernel」。**这是本库构造的仿真构建器**，不是任何真实实现。

    ★ 它刻意把**全部六个** affects_build 字段都写进 kernel 的身份——
      这样一来，「某个字段没进 key」的后果就是**可观察的**。
    """
    parts = [
        f"dtype={call['dtype']}",
        f"layout={call['layout_stride']}",
        f"out={call['output_precision']}",
        f"det={'on' if call['determinism'] == 'bitwise_reproducible' else 'off'}",
        f"epilogue={'fused' if call['optional_params'] == 'present' else 'plain'}",
        f"bounds={call['boundary_behavior']}",
    ]
    return "KERNEL[" + " ".join(parts) + "]"


def cache_key(call, contract):
    """按契约声明的 key 字段算哈希键。**它只看得见 key 里有的字段。**"""
    in_key = set(contract["cache_key_fields"])
    return tuple(sorted((f, call[f]) for f in in_key))


class KernelCache:
    """最小的 kernel cache：键相同 ⇒ 直接返回已构建的那个 kernel。"""

    def __init__(self, contract):
        self.contract = contract
        self.entries = {}
        self.builds = 0

    def get(self, call):
        k = cache_key(call, self.contract)
        if k not in self.entries:
            self.entries[k] = build_kernel(call)
            self.builds += 1
        return self.entries[k]


BASE_CALL = {
    "shape_rule": "rank_2",
    "dtype": "fp16",
    "layout_stride": "row_major_contiguous",
    "output_precision": "same_as_input",
    "determinism": "run_to_run_stable",
    "optional_params": "absent",
    "boundary_behavior": "clamp",
}
CALL_A = dict(BASE_CALL, optional_params="absent")
CALL_B = dict(BASE_CALL, optional_params="present")   # ★ 唯一的不同

print(f"    调用 A：optional_params = {CALL_A['optional_params']}")
print(f"    调用 B：optional_params = {CALL_B['optional_params']}")
diff = [k for k in BASE_CALL if CALL_A[k] != CALL_B[k]]
print(f"    两者的差异字段（应当**恰好**是 ['optional_params']）= {diff}")
check("C.1 两个调用只在 `optional_params` 上不同", diff == ["optional_params"], f"{diff}")

sub("C.2  有缺陷的契约（key 漏 `optional_params`）下，两次调用**命中同一个条目**")
buggy = contract_missing_key_field()
cache_buggy = KernelCache(buggy)
k_a, k_b = cache_key(CALL_A, buggy), cache_key(CALL_B, buggy)
got_a = cache_buggy.get(CALL_A)
got_b = cache_buggy.get(CALL_B)
print(f"    key(A) = {k_a}")
print(f"    key(B) = {k_b}")
print(f"    key(A) == key(B) ?  {k_a == k_b}")
print(f"    → 缓存条目数 = {len(cache_buggy.entries)}；实际构建次数 = {cache_buggy.builds}")
print(f"    A 拿到的 kernel：{got_a}")
print(f"    B 拿到的 kernel：{got_b}")
check("C.2 两个调用哈希到**同一个** cache key", k_a == k_b, f"{k_a}")
check("C.2 只发生**一次**构建（第二次直接复用）", cache_buggy.builds == 1,
      f"{cache_buggy.builds}")
check("C.2 两次拿到的是**同一个** kernel 对象", got_a is got_b, f"{got_a!r} vs {got_b!r}")

sub("C.3  而这**是错的**：正确的构建本该产出两个不同的 kernel")
want_a, want_b = build_kernel(CALL_A), build_kernel(CALL_B)
print(f"    该给 A 的：{want_a}")
print(f"    该给 B 的：{want_b}")
print(f"    两者是否相同？ {want_a == want_b}")
check("C.3 ★ 正确的两个 kernel **不相等**（⇒ 复用是错的，不是「恰好相同」）",
      want_a != want_b, f"{want_a} vs {want_b}")
check("C.3 ★ 实际返回给 B 的 kernel **不等于** B 应得的那个（**静默错误成立**）",
      got_b != want_b, f"{got_b} vs {want_b}")
check("C.3 ★ 差异**恰好**落在 `optional_params` 那一段（epilogue=fused vs plain）",
      ("epilogue=fused" in want_b) and ("epilogue=plain" in got_b),
      f"{want_b} / {got_b}")
print()
print("    ★ 这就是 §4 坑 ①/③ 的形态：**没有任何异常、没有任何警告、返回值看起来完全正常**。")
print("      B 要求 `optional_params=present`（它期望一条融合的 epilogue），")
print("      而它拿到的是一个 `plain` 的 kernel——**静默用错 kernel**。")

sub("C.4  反事实：把字段放回 key，同一个场景下错误**消失**")
cache_ok = KernelCache(good_contract())
got_a2 = cache_ok.get(CALL_A)
got_b2 = cache_ok.get(CALL_B)
print(f"    缓存条目数 = {len(cache_ok.entries)}；实际构建次数 = {cache_ok.builds}")
print(f"    key(A) != key(B) ? {cache_key(CALL_A, good_contract()) != cache_key(CALL_B, good_contract())}")
check("C.4 修好 key 之后发生**两次**构建", cache_ok.builds == 2, f"{cache_ok.builds}")
check("C.4 两个调用拿到**不同**的 kernel", got_a2 != got_b2, f"{got_a2} vs {got_b2}")
check("C.4 且各自都等于它应得的那个（错误消失）",
      got_a2 == want_a and got_b2 == want_b, f"{got_a2} / {got_b2}")
print("    ⇒ **这一条使 C 类可证伪**：若把 `optional_params` 加回 key 而错误不消失，")
print("      说明我的缺陷模型没抓住因果，C.2/C.3 的结论就不成立。")

sub("C.5  反方向的错误：把**不该进 key** 的字段放进去 ⇒ 缓存一次都不命中")
# ★ 两个调用**都**带上 `stream_device_context`（它是**调用级输入**，不是契约字段），
#   且**只**在这一项上不同。
CALL_A1 = dict(CALL_A, stream_device_context="stream#1")
CALL_A2 = dict(CALL_A, stream_device_context="stream#7")   # 只是**另一个 stream**
cache_over = KernelCache(good_contract())
# ❌ 过度覆盖：把这一个**不改变构建结果**的运行期字段也塞进了 key
cache_over.contract = dict(good_contract(),
                           cache_key_fields=sorted(EXPECTED_KEY_FIELDS)
                           + ["stream_device_context"])
over_a = cache_over.get(CALL_A1)
over_a2 = cache_over.get(CALL_A2)
print(f"    同一个 kernel 该不该复用？（`stream_device_context` 不改变构建结果）")
print(f"    A1 与 A2（只换了 stream）的 kernel：{over_a}  /  {over_a2}")
check("C.5 两者**构建出同一个 kernel**（stream 不改变构建结果）", over_a == over_a2,
      f"{over_a} vs {over_a2}")
check("C.5 但由于 key 里多了 stream ⇒ **两次都构建**（缓存失效）",
      cache_over.builds == 2, f"{cache_over.builds}")
print("    ⇒ **两个方向都是病**：漏字段 ⇒ 用错 kernel；多字段 ⇒ 永远重编。")
print("      B 级那条规则（B.3 的最后一 check）是**双向**的，不是「多多益善」。")


# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")
# ===========================================================================

sub("D.1  `VLLM_IR_RESERVED_PROVIDERS` 与两层检查的分工")
check("D.1 保留 provider 恰好是 ('native','unfused')",
      SP.VLLM_IR_RESERVED_PROVIDERS == ("native", "unfused"),
      f"{SP.VLLM_IR_RESERVED_PROVIDERS}")
check("D.1 ★ `native` 在保留名里 ⇒ 优先级列表**永远有兜底**",
      "native" in SP.VLLM_IR_RESERVED_PROVIDERS)
check("D.1 `register_impl` 的关键字恰好是 supported/supports_args/inplace",
      SP.VLLM_IR_REGISTER_IMPL_KWARGS == ("supported", "supports_args", "inplace"),
      f"{SP.VLLM_IR_REGISTER_IMPL_KWARGS}")
check("D.1 ★ 两层检查恰好是 ('supported','supports_args')（不是一层、也不是三层）",
      SP.VLLM_IR_TWO_LEVEL_SUPPORT_CHECKS == ("supported", "supports_args"),
      f"{SP.VLLM_IR_TWO_LEVEL_SUPPORT_CHECKS}")
check("D.1 两层检查的两个名字都出现在 `register_impl` 的关键字里（不是编的）",
      set(SP.VLLM_IR_TWO_LEVEL_SUPPORT_CHECKS) <= set(SP.VLLM_IR_REGISTER_IMPL_KWARGS),
      f"{SP.VLLM_IR_TWO_LEVEL_SUPPORT_CHECKS} ⊆ {SP.VLLM_IR_REGISTER_IMPL_KWARGS}")

sub("D.2  `supported` 是 import 时求值；`supports_args` **不得**查 batch size")
check("D.2 `supported` 只在 import 时求值一次", SP.VLLM_IR_SUPPORTED_IS_IMPORT_TIME_ONLY
      is True)
check("D.2 ★★ `supports_args` **不得**检查 batch size（本篇最可证伪的一条 B 级事实）",
      SP.VLLM_IR_SUPPORTS_ARGS_MUST_NOT_CHECK_BATCH is True)
print("    ★ 这一条与「凡改变构建结果的字段都要进 key」**不矛盾**：")
print("      batch size 改变的是**同一个 kernel 的调用参数**，不是**构建出哪个 kernel**。")
print("      ⇒ 两者是同一条规则的**两个方向**（B 类与 C.5 各是一条）。")
check("D.2 优先级配置的 CLI 前缀是 --ir-op-priority",
      SP.VLLM_IR_PRIORITY_CLI_PREFIX == "--ir-op-priority",
      f"{SP.VLLM_IR_PRIORITY_CLI_PREFIX}")

sub("D.3  B 类的现实形态：**机制上看不见**那个字段")
check("D.3 `IrOpImpl.uuid` 是**源文件内容**的哈希", SP.VLLM_IR_IMPL_UUID_IS_SOURCE_HASH is True)
check("D.3 ★ Dynamo **看不见**实现源文件（hidden from the traced files list）",
      SP.VLLM_IR_DYNAMO_HIDES_IMPL_SOURCES is True)
check("D.3 ⇒ 它必须被**手工**加进 hash factors（两者是一对）",
      SP.VLLM_IR_IMPL_UUID_MANUALLY_IN_HASH is True)
print("    ★ 现实的失败形态不是「有人忘了写」，而是**机制看不见 ⇒ 必须手工补**。")
print("      本程序 §C 的仿真是这个机制后果的**最小可运行模型**。")

sub("D.4  反方向：D 级来源（单一社区项目）给的一例")
check("D.4 那个字段名是 fpm_worker_id", SP.VLLM_DYNAMO_BUSTING_FIELD == "fpm_worker_id",
      f"{SP.VLLM_DYNAMO_BUSTING_FIELD}")
check("D.4 ★ 它**不**是编译输入（⇒ 放进 hash 就是过度覆盖，缓存永不命中）",
      SP.VLLM_DYNAMO_BUSTING_FIELD_IS_COMPILE_INPUT is False)
print("    ⚠ **D 级标注**：这一条来自**一个社区仓库的 issue**（ai-dynamo/dynamo #10119），")
print("      是**单一主体、未被独立复现**的报告。正文出现它时必须写明这两点，")
print("      且它**只**用来与 B 级那条规则**同向**印证，**不单独支撑**任何结论。")

sub("D.5  hash 规则的双向性，以及**引用等级**的自我声明")
check("D.5 规则双向：编译相关字段**要**进 hash",
      SP.VLLM_KERNELCONFIG_HASH_INCLUDES_COMPILE_AFFECTING_FIELDS is True)
check("D.5 规则双向：与编译**无关**的字段**要排除**",
      SP.VLLM_KERNELCONFIG_HASH_EXCLUDES_NON_COMPILE_FIELDS is True)
check("D.5 ★★ 这条规则**不够格标「逐字」**（只经一次检索）⇒ 布尔为 False",
      SP.VLLM_KERNELCONFIG_HASH_RULE_IS_VERBATIM is False)
print("    ★ 与 02 篇成对读：`PTX_WMMA_UNSPECIFIED_QUOTE_IS_VERBATIM = True`（两次检索一致）")
print("      而这一条是 `False`。**这个对照就是引用纪律的执行记录**——")
print("      不是「我引用了」就够，要**够格**才写成逐字。")

sub("D.6  A 级事实 `VLLM_IR_*` 的层级声明：它们只支撑「vLLM 这么做」")
check("D.6 优先级前缀与 `requirement-discovery` 的那一条**同值**（跨主题一致）",
      SP.VLLM_IR_PRIORITY_CLI_PREFIX == "--ir-op-priority")
print("    ★ P10 的执行点：本节全部常量（除 D.4 那条 D 级外）都是 **[B] 某一实现的规格**，")
print("      ⇒ 正文只能写「vLLM 的做法是……」，**不得**写「业界通常」。")

sub("D.7  被仿真的工件本身：七个契约字段")
check("D.7 字段恰好七个，且与文档 §1 的中文名一一对应",
      set(FIELD_NAMES_ZH) == set(GOOD_DOMAINS)
      and len(FIELD_NAMES_ZH) == 7, f"{sorted(FIELD_NAMES_ZH)}")
check("D.7 应进 key 的字段恰好六个（`shape_rule` 不进）",
      EXPECTED_KEY_FIELDS == frozenset(FIELD_NAMES_ZH) - {"shape_rule"},
      f"{sorted(EXPECTED_KEY_FIELDS)}")
check("D.7 `determinism` 在必填清单里（A.2 的断言依赖这一点）",
      "determinism" in EXPECTED_ALL_REQUIRED)
check("D.7 与「累加顺序未规定」冲突的可选参数恰好一个",
      OPTIONAL_PARAMS_WITH_UNSPECIFIED_ORDER == ("use_atomics",),
      f"{OPTIONAL_PARAMS_WITH_UNSPECIFIED_ORDER}")

sub("D.8  `UNSOURCED` 逐条比对（防止「既保留数字又记 UNSOURCED」）")
UNSOURCED_EXPECTED_01 = (
    "address_and_index_register_budget",
    "ungrounded_saturation_claims",
    "cudnn_fast_float_for_fp8_error_magnitude",
    "wmma_fragment_element_mapping_is_unspecified",
    "real_world_wrong_kernel_by_missing_key_field",
    "real_world_tail_defect_inventory",
)
check("D.8 UNSOURCED 元组与逐条复核的结果完全一致",
      tuple(SP.UNSOURCED) == UNSOURCED_EXPECTED_01, f"{SP.UNSOURCED}")
check("D.8 ★ 第 5 项存在 ⇒ §C 的仿真是**本库构造的**（正文不得暗示它是真实故障）",
      "real_world_wrong_kernel_by_missing_key_field" in SP.UNSOURCED)
check("D.8 ★ 第 6 项存在 ⇒ 03 篇 §C 的三个缺陷模型也是**本库构造的**",
      "real_world_tail_defect_inventory" in SP.UNSOURCED)
check("D.8 UNSOURCED 里没有任何一项出现在正文常量中（二者只能取其一）",
      not (set(SP.UNSOURCED) & {n for n in dir(SP) if n.isupper() and n != "UNSOURCED"}),
      f"{sorted(set(SP.UNSOURCED))}")


# ---------------------------------------------------------------------------
sub("D.9  ★ 全模块常量覆盖的**机械检查**（本域 C6 的机器化）")
# ★★ 上一轮（`requirement-discovery`）最贵的教训：`specs.py` 里有 **19 条常量**
#    在三个程序里 **0 引用**——它们只能被 `print`，于是任何变异测试都必然漏网。
#    ⇒ 本程序用 `ast` 读出**同目录另外两个程序**里的 `COVERED_02` / `COVERED_03`
#      元组，断言 **三者的并集 == specs.py 的全部模块级常量**。
#    ⇒ **在 specs.py 里加一个常量而没人断言它，`make run` 会立刻失败。**

COVERED_01 = (
    # §1 契约的落点（B 级 · vLLM 的做法）
    "VLLM_IR_RESERVED_PROVIDERS",
    "VLLM_IR_REGISTER_IMPL_KWARGS",
    "VLLM_IR_TWO_LEVEL_SUPPORT_CHECKS",
    "VLLM_IR_SUPPORTED_IS_IMPORT_TIME_ONLY",
    "VLLM_IR_SUPPORTS_ARGS_MUST_NOT_CHECK_BATCH",
    "VLLM_IR_PRIORITY_CLI_PREFIX",
    "VLLM_IR_IMPL_UUID_IS_SOURCE_HASH",
    "VLLM_IR_DYNAMO_HIDES_IMPL_SOURCES",
    "VLLM_IR_IMPL_UUID_MANUALLY_IN_HASH",
    "VLLM_KERNELCONFIG_HASH_INCLUDES_COMPILE_AFFECTING_FIELDS",
    "VLLM_KERNELCONFIG_HASH_EXCLUDES_NON_COMPILE_FIELDS",
    "VLLM_KERNELCONFIG_HASH_RULE_IS_VERBATIM",
    # D 级（反方向对照）
    "VLLM_DYNAMO_BUSTING_FIELD",
    "VLLM_DYNAMO_BUSTING_FIELD_IS_COMPILE_INPUT",
    # 待办清单
    "UNSOURCED",
)

print(f"    01 覆盖 {len(COVERED_01)} 条")
for _n in COVERED_01:
    print(f"      {_n:52s} 有断言")


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
c02 = _covered_names_in("02_error_budget.py", "COVERED_02")
c03 = _covered_names_in("03_tiling_check.py", "COVERED_03")
union = frozenset(COVERED_01) | frozenset(c02) | frozenset(c03)
print(f"    specs.py 的模块级常量共 {len(ALL_CONSTANTS)} 条")
print(f"    三个程序的覆盖：01={len(COVERED_01)} 02={len(c02)} 03={len(c03)}"
      f" 并集={len(union)}")
print(f"    未被任何程序断言 = {sorted(ALL_CONSTANTS - union)}")
print(f"    断言了但 specs 里没有 = {sorted(union - ALL_CONSTANTS)}")
check("D.9 ★ 三程序覆盖的**并集**恰好等于 specs.py 的全部模块级常量（无漏网）",
      union == ALL_CONSTANTS, f"漏 {sorted(ALL_CONSTANTS - union)}")
check("D.9 三个程序之间**没有**重复覆盖（每条常量恰好一个归属）",
      len(COVERED_01) + len(c02) + len(c03) == len(union),
      f"{len(COVERED_01)}+{len(c02)}+{len(c03)} vs {len(union)}")
check("D.9 COVERED_01 里没有拼错的常量名",
      set(COVERED_01) <= ALL_CONSTANTS,
      f"{sorted(set(COVERED_01) - ALL_CONSTANTS)}")
check("D.9 `ALL_CONSTANTS` 非空（否则上面两条会退化成恒真）",
      len(ALL_CONSTANTS) > 20, f"{len(ALL_CONSTANTS)}")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 4 组 · B 类 3 组 · C 类 5 组 · D 类 9 组）")
print()
print("  ★ 本程序的对象是**本库构造的契约与仿真构建器**，不是任何真实框架的实现；")
print("    外部事实全部来自 specs.py §1（B 级：某一实现的官方规格，只支撑「vLLM 这么做」）；")
print("    §C 的「静默用错 kernel」是**本库构造的仿真**——检索未找到任何真实案例")
print("    （specs.UNSOURCED 第 5 项），正文不得暗示它是 vLLM 的故障。")
print(f"    python {sys.version.split()[0]} / 纯标准库（未 import numpy、未 import torch）")
