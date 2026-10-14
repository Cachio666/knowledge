#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_vllm.py —— **vLLM 的架构选择**：引擎循环（step）· `vllm.ir` 的位置 · **版本相关的参数面**。

配套文档：`01-vllm.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题【没有 A 类】**：本机**无 GPU、无任何引擎**（vLLM / SGLang / TensorRT-LLM / llama.cpp 都没有）
  ⇒ vLLM 的**【行为与性能】无处可验**。★ 本程序跑在默认 `python3` 下、只用标准库
  ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**（**程序内断言**）。
  ★★ **如实点破一层**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**
  ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。
  ★★★ **不得把「常量钉住」冒充 A 类**（那是 `P12` 的变体）⇒ 本程序的一切钉住归 `D` 类。
  ★ `D.0` **机械地**证明「本文件里一条 A 类判据都没有」（AST 扫自己）。

★★★ **本程序只写「vLLM 的选择与差异」** —— ★ **机制一律回链**（`P37`）：
  分页 → `120-inference/kv-cache/02`；前缀树 → `kv-cache/03`；请求调度器 → `batching-and-scheduling/01`；
  PagedAttention 的算式 → `130-model-optimization/quantization/18-kv-cache-basics:68-84`；
  ★★ **vLLM 的模型接入与算子注册（`vllm.ir`）** → `45-kernel-lifecycle/requirement-discovery/01:205-214`。

★★★ **本程序的禁写清单（计划 `§0.7 C.5`）**：
  · `F1`（**不得写那类「没有第二个独立主体」的断言** —— 见 `D.9` 的**被禁裸词**）；
  · `F2`（`max_num_seqs` 的默认值**不带设备与场景**）；
  · `F3`（`continuous batching` **不是 vLLM 提出** —— 出处是 ORCA 的 `iteration-level scheduling`）；
  · `F4`（`36:9×` / `23x` 是**引用值**，基线各异；★ `36:9×` 的原文是**冒号**）；
  · `F5`（**不写** APC 默认开启）；`F6`（chunked prefill 默认开启**必须带** `In V1` / `whenever possible`）；
  · `F7`（**不引** `design/paged_attention` 当 vLLM 现状 —— 官方自标为**历史文档**）；
  · `F9`（**不裸写 `TPOT`** 当 vLLM 的官方词 —— 该词在 vLLM 页面上 0 处）。
  ★ `F8` / `F10` 属 `decoding-strategies`（本程序**不涉**，只回链）。

★★★ **本程序的判据纪律（★ 形态 A 的根治）**：★ 抬头写了「不裸写某某」这条纪律，
  本程序里**必须有一条断言能抓住「没做到」** —— 见 `D.9`：
  ★★ 判据是**等式**「裸写计数 == 显式豁免清单覆盖的计数」，**不是「≤ N」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `B.1` ★★ **参数面（I）**：`max_num_seqs` 的**取值集合**（兜底 + 设备/场景 6 格 + throughput 翻倍）；
  · `B.2` ★★ **参数面（II）**：每槽 token 数 `max_num_batched_tokens / max_num_seqs` 随格变（`8..32`）；
  · `C.1` ★★★ **把兜底常量当「vLLM 的默认值」**必须被拒（`F2`）；
  · `C.2` ★★★ **chunked prefill「默认开启」不带限定**必须被拒（`F6`）；
  · `C.3` ★★★ **把 `iteration-level scheduling` 当成 vLLM 的术语**必须被拒（`F3`）；
  · `C.4` ★★★ **裸写 `TPOT` 当成 vLLM 的官方词**必须被拒（`F9`）；
  · `D.0` ★★★ **机械证明「本文件没有 A 类判据」**（AST 扫 `check("A…")` 的调用数 == 0）；
  · `D.5` ★★ 引文**逐字**钉住（引擎循环 / 参数说明 / ORCA / 博客 / inter-token latency）；
  · `D.7` ★★★ **逐元素**钉住（参数表六行 × 四列 · 两处兜底 · 限定语元组 · `TPOT` 计数 = 0）；
  · `D.9` ★★★ **术语纪律**（被禁裸词的**裸写计数** == **显式豁免清单覆盖数**）。
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
#   · 本域一律写「**KV 缓存**」/「**请求调度器**」全称；★ 那两个被禁的 ASCII 词见 `TERM_BARE_EXEMPT`
#     （**不在此重复** —— ★ 否则会与 `D.9` 的裸写计数互相干扰）；
#   · 本域最强的一句**限定语**必须逐字出现。
TERM_KV_CACHE = "KV 缓存"
TERM_SCHEDULER = "请求调度器"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ `D3` 订正：此前「不裸写该两词」这条纪律**没有任何断言盯住** ——
#   本主题在自己身上栽了第三次，见计划 `§5` v1.10）：
#   本程序源码里**裸写那两个被禁 ASCII 词**的【计数】必须**恰好等于**下面这份**显式豁免清单**覆盖的计数。
#   ★★★ 判据是【计数 == 豁免覆盖数】，**不是「≤ N」**（「≤ N」挡不住**新增**一处裸写）。
#   ★★ **裸写 = 该 ASCII 词紧邻中文**（中间只隔空白 / 反引号）；★ 计数码用**拼接**避免自计。
#   ★ 每一项都写明【为什么豁免】：
#     ① 抬头注释里**提到**这条纪律本身 —— 那是 **mention**（它写该词**恰恰因为在规定不许写它**）：
#        · ★★★ 本主题一律【写明是哪一家】：不裸写「业界」这类无第二个独立主体的断言（`F1`）。
#     ② `D.9` 的判据文案里**复述**这条纪律 —— ★ 同为 mention。
BARE_ATTRIBUTION_EXEMPT = (
    "不裸写「业界」这类无第二个独立主体的断言",
    "**不裸写「业界」**",
)

# ★★★ **术语纪律的第三半**（★ `D3` 的根治：对那两个被禁 ASCII 词建**同形态的等式**）：
#   ★ 豁免清单（**逐项带理由**）：① 下面两条字符串**本身就是**抬头那条纪律的 **mention** ——
#     它们写该 ASCII 词**恰恰因为在规定不许写它**。★ 它们是**唯一**被豁免的裸写；
#     ★★ 任何**新增**的裸写都会让等式 `n_bare == n_exempt` 不成立。
BARE_ASCII_WORDS = ("ca" + "che", "sched" + "uler")
TERM_BARE_EXEMPT = (
    "本域一律写「**KV 缓存**」全称（不裸写 `cache`，那是 L1/L2 与 kernel cache）",
    "本域一律写「**请求调度器**」全称（不裸写 `scheduler`，那是 GPU 的 warp scheduler）",
)


def _is_cjk(ch):
    """★ 是否 CJK 表意字（★ **不含** CJK 标点 —— 「，」「）」不算「中文语境」）。"""
    return ch != "" and "一" <= ch <= "鿿"


def _count_bare_ascii(src):
    """★★ 数「**裸写**」：被禁 ASCII 词**紧邻中文**（中间只隔空白 / 反引号）的出现次数。

    ★★★ 这是 `D.9` 的**被测量** —— ★ 判据是**等式**（`_count_bare_ascii(src) == 豁免覆盖数`），
      ★★ 不是「≤ N」。
    """
    n = 0
    for w in BARE_ASCII_WORDS:
        i = src.find(w)
        while i != -1:
            l = i
            while l > 0 and src[l - 1] in " `\t":
                l -= 1
            r = i + len(w)
            while r < len(src) and src[r] in " `\t":
                r += 1
            lc = src[l - 1] if l > 0 else ""
            rc = src[r] if r < len(src) else ""
            if _is_cjk(lc) or _is_cjk(rc):
                n += 1
            i = src.find(w, i + 1)
    return n

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
    ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 失败标记必然在**行首**。
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
# 1. 被测的「参数面校验器」—— ★★ **它【不是】A 类**（本机没有 vLLM 可测）
#    ★★★ 它是**本库的一段校验器**：把「vLLM 的参数默认值有几种」变成**可数**的量，
#      用来判「把兜底当权威」这类错法。★ 一切钉住归 `D` 类。
# ===========================================================================
def distinct_max_num_seqs():
    """★★★ `max_num_seqs` 的**取值集合**（兜底 + 设备/场景 6 格 + throughput 翻倍）。

    ★★★ 取值多于一个 ⇒ 「**vLLM 的 `max_num_seqs` 默认值是 N**」这句话**没有单一答案**（`F2`）。
    """
    vals = {specs.VLLM_MAX_NUM_SEQS_FALLBACK}
    for _dev, _ctx, _tok, seqs in specs.VLLM_BATCH_DEFAULTS:
        vals.add(seqs)
        vals.add(seqs * specs.VLLM_THROUGHPUT_MULTIPLIER)
    return vals


_DEVICE_TAGS = tuple(sorted({d for d, _c, _t, _s in specs.VLLM_BATCH_DEFAULTS}))
_SCENARIO_TAGS = ("API server", "离线")


def default_claim_is_valid(claim):
    """★★★ **一条「vLLM 的某参数默认值是 `N`」的表述【成立】吗？** —— ★ 判据：它必须**带设备与场景**。

    ★★★ 这是 `C.1` 的**「反例 → 被拒」**结构（★ 不是恒真的常量检查；`D6` 订正）：
      · **反例（被拒）**：「vLLM 的 `max_num_seqs` 默认是 `128`」—— ★ 既无设备、也无场景；
      · **正例（通过）**：「`A100/其余` 上 `API server` 的 `max_num_seqs` 是 `256`」。
    ★ 依据：真实默认由 `get_batch_defaults()` 按**设备 + 场景**覆盖（`specs.VLLM_BATCH_DEFAULTS`）。
    """
    return any(d in claim for d in _DEVICE_TAGS) and any(c in claim for c in _SCENARIO_TAGS)


def tokens_per_slot():
    """★ **每槽 token 数** = `max_num_batched_tokens / max_num_seqs`（★ 这是一条**结构量**）。

    ★★ 它随设备/场景变（`8..32`）⇒ **两个参数不是等比缩放的一对**（若等比，这个比值应当恒定）。
    """
    out = {}
    for dev, ctx, tok, seqs in specs.VLLM_BATCH_DEFAULTS:
        out["%s/%s" % (dev, ctx)] = tok / seqs
    return out


def chunked_prefill_claim_is_complete(claim):
    """★ **「chunked prefill 默认开启」这句话【完整】吗？** —— 必须同时含两个限定（`F6`）。

    ★★★ `In V1` 与 `whenever possible` **缺一不可**；★ encoder-decoder 模型还会禁用 ⇒
      「不带限定」就是不完整（`C.2` 判它）。
    """
    return all(q in claim for q in specs.VLLM_CHUNKED_PREFILL_QUALIFIERS)


# ★★ **术语 → 原始出处**（`C.3` / `C.4` 的「反例 → 被拒」结构）。
TERM_ORIGIN = {"iteration-level scheduling": "ORCA (OSDI '22)"}


def term_origin_is(term, owner):
    """★★★ **某条术语的原始出处【是】`owner` 吗？** —— ★ 本库记录：`iteration-level scheduling`
    的原始出处是 **ORCA（OSDI '22）**，**不是 vLLM**（`F3`）。★ **归错主体 ⇒ 被拒**。
    """
    return TERM_ORIGIN.get(term) == owner


def is_official_vllm_term(term):
    """★ **`term` 是 vLLM 的【官方词】吗？** —— ★ 只有 `inter-token latency` 是。

    ★★★ `TPOT` 是**社区缩写**（★ 在 vLLM 两个 metrics 页面上**出现 0 处**）
      ⇒ ★ 把它当成官方词 ⇒ **被拒**（`F9`，`C.4` 判它）。
    """
    return term in specs.VLLM_INTER_TOKEN_LATENCY_QUOTE


# ===========================================================================
# 2. A 类 —— ★★★ 【本主题没有 A 类】
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ 【本主题没有 A 类】—— 本机无 GPU / 无任何引擎")
    print("=" * 78)
    print("  ★★★ 「vLLM 的行为与性能」**无处可验**：本机既没有 GPU 也没有 vLLM。")
    print("      ⇒ 本程序**跑在默认 `python3` 下** ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**")
    print("        （**程序内断言**）；★ 而「**把不可测的那一档写明**」是【另一条】纪律（不是 §2 的形态）。")
    print("      ★★ **如实点破**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**")
    print("        ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。")
    print("  ★★★ **不得把「常量钉住」冒充 A 类**（`P12` 的变体）⇒ ★★ 钉住一律归 `D` 类。")
    print("  ★★ 本程序只守一件事：**参数面有几种取值、两条禁写真的会被拒**。")
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
# 3. B 类：**参数面的结构性账**
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `[B]` 级的账：**参数面**（`max_num_seqs` 的取值集合 · 每槽 token 数）")
    print("=" * 78)
    print("  ★★ 载体 = vLLM 官方文档 `configuration/engine_args/`（参数说明）"
          " + 源码 `vllm/config/scheduler.py` / `vllm/engine/arg_utils.py`（main 分支），")
    print("      版本 **0.31.0**（文档站 = latest developer preview），抓取日 **%s**。"
          % specs.ENGINES_FETCH_DAY)

    # ── B.1 ★★ 参数面（I）：取值集合 ─────────────────────────────────────
    print()
    print("  ── B.1 ★★ **`max_num_seqs` 的取值集合**（设备/场景 × throughput 翻倍）──")
    print("     %-16s %-20s %-22s %-14s" % ("设备档", "场景", "max_num_batched_tokens", "max_num_seqs"))
    for dev, ctx, tok, seqs in specs.VLLM_BATCH_DEFAULTS:
        print("     %-16s %-20s %-22d %-14d" % (dev, ctx, tok, seqs))
    print("     ★★ **兜底类常量**：`DEFAULT_MAX_NUM_SEQS = %d` · `DEFAULT_MAX_NUM_BATCHED_TOKENS = %d`"
          % (specs.VLLM_MAX_NUM_SEQS_FALLBACK, specs.VLLM_MAX_NUM_BATCHED_TOKENS_FALLBACK))
    vals = sorted(distinct_max_num_seqs())
    print("     ★★★ **取值集合**（含兜底与 throughput 翻倍 ×%d）：`%s` ⇒ **%d 个不同取值**"
          % (specs.VLLM_THROUGHPUT_MULTIPLIER, vals, len(vals)))
    print("     ★★ 文档自己说这个 default 是「**mainly for convenience when testing**」——")
    print("        ⇒ ★★★ **「vLLM 的 `max_num_seqs` 默认是 N」这句话没有单一答案**（`F2`）。")
    check("B.1", len(specs.VLLM_BATCH_DEFAULTS) == 6 and len(vals) == 5
          and vals == [128, 256, 512, 1024, 2048]
          and specs.VLLM_MAX_NUM_SEQS_FALLBACK == 128
          and specs.VLLM_MAX_NUM_BATCHED_TOKENS_FALLBACK == 2048,
          "★★ **判据成立**：参数表 **6 格**（3 设备档 × 2 场景）+ 兜底 ⇒ `max_num_seqs` 的"
          "**取值集合有 %d 个**（`%s`）—— ★★ 只引兜底 `128` 会把 **A100 上 API server 的真实值 `256`** 写成 128"
          % (len(vals), vals),
          "**参数面被动过** —— ★ 先查 `VLLM_BATCH_DEFAULTS` / `VLLM_MAX_NUM_SEQS_FALLBACK`")

    # ── B.2 ★★ 参数面（II）：每槽 token 数 ───────────────────────────────
    print()
    print("  ── B.2 ★★ **每槽 token 数** = `max_num_batched_tokens / max_num_seqs` ──")
    tps = tokens_per_slot()
    for k in sorted(tps):
        print("     %-30s 每槽 %6.2f token" % (k, tps[k]))
    tset = sorted(set(tps.values()))
    print("     ★★ 取值 `%s` ⇒ **跨格从 %.0f 到 %.0f（%d 倍）**" % (tset, tset[0], tset[-1], tset[-1] / tset[0]))
    print("     ★★★ **读法**：若两个参数是**等比缩放的一对**，这个比值应当**恒定** ——")
    print("        它不恒定 ⇒ ★ **它们是**两个**各自**按设备/场景定的**独立预算**（`B.1` 的补充）。")
    check("B.2", tset == [8.0, 16.0, 32.0] and tps["A100/其余/LLM_CLASS"] == 32.0
          and tps["H100/H200/OPENAI_API_SERVER"] == 8.0,
          "★★ **判据成立**：每槽 token 数取 **`%s`**（`8..32`，跨 **4 倍**）⇒ ★ 两个参数不是等比缩放的一对；"
          "★★ 注：throughput 模式**两个都翻倍** ⇒ **比值不变**（这一点与 `B.1` 的取值集合互补）" % tset,
          "**每槽 token 数被动过** —— ★ 先查 `VLLM_BATCH_DEFAULTS` 的第三、四列")


# ===========================================================================
# 4. C 类：**四条禁写的真现场**（`F2`/`F6`/`F3`/`F9`）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**把兜底当权威** · **缺限定的默认** · **术语误归** · **裸写 TPOT**")
    print("=" * 78)
    print("  ★★★ 四条都对应计划 `§0.7 C.5` 的**明文禁写** —— ★ 本节的判据让它们**真的会失败**。")

    # ── C.1 ★★★ 把兜底常量当「vLLM 的默认值」（`F2`）────────────────────
    print()
    print("  ── C.1 ★★★ **把兜底常量当成「vLLM 的默认值」**（`F2`）──")
    bare_claim = "vLLM 的 max_num_seqs 默认是 128"
    good_claim = "A100/其余 上 API server 的 max_num_seqs 是 256"
    offline_claim = "H100/H200 上 离线 的 max_num_seqs 是 1024"
    print("     ★★★ **反例（不带设备与场景）**：`%s`" % bare_claim)
    print("        ⇒ 校验器判**成立**？ **%s**（★ 必须是 `False` —— **被拒**）"
          % default_claim_is_valid(bare_claim))
    print("     ★ **正例（带设备 + 场景）**：`%s` ⇒ **%s**（应 `True`）"
          % (good_claim, default_claim_is_valid(good_claim)))
    print("       ★ 另一场景正例（离线）：`%s` ⇒ **%s**（应 `True`）"
          % (offline_claim, default_claim_is_valid(offline_claim)))
    print("     取值集合：`%s`（%d 个）" % (sorted(distinct_max_num_seqs()), len(distinct_max_num_seqs())))
    print("     ★★★ 关键：那 `128` 只是**类常量兜底**；`get_batch_defaults()` 会按**设备 + 场景**覆盖它 ——")
    print("       ★★ **A100 上 API server 的真实值是 `256`**（★ 写成 `128` 就是「规范与实现不分」）。")
    check("C.1", (not default_claim_is_valid(bare_claim)) and default_claim_is_valid(good_claim)
          and default_claim_is_valid(offline_claim)
          and specs.VLLM_MAX_NUM_SEQS_FALLBACK == 128
          and len(distinct_max_num_seqs()) > 1,
          "★★★ **判据成立**：把兜底 `%d` 当成「vLLM 的默认值」（**不带设备与场景**）**被拒**；"
          "★ 带上设备与场景才通过（★ 两个场景词各验一次）—— 真实取值有 **%d 个**（`F2`）"
          % (specs.VLLM_MAX_NUM_SEQS_FALLBACK, len(distinct_max_num_seqs())),
          "**「单一默认值」没被拒绝** —— ★ 先查 `VLLM_BATCH_DEFAULTS` / `default_claim_is_valid`")

    # ── C.2 ★★★ chunked prefill「默认开启」缺限定（`F6`）────────────────
    print()
    print("  ── C.2 ★★★ **chunked prefill「默认开启」不带限定**（`F6`）──")
    full = "In V1, " + specs.VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE
    bare = "chunked prefill is enabled by default"
    print("     逐字引文：`%s`" % specs.VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE)
    print("     两个限定：`%s`（★ 缺一不可）" % (specs.VLLM_CHUNKED_PREFILL_QUALIFIERS,))
    print("     判定：完整表述 `%s` ⇒ **%s**（应 `True`）；裸表述 `%s` ⇒ **%s**（应 `False`）"
          % ("In V1, ...whenever possible", chunked_prefill_claim_is_complete(full),
             bare, chunked_prefill_claim_is_complete(bare)))
    print("     ★★★ **读法**：`whenever possible` 是句子的一部分 —— encoder-decoder 模型会**禁用**；")
    print("       ★★ 去掉限定就**在别的语义下是错的**（`V0` 语义 / 无条件默认）。")
    check("C.2", chunked_prefill_claim_is_complete(full)
          and not chunked_prefill_claim_is_complete(bare)
          and specs.VLLM_CHUNKED_PREFILL_QUALIFIERS == ("In V1", "whenever possible"),
          "★★★ **判据成立**：完整表述（带 `In V1` + `whenever possible`）通过；"
          "**裸表述 `%s` 被拒** —— ★★ 缺限定 ⇒ `F6` 的错法真的会失败" % bare,
          "**「默认开启」的限定被动过** —— ★ 先查 `VLLM_CHUNKED_PREFILL_QUALIFIERS` 与引文")

    # ── C.3 ★★★ 把 iteration-level scheduling 当成 vLLM 的术语（`F3`）───
    print()
    print("  ── C.3 ★★★ **把 `iteration-level scheduling` 当成 vLLM 造的术语**（`F3`）──")
    print("     逐字（ORCA）：`%s`" % specs.ORCA_ITERATION_LEVEL_QUOTE)
    mis = term_origin_is("iteration-level scheduling", "vLLM")
    print("     ★★★ **反例（归错主体）**：把 `iteration-level scheduling` 归给 **vLLM** ⇒ **%s**（应 `False`）" % mis)
    print("     ★ **正例**：归给 **ORCA (OSDI '22)** ⇒ **%s**（应 `True`）"
          % term_origin_is("iteration-level scheduling", "ORCA (OSDI '22)"))
    print("     ★★★ 依据：**原始术语**出自 **ORCA（OSDI '22）**，vLLM 首页只把它当**特性名**列出，")
    print("       **没有机制定义**（`§0.7 C` 事实 6/7）⇒ 引用必须写**出处是 ORCA**。")
    check("C.3", (not mis) and term_origin_is("iteration-level scheduling", "ORCA (OSDI '22)")
          and "granularity of iteration" in specs.ORCA_ITERATION_LEVEL_QUOTE,
          "★★★ **判据成立**：把 `iteration-level scheduling` 归给 vLLM 会被拒（**反例 → 被拒**）—— "
          "它的出处是 **ORCA（OSDI '22）**（`F3`）",
          "**术语出处被动过** —— ★ 先查 `ORCA_ITERATION_LEVEL_QUOTE` 与 `term_origin_is`")

    # ── C.4 ★★★ 裸写 TPOT 当成 vLLM 的官方词（`F9`）────────────────────
    print()
    print("  ── C.4 ★★★ **把 `TPOT` 当成 vLLM 的官方词**（`F9`）──")
    print("     vLLM 页面上该缩写的出现次数 = **%d**（★ 必须是 `0`）" % specs.VLLM_TPOT_OCCURRENCES)
    print("     vLLM 的官方词逐字：`%s`" % specs.VLLM_INTER_TOKEN_LATENCY_QUOTE)
    print("     ★★★ **反例（把社区缩写当官方词）**：`is_official_vllm_term(\"TPOT\")` ⇒ **%s**"
          "（应 `False` —— **被拒**）" % is_official_vllm_term("TPOT"))
    print("     ★ **正例**：`is_official_vllm_term(\"inter-token latency\")` ⇒ **%s**（应 `True`）"
          % is_official_vllm_term("inter-token latency"))
    print("     ★★★ **读法**：该缩写是**社区常用词**；vLLM 的词是 **inter-token latency** ——")
    print("       引用它时**必须注明**这一点，否则就是把社区词冒充官方词。")
    check("C.4", (not is_official_vllm_term("TPOT")) and is_official_vllm_term("inter-token latency")
          and specs.VLLM_TPOT_OCCURRENCES == 0
          and "inter-token latency" in specs.VLLM_INTER_TOKEN_LATENCY_QUOTE,
          "★★★ **判据成立**：把 `TPOT` 当成 **vLLM 的官方词**会被拒（**反例 → 被拒**）—— "
          "该词在 vLLM 页面上 **%d 处**；官方词是 **inter-token latency**（`F9`）"
          % specs.VLLM_TPOT_OCCURRENCES,
          "**TPOT 的计数或官方词被动过** —— ★ 先查 `VLLM_TPOT_OCCURRENCES`（须为 0）")


# ===========================================================================
# 5. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何引擎。")
    print("  ★★ 本程序**没有 A 类**（`D.0` 机械证明），**一切钉住归 `D` 类**。")
    before = len(FAILS)

    # ── D.1 ★ 标记自检（AST 读出，**不写死**）──────────────────────────────
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

    # ── D.2 ★ 标记自检（字面量比较）────────────────────────────────────────
    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: engines" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: engines`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住；"
          "★ 五个标记**两两不互为前缀**")

    # ── D.3 ★ 载体元数据 + 回链锚点 ───────────────────────────────────────
    check("D.3", specs.ENGINES_FETCH_DAY == "2026-10-09"
          and specs.VLLM_IR_REF == "45-kernel-lifecycle/requirement-discovery/01:205-214"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★★ **`vllm.ir` 的回链锚点** `%s`"
          "（★ **机制正文在彼处**，本主题**一行都不重写**）；★ `UNSOURCED == ()` 一并钉住"
          % (specs.ENGINES_FETCH_DAY, specs.VLLM_IR_REF))

    # ── D.3b ★★★ 口径守卫：参数表**整表逐元素钉住**（否则 B/C 会跑偏）──────
    #   ★★★ 它**必须跑在任何计算之前**（本库 `P23`）：★ 一个被改过的**设备/场景字符串**会让
    #     `tokens_per_slot()` 的键名变掉 ⇒ `B.2` 会 `KeyError` **崩溃**（而不是断言失败）。
    #     ⇒ ★★ 把**整表**（6 行 × 4 列 = 24 个叶子）在**这里**钉住，让「改常量」变成**断言捕获**。
    ok_tbl = all(seqs >= 1 and tok >= seqs for _d, _c, tok, seqs in specs.VLLM_BATCH_DEFAULTS) \
        and len(specs.VLLM_BATCH_DEFAULTS) > 0 and specs.VLLM_THROUGHPUT_MULTIPLIER > 1
    print("     ★★ 参数表非空、每行 `max_num_batched_tokens ≥ max_num_seqs`、倍率 > 1 = %s" % ok_tbl)
    check("D.3b", ok_tbl
          and specs.VLLM_BATCH_DEFAULTS == (
              ("B200/B300", "OPENAI_API_SERVER", 16384, 1024),
              ("B200/B300", "LLM_CLASS", 16384, 1024),
              ("H100/H200", "OPENAI_API_SERVER", 8192, 1024),
              ("H100/H200", "LLM_CLASS", 16384, 1024),
              ("A100/其余", "OPENAI_API_SERVER", 2048, 256),
              ("A100/其余", "LLM_CLASS", 8192, 256))
          and specs.VLLM_MAX_NUM_SEQS_FALLBACK == 128
          and specs.VLLM_MAX_NUM_BATCHED_TOKENS_FALLBACK == 2048
          and specs.VLLM_THROUGHPUT_MULTIPLIER == 2,
          "★★★ **口径守卫 + 整表逐元素钉住**：参数表 **6 行 × 4 列 = 24 个叶子** · 两处兜底 · 倍率 `%d`"
          "（★ 改掉任一**设备/场景字符串**会让 `tokens_per_slot()` 的键名变掉 ⇒ 必须在**这里**先钉住，"
          "否则 `B.2` 会 `KeyError` **崩溃**而非断言失败）" % specs.VLLM_THROUGHPUT_MULTIPLIER,
          "**参数表或倍率被动过** —— ★ 先查 `VLLM_BATCH_DEFAULTS` 的元素与 `VLLM_THROUGHPUT_MULTIPLIER`")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：引文逐字 · 逐元素 · 术语纪律
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · **逐元素**钉住 · 术语纪律（★ 纪律与断言成对）")
    print("=" * 78)

    # ── D.5 ★★ 五条引文逐字 ─────────────────────────────────────────────
    check("D.5", specs.VLLM_ARCH_BUSY_LOOP_QUOTE
          == "a busy loop that continuously schedules requests and dispatches work to the GPU workers"
          and specs.VLLM_MAX_NUM_SEQS_QUOTE
          == ("Maximum number of sequences to be processed in a single iteration. The default value "
              "here is mainly for convenience when testing. In real usage, this should be set in "
              "EngineArgs.create_engine_config.")
          and specs.VLLM_HOME_FEATURES_QUOTE
          == "Continuous batching of incoming requests, chunked prefill, prefix caching"
          and specs.ORCA_THROUGHPUT_QUOTE
          == "36:9× throughput improvement at the same level of latency"
          and specs.VLLM_HOME_23X_BLOG_QUOTE
          == ("How continuous batching enables 23x throughput in LLM inference while reducing "
              "p50 latency")
          and specs.VLLM_INTER_TOKEN_LATENCY_QUOTE
          == ("Mean inter-token latency (average time between successive output tokens) "
              "during the decode phase.")
          and specs.VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE
          == "chunked prefill is enabled by default whenever possible"
          and specs.ORCA_ITERATION_LEVEL_QUOTE
          == "schedules execution at the granularity of iteration (instead of request)",
          "★★ **六条引文逐字钉住**：引擎循环（`design/arch_overview`）· 参数说明（`engine_args`）· "
          "首页特性名（`/`）· ORCA 的 `36:9×`（**冒号**）· `23x` 博客标题 · inter-token latency\n"
          "        ★★★ **`36:9×` 与 `23x` 是【两个不同来源】**（`F4`）—— ★ 都是**引用值**，"
          "基线各自不同，**不得当作本库的测量**\n"
          "        ★★ 载体抓取日 %s" % specs.ENGINES_FETCH_DAY,
          "**六条引文之一被动过** —— ★ 先查 `VLLM_ARCH_BUSY_LOOP_QUOTE` / `ORCA_THROUGHPUT_QUOTE` 等")

    # ── D.6 ★★ PagedAttention 的载体（论文 vs 历史文档）──────────────────
    check("D.6", specs.PAGEDATTENTION_DEFINITION_QUOTE
          == ("we propose PagedAttention, an attention algorithm inspired by the classical virtual "
              "memory and paging techniques in operating systems.")
          and specs.PAGEDATTENTION_PAPER_CITATION
          == ("Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, Efficient Memory "
              "Management for Large Language Model Serving with PagedAttention, arXiv:2309.06180")
          and specs.VLLM_PAGED_ATTENTION_HISTORICAL_QUOTE
          == ("This is a historical document based on the original paper for vLLM. "
              "It no longer describes the code used in vLLM today."),
          "★★ **PagedAttention 的两条载体逐字钉住**：定义来自 **论文**（`[C]`，arXiv:2309.06180）；"
          "★★★ 而 vLLM 的 `design/paged_attention` 页**自标为历史文档** ⇒ **不得拿它当 vLLM 现状**（`F7`）\n"
          "        ★★ 机制正文（分页 / 算式）在 `kv-cache/02` 与 `130-model-optimization/quantization/18-kv-cache-basics:68-84`",
          "**PagedAttention 的载体被动过** —— ★ 先查 `PAGEDATTENTION_DEFINITION_QUOTE` 与历史文档引文")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.VLLM_MAX_NUM_SEQS_FALLBACK == 128
          and specs.VLLM_MAX_NUM_BATCHED_TOKENS_FALLBACK == 2048
          and specs.VLLM_THROUGHPUT_MULTIPLIER == 2
          and specs.VLLM_TPOT_OCCURRENCES == 0
          and specs.VLLM_BATCH_DEFAULTS == (
              ("B200/B300", "OPENAI_API_SERVER", 16384, 1024),
              ("B200/B300", "LLM_CLASS", 16384, 1024),
              ("H100/H200", "OPENAI_API_SERVER", 8192, 1024),
              ("H100/H200", "LLM_CLASS", 16384, 1024),
              ("A100/其余", "OPENAI_API_SERVER", 2048, 256),
              ("A100/其余", "LLM_CLASS", 8192, 256))
          and specs.VLLM_CHUNKED_PREFILL_QUALIFIERS == ("In V1", "whenever possible")
          and specs.ENGINES_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：参数表 **6 行 × 4 列 = 24 个叶子** · 两处兜底 · 倍率 · `TPOT` 计数 · "
          "限定语元组 · 抓取日；★★ `UNSOURCED == ()` 一并钉住\n"
          "        —— ★ 只钉「整体」会让**元素级**变异漏网（本库实测过：一次报出 58 条漏网）",
          "**某个常量被动过** —— ★ 先查 `VLLM_BATCH_DEFAULTS` 的元素")

    # ── D.9 ★★★ 术语纪律（★ 形态 A 的根治：计数 == 豁免清单）──────────────
    src = open(__file__, encoding="utf-8").read()
    # ★★ 被禁裸词（**刻意用拼接**而不是字面量）：★ 若这里写成字面量，**计数代码自己**也会被算进去，
    #   于是「计数 == 豁免覆盖数」这条**等式**永远差 1。★ 拼接后计数行里**不再出现该词的字面量**。
    WORD = "业" + "界"
    n_bare = src.count(WORD)
    n_exempt = sum(src.count(s) * s.count(WORD) for s in BARE_ATTRIBUTION_EXEMPT)
    # ★★★ 第三半：**被禁 ASCII 词**的裸写计数（★ `D3` 的根治）—— 判据同样是**等式**。
    n_bare_terms = _count_bare_ascii(src)
    n_exempt_terms = sum(src.count(s) * _count_bare_ascii(s) for s in TERM_BARE_EXEMPT)
    print()
    print("  ── D.9 ★★★ **术语纪律**：一律写「%s」/「%s」全称；限定语逐字出现 ──"
          % (TERM_KV_CACHE, TERM_SCHEDULER))
    print("     ★★ 源码里**裸写那个被禁裸词** = **%d** 处；豁免清单覆盖 = **%d**（清单 %d 项，逐项理由见模块头）"
          % (n_bare, n_exempt, len(BARE_ATTRIBUTION_EXEMPT)))
    print("     ★★★ 源码里**裸写那两个被禁 ASCII 词**（★ 紧邻中文）= **%d** 处；豁免清单覆盖 = **%d**"
          % (n_bare_terms, n_exempt_terms))
    check("D.9", TERM_KV_CACHE == "KV 缓存" and TERM_SCHEDULER == "请求调度器"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_KV_CACHE in src
          and BARE_ATTRIBUTION_EXEMPT
          == ("不裸写「" + WORD + "」这类无第二个独立主体的断言", "**不裸写「" + WORD + "」**")
          and n_bare == n_exempt
          and all(s in src for s in BARE_ATTRIBUTION_EXEMPT)
          and BARE_ASCII_WORDS == ("ca" + "che", "sched" + "uler")
          and TERM_BARE_EXEMPT == (
              "本域一律写「**KV 缓存**」全称（不裸写 `" + BARE_ASCII_WORDS[0] + "`，那是 L1/L2 与 kernel "
              + BARE_ASCII_WORDS[0] + "）",
              "本域一律写「**请求调度器**」全称（不裸写 `" + BARE_ASCII_WORDS[1] + "`，那是 GPU 的 warp "
              + BARE_ASCII_WORDS[1] + "）")
          and n_bare_terms == n_exempt_terms
          and all(s in src for s in TERM_BARE_EXEMPT),
          "★★ **判据成立**：两条术语常量 `%r` / `%r`（★ **不裸写**那两个 ASCII 词 —— "
          "它们在本库都另有含义）；★ 限定语 `%r` **逐字出现在本程序源码里**；"
          " ★★ **被禁裸词的裸写计数 `%d` 恰好等于豁免清单覆盖的 `%d`**；"
          " ★★★ **被禁 ASCII 词的裸写计数 `%d` 恰好等于其豁免清单覆盖的 `%d`**"
          "（★ 清单之外的任何一处裸写都会让等式不等 ⇒ **纪律真的被断言盯住了**）"
          % (TERM_KV_CACHE, TERM_SCHEDULER, LIMIT_QUOTE, n_bare, n_exempt, n_bare_terms, n_exempt_terms),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写** —— "
          "★ 本域一律写全称；★ 新增的裸写要么改全称、要么进豁免清单并写明理由")


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
    print("★ 依据在 `specs.py` 的 `# 来源：`（vLLM 官方文档 + 源码，抓取日 %s；ORCA 论文）。"
          % specs.ENGINES_FETCH_DAY)
    print("★ 本程序只核对「参数面有几种取值」与「四条禁写真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_vllm.py —— vLLM 的架构选择：引擎循环 · vllm.ir 的位置 · 版本相关的参数面")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何引擎。")
    print("  ★★★ **本主题没有 A 类**（`D.0` 机械证明）；★★ **机制一律回链**（分页 / 前缀树 / 调度器 / `vllm.ir`）。")
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
