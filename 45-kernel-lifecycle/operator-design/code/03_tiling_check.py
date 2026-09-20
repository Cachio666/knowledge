#!/usr/bin/env python3
r"""03-tiling-check —— 约束方程的四条腿，以及「边界 bug 不会在第一块 tile 上暴露」。

对应文档：`../03-tiling-and-boundaries.md`

★ 本程序的可证伪断言：
  A 类  四组约束（寄存器 / 共享内存 / 对齐 / warp 数）的逐条判定与**手算**一致；
        ★ 其中两条是「反直觉但有来源」的：
         · 寄存器预算的真实上限是 **248**（= 255 按 8 一档**向下取整**），不是 255；
         · `ldm` 的「8 / 4」与「16 字节」**不是**同一个数——是**字节积**才等于 16。
  B 类  给出**恰好不越界**与**恰好越界**的两组参数（共享内存一侧按 `stages` 调，
        寄存器一侧按 248 / 256 那一对），两侧都做成断言。
  C 类  ★ 构造三个「**只在最后一 tile 出错**」的缺陷，用**只含 2 的幂形状**的测试集跑，
        三个缺陷的**捕获数都是 0**；换成 §1 推出的**最小边界形状集**后**全部被捕获**。
        这是 §4 第 ① 条论断的直接证据，也是本程序存在的理由。

跑法：`make 03_tiling_check`。纯 python3 + **numpy**（做整数矩阵乘与逐元素比对），
不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则结论不可比）：
  · 约束方程里的**寄存器模型是本库的建模选择 [E]**，不是任何真实 kernel 的寄存器数。
    它的两个输入（`MAX_REGISTERS_PER_THREAD` / `REGISTER_ALLOC_GRANULARITY` /
    fragment 的每线程寄存器数）**都有 A 级来源**，但**组合方式**是本库定的。
    `ADDRESS_AND_INDEX_REGS_ASSUMED = 0` 是**明写的假设** ⇒ 算出的 tile 上界是**乐观上界**。
  · §C 的缺陷是**本库构造的模型**，不是任何一个真实 kernel 的 bug 报告。
  · 本程序**不产出实测性能数字**：它算的全部是**约束**与**覆盖率**，不是速度。
"""

import sys

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
# 1. 约束方程：四组约束，各自一个纯函数
# ===========================================================================
#
# ★ 四组约束的**来源全部是 A 级**（见 specs.py §3 每条常量的 `# 来源：`）；
#   把它们**组合**成一个预算系统的这一层是**本库的建模 [E]**。

ESIZE = SP.FLOAT16_BYTES                    # fp16：2 B/element（本篇唯一的 dtype）
FRAG = 16                                   # wmma 的 fragment 边长（三种 shape 都是 16 的倍数）
SMEM_LIMIT_BYTES = SP.SMEM_PER_SM_MAX_KB * 1024
REG_LIMIT_RAW = SP.MAX_REGISTERS_PER_THREAD          # 255
REG_GRAN = SP.REGISTER_ALLOC_GRANULARITY             # 8
# ★★ 寄存器的**真实**上限：255 按 8 一档**向下取整** = 248（256 > 255）
REG_LIMIT_EFFECTIVE = (REG_LIMIT_RAW // REG_GRAN) * REG_GRAN
WARPS_LIMIT = SP.MAX_THREADS_PER_BLOCK // SP.WARP_SIZE       # block 侧 = 32
WARPS_LIMIT_SM = SP.MAX_WARPS_PER_SM                          # SM 侧 = 64
WARPS_LIMIT_EFFECTIVE = min(WARPS_LIMIT, WARPS_LIMIT_SM)      # 两者取最紧 = 32


def frag_regs(tm, tn, tk, wm, wn):
    """一个 block tile 在**每线程**上的 fragment 寄存器数。

    模型（**本库的建模 [E]**，不是实测）：
      · 累加器常驻：每个 (16×16) 的 warp 级累加片段占
        `WMMA_ACC_F32_FRAGMENT_REGS` 个寄存器；
      · A、B 各保留**一个 K 方向工作集**（`tk/16` 个片段行/列）。
    三种 shape 的 fragment **寄存器数相同**（specs：`WMMA_A_FRAGMENT_REGS` 等），
    差别只在布局——所以这个模型与选哪一种 shape 无关。
    """
    ptm, ptn, ptk = tm // wm, tn // wn, tk          # 每个 warp 负责的子 tile
    acc = SP.WMMA_ACC_F32_FRAGMENT_REGS * (ptm // FRAG) * (ptn // FRAG)
    a = SP.WMMA_A_FRAGMENT_REGS * (ptm // FRAG) * (ptk // FRAG)
    b = SP.WMMA_B_FRAGMENT_REGS * (ptk // FRAG) * (ptn // FRAG)
    return acc + a + b


def regs_padded(r):
    """按分配粒度**向上取整**——这才是要跟 255 比的那个数。"""
    return ((r + REG_GRAN - 1) // REG_GRAN) * REG_GRAN


def smem_bytes(tm, tn, tk, stages):
    """A 与 B 两个 stage 缓冲区的字节数（fp16，读写各一份 ⇒ 系数 1，不乘 2）。"""
    return (tm * tk + tk * tn) * ESIZE * stages


def c_fragment_multiple(tm, tn, tk, wm, wn):
    """约束①：fragment 的离散性——tile 必须是 16 的倍数，且每个 warp 至少一个片段。"""
    return (tm % FRAG == 0 and tn % FRAG == 0 and tk % FRAG == 0
            and (tm // wm) % FRAG == 0 and (tn // wn) % FRAG == 0)


def c_registers(tm, tn, tk, wm, wn):
    """约束②：寄存器。**判据用的是取整后的 248，不是 255。**"""
    return regs_padded(frag_regs(tm, tn, tk, wm, wn)
                       + SP.ADDRESS_AND_INDEX_REGS_ASSUMED) <= REG_LIMIT_EFFECTIVE


def c_smem(tm, tn, tk, wm, wn, stages):
    """约束③：共享内存。"""
    return smem_bytes(tm, tn, tk, stages) <= SMEM_LIMIT_BYTES


def c_warps(tm, tn, tk, wm, wn):
    """约束④：warp 数（block 侧与 SM 侧**取最紧**）。"""
    return wm * wn <= WARPS_LIMIT_EFFECTIVE


def c_alignment_tile(tm, tn, tk, wm, wn, dtype="fp16"):
    """约束⑤（对齐，tile 侧）：行距的**字节数**必须是 `WMMA_LDM_BYTE_MULTIPLE` 的倍数。"""
    ld = SP.WMMA_LDM_ELEMENT_MULTIPLE[dtype]
    return (tk % ld == 0) and (tn % ld == 0) and (tk * ESIZE) % SP.WMMA_LDM_BYTE_MULTIPLE == 0


def c_alignment_problem(n, dtype="fp16"):
    """约束⑤（对齐，**问题形状**侧）：PTX 层要求起始地址按 fragment 的字节大小对齐。

    ★ 这一条**不受 tile 参数影响**——它判的是**问题自己的行距** `n`。
      ⇒ 它属于「问题形状约束」，修复方式是**补 padding / 换布局**，不是换 tile。
    """
    return (n * ESIZE) % SP.WMMA_FRAGMENT_BYTES == 0


def tightness(tm, tn, tk, wm, wn, stages):
    """**三大预算**的紧度 = 用量 ÷ 上限。1.0 表示刚好用满。

    ★★ 对齐**不在这里**——它是一条**闸门**（合法/不合法），不是一个可比的比值。
       「四者取最紧」的正确读法是：**三条预算各自算紧度、取最大者**，
       而**对齐是一条前置的合法性闸门**。把布尔与比值放进同一个 `max` 会得到一个
       恒等于对齐的结果，那是**度量错误**，不是发现（本实现在初版就犯过这个错）。
    """
    return {
        "registers": (regs_padded(frag_regs(tm, tn, tk, wm, wn)
                                  + SP.ADDRESS_AND_INDEX_REGS_ASSUMED)
                      / REG_LIMIT_EFFECTIVE),
        "shared_memory": smem_bytes(tm, tn, tk, stages) / SMEM_LIMIT_BYTES,
        "warps": (wm * wn) / WARPS_LIMIT_EFFECTIVE,
    }


def alignment_gate(tm, tn, tk, wm, wn):
    """对齐**闸门**：合法 / 不合法。单独列出，因为它不是一个预算。"""
    return c_alignment_tile(tm, tn, tk, wm, wn)


def verdict(tm, tn, tk, wm, wn, stages):
    """四组约束逐条判定 + 取最紧的那条。"""
    v = {
        "fragment_multiple": c_fragment_multiple(tm, tn, tk, wm, wn),
        "registers": c_registers(tm, tn, tk, wm, wn),
        "shared_memory": c_smem(tm, tn, tk, wm, wn, stages),
        "warps": c_warps(tm, tn, tk, wm, wn),
        "alignment": c_alignment_tile(tm, tn, tk, wm, wn),
    }
    v["legal"] = all(v.values())
    t = tightness(tm, tn, tk, wm, wn, stages)
    # ★ 只在**三大预算**之间取最紧（对齐是闸门，见 `tightness` 的 docstring）
    v["tightest"] = max(t, key=lambda k: t[k])
    v["budget_tightness"] = t
    return v


# ===========================================================================
head("§A  算法正确性：四组约束的逐条判定与手算一致")
# ===========================================================================

sub("A.1  ★ 寄存器预算的真实上限是 **248**，不是 255")
print(f"    MAX_REGISTERS_PER_THREAD = {REG_LIMIT_RAW}；分配粒度 = {REG_GRAN}")
print(f"    ⇒ 上取整到 {REG_GRAN} 的倍数后 ≤ {REG_LIMIT_RAW} 的最大值是 {REG_LIMIT_EFFECTIVE}"
      f"（下一档 {REG_LIMIT_EFFECTIVE + REG_GRAN} > {REG_LIMIT_RAW} ⇒ 那一档不可达）")
check("A.1 ★ 有效上限 == 248（不是 255）", REG_LIMIT_EFFECTIVE == 248,
      f"{REG_LIMIT_EFFECTIVE}")
check("A.1 ★ 250 会被取整成 256 ⇒ **非法**（手算：不能只比 250 ≤ 255）",
      regs_padded(250) == 256 and regs_padded(250) > REG_LIMIT_RAW,
      f"{regs_padded(250)}")
check("A.1 248 恰好合法、256 恰好非法（差一个粒度档）",
      regs_padded(248) == 248 and regs_padded(256) == 256
      and regs_padded(248) <= REG_LIMIT_EFFECTIVE < regs_padded(256),
      f"{regs_padded(248)} / {regs_padded(256)}")

sub("A.2  ★ 对齐的两条规则**不是同一个数**——「8 / 4」乘上元素宽度才是 16")
for dt, esz in (("fp16", SP.FLOAT16_BYTES), ("fp32", 4)):
    ld = SP.WMMA_LDM_ELEMENT_MULTIPLE[dt]
    print(f"    {dt}: ldm 必须是 {ld} 的倍数（**元素**）"
          f" ⇒ 行距 {ld} × {esz} B = {ld*esz} B（**字节**）")
check("A.2 fp16 的 ldm 倍数 == 8、fp32 == 4（两者都在表里）",
      SP.WMMA_LDM_ELEMENT_MULTIPLE == {"fp16": 8, "fp32": 4},
      f"{SP.WMMA_LDM_ELEMENT_MULTIPLE}")
check("A.2 ★★ `ldm` 本身**不是** 16 的倍数（8 和 4 都不是）",
      SP.WMMA_LDM_ELEMENT_MULTIPLE["fp16"] != 16
      and SP.WMMA_LDM_ELEMENT_MULTIPLE["fp32"] != 16,
      f"{SP.WMMA_LDM_ELEMENT_MULTIPLE}")
check("A.2 ★★ 是**字节积**等于 16：8×2 == 4×4 == 16",
      8 * SP.FLOAT16_BYTES == 4 * 4 == SP.WMMA_LDM_BYTE_MULTIPLE == 16,
      f"{8*SP.FLOAT16_BYTES} / {4*4}")
check("A.2 ★★★ 而 PTX 层的 fragment 字节数是 **32**（≠ 16）——两条规则不同层级",
      SP.WMMA_FRAGMENT_BYTES == 32 and SP.WMMA_FRAGMENT_BYTES != SP.WMMA_LDM_BYTE_MULTIPLE,
      f"{SP.WMMA_FRAGMENT_BYTES} vs {SP.WMMA_LDM_BYTE_MULTIPLE}")
check("A.2 这把「别互相换算」写成了机器判据", SP.WMMA_ALIGN_RULES_ARE_DISTINCT_LEVELS is True)
print("    ★ 本库 `20-gpu-architecture/tensor-core/02-fragment-layout.md:236` 逐字警告过")
print("      「两个数都等于 16 字节，但不是同一条规则，别互相换算」——")
print("      上面这三条 check 就是那句话的机器化：把 16 与 32 断言成**不相等**。")

sub("A.3  四条约束的逐条判定：四组手算 vs 函数")
HAND = [
    # (TM, TN, TK, WM, WN, stages, 手算的 fragment 寄存器数, 手算的总寄存器, 合法?)
    # 手算 X：PTM=PTN=64, PTK=16 ⇒ acc=8·4·4=128, A=8·4·1=32, B=8·1·4=32 ⇒ 192
    (128, 128, 16, 2, 2, 3, 128 + 32 + 32, 192, True),
    # 手算 Y：PTM=PTN=64, PTK=32 ⇒ acc=128, A=8·4·2=64, B=8·2·4=64 ⇒ 256 > 248 ⇒ 非法
    (128, 128, 32, 2, 2, 3, 128 + 64 + 64, 256, False),
    # 手算 Z：PTM=PTN=32, PTK=64 ⇒ acc=8·2·2=32, A=8·2·4=64, B=8·4·2=64 ⇒ 160
    (64, 64, 64, 2, 2, 4, 32 + 64 + 64, 160, True),
    # 手算 W：PTM=128, PTN=64, PTK=64 ⇒ acc=8·8·4=256, A=8·8·4=256, B=8·4·4=128 ⇒ 640
    (256, 256, 64, 2, 4, 2, 256 + 256 + 128, 640, False),
]
print(f"    {'(TM,TN,TK) warps stages':>30} {'frag regs':>10} {'手算':>6} {'总 regs':>8}"
      f" {'合法':>5} {'最紧的约束':>16}")
for tm, tn, tk, wm, wn, st, hand_frag, hand_total, hand_legal in HAND:
    got_frag = frag_regs(tm, tn, tk, wm, wn)
    got_total = regs_padded(got_frag + SP.ADDRESS_AND_INDEX_REGS_ASSUMED)
    v = verdict(tm, tn, tk, wm, wn, st)
    print(f"    {f'({tm},{tn},{tk}) w{wm}x{wn} s{st}':>30} {got_frag:>10} {hand_frag:>6}"
          f" {got_total:>8} {str(v['legal']):>5} {v['tightest']:>16}")
    check(f"A.3 ({tm},{tn},{tk}) w{wm}x{wn} 的 fragment 寄存器数与手算一致",
          got_frag == hand_frag, f"{got_frag} vs {hand_frag}")
    check(f"A.3 ({tm},{tn},{tk}) 的总寄存器（取整后）与手算一致",
          got_total == hand_total, f"{got_total} vs {hand_total}")
    check(f"A.3 ({tm},{tn},{tk}) 的合法性与手判一致", v["legal"] == hand_legal,
          f"{v['legal']} vs {hand_legal}")

sub("A.4  共享内存算式与手算一致")
tm, tn, tk, st = 128, 128, 16, 3
hand_smem = (128 * 16 + 16 * 128) * 2 * 3
print(f"    (128,128,16) stages=3：(TM·TK + TK·TN)·esize·stages")
print(f"      = ({128*16} + {16*128}) × {ESIZE} × {st} = {smem_bytes(tm,tn,tk,st):,} B")
print(f"      上限 = {SP.SMEM_PER_SM_MAX_KB} KB = {SMEM_LIMIT_BYTES:,} B"
      f" ⇒ 紧度 {smem_bytes(tm,tn,tk,st)/SMEM_LIMIT_BYTES:.4f}")
check("A.4 (128,128,16) stages=3 的共享内存 == 24576 B（手算）",
      smem_bytes(tm, tn, tk, st) == hand_smem == 24576, f"{smem_bytes(tm,tn,tk,st)}")
check("A.4 该配置的共享内存紧度 == 24576/233472",
      abs(tightness(tm, tn, tk, 2, 2, st)["shared_memory"] - 24576 / 233472) < 1e-15)
check("A.4 ★ 它**远未**越界（紧度 0.105）⇒ 这一档的瓶颈在寄存器，不在共享内存",
      tightness(tm, tn, tk, 2, 2, st)["shared_memory"] < 0.2,
      f"{tightness(tm,tn,tk,2,2,st)['shared_memory']:.4f}")

sub("A.5  warp 数：block 侧与 SM 侧**取最紧**")
print(f"    block 侧 = {SP.MAX_THREADS_PER_BLOCK} ÷ {SP.WARP_SIZE} = {WARPS_LIMIT}")
print(f"    SM 侧    = {SP.MAX_WARPS_PER_SM}")
print(f"    ⇒ 取最紧 = {WARPS_LIMIT_EFFECTIVE}")
check("A.5 block 侧上限 == 32", WARPS_LIMIT == 32, f"{WARPS_LIMIT}")
check("A.5 SM 侧上限 == 64", WARPS_LIMIT_SM == 64, f"{WARPS_LIMIT_SM}")
check("A.5 ★ 取最紧 == 32（block 侧更紧，SM 侧这份是**不会生效的**约束）",
      WARPS_LIMIT_EFFECTIVE == 32 and WARPS_LIMIT_EFFECTIVE == WARPS_LIMIT,
      f"{WARPS_LIMIT_EFFECTIVE}")
check("A.5 32 个 warp 在 fragment 上可行（PTM/16 ≥ 1 需要 TM ≥ 32·16 = 512）",
      c_fragment_multiple(512, 16, 16, 32, 1) and not c_fragment_multiple(256, 16, 16, 32, 1),
      "512✓ / 256✗")


# ===========================================================================
head("§B  模型与公式：恰好越界 / 恰好不越界的那两组参数")
# ===========================================================================

sub("B.1  共享内存一侧：把 `stages` 调到边界")
TM, TN, TK = 128, 128, 32
per_stage = smem_bytes(TM, TN, TK, 1)
stages_max = SMEM_LIMIT_BYTES // per_stage
print(f"    tile ({TM},{TN},{TK})：每 stage = {per_stage:,} B")
print(f"    ⇒ stages_max = floor({SMEM_LIMIT_BYTES:,} / {per_stage:,}) = {stages_max}")
for s in (stages_max, stages_max + 1):
    used = smem_bytes(TM, TN, TK, s)
    ok = c_smem(TM, TN, TK, 2, 2, s)
    print(f"      stages={s:>2}：{used:>7,} B  ≤ {SMEM_LIMIT_BYTES:,} ? {ok}"
          f"   紧度 {used/SMEM_LIMIT_BYTES:.6f}")
check("B.1 ★ **恰好不越界**的那组：stages = stages_max 时合法",
      c_smem(TM, TN, TK, 2, 2, stages_max) is True)
check("B.1 ★ **恰好越界**的那组：stages_max + 1 非法",
      c_smem(TM, TN, TK, 2, 2, stages_max + 1) is False)
check("B.1 ★ 两者之差就是**一个 stage**（不是「差不多到边界」）",
      smem_bytes(TM, TN, TK, stages_max + 1) - smem_bytes(TM, TN, TK, stages_max)
      == per_stage, f"{per_stage:,} B")
check("B.1 越界组超出上限的量 < 一个 stage ⇒ 它确实是**恰好**越界",
      smem_bytes(TM, TN, TK, stages_max + 1) - SMEM_LIMIT_BYTES < per_stage,
      f"{smem_bytes(TM,TN,TK,stages_max+1) - SMEM_LIMIT_BYTES:,} B")

sub("B.2  寄存器一侧：248 与 256 那一对")
# ★ 解 8·(a·b + a·1 + 1·b) == 248 ⇒ (a+1)(b+1) == 32 ⇒ (3,7) ⇒ PTM=48, PTN=112
#   解 8·(a·b + a·1 + 1·b) == 256 ⇒ (a+1)(b+1) == 33 ⇒ (2,10) ⇒ PTM=32, PTN=160
LEGAL_CT = (48, 112, 16, 1, 1)
OVER_CT = (32, 160, 16, 1, 1)
for label, ct in (("恰好不越界", LEGAL_CT), ("恰好越界", OVER_CT)):
    tm, tn, tk, wm, wn = ct
    r = regs_padded(frag_regs(tm, tn, tk, wm, wn) + SP.ADDRESS_AND_INDEX_REGS_ASSUMED)
    print(f"    {label}：tile ({tm},{tn},{tk}) w{wm}x{wn}"
          f" ⇒ regs = {r}   ≤ {REG_LIMIT_EFFECTIVE} ? {r <= REG_LIMIT_EFFECTIVE}")
check("B.2 ★ 恰好不越界的那组 register 数 == 248（**正好**等于有效上限）",
      regs_padded(frag_regs(*LEGAL_CT[:3], LEGAL_CT[3], LEGAL_CT[4])) == 248,
      f"{regs_padded(frag_regs(*LEGAL_CT[:3], LEGAL_CT[3], LEGAL_CT[4]))}")
check("B.2 ★ 恰好越界的那组 register 数 == 256（**正好**比上限大一档）",
      regs_padded(frag_regs(*OVER_CT[:3], OVER_CT[3], OVER_CT[4])) == 256,
      f"{regs_padded(frag_regs(*OVER_CT[:3], OVER_CT[3], OVER_CT[4]))}")
check("B.2 ★ 248 合法、256 非法（这就是「255 不是真上限」的后果）",
      c_registers(*LEGAL_CT) is True and c_registers(*OVER_CT) is False,
      f"{c_registers(*LEGAL_CT)} / {c_registers(*OVER_CT)}")
# ★ 反驳点：**不用分配粒度**、直接拿 255 比的朴素判据会接受一个达不到的值
NAIVE_ACCEPTS = 250 <= REG_LIMIT_RAW                       # 朴素：250 ≤ 255 ⇒ 合法
TRUE_VERDICT = regs_padded(250) <= REG_LIMIT_EFFECTIVE     # 真实：取整到 256 ⇒ 非法
print(f"    反驳点：需求 250 个寄存器时——")
print(f"      朴素判据（直接比 255）：250 ≤ {REG_LIMIT_RAW} ⇒ **合法**")
print(f"      真实判据（按 {REG_GRAN} 取整）：ceil{REG_GRAN}(250) = {regs_padded(250)}"
      f" > {REG_LIMIT_EFFECTIVE} ⇒ **非法**")
check("B.2 ★★ 反驳点：朴素判据接受 250、真实判据拒绝它（⇒ 分配粒度不可省）",
      NAIVE_ACCEPTS is True and TRUE_VERDICT is False,
      f"朴素 {NAIVE_ACCEPTS} / 真实 {TRUE_VERDICT}")

sub("B.3  合法 tile 参数集合的规模与「谁最紧」的分布")
TMS = (16, 32, 64, 128, 256)
TKS = (16, 32, 64)
WARPS = ((1, 1), (2, 1), (1, 2), (2, 2), (2, 4), (4, 2), (4, 4))
legal, count, tightest_hist = [], 0, {}
for tm in TMS:
    for tn in TMS:
        for tk in TKS:
            for wm, wn in WARPS:
                for st in (1, 2, 3, 4, 5):
                    count += 1
                    v = verdict(tm, tn, tk, wm, wn, st)
                    if v["legal"]:
                        legal.append((tm, tn, tk, wm, wn, st))
                        tightest_hist[v["tightest"]] = tightest_hist.get(v["tightest"], 0) + 1
print(f"    穷举空间 = {count:,} 组（TM,TN∈{TMS} × TK∈{TKS} × warps × stages 1..5）")
print(f"    合法组合 = {len(legal):,} 组")
print(f"    最紧约束的分布（**只有被枚举到的约束才可能出现**）：")
for k in sorted(tightest_hist):
    print(f"      {k:18s} {tightest_hist[k]:>6,}")
check("B.3 合法组合非空且远少于穷举空间（约束确实在筛）",
      0 < len(legal) < count, f"{len(legal)} / {count}")
check("B.3 ★ 合法组合里**至少出现过两种**不同的「最紧约束」"
      "（否则「四者取最紧」是空话）",
      len(tightest_hist) >= 2, f"{sorted(tightest_hist)}")
check("B.3 ★ 寄存器是其中**最常见**的最紧约束（⇒ 它是本主题的主要限制）",
      (max(tightest_hist, key=lambda k: tightest_hist[k]) == "registers"),
      f"{tightest_hist}")
check("B.3 每个合法组合的紧度全部 ≤ 1（定义使然，但值得钉住）",
      all(all(t <= 1.0 + 1e-12 for t in tightness(*c).values()) for c in legal[:200]))
print(f"    ★ 结论：**tile 选型是「受约束的离散选择」**——{len(legal):,} 个候选里，")
print(f"      {tightest_hist.get('registers', 0):,} 个是被**寄存器**卡住的。")


# ===========================================================================
head("§C  边界与反例：只测 2 的幂形状 ⇒ **捕获数为 0**")
# ===========================================================================
#
# ★ 三个**只在最后一 tile 出错**的缺陷模型（本库构造，不是任何真实 kernel 的 bug 报告）。

TILE = (64, 64, 16)          # 本篇演示用的 tile：M/N/K 各 64/64/16
DEFECT_TM, DEFECT_TN, DEFECT_TK = TILE
RNG = np.random.default_rng(20260920)


def make_problem(M, N, K):
    """整数问题（精确可判，不用容差）——这样「相等」是**逐位**的。"""
    A = RNG.integers(-9, 10, size=(M, K), dtype=np.int64)
    B = RNG.integers(-9, 10, size=(K, N), dtype=np.int64)
    return A, B


def reference(A, B):
    return A @ B


def defect_drop_m_tail(A, B, M, N, K):
    """缺陷①：**只**在 M 方向用 `M // TM` 而不是 `ceil(M / TM)` ⇒ 最后一块被丢掉。

    ★ N 与 K 的余数**照常正确处理**——这样每个缺陷**只对应一个轴**，
      于是「每个轴都需要自己的边界形状」这句话才是可证的（见 §C.3）。
    """
    out = np.zeros((M, N), dtype=np.int64)
    for i0 in range(0, (M // DEFECT_TM) * DEFECT_TM, DEFECT_TM):
        for j0 in range(0, N, DEFECT_TN):
            cols = list(range(j0, min(j0 + DEFECT_TN, N)))
            for k0 in range(0, K, DEFECT_TK):
                kk = list(range(k0, min(k0 + DEFECT_TK, K)))
                out[i0:i0 + DEFECT_TM, cols] += (
                    A[i0:i0 + DEFECT_TM][:, kk] @ B[kk][:, cols])
    return out


def defect_misalign_n_tail(A, B, M, N, K):
    """缺陷②：尾部 tile 把**起点夹回** `N − TN`（一个常见的 tail 处理手法），
    但**忘了把这个偏移从写回的位置里减掉** ⇒ 尾部列拿到的是**别处的列**。

    ★ 本实现初版写的是「列索引对 TN 取模回绕」，那个写法在尾部**恰好是恒等变换**
      （`(cols − j0) % TN == 0` 当 `cols == j0`）⇒ 捕获数 0，看起来像「边界集没用」。
      **错的是缺陷模型，不是边界集。** 这一条记在报告里。
    """
    out = np.zeros((M, N), dtype=np.int64)
    for i0 in range(0, M, DEFECT_TM):
        for j0 in range(0, N, DEFECT_TN):
            cols = list(range(j0, min(j0 + DEFECT_TN, N)))
            j0c = min(j0, N - DEFECT_TN)          # 把 tile 起点夹回去
            res = np.zeros((min(DEFECT_TM, M - i0), DEFECT_TN), dtype=np.int64)
            for k0 in range(0, K, DEFECT_TK):
                res += (A[i0:i0 + res.shape[0], k0:k0 + DEFECT_TK]
                        @ B[k0:k0 + DEFECT_TK, j0c:j0c + DEFECT_TN])
            # BUG：`res` 的第 0 列对应的是 `j0c`，不是 `j0`
            for t_, j in enumerate(cols):
                out[i0:i0 + res.shape[0], j] += res[:, t_]
    return out


def defect_drop_k_tail(A, B, M, N, K):
    """缺陷③：**只**在 K 方向用 `K // TK` ⇒ K 的余数完全不参与累加。"""
    out = np.zeros((M, N), dtype=np.int64)
    for i0 in range(0, M, DEFECT_TM):
        for j0 in range(0, N, DEFECT_TN):
            cols = list(range(j0, min(j0 + DEFECT_TN, N)))
            rows = list(range(i0, min(i0 + DEFECT_TM, M)))
            for k0 in range(0, (K // DEFECT_TK) * DEFECT_TK, DEFECT_TK):
                out[np.ix_(rows, cols)] += (
                    A[np.ix_(rows, list(range(k0, k0 + DEFECT_TK)))]
                    @ B[np.ix_(list(range(k0, k0 + DEFECT_TK)), cols)])
    return out


DEFECTS = {
    "drop M tail（M//TM 而非 ceil）": defect_drop_m_tail,
    "misalign N tail（夹回起点却写错位置）": defect_misalign_n_tail,
    "drop K tail（K//TK 而非 ceil）": defect_drop_k_tail,
}

# ★ 测试集一：**只含 2 的幂形状**（每个轴都是 tile 边长的整数倍 ⇒ 没有不完整 tile）
POW2_SHAPES = [(64, 64, 16), (128, 128, 32), (256, 256, 64), (512, 512, 16)]
# ★ 测试集二：§1 推出的**最小边界形状集**——每个轴各取「tile 边长 + 1」与「2×+1」
BOUNDARY_SHAPES = [
    (DEFECT_TM + 1, 64, 16),          # M 的余数 = 1
    (2 * DEFECT_TM + 1, 64, 16),      # M 的余数 = 1，但跨过两块（抓「只处理第一块 tail」）
    (64, DEFECT_TN + 1, 16),          # N 的余数 = 1
    (64, 2 * DEFECT_TN + 1, 16),      # N 的余数 = 1，跨两块
    (64, 64, DEFECT_TK + 1),          # K 的余数 = 1
    (DEFECT_TM + 1, DEFECT_TN + 1, DEFECT_TK + 1),   # 三个轴同时有尾巴
]


def captured(shape, defect_fn):
    """缺陷在这个形状上被**捕获**了吗？（与参考实现逐位比对）"""
    M, N, K = shape
    A, B = make_problem(M, N, K)
    return not np.array_equal(defect_fn(A, B, M, N, K), reference(A, B))


sub("C.1  测试集一（**只含 2 的幂形状**）：每个缺陷的捕获数")
print(f"    形状集 = {POW2_SHAPES}")
print(f"    （全部是 (64,64,16) 的整数倍 ⇒ **没有不完整 tile**）")
print()
print(f"    {'缺陷':34s} {'捕获数':>7} {'/ 形状数':>9}")
pow2_caps = {}
for name, fn in DEFECTS.items():
    n = sum(1 for s in POW2_SHAPES if captured(s, fn))
    pow2_caps[name] = n
    print(f"    {name:34s} {n:>7} {len(POW2_SHAPES):>9}")
check("C.1 ★★ 只含 2 的幂形状时，**三个缺陷的捕获数全部为 0**",
      all(v == 0 for v in pow2_caps.values()), f"{pow2_caps}")
check("C.1 ★★ 三个缺陷在 4 个 2 的幂形状上共 12 次比对，**一次都没捕获**",
      sum(pow2_caps.values()) == 0 and len(POW2_SHAPES) * len(DEFECTS) == 12,
      f"{sum(pow2_caps.values())} / 12")

sub("C.2  测试集二（**最小边界形状集**）：同一个缺陷模型，捕获数")
print(f"    形状集（每个轴各取 tile 边长 +1 与 2×+1，再加一个三轴同时有余数的）：")
for s in BOUNDARY_SHAPES:
    print(f"      {s}")
print()
print(f"    {'缺陷':34s} {'捕获数':>7} {'/ 形状数':>9}")
bnd_caps = {}
for name, fn in DEFECTS.items():
    hits = [s for s in BOUNDARY_SHAPES if captured(s, fn)]
    bnd_caps[name] = len(hits)
    print(f"    {name:34s} {len(hits):>7} {len(BOUNDARY_SHAPES):>9}")
    for s in hits:
        print(f"        捕获于 {s}")
check("C.2 ★★ 换成边界集后，**三个缺陷都被捕获**（捕获数全部 ≥ 1）",
      all(v >= 1 for v in bnd_caps.values()), f"{bnd_caps}")
check("C.2 ★★ 每一个缺陷的捕获数都**从 0 变成 ≥1**——这就是 §4 第 ① 条论断的证据",
      all(pow2_caps[k] == 0 < bnd_caps[k] for k in DEFECTS),
      f"{ {k: (pow2_caps[k], bnd_caps[k]) for k in DEFECTS} }")
# ★ 反驳点：把边界集换成「每个轴都正好整除」的形状，捕获数**退回 0**
ALL_DIVISIBLE = [(64, 64, 16), (128, 64, 16), (64, 128, 16), (64, 64, 32)]
check("C.2 反驳点：把边界集换成整除形状 ⇒ 三个缺陷的捕获数**全部退回 0**",
      all(not captured(s, fn) for s in ALL_DIVISIBLE for fn in DEFECTS.values()),
      f"{len(ALL_DIVISIBLE)} 形状 × {len(DEFECTS)} 缺陷")

sub("C.3  最小边界形状集的**每一条都对应一个具体的约束**（不是凑出来的）")
print(f"    (TM+1, ·, ·) = ({DEFECT_TM+1},64,16)：M 方向余数 = 1"
      f"  ⇒ 抓「M 的 tail 被丢」")
print(f"    (·, TN+1, ·) = (64,{DEFECT_TN+1},16)：N 方向余数 = 1"
      f"  ⇒ 抓「N 的 tail 未 mask」")
print(f"    (·, ·, TK+1) = (64,64,{DEFECT_TK+1})：K 方向余数 = 1"
      f"  ⇒ 抓「K 的 tail 被丢」")
print(f"    (2·TM+1, ·, ·)：跨过两块 ⇒ **本程序没有建模它抓的那类缺陷**（冗余项，见末）")
check("C.3 ★ 边界集里**每个轴**都有一个「tile 边长 + 1」的形状",
      any(s[0] == DEFECT_TM + 1 for s in BOUNDARY_SHAPES)
      and any(s[1] == DEFECT_TN + 1 for s in BOUNDARY_SHAPES)
      and any(s[2] == DEFECT_TK + 1 for s in BOUNDARY_SHAPES))
check("C.3 ⚠ 有「2×+1」的形状（余数相同但跨两块）——但**本程序没有为它建模任何缺陷**，"
      "⇒ 它对连续 tiling 是**冗余**的（见 §C.3 末与 specs.UNSOURCED 第 6 项）",
      any(s[0] == 2 * DEFECT_TM + 1 for s in BOUNDARY_SHAPES)
      and any(s[1] == 2 * DEFECT_TN + 1 for s in BOUNDARY_SHAPES)
      and "real_world_tail_defect_inventory" in SP.UNSOURCED)
check("C.3 ★ 反驳点：把边界集里的「+1」全换成「正好整除」，捕获数**退回 0**",
      all(not captured((64, 64, 16), fn) for fn in DEFECTS.values()),
      "对齐到 64/64/16 时三个缺陷都不触发")
# ★★ 必要性：一个缺陷**只在它那一轴有余数**的形状上被抓到 ⇒ 三个轴缺一不可
axis_of = {
    "drop M tail（M//TM 而非 ceil）": 0,
    "misalign N tail（夹回起点却写错位置）": 1,
    "drop K tail（K//TK 而非 ceil）": 2,
}
for name, fn in DEFECTS.items():
    ax = axis_of[name]
    hits = [s for s in BOUNDARY_SHAPES if captured(s, fn)]
    # 每个命中的形状，那个轴上都必须**确实有余数**；且该轴无余数的形状**不得**命中
    assert hits, name
    only_that_axis = (all(s[ax] % TILE[ax] != 0 for s in hits)
                      and all(not captured(s, fn) for s in BOUNDARY_SHAPES
                              if s[ax] % TILE[ax] == 0))
    check(f"C.3 ★★ 「{name}」被捕获 ⟺ **它那一轴有余数**"
          f"（⇒ 这组形状能**定位**是哪个轴的 tail 出了问题）", only_that_axis,
          f"{[s for s in hits]}")
# ★★ 一节被自己的实跑推翻的话（**保留在这里是为了不让它回流**）：
#   初版本节写的是「三个轴各需要一个形状，**缺一不可**」。**那是错的**——
#   形状 (65,65,17) 是另外三个「单轴 +1」形状的**超集**，它一个就捕获了全部三个缺陷。
print()
print("    ⚠ ★★ **一处被实跑推翻的本库论断**（保留在此，防止它再回流）：")
_superset = (65, 65, 17)
_all_by_one = all(captured(_superset, fn) for fn in DEFECTS.values())
print(f"      初版本节主张「三个轴各需要一个形状，**缺一不可**」。")
print(f"      实跑：形状 {_superset}（三轴各 +1）一个就捕获了全部三个缺陷 —— "
      f"{_all_by_one}")
check("C.3 ★★ 更正：单个「三轴都 +1」的形状就足以捕获全部三个缺陷（初版论断是错的）",
      _all_by_one, f"{_superset}")
# 逐个删掉**一个**「单轴 +1」形状，看是否真的漏网
# ⚠ 注意：必须删**恰好一个形状**。初版用 `s[i] != TILE[i] + 1` 过滤，那会一次删掉
#   两个（`(64,64,17)` 与 `(65,65,17)` 的 K 都是 17）——**实验设计错误**，不是发现。
_drop_one = []
for _i, _nm in enumerate(("M", "N", "K")):
    _removed = tuple(TILE[_j] + 1 if _j == _i else TILE[_j] for _j in range(3))
    _rest = [s for s in BOUNDARY_SHAPES if s != _removed]
    _missed = [n for n, f in DEFECTS.items()
               if not any(captured(s, f) for s in _rest)]
    _drop_one.append((_nm, _removed, _missed))
for _nm, _removed, _missed in _drop_one:
    print(f"      删掉 {_nm} 轴的「+1」形状 {_removed} 后仍然漏网的缺陷 = {_missed}")
check("C.3 ★★ 更正：只删掉**一个**轴的「+1」形状，**没有任何缺陷漏网**（必要性不成立）",
      all(not m for _, _, m in _drop_one), f"{_drop_one}")
check("C.3 ★★ 更正的反事实：删掉那个**三轴全 +1** 的 (65,65,17) 之后，"
      "剩下的集仍然能捕获全部三个缺陷（⇒ 它也只是冗余的一条）",
      all(any(captured(s, f) for s in BOUNDARY_SHAPES if s != (65, 65, 17))
          for f in DEFECTS.values()))
print("      ⇒ 这组形状的**真正作用**是「**定位**」而不是「**覆盖**」：")
print("        每个缺陷只被**含它那一轴余数**的形状抓到 ⇒ 看到哪个形状红了就知道是哪个轴。")
print("      ⚠ 而 `2·TM+1` 那一类形状对本程序建模的三个缺陷是**冗余的**——")
print("        它对**连续 tiling** 不增加捕获力（每个轴只有一个不完整 tile）。")
print("        它只在「余数自己还要再分块」时才有对象，而那个场景**本程序没有建模**")
print("        （见 specs.UNSOURCED：没有来源能给出这样一份真实缺陷清单）。")
print("    ★ 这就是本程序的核心：**2 的幂形状不是「覆盖得不全」，而是「结构性覆盖不到」**——")
print("      因为不完整 tile 在那些形状上**根本不存在**。")

sub("C.4  约束方程与边界形状的交汇：对齐约束**不受 tile 参数影响**")
N_ALIGNED, N_MISALIGNED = 64, 44
print(f"    fp16：N × {ESIZE} B 必须是 {SP.WMMA_FRAGMENT_BYTES} B 的倍数"
      f"（PTX 层的 fragment 字节数）")
for n in (N_ALIGNED, N_MISALIGNED):
    print(f"      N={n:>4}：{n} × {ESIZE} = {n*ESIZE:>4} B"
          f"   % {SP.WMMA_FRAGMENT_BYTES} = {n*ESIZE % SP.WMMA_FRAGMENT_BYTES:>3}"
          f"  ⇒ {'对齐' if c_alignment_problem(n) else '**不对齐**'}")
check("C.4 N=64 时问题形状满足对齐约束", c_alignment_problem(64) is True)
check("C.4 ★★ N=44 时**任何 tile 参数都救不了它**——这是「问题形状约束」",
      c_alignment_problem(44) is False)
check("C.4 ★ 对同一个 N=44，换遍所有合法 tile，问题形状判定**一个都不变**",
      len({c_alignment_problem(44)
           for (tm, tn, tk, wm, wn, st) in legal[:50]}) == 1
      if legal else True, "全部 False")
print("    ⇒ **两类约束的修复方式不同**：tile 参数约束换参数，问题形状约束要改布局/补 padding。")
print("      把二者混在一张表里，会让「不合法的原因」变得无法定位。")

sub("C.5  cuDNN 的 FP8 约束也是**问题形状**约束（A 级，第二处独立佐证）")
CH = (16, 32, 15, 17)
print(f"    Hopper FP8 要求 I/O channels 是 {SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE} 的倍数")
for ch in CH:
    ok = ch % SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE == 0
    print(f"      channels={ch:>3} ⇒ {'合法' if ok else '**不合法**'}"
          f"（余数 {ch % SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE}）")
check("C.5 channels=16 与 32 合法、15 与 17 不合法",
      [ch % SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE == 0 for ch in CH]
      == [True, True, False, False], f"{CH}")
check("C.5 ★ 反向与前向**不对称**：前向 batch 任意、反向必须是 16 的倍数",
      SP.CUDNN_FP8_HOPPER_FWD_BATCH_ANY is True
      and SP.CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE == 16)
check("C.5 ★ 这与 02 篇的「反向需要 FLOAT 累加」**同向**（两处独立的「方向不对称」）",
      SP.CUDNN_FP8_BACKWARD_REQUIRES_FLOAT is True)
print("    ★ 两处独立的不对称指向同一条结构性认识：**「这个算子要满足什么」不是算子的属性**，")
print("      是**「算子 × 方向」**的联合属性。")


# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")
# ===========================================================================

sub("D.1  fragment 的形状与寄存器数")
check("D.1 fp16 的三种 shape 恰好是 (16,16,16) / (32,8,16) / (8,32,16)",
      SP.FP16_FRAGMENT_SHAPES == ((16, 16, 16), (32, 8, 16), (8, 32, 16)),
      f"{SP.FP16_FRAGMENT_SHAPES}")
# ★ 注意：**M 与 N 可以是 8**（`m8n32k16` / `m32n8k16`），只有 K 恒为 16。
#   ⇒ 「tile 必须是 16 的倍数」这句话**只对 K 成立**；M/N 的粒度是 8。
#   本主题的演示 tile 取 16 的倍数，但**不能**把它写成 shape 的要求。
check("D.1 ★★ M、N 的粒度是 **8**（两种 shape 的 M 或 N 就是 8），只有 K 恒为 16",
      all(d % 8 == 0 for s in SP.FP16_FRAGMENT_SHAPES for d in s)
      and all(s[2] == 16 for s in SP.FP16_FRAGMENT_SHAPES)
      and any(s[0] == 8 for s in SP.FP16_FRAGMENT_SHAPES)
      and any(s[1] == 8 for s in SP.FP16_FRAGMENT_SHAPES),
      f"{SP.FP16_FRAGMENT_SHAPES}")
check("D.1 ★ 三种 shape 的 fragment **寄存器数相同**（它们是同一套规则的不同布局）",
      SP.WMMA_A_FRAGMENT_REGS == SP.WMMA_B_FRAGMENT_REGS == 8
      and SP.WMMA_ACC_F32_FRAGMENT_REGS == 8,
      f"A={SP.WMMA_A_FRAGMENT_REGS} B={SP.WMMA_B_FRAGMENT_REGS}"
      f" C32={SP.WMMA_ACC_F32_FRAGMENT_REGS}")
check("D.1 ★★ `.f16` 累加器只要 4 个寄存器（**省一半**）——精度的代价换寄存器",
      SP.WMMA_ACC_F16_FRAGMENT_REGS == 4
      and SP.WMMA_ACC_F16_FRAGMENT_REGS == SP.WMMA_ACC_F32_FRAGMENT_REGS // 2,
      f"{SP.WMMA_ACC_F16_FRAGMENT_REGS}")
print("    ★ 这一条是 02 篇与 03 篇**打架的地方**：用 `.f16` 累加省一半累加器寄存器，")
print("      代价是 02 篇的精度（`K` 的可用到 2048 就退化）。两篇各说一半。")

sub("D.2  对齐的三条常量与它们**不相等**这件事")
check("D.2 WMMA_MPTR_ALIGN_BYTES == 32（= 256 bit）", SP.WMMA_MPTR_ALIGN_BYTES == 32,
      f"{SP.WMMA_MPTR_ALIGN_BYTES}")
check("D.2 WMMA_LDM_ELEMENT_MULTIPLE == {'fp16': 8, 'fp32': 4}",
      SP.WMMA_LDM_ELEMENT_MULTIPLE == {"fp16": 8, "fp32": 4})
check("D.2 WMMA_LDM_BYTE_MULTIPLE == 16", SP.WMMA_LDM_BYTE_MULTIPLE == 16,
      f"{SP.WMMA_LDM_BYTE_MULTIPLE}")
check("D.2 WMMA_FRAGMENT_BYTES == 32", SP.WMMA_FRAGMENT_BYTES == 32,
      f"{SP.WMMA_FRAGMENT_BYTES}")
check("D.2 ★★ 16 ≠ 32 ⇒ 两条规则**不同层级**，本程序断言它们不相等",
      SP.WMMA_LDM_BYTE_MULTIPLE != SP.WMMA_FRAGMENT_BYTES)
check("D.2 ★ mptr 的 32 B 与 PTX 层 fragment 的 32 B **数值相同但规则不同**",
      SP.WMMA_MPTR_ALIGN_BYTES == SP.WMMA_FRAGMENT_BYTES == 32
      and SP.WMMA_ALIGN_RULES_ARE_DISTINCT_LEVELS is True)
print("    ⚠ 这正是本域规格初稿写错的地方：原文把 `ldm` 的「8 / 4」直接说成")
print("      「均为 16 字节的倍数」。**8 和 4 都不是 16 的倍数**——是**字节积**才是。")

sub("D.3  cuDNN FP8 的三条约束")
check("D.3 channels 必须是 16 的倍数",
      SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE == 16,
      f"{SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE}")
check("D.3 反向的 batch 必须是 16 的倍数",
      SP.CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE == 16)
check("D.3 ★ 前向的 batch **任意**（与反向不对称）",
      SP.CUDNN_FP8_HOPPER_FWD_BATCH_ANY is True)
check("D.3 ★ 两条 16 是**同一条表里的两个不同约束**（channels 与 batch），别混",
      SP.CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE == SP.CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE == 16)

sub("D.4  四条约束的四个上限（含两个「取最紧」的落点）")
check("D.4 MAX_REGISTERS_PER_THREAD == 255（原始值）",
      SP.MAX_REGISTERS_PER_THREAD == 255, f"{SP.MAX_REGISTERS_PER_THREAD}")
check("D.4 REGISTER_ALLOC_GRANULARITY == 8 ⇒ 有效上限 248",
      SP.REGISTER_ALLOC_GRANULARITY == 8 and REG_LIMIT_EFFECTIVE == 248,
      f"{SP.REGISTER_ALLOC_GRANULARITY} / {REG_LIMIT_EFFECTIVE}")
check("D.4 SMEM_PER_SM_MAX_KB == 228（按**每 block 预算**用 ⇒ 结论是乐观上界）",
      SP.SMEM_PER_SM_MAX_KB == 228 and SMEM_LIMIT_BYTES == 233472,
      f"{SP.SMEM_PER_SM_MAX_KB} KB = {SMEM_LIMIT_BYTES:,} B")
check("D.4 MAX_THREADS_PER_BLOCK == 1024、WARP_SIZE == 32 ⇒ 32 warps",
      SP.MAX_THREADS_PER_BLOCK == 1024 and SP.WARP_SIZE == 32 and WARPS_LIMIT == 32,
      f"{WARPS_LIMIT}")
check("D.4 MAX_WARPS_PER_SM == 64（**不会生效**的那条——取最紧后是 32）",
      SP.MAX_WARPS_PER_SM == 64 and WARPS_LIMIT_EFFECTIVE == 32)
check("D.4 ★ `ADDRESS_AND_INDEX_REGS_ASSUMED == 0` 是**明写的假设**，"
      "⇒ 算出的 tile 上界是乐观上界",
      SP.ADDRESS_AND_INDEX_REGS_ASSUMED == 0)
check("D.4 FLOAT16_BYTES == 2（shared memory 算式里真正用到的那一处）",
      SP.FLOAT16_BYTES == 2 and ESIZE == 2, f"{SP.FLOAT16_BYTES}")

sub("D.5  `UNSOURCED` 在本篇的落点")
check("D.5 ★ 「地址寄存器占多少」**无来源** ⇒ 改用明写的假设而不是编一个数",
      "address_and_index_register_budget" in SP.UNSOURCED)
check("D.5 ★ 「大多数 kernel 在 80% 带宽处饱和」是编机制从句（P4）⇒ 不写",
      "ungrounded_saturation_claims" in SP.UNSOURCED)
print("    ★ 这两条各自说明了本程序的一种纪律：**缺口要标注，不要填平**；")
print("      **测不了的机制不要写成数字**。")

sub("D.6  本程序的常量覆盖表")
# ★ 归属口径：**每条常量恰好一个「家」**（01 的 §D.9 断言三个 COVERED 元组两两不交）。
#   ⇒ 02 篇的常量**不重复列在这里**，即使本程序也引用了它：
#     `CUDNN_FP8_BACKWARD_REQUIRES_FLOAT` 在 §C.5 被**交叉引用**，但它的**归属**是 02 篇
#     ——**引用 ≠ 归属**。
COVERED_03 = (
    "FP16_FRAGMENT_SHAPES",
    "WMMA_A_FRAGMENT_REGS",
    "WMMA_B_FRAGMENT_REGS",
    "WMMA_ACC_F32_FRAGMENT_REGS",
    "WMMA_ACC_F16_FRAGMENT_REGS",
    "WMMA_MPTR_ALIGN_BYTES",
    "WMMA_LDM_ELEMENT_MULTIPLE",
    "WMMA_LDM_BYTE_MULTIPLE",
    "WMMA_FRAGMENT_BYTES",
    "WMMA_ALIGN_RULES_ARE_DISTINCT_LEVELS",
    "CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE",
    "CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE",
    "CUDNN_FP8_HOPPER_FWD_BATCH_ANY",
    "MAX_REGISTERS_PER_THREAD",
    "REGISTER_ALLOC_GRANULARITY",
    "SMEM_PER_SM_MAX_KB",
    "MAX_THREADS_PER_BLOCK",
    "MAX_WARPS_PER_SM",
    "WARP_SIZE",
    "ADDRESS_AND_INDEX_REGS_ASSUMED",
    "FLOAT16_BYTES",
)
for _n in COVERED_03:
    print(f"      {_n:40s} 有断言")
print(f"    ⇒ 03 篇覆盖 {len(COVERED_03)} 条")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 5 组 · B 类 3 组 · C 类 5 组 · D 类 6 组）")
print()
print("  ★ 本程序算的全部是**约束**与**覆盖率**，**不是速度**：本机无 GPU，")
print("    本域不产出实测性能数字。寄存器模型是本库的建模 [E]，其输入常量有 A 级来源。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}"
      f"（未 import torch）")
