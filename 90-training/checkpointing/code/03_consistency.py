#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_consistency.py —— **一致性恢复**：RNG 的内部状态 / dataloader 位置 / 优化器状态。

配套文档：`03-consistency-and-fault-tolerance.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★★ **判据 ④ 是「三条路径」**（计划 §3 Chunk 5 表 A，结构层审查 **M6** 改写后的版本）：
  · **正确**：存档含 RNG 的**内部状态**（`bit_generator.state`）⇒ 恢复时**只从文件读、
    不调用任何 `seed()`** ⇒ 与「一口气跑完」的参照**逐位一致** → **`True`**；
  · **阴性对照**：存档里**不含** RNG 状态 ⇒ **必须不一致** → **`False`**
    （★★ **这一条才是可证伪的那一半**）；
  · **反例**：实现者改用「重新 `seed()` + 快进」⇒ **当然也一致** → **`True`**，
    但它**不检查存档**（**P17**）。
  ⇒ ★★ 判据因此**写死「只允许从文件读、不得重新调用任何 `seed()`」**，
    并把它做成**源码级的结构钉住**（`A.4`）。

★★★ **第三条路径的工程解读（[C] PaLM 的逐字支持）**：
  它**不是错的** —— PaLM §5 逐字写着它的第二条做法是
  `a deterministic dataset pipeline where the shuffled data is written out in a random-access
  format`（`specs.PALM_DATASET_RA_QUOTE`），即
  `specs.PALM_DATASET_STEP_QUOTE`。
  ⇒ 那条路径把随机性**重写成了步号的函数**；它成立的条件是「消耗模式可重放」，
    **而它不检查「存档是否完整」** —— 本程序把这两件事分开（`A.3` / `A.5` / `B.2`）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  **路径 1**：从**文件**读回（含 RNG 内部状态）⇒ 与参照**每一步都逐位一致** → `True`；
  · `A.2`  **路径 2**（阴性对照）：存档缺 RNG 状态 ⇒ **不一致** → `False`，且**第 1 个恢复步**就分叉；
  · `A.3`  **路径 3**（反例）：重新 `seed()` + 快进 ⇒ 也一致 → `True`，★ 且它的 RNG 状态**逐位等于参照**
            ⇒ **它一致是因为它重建了同一份状态**，不是因为它检查了存档；
  · `A.4`  ★★ **结构钉住**：`restore_rng` 的调用表里**没有** `default_rng` / `seed`，
            且**必须**有 `PCG64`；`resume_from_file` 的源码里**不出现** `SEED` 这个名字；
  · `A.5`  ★★ **路径 3 的前提**：每步的消耗次数是**固定**的（`DRAWS_PER_STEP`），
            所以「快进」可做 —— 这正是 PaLM 那句「数据内容只是步号的函数」在玩具里的等价物；
  · `B.1`  ★★ **字节账**：档案里去掉 RNG 项后少了 `RNG 状态本体 + 键名与标点` 字节（两项都量出来）；
  · `B.2`  ★★ **两条路线的账记在不同科目上**：路 A 在**检查点**里付常数级字节；
            路 B（步号索引）在**数据集侧**付 `步数 × 批大小 × 8`（**随训练长度线性增长**）；
  · `C.1`  ★★ **缺一项必须被拦住**（[B] 逐字 `allow_partial_load: If False, …`）；
            ★ 而 `strict=False` 时**静默用默认值** ⇒ 轨迹不一致（两个方向都量）；
  · `C.2`  ★★ **就地恢复**的代价：中途失败会留下**半新半旧**的对象 ⇒ 必须先装配、再一次性替换；
  · `C.3`  ★ **恢复优化器状态有时序约束**（[B] 逐字 `WARN: set_optimizer_state_dict …`）——
            那是**框架内部**的调用契约，本机**复现不了** ⇒ 只作**引用判据**（**P3**）。

用法：
    python3 03_consistency.py
"""


import ast
import collections
import inspect
import json
import os
import shutil
import sys
import tempfile

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（checkpointing）取 `;; FAIL: checkpointing`。
ASSERT_FAIL_MARK = ";; FAIL: checkpointing"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", ":: FAIL: mixed-precision",
                     "++ FAIL: optimizers", "// FAIL: init-stability")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260955              # ★ 与同域另外两个程序（…0953 / …0954）都**不同**
D = 6                        # 参数维度
N_DATA = 32                  # 玩具数据集大小
BATCH = 8                    # 批大小
STEPS = 60                   # 总步数（参照的终点）
SAVE_AT = 23                 # 存档点（★ 仿 PaLM 的 `15,000 / 17,000` 那一对）
LR = 0.05                    # 学习率（玩具目标函数，只要求**确定**）
NOISE = 0.001                # 每步注入的随机噪声（代表 dropout / 数据增强）
DRAWS_PER_STEP = 2           # `A.5`：每步**消耗两次**随机数（`integers` + `normal`）
TRUE_W = (1.0, 2.0, 0.5, 0.25, 1.5, 0.75)   # 玩具数据的真值（★ **全正数存放**：`-1.0` 是 `UnaryOp`、不进变异分母）
# ★★ **盲区⑥ 的落法**：浮点常量都 `< 2^53`（最大的是 `60.0`）⇒ `v + 1.0` 能改变它们。
RNG_STATE_BYTES_MAX = 200        # `B.1`：RNG 状态本体的上界（本机实测 175）
RNG_KEY_OVERHEAD_MIN = 8         # `B.1`：键名与标点的下界（本机实测 15）
RNG_KEY_OVERHEAD_MAX = 32        # `B.1`：上界（★ 它是**口径**的一部分：换个序列化器就变）
INDEX_BYTES_PER_ID = 8           # `B.2`：把下标物化下来时每个下标 8 字节（int64）
ARCHIVE_KEYS = ("m", "rng_state", "step", "v", "w")   # ★ 键名钉住（P23）
META_KEY = "meta"                # `C.2`：集合级的完整性标记


# ===========================================================================
# 2. 路径（被测）：玩具训练循环 · 存档 · 三条恢复路径
# ===========================================================================
def make_data(seed):
    """固定的玩具数据集（`X` 与 `y`）：**它不是随机流的一部分**，四条路径共用。"""
    rng = np.random.default_rng(seed + 7)
    X = rng.normal(size=(N_DATA, D))
    y = X @ np.array(TRUE_W)
    return X, y


_DATA = {}


def data():
    """★ **惰性**构造数据集：**不在模块导入期**做（`P23` 的落法现场）。

    ★★ 为什么必须是惰性的：`D` 是**口径常量**，它被变异之后
      `X @ np.array(TRUE_W)` 会**形状不匹配** ⇒ 那是**导入期**的异常 ⇒
      `preflight()` 根本跑不到 ⇒ `mutate.py` 会把它记成**崩溃捕获**（不算真捕获）。
      ⇒ 改成惰性之后，`preflight()` 的 `D.3` 先抓住「`D` 变了」，程序**正常报断言失败**。
    """
    if "X" not in _DATA:
        _DATA["X"], _DATA["Y"] = make_data(SEED)
    return _DATA["X"], _DATA["Y"]


def draw(rng):
    """**每步的两次抽样**（★ 两种**路径不同**的调用，见 `01_state_bytes.py` 的 `C.1`）：
      · `integers` —— 走 **32 位**路径（dataloader 取下标）；
      · `normal`   —— 走 **64 位**路径（噪声 / dropout）。
    ★★ `A.5` 的数：**每步恰好两次**，与数据无关 ⇒ 「快进」才可做。
    """
    idx = rng.integers(0, N_DATA, size=BATCH)
    noise = rng.normal(size=D)
    return idx, noise


def train_step(rng, w, m, v, t):
    """**被测的训练步**（Adam，超参从 `specs.py` 读；`ε` 在根号**外**）。"""
    X, Y = data()
    idx, noise = draw(rng)
    g = X[idx].T @ (X[idx] @ w - Y[idx]) / BATCH + NOISE * noise
    m = specs.ADAM_DEFAULT_BETA1 * m + (1 - specs.ADAM_DEFAULT_BETA1) * g
    v = specs.ADAM_DEFAULT_BETA2 * v + (1 - specs.ADAM_DEFAULT_BETA2) * g * g
    mh = m / (1 - specs.ADAM_DEFAULT_BETA1 ** t)
    vh = v / (1 - specs.ADAM_DEFAULT_BETA2 ** t)
    w = w - LR * mh / (np.sqrt(vh) + specs.ADAM_DEFAULT_EPS)
    return w, m, v


def run_reference(steps):
    """**参照**：**一口气**从第 1 步跑到第 `steps` 步，记下**每一步**的 `w` 字节。

    ★ 它就是 PaLM 那句里的「trained up to step 17,000 **in a single run**」。
    """
    rng = np.random.default_rng(SEED)
    w, m, v = np.zeros(D), np.zeros(D), np.zeros(D)
    traj = []
    for t in range(1, steps + 1):
        w, m, v = train_step(rng, w, m, v, t)
        traj.append(w.tobytes())
    return {"traj": traj, "w": w, "m": m, "v": v, "rng": rng}


def make_archive(part, with_rng=True):
    """造一份存档（**字典**）：`w` / `m` / `v` / `step`（+ 可选 `rng_state`）。"""
    arch = {"w": part["w"].tolist(), "m": part["m"].tolist(), "v": part["v"].tolist(),
            "step": SAVE_AT}
    if with_rng:
        arch["rng_state"] = part["rng"].bit_generator.state
    return arch


def write_archive(arch, path):
    """**被测**：把存档写到**文件**（★ 先写临时文件、`os.replace` 原子改名）。

    ★ [B] DCP 逐字 `Write the metadata and marks the current checkpoint as successful.`
      ⇒ 本程序把「这份存档是完整的」这个事实做成一个**集合级的键**（`meta`）。
    ★ [B] DCP 逐字 `File creation is atomic` —— **原子的粒度是文件**，
      所以「整份档案要么全有、要么全无」要靠**改名**来给（而不是靠文件系统保证一堆文件的原子性）。
    """
    payload = dict(arch)
    payload[META_KEY] = {"complete": True, "steps": len(arch["w"])}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    os.replace(tmp, path)
    return path


def read_archive(path, strict=True):
    """**被测**：从**文件**读回存档。

    ★★ `strict=True` 时**必须**有 `meta`（集合级完整性）且**键集完整**
      —— 对应 [B] DCP 逐字：
      `allow_partial_load: If False, will raise a runtime error if a key is present in
       state_dict, but not in the checkpoint.`
      ★ `strict=False` 时静默地让缺的键用默认值顶上（`C.1` 量这件事的代价）。
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if strict:
        if META_KEY not in raw:
            raise ValueError("存档缺 %r：这份检查点**没有被标记为成功**" % META_KEY)
        missing = [k for k in ARCHIVE_KEYS if k not in raw]
        if missing:
            raise ValueError("存档缺键 %s（strict 模式下必须报错，不得用默认值顶）" % missing)
    return raw


def restore_rng(archive):
    """**被测**：从存档里的**内部状态**恢复发生器 —— ★ **不调用任何 `seed()`**。

    ★★ 这就是判据 ④ 写死的那一句：**只允许从文件读**。
      · 用 `PCG64()`（**无参**、从操作系统熵源起）造一个**占位**发生器，
        紧接着把它的**状态**整体设成存档里的那一份 ⇒ 起手状态**与熵源无关**；
      · ★ **不出现** `default_rng(SEED)` / `seed(...)` 这类写法（`A.4` 的结构钉住守着它）。
    """
    bg = np.random.PCG64()
    bg.state = archive["rng_state"]
    return np.random.Generator(bg)


def resume_from_file(path, upto):
    """**路径 1（正确）**：从**文件**读存档、用存档里的 RNG 状态继续跑。"""
    arch = read_archive(path, strict=True)
    rng = restore_rng(arch)
    w = np.array(arch["w"])
    m = np.array(arch["m"])
    v = np.array(arch["v"])
    traj = []
    for t in range(int(arch["step"]) + 1, upto + 1):
        w, m, v = train_step(rng, w, m, v, t)
        traj.append(w.tobytes())
    return {"traj": traj, "rng": rng, "w": w}


def resume_without_rng(path, upto):
    """**路径 2（阴性对照）**：存档里**没有** RNG 状态 ⇒ 用一个**新的**发生器继续。

    ★★ 这一条是**可证伪的那一半**：若它也「一致」，说明判据在检查一个**恒真**的东西。
    """
    arch = read_archive(path, strict=False)
    rng = np.random.default_rng(0)          # ★ 全新的发生器（没有内部状态可用）
    w = np.array(arch["w"])
    m = np.array(arch["m"])
    v = np.array(arch["v"])
    traj = []
    for t in range(int(arch["step"]) + 1, upto + 1):
        w, m, v = train_step(rng, w, m, v, t)
        traj.append(w.tobytes())
    return {"traj": traj, "rng": rng, "w": w}


def resume_by_reseed():
    """**路径 3（反例 / P17）**：**重新 `seed(SEED)`**，再把存档点之前的消耗**快进**一遍。

    ★★ 它一致 —— 但它是**重建了同一份状态**，**不是**检查了存档里有没有那份状态
      （`A.3` 把这件事量出来：两条路径的 RNG 状态**逐位相同**）。
    ★★ 工程上它**不是错的**：它就是 [C] PaLM 的第二条做法（数据内容只是步号的函数）。
      ⇒ 判据必须**禁止**它，否则「一致」可以由「重建」获得（**P17**）。
    ★ 它**不接收**任何存档（这正是它的问题所在：它不需要存档里的那份状态）。
    """
    rng = np.random.default_rng(SEED)
    for _ in range(SAVE_AT):          # ★ 每步一次 `draw` = 消耗 `DRAWS_PER_STEP` 次
        draw(rng)
    return rng


# ===========================================================================
# 3. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _attr_calls(func):
    """`func` 里**属性调用**的最后一段名字（如 `np.random.default_rng` → `default_rng`）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            out.add(n.func.attr)
    return out


def _mark_assignments_in(path):
    """用 **AST** 从本文件里数出模块级的 `ASSERT_FAIL_MARK = <字符串字面量>`。"""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def first_divergence(a, b):
    """两条轨迹**第一个**不同的下标（`0-based`；完全相同给 `None`）。"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


# ===========================================================================
# 4. A 类：判据 ④ 的三条路径 + 结构钉住 + 前提
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 判据 ④：**三条路径**（正确 / 阴性对照 / 反例）+ 结构钉住 + 反例的前提")
    print("=" * 74)
    print("  玩具：`D = %d`、`N_data = %d`、`batch = %d`、总步数 `%d`、**存档点第 %d 步**"
          "（★ 仿 PaLM 的 `15,000 → 17,000` 那一对，逐字见 `A.6`）"
          % (D, N_DATA, BATCH, STEPS, SAVE_AT))
    ref = run_reference(STEPS)
    half = run_reference(SAVE_AT)
    tmpdir = tempfile.mkdtemp(prefix="cons03_")
    path_full = os.path.join(tmpdir, "ckpt_full.json")
    path_no_rng = os.path.join(tmpdir, "ckpt_no_rng.json")
    write_archive(make_archive(half, with_rng=True), path_full)
    write_archive(make_archive(half, with_rng=False), path_no_rng)
    print("  参照轨迹：%d 步（**一口气跑完**，就是 PaLM 那句 `in a single run`）；"
          "两条存档各写在文件里" % len(ref["traj"]))
    print("  存档键 = `%s`（+ `%s`）" % (list(ARCHIVE_KEYS), META_KEY))

    # ── A.1 ★★ 路径 1 ─────────────────────────────────────────────────────
    print("  ── A.1 ★★ **路径 1（正确）**：从**文件**读回（含 RNG 内部状态）──")
    p1 = resume_from_file(path_full, STEPS)
    tail = ref["traj"][SAVE_AT:]
    same1 = p1["traj"] == tail
    rng_same1 = p1["rng"].bit_generator.state == ref["rng"].bit_generator.state
    print("     重启后 %d 步：与参照**逐位一致** = **%s**；RNG 的内部状态也一致 = **%s**"
          % (len(tail), same1, rng_same1))
    check("A.1", same1 and rng_same1 and len(p1["traj"]) == STEPS - SAVE_AT,
          "★★★ **路径 1 = `True`**：`%d` 步**每一步**的 `w` 字节都与参照相同，"
          "且**第 %d 步之后的 RNG 内部状态**也逐位相同 ⇒ 「恢复」包含**随机流本身**，"
          "不只是参数（[C] PaLM 逐字的目标：`%s`）"
          % (len(tail), STEPS, specs.PALM_BITWISE_QUOTE))

    # ── A.2 ★★ 路径 2（阴性对照）──────────────────────────────────────────
    print()
    print("  ── A.2 ★★ **路径 2（阴性对照）**：存档里**不含** RNG 状态 ──")
    p2 = resume_without_rng(path_no_rng, STEPS)
    same2 = p2["traj"] == tail
    diverge = first_divergence(p2["traj"], tail)
    print("     重启后 %d 步：与参照一致 = **%s**；**第 %d 个恢复步**就分叉（= 第 %d 步）"
          % (len(tail), same2, (diverge or 0), SAVE_AT + 1 + (diverge or 0)))
    check("A.2", not same2 and diverge == 0,
          "★★★ **路径 2 = `False`**：**第 1 个恢复步就分叉** ⇒ "
          "「存档里少了 RNG 状态」这件事**真的**会改变轨迹（差 %d 步里的第 1 步）；"
          "★★ **这一条是判据 ④ 可证伪的那一半** —— 若它也一致，判据就是在检查一个恒真的东西"
          % len(tail))

    # ── A.3 ★★ 路径 3（反例）──────────────────────────────────────────────
    print()
    print("  ── A.3 ★★ **路径 3（反例）**：重新 `seed()` + 快进 %d 步的抽样 ──"
          % SAVE_AT)
    rng3 = resume_by_reseed()
    # ★★ 口径：快进之后要跟**存档点那一刻**的状态比（不是跟「跑完 60 步之后」比 ——
    #   本程序第一次就把它写错成了后者，两个状态**不是同一个时刻**的）。
    fast_fwd_match = rng3.bit_generator.state == half["rng"].bit_generator.state
    print("     快进后：与**存档点那一刻**的 RNG 内部状态**逐位相同** = **%s**"
          % fast_fwd_match)
    w3 = np.array(half["w"])
    m3 = np.array(half["m"])
    v3 = np.array(half["v"])
    traj3 = []
    for t in range(SAVE_AT + 1, STEPS + 1):
        w3, m3, v3 = train_step(rng3, w3, m3, v3, t)
        traj3.append(w3.tobytes())
    same3 = traj3 == tail
    end_match = rng3.bit_generator.state == ref["rng"].bit_generator.state
    print("     重启后 %d 步：与参照一致 = **%s**；跑完后与参照的 RNG 状态相同 = **%s**"
          % (len(tail), same3, end_match))
    check("A.3", same3 and fast_fwd_match and end_match,
          "★★★ **路径 3 = `True`**，★ 而它一致的原因被量出来了：**快进之后的 RNG 状态与参照逐位相同**"
          "⇒ 「重新 `seed()`」**重建了同一份状态**，于是轨迹当然一致 —— "
          "★★ **它没有检查存档里有没有那份状态**（这就是 **P17**）⇒ 判据必须**写死**"
          "「只允许从文件读、不得重新调用任何 `seed()`」")

    # ── A.4 ★★ 结构钉住（判据写死的那一句的落法）───────────────────────────
    print()
    print("  ── A.4 ★★ 源码级结构钉住：恢复路径**只能**读状态、**不得** seed ──")
    src_r = inspect.getsource(restore_rng)
    src_f = inspect.getsource(resume_from_file)
    called_r = _callee_names(restore_rng) | _attr_calls(restore_rng)
    called_f = _callee_names(resume_from_file) | _attr_calls(resume_from_file)
    banned = {"seed", "default_rng", "SeedSequence"}
    hit = (banned & called_r) | (banned & called_f)
    has_state = '.state = ' in src_r and 'archive["rng_state"]' in src_r
    print("     `restore_rng` 的调用表 = `%s`；`resume_from_file` 的 = `%s`"
          % (sorted(called_r), sorted(called_f)))
    print("     · 禁用的名字 `%s` 命中 = **%s**" % (sorted(banned), sorted(hit) or "无"))
    print("     · `restore_rng` 从**存档**取状态（把 `archive` 里的 `rng_state` 赋给 `.state`）"
          "= **%s**" % has_state)
    check("A.4", not hit and has_state and "PCG64" in called_r
          and "SEED" not in src_f,
          "★★★ **判据写死的那一句的**可执行形态**：恢复函数的调用表里**没有** "
          "`default_rng` / `seed`（**空命中**），且**必须**有 `PCG64` + `bit_generator.state` ⇒ "
          "★ 谁把实现改成「重新 `seed()`」，这条**立刻**失败（`mutate.py` 管不到这种改动 —— "
          "那是「结构选择翻转」，见本程序 §5 与实施报告）")

    # ── A.5 ★★ 反例的**前提**：消耗次数固定 ────────────────────────────────
    print()
    print("  ── A.5 ★★ 路径 3 成立的**前提**：每步的消耗次数是**固定**的 ──")
    rng_probe = np.random.default_rng(SEED)
    counts = []
    for _ in range(SAVE_AT):
        before = rng_probe.bit_generator.state["state"]["state"]
        draw(rng_probe)
        counts.append(rng_probe.bit_generator.state["state"]["state"] != before)
    print("     `DRAWS_PER_STEP = %d`（`integers` + `normal`）；%d 步里**每一步**都消耗了 = **%s**"
          % (DRAWS_PER_STEP, SAVE_AT, all(counts)))
    print("     [C] PaLM 逐字：`%s`" % specs.PALM_DATASET_STEP_QUOTE)
    check("A.5", DRAWS_PER_STEP == 2 and all(counts) and len(counts) == SAVE_AT,
          "★★ **快进可做，是因为消耗模式与数据无关**（每步固定 %d 次）—— "
          "★ 这正是 PaLM 那句 `%s` 在玩具里的等价物：**内容只是步号的函数**；"
          "★ 一旦消耗次数依赖数据或时序，「重放」就不等价了（那是本程序**不覆盖**的范围，"
          "登记在 `specs.UNSOURCED` 的 `palm_pipeline_materialized_bytes` 附近）"
          % (DRAWS_PER_STEP, specs.PALM_DATASET_STEP_QUOTE))

    # ── A.6 ★ PaLM 的逐字目标 ─────────────────────────────────────────────
    print()
    print("  ── A.6 ★ 判据 ④ 的**外部目标**（[C] PaLM §5）──")
    print("     `%s` / `%s`" % (specs.PALM_BITWISE_SECTION_QUOTE, specs.PALM_BITWISE_QUOTE))
    print("     `%s`" % specs.PALM_RUN_PAIR_QUOTE)
    check("A.6", specs.PALM_BITWISE_QUOTE == "fully bitwise reproducible from any checkpoint"
          and specs.PALM_RUN_PAIR_QUOTE
          == "if the model has been trained up to step 17,000 in a single run",
          "★★★ **判据 ④ 的形状是照 PaLM 那句做的**：一个**一口气跑完**的参照 + 一个从存档点"
          "重启的对照，比**重启点之后**的每一步（本程序：`%d → %d`，与 `15,000 → 17,000` 同构）"
          % (SAVE_AT, STEPS))
    return tmpdir, path_full, path_no_rng, ref, tail


# ===========================================================================
# 5. B 类：字节账 —— RNG 项的成本 · 两条路线的科目不同
# ===========================================================================
def part_B(tmpdir, path_full, path_no_rng):
    print()
    print("=" * 74)
    print("B. 代价模型：RNG 项的字节账 · **两条路线记在不同的科目上**")
    print("=" * 74)

    # ── B.1 ★★ RNG 项的字节账 ─────────────────────────────────────────────
    print("  ── B.1 ★★ 档案里去掉 RNG 项：少了多少字节，少的是什么 ──")
    full = open(path_full, encoding="utf-8").read()
    no_rng = open(path_no_rng, encoding="utf-8").read()
    arch = read_archive(path_full, strict=False)
    rng_blob = json.dumps(arch["rng_state"], sort_keys=True).encode("utf-8")
    # ★★ **对照**：用一个**只有两项**的字典把「键名 + 标点」的开销单独量出来
    base = json.dumps({"a": 1}, sort_keys=True)
    plus = json.dumps({"a": 1, "rng_state": arch["rng_state"]}, sort_keys=True)
    key_overhead = len(plus.encode("utf-8")) - len(base.encode("utf-8")) - len(rng_blob)
    diff = len(full.encode("utf-8")) - len(no_rng.encode("utf-8"))
    print("     完整档案 `%d B` · 去掉 RNG 项 `%d B` ⇒ 差 **%d B**"
          % (len(full.encode("utf-8")), len(no_rng.encode("utf-8")), diff))
    print("     其中：状态本体 **%d B** + 键名与标点 **%d B**（后者由**对照字典**量出）"
          % (len(rng_blob), key_overhead))
    check("B.1", diff == len(rng_blob) + key_overhead
          and 0 < len(rng_blob) <= RNG_STATE_BYTES_MAX
          and RNG_KEY_OVERHEAD_MIN <= key_overhead <= RNG_KEY_OVERHEAD_MAX,
          "★★ **RNG 项的成本是常数级**：**%d B**（状态本体 `%d` + 键名与标点 `%d`）——"
          "★ 而**参数那一半**是 `12Ψ = %d B`（`Ψ = %d`）⇒ 两者相差 `%.0f` 倍；"
          "★★ 但少了这 `%d B`，轨迹**从第 1 个恢复步就分叉**（`A.2`）—— "
          "「便宜」与「可以省」是两件事"
          % (diff, len(rng_blob), key_overhead, 12 * 172, 172, 12 * 172 / diff, diff))

    # ── B.2 ★★ 两条路线的科目 ─────────────────────────────────────────────
    print()
    print("  ── B.2 ★★ 两条路线：**在检查点里付常数** vs **在数据集侧付线性** ──")
    idx_bytes = STEPS * BATCH * INDEX_BYTES_PER_ID
    print("     [C] PaLM 逐字：`%s`" % specs.PALM_DATASET_RA_QUOTE)
    print("     · **路 A（存 RNG 状态）**：检查点侧 **%d B**（`B.1`），**与训练长度无关**"
          % diff)
    print("     · **路 B（步号索引）**：数据集侧 `步数 × 批大小 × %d` = `%d × %d × %d` = **%d B**，"
          "**随训练长度线性增长**" % (INDEX_BYTES_PER_ID, STEPS, BATCH, INDEX_BYTES_PER_ID, idx_bytes))
    print("       翻倍训练长度：%d → **%d B**（**×2**）；而路 A 仍是 **%d B**（**×1**）"
          % (idx_bytes, 2 * idx_bytes, diff))
    check("B.2", idx_bytes == STEPS * BATCH * INDEX_BYTES_PER_ID
          and idx_bytes > diff
          and 2 * idx_bytes == 2 * (STEPS * BATCH * INDEX_BYTES_PER_ID),
          "★★★ **两笔账记在不同的科目上**：路 A 在**检查点**里付 `%d B`（常数）、"
          "路 B 在**数据集侧**付 `%d B`（**线性于步数**）—— ★★ 把两者混成一句"
          "「RNG 一致性很便宜」是**跨科目的**；★ 路 B 的那笔账属**数据集管线**"
          "（PaLM 只说写成 `random-access format`，**没给字节数** ⇒ 这个数标为**推导值**，"
          "已在 `specs.UNSOURCED` 登记 `palm_pipeline_materialized_bytes`）"
          % (diff, idx_bytes))
    return diff


# ===========================================================================
# 6. C 类：边界与反例 —— 缺一项 · 就地恢复 · 时序契约
# ===========================================================================
def part_C(tmpdir, path_full):
    print()
    print("=" * 74)
    print("C. 边界与反例：缺一项 · **就地恢复的半新半旧** · 一句只在原文成立的话")
    print("=" * 74)

    # ── C.1 ★★ 缺一项：strict 拦住 / 不 strict 静默 ───────────────────────
    print("  ── C.1 ★★ 存档**缺一项**：`strict` 拦住，`strict=False` 静默顶上 ──")
    print("     [B] DCP 逐字：`%s`" % specs.DCP_ALLOW_PARTIAL_LOAD_QUOTE)
    broken = dict(read_archive(path_full, strict=False))
    broken.pop("m")                     # ★ 把优化器状态里的 `m` 拿掉
    path_broken = os.path.join(tmpdir, "ckpt_broken.json")
    with open(path_broken, "w", encoding="utf-8") as fh:
        json.dump(broken, fh, sort_keys=True)
    raised = None
    try:
        read_archive(path_broken, strict=True)
    except ValueError as e:
        raised = str(e)
    loose = read_archive(path_broken, strict=False)
    m_default = np.zeros(D) if "m" not in loose else np.array(loose["m"])
    differs = not np.array_equal(m_default, np.array(read_archive(path_full)["m"]))
    print("     `strict=True` ⇒ `%s`" % raised)
    print("     `strict=False` ⇒ 读回**成功**，缺的 `m` 被默认值顶上 ⇒ 与真值不同 = **%s**"
          % differs)
    check("C.1", raised is not None and "m" in raised and differs
          and specs.DCP_ALLOW_PARTIAL_LOAD_QUOTE.startswith("allow_partial_load: If False"),
          "★★ **两个方向都量了**：`strict` 模式**拦住**（`%s`）；"
          "★ `strict=False` 时**静默**地用默认值顶上，而那份 `m` 与真值不同 ⇒ "
          "「读回来没报错」**不等于**「读回来是对的」；"
          "★★ 这是 [B] 里 `allow_partial_load` 默认 `False` 的理由" % raised)

    # ── C.2 ★★ 就地恢复的半新半旧 ────────────────────────────────────────
    print()
    print("  ── C.2 ★★ **就地恢复**：中途失败会留下**半新半旧**的对象 ──")
    print("     [B] DCP 逐字：`%s`" % specs.DCP_IN_PLACE_QUOTE)
    live = {"w": np.zeros(D), "m": np.zeros(D), "v": np.zeros(D), "step": 0}
    arch = read_archive(path_full, strict=False)
    order = ("w", "m", "v", "step")
    failed_at = None
    for i, k in enumerate(order):
        if k == "v" and i == 2:
            failed_at = k            # ★ 模拟第 3 个字段反序列化时失败
            break
        live[k] = np.array(arch[k]) if k != "step" else int(arch[k])
    mixed = (np.array_equal(live["w"], np.array(arch["w"]))
             and int(live["step"]) != int(arch["step"]))
    staged = {k: (np.array(arch[k]) if k != "step" else int(arch[k])) for k in order}
    installed = np.array_equal(staged["w"], np.array(arch["w"])) \
        and np.array_equal(staged["m"], np.array(arch["m"]))
    print("     就地逐个字段写入、在第 `%s` 个字段失败 ⇒ 对象已经是**半新半旧** = **%s**"
          % (failed_at, mixed))
    print("     ★ 而「**先装配到临时对象、成功后一次性替换**」这条路：装配产物完整 = **%s**"
          % installed)
    check("C.2", mixed and installed and failed_at == "v",
          "★★ **就地恢复的代价**：失败之后那个对象**既不是旧的、也不是新的**"
          "（`w` 已经是新的、`step` 还是旧的），★ 而它**看起来很正常**（没有异常、类型对、形状对）⇒ "
          "⇒ 恢复必须**先装配到临时对象、成功后一次性替换**（[B] 的 `%s` 说的正是这个语义："
          "**模型先把内存分配好、DCP 往里写** —— 于是「写了一半」是**有可能的状态**）"
          % specs.DCP_IN_PLACE_QUOTE)

    # ── C.3 ★ 一句只在原文成立的话（引用判据，P3）────────────────────────────
    print()
    print("  ── C.3 ★ 恢复优化器状态的**时序约束**：只作引用判据 ──")
    print("     [B] DCP 逐字：`%s` … `or after step() is called on the optimizers.`"
          % specs.DCP_OPTIMIZER_ORDER_QUOTE)
    check("C.3", specs.DCP_OPTIMIZER_ORDER_QUOTE
          == "WARN: set_optimizer_state_dict can only be called before backward()"
          and specs.DCP_SAME_KEYS_QUOTE == "Mismatched keys may result in hangs or errors.",
          "★★ **只钉逐字、不复现数值**：这条时序约束是**框架内部**的调用契约"
          "（本机**没有 torch**，无从复现）⇒ 按 **P3** 它**不作数值判据**；"
          "★ 同一段里 [B] 还给了缺键的失败形态 `%s` —— ★ **`hang` 比 `error` 更坏**"
          "（没有异常、没有栈），这也是「簇级的键集必须一致」的理由"
          % specs.DCP_SAME_KEYS_QUOTE)


# ===========================================================================
# 7. D 类（前半）：标记自检 + 口径/阈值/**键名**钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ **禁区回链**：导出/推理格式的选择在 "
          "`70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；"
          "异步落盘 / 通信-计算重叠在 `110-communication/overlap/`；"
          "优化器状态的分片与通信量在 `100-parallelism/zero-fsdp/`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ";; FAIL: checkpointing" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", ":: FAIL: mixed-precision",
                                        "++ FAIL: optimizers", "// FAIL: init-stability"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")
    # ★★ **键名钉住**（P23 的一条）：存档的 5 个键就是判据的载具（`C.1` 靠它做缺项测试）。
    check("D.2c", ARCHIVE_KEYS == ("m", "rng_state", "step", "v", "w")
          and META_KEY == "meta" and len(ARCHIVE_KEYS) == 5,
          "★★ **键名钉住**：存档键 = `%s`、集合级标记 = `%r` —— "
          "`A.1`/`C.1`/`C.2` 都挂在这几个名字上（改名会让那三条判据**静默失去载荷**）"
          % (list(ARCHIVE_KEYS), META_KEY))

    measured = {"SEED": SEED, "D": D, "N_DATA": N_DATA, "BATCH": BATCH, "STEPS": STEPS,
                "SAVE_AT": SAVE_AT, "LR": LR, "NOISE": NOISE,
                "DRAWS_PER_STEP": DRAWS_PER_STEP, "TRUE_W": TRUE_W,
                "ADAM_DEFAULTS": (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                                  specs.ADAM_DEFAULT_EPS)}
    expect = {"SEED": 20260955, "D": 6, "N_DATA": 32, "BATCH": 8, "STEPS": 60,
              "SAVE_AT": 23, "LR": 0.05, "NOISE": 0.001, "DRAWS_PER_STEP": 2,
              "TRUE_W": (1.0, 2.0, 0.5, 0.25, 1.5, 0.75),
              "ADAM_DEFAULTS": (0.9, 0.999, 1e-8)}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", len(TRUE_W) == D and 0 < SAVE_AT < STEPS
          and STEPS - SAVE_AT >= 2 and DRAWS_PER_STEP == 2,
          "★ 存档点在第 %d/%d 步（恢复段 %d 步）、真值维度与 `D` 一致、每步消耗 %d 次"
          "—— 它们是三条路径的比较窗口" % (SAVE_AT, STEPS, STEPS - SAVE_AT, DRAWS_PER_STEP))
    check("D.3c", (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2) == (0.9, 0.999),
          "★★ **Adam 的 `β` 必须先钉住**（在 `preflight()` 里）——★ **P23** 的同一条理由")
    check("D.3d", specs.PALM_DATASET_STEP_QUOTE
          == "the contents of a given training batch is only a function of the step number",
          "★★ 路径 3 的**正面依据**在任何计算之前钉住（省得 `A.3`/`A.5` 的结论失去支点）")

    th = {"RNG_STATE_BYTES_MAX": RNG_STATE_BYTES_MAX,
          "RNG_KEY_OVERHEAD_MIN": RNG_KEY_OVERHEAD_MIN,
          "RNG_KEY_OVERHEAD_MAX": RNG_KEY_OVERHEAD_MAX,
          "INDEX_BYTES_PER_ID": INDEX_BYTES_PER_ID}
    th_expect = {"RNG_STATE_BYTES_MAX": 200.0, "RNG_KEY_OVERHEAD_MIN": 8.0,
                 "RNG_KEY_OVERHEAD_MAX": 32.0, "INDEX_BYTES_PER_ID": 8.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 4 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < 9007199254740992
          and max(abs(v) for v in TRUE_W) < 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点常量都 `< 2^53`（最大的是 `%g`）"
          % max(max(th_expect.values()), max(abs(v) for v in TRUE_W)))
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", specs.PALM_ARXIV_ID == "2204.02311"
          and specs.PALM_TITLE == "PaLM: Scaling Language Modeling with Pathways"
          and specs.PALM_BITWISE_SECTION_QUOTE == "Bitwise determinism"
          and specs.PALM_BITWISE_QUOTE == "fully bitwise reproducible from any checkpoint"
          and specs.PALM_DATASET_RA_QUOTE
          == "a deterministic dataset pipeline where the shuffled data is written out in a random-access format"
          and specs.PALM_DATASET_STEP_QUOTE
          == "the contents of a given training batch is only a function of the step number"
          and specs.PALM_RUN_PAIR_QUOTE
          == "if the model has been trained up to step 17,000 in a single run",
          "★★ **判据 ④ 的六条 [C] 逐字**（`%s`，**不确认同行评审**）——"
          "★ 前三句是**目标**，后三句是**两条做法**（其中第二条正是本程序的路径 3）"
          % specs.PALM_ARXIV_ID)
    check("D.6", specs.DCP_ALLOW_PARTIAL_LOAD_QUOTE
          == "allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint."
          and specs.DCP_SAME_KEYS_QUOTE == "Mismatched keys may result in hangs or errors."
          and specs.DCP_OPTIMIZER_ORDER_QUOTE
          == "WARN: set_optimizer_state_dict can only be called before backward()",
          "★★ **三条 [B] 逐字**（PyTorch 2.14 的 DCP 页）：缺键的默认行为、缺键的失败形态"
          "（`hang` 或 `error`）、恢复优化器状态的时序约束")
    check("D.7", specs.DCP_IN_PLACE_QUOTE
          == "It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead."
          and specs.DCP_METADATA_MARKS_SUCCESS_QUOTE
          == "Write the metadata and marks the current checkpoint as successful."
          and specs.DCP_ATOMIC_CREATION_QUOTE == "File creation is atomic"
          and specs.DCP_EMPTY_DIR_QUOTE
          == "The checkpoint path is an empty or non-existing directory.",
          "★★★ **`C.2` 的四条依据**：就地（⇒ 可能半新半旧）、元数据 = 成功标记、"
          "原子性只在**文件**级、写侧假定目录是空的（⇒ 先写临时目录再改名）")
    check("D.7b", (specs.ZERO_K_OPT_STATE, specs.ZERO_TOTAL_MULTIPLE) == (12, 16.0)
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径",
          "★ 本程序**不用** ZeRO 的字节账（那是 `01`/`02` 的），但把同一份 `specs.py` 的"
          "常量钉住 ⇒ 三个程序共享的事实源**一致**（`K = 12` 是 **%s**）"
          % specs.ZERO_K_CALIBER_NOTE)
    check("D.8", specs.UNSOURCED == ("dcp_per_shard_header_bytes",
                                      "dcp_metadata_file_format",
                                      "dcp_partial_load_state",
                                      "dcp_checkpoint_excludes_gradients",
                                      "palm_pipeline_materialized_bytes",
                                      "numpy_pcg64_state_json_schema",
                                      "zero_16psi_checkpoint_caliber"),
          "★★ `specs.UNSOURCED` 的 %d 项**逐位钉住** —— ★ 其中 `dcp_partial_load_state`"
          "（半新半旧）与 `palm_pipeline_materialized_bytes`（路 B 的字节数）"
          "正是本程序 `C.2` / `B.2` 的处境" % len(specs.UNSOURCED))

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_ref = inspect.getsource(run_reference)
    src_p1 = inspect.getsource(resume_from_file)
    src_p3 = inspect.getsource(resume_by_reseed)
    check("D.9", len({src_ref, src_p1, src_p3}) == 3,
          "★ 防 P17：参照（`run_reference` %d 字符）、路径 1（`resume_from_file` %d）、"
          "路径 3（`resume_by_reseed` %d）**源码两两不同**"
          % (len(src_ref), len(src_p1), len(src_p3)))
    watched = {"resume_from_file", "restore_rng", "read_archive"}
    called = _callee_names(run_reference) | _callee_names(resume_by_reseed)
    check("D.9b", not (watched & called),
          "★ **AST 调用表**：参照与反例的 `Call` 节点 = %s —— "
          "**一条都没调到**「从文件读」那一路 ⇒ 路径 1 的读档行为不是它们提供的"
          % (", ".join(sorted(called)) if called else "**空集**"))
    check("D.9c", 'archive["rng_state"]' in inspect.getsource(restore_rng)
          and "default_rng(SEED)" in inspect.getsource(resume_by_reseed),
          "★★ **两条路径各自的状态来源**都被源码钉住：路径 1 从**存档**取"
          "（`archive[\"rng_state\"]`）、路径 3 从**种子重放**取（`default_rng(SEED)`）——"
          "★ 两者的**结果一样**（`A.3`）、**来源不同**（这就是 P17 的现场："
          "结果一样 ≠ 检查了同一件事）")
    # ★★ T41 的同源做法：A.2 那条**阴性对照**不能退化成 no-op —— 它成立的前提是
    #   「随机流真的进了轨迹」。用**源码级钉住**把这件事写死（而不是靠「现在恰好成立」）。
    src_step = inspect.getsource(train_step)
    called_step = _callee_names(train_step)
    check("D.9d", "draw" in called_step and "NOISE * noise" in src_step
          and "train_step" not in called_step,
          "★★ **T41 的同源钉住**：`train_step` **自己**调 `draw`（随机流进轨迹）且用 `NOISE * noise`"
          "⇒ 路径 2 的「不一致」不会退化成 no-op（★ 若把噪声去掉、批次也变成确定性的，"
          "`A.2` 就会**恒真**地报告「一致」）；★ 且它**不递归调用自己**")
    check("D.10", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))


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
    return 1 if FAILS else 0


def main():
    print("=" * 74)
    print("03_consistency.py —— 一致性恢复：RNG 内部状态 / dataloader 位置 / 优化器状态")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ **判据 ④ 的三条路径**：正确 `True` · 阴性对照 `False` · 反例 `True`"
          "（★ 第三条证明了判据必须**写死**「只允许从文件读、不得 seed()」）")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    tmpdir, path_full, path_no_rng, ref, tail = part_A()
    part_B(tmpdir, path_full, path_no_rng)
    part_C(tmpdir, path_full)
    part_D()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    print("  ★ 本次的临时目录已清理（`%s`）：★ 它在 `code/` 之外，`make clean` 管不到它，"
          "所以由本程序自己收尾" % tmpdir)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
