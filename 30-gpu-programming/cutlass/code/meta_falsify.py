#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""meta_falsify.py —— **元检验**（`make falsify`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 它回答什么问题
===============================================================================

本主题的 A 类断言全部是「**本仓库自己写的一份实现** vs **外部工件**」：

  · `01_layout_algebra.py` 的 `RefLayout` 系列 vs `pycute`；
  · `02_tiling_decompose.py` 的 `our_cta_products` / `our_decode_global_level` vs `cutlass_library`；
  · `03_swizzle_probe.py` 的 `our_swizzle` vs `pycute.Swizzle`。

于是有一个致命质疑（**与 P12 同构，只是「被测对象」换成了「本仓库的参照实现」**）：

  > **如果那份「参照实现」本身写错了呢？对拍还会成功吗？**

  更糟的一种：如果对拍的两个实现是**同一段代码**（或都被同一个错误影响），
  那么对拍**永远成功**，而它什么也没证明。

本文件用**最直接的方式**回答它：**把本仓库那份实现改坏，看断言会不会叫。**
做法是给某个编号程序的源码打一个**单点补丁** → 复制到临时目录 → 跑 →
**断言退出码非 0**。若某个缺陷下程序仍然退出 0，则那条对拍是**空操作**，
本文件会把它标成 `**没失败，这条 A 类断言无效**` 并计一次失败。

★★ 本文件还带**两条方向相反**的缺陷（`D8` / `D9`）：

  · `D8` 把本仓库的 `coalesce` 的合并条件改成**照抄一句看起来更严格的条件**；
  · `D9` 把 `our_swizzle` 的掩码位置改成**另一个候选语义**（shift 的方向反过来）。
  它们不是「改错」，是**改成另一个看起来同样合理的写法** —— 必须被抓住。
  如果换成另一个候选语义而程序照样通过，说明「本实现选的是哪一个」**没有被任何断言钉住**。

===============================================================================
★ 五条硬纪律
===============================================================================

1. **不修改**仓库里的任何原文件 —— 每个变异体都在自己的 `tempfile` 目录里跑。
2. **基线必须全过**：不打补丁时三个编号程序必须都退出 0，否则读数毫无意义。
3. **锚点必须恰好出现 1 次**：锚点漂了（`src.count(old) != 1`）补丁打不上，
   那会让元检验**静默变成空操作** —— 这里直接把它记成失败。
4. **临时目录里要放齐全部支撑模块**（含 `meta_falsify.py` 自己）——
   编号程序会用 `ast` 读同目录的其它文件（`01` 的 `A.11` 要读三个程序的标签），
   副本里少一个，基线就会 rc≠0，整个元检验的读数作废（`triton` 主题的第一版就这么失败的）。
5. **探测到的外部工件路径必须写进子进程的 env**（`child_env()`，见下一节）——
   本文件在**临时副本**里跑编号程序，副本里那些**相对**候选（`../cutlass` 等）
   一个都解析不到；而 `Makefile` 的 `CUTLASS_PATH ?=` 默认为空，
   「**空值原样透传等于没传**」。
   ★ 探测失败时**报出是哪一步失败**并以 rc=2 退出（不是让三个程序各自炸一遍、
   再汇总成一句含糊的话）。`env_selftest()` 把这一条做成**可证伪**的。

===============================================================================
★ 两道「harness 自己」的自证（都不是「跑一遍看看」，而是**可证伪的判据**）
===============================================================================

· **步骤 2/5 `env_selftest()`** —— 「不设 `CUTLASS_PATH` 且不在原位」**可被程序检出**；
· **步骤 3/5 `candidate_selftest()`** —— 候选列表（`specs.CUTLASS_CANDIDATES` /
  `specs.CUTLASS_REQUIRED`）的**正确性有断言钉住，且钉得住**：
  ① 候选里不得含**绝对路径**、② 必需文件必须全为**相对**路径（否则 `os.path.join`
  会把根丢掉 ⇒ `is_cutlass_root()` 与传进来的那棵树无关）、
  ③ 候选里必须至少有一条**相对本目录**的（不是 `~` 家目录项）、
  ④ 拿**三张合成的坏表**证明 ①②③ **各自有区分力**、
  ⑤ 把 `os.getcwd` 换成抛异常的桩后 `find_cutlass()` **不得**读 cwd、
  ⑥ 用**合成假树 + 一条相对候选**证明「三个 cwd 下选中同一棵」（**不依赖本机有没有
  CUTLASS**，所以它在任何机器上都有区分力）。

  为什么要有 3/5：`CUTLASS_PATH` 是候选列表的**第一顺位**，它一命中，候选列表
  **根本不会被问到** ⇒ `make mutate` 的叶子级变异里 `CUTLASS_CANDIDATES[*]`
  **六个叶子全在漏网清单**（把字符串改成任何别的东西，`make run` 照样全过）。
  修掉的是「**改坏了没有任何东西会叫**」，用的是**断言**而不是变异体。
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# ★ 与三个编号程序、与 `Makefile` **共用同一份候选列表**：直接 import 同目录的
#   `specs.py`（它 import 时**不做任何文件系统探测**，见 `specs.py` 模块头）。
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import specs                                    # noqa: E402

PROGRAMS = ("01_layout_algebra.py", "02_tiling_decompose.py", "03_swizzle_probe.py")
AUX = ("specs.py", "meta_falsify.py", "mutate.py")

# ★★ 失败标记：本主题三个程序统一打印 `  [FAIL] <标签> …`。
#   ★ 这条正则**不是**照抄任何别的主题的：它是**本目录自己的输出形态**。
#     抄 `cuda` 主题的 `\[[ABC]/FAIL|FAILED: ` 会**一条都匹配不到**，
#     于是「断言捕获」会被误判成「崩溃捕获」（Chunk 2 踩过：123 个捕获全被误分类）。
ASSERT_FAIL_MARK = "[FAIL]"

# (缺陷名, 被改的文件, 原文, 替换成什么, 期望失败的程序)
# ★ 每一条都对应一个**正文里的承重结论** —— 改坏它，那条结论的载体必须叫。
DEFECTS = [
    ("D1 `RefLayout.coalesce` 的合并条件被删（无条件合并相邻模式）",
     "01_layout_algebra.py",
     "        elif rs[-1] * rt[-1] == stride:      # ★ 合并条件",
     "        elif True:                          # DEFECT: 合并条件被删",
     ["01_layout_algebra.py"]),

    ("D2 `RefLayout.complement` 的 `ceil_div` 被换成直接取 max_idx",
     "01_layout_algebra.py",
     "    rs.append(-(-max_idx // cur))            # ceil_div(max_idx, cur)",
     "    rs.append(max_idx)                       # DEFECT: 不做 ceil_div",
     ["01_layout_algebra.py"]),

    ("D3 `RefLayout.logical_divide` 里丢掉 `complement` 那一项",
     "01_layout_algebra.py",
     "    return n_composition(A, n_make_layout(B, n_complement(B, n_size(A))))",
     "    return n_composition(A, n_make_layout(B))  # DEFECT: 丢掉 complement",
     ["01_layout_algebra.py"]),

    ("D4 `RefLayout.right_inverse` 不做前缀满射检查（`break` 被删）",
     "01_layout_algebra.py",
     "        if cur != stride:                    # ★ 非前缀满射 ⇒ 停止累积",
     "        if False:                            # DEFECT: 不检查前缀满射",
     ["01_layout_algebra.py"]),

    ("D5 `RefLayout._eval` 用坐标**和**而不是「模式 + 步长」的加权和",
     "01_layout_algebra.py",
     "        if not n_is_tuple(shape):\n            return crd * stride",
     "        if not n_is_tuple(shape):\n            return crd + stride  # DEFECT: 加法",
     ["01_layout_algebra.py"]),

    ("D6 `our_decode_global_level` 的移位基数改成 `//5`",
     "02_tiling_decompose.py",
     "        digits.append(v % 10)\n        v //= 10",
     "        digits.append(v % 10)\n        v //= 5   # DEFECT: 基数错",
     ["02_tiling_decompose.py"]),

    ("D7 `our_cta_products` 的 CTA 用**加法**而不是逐元素乘法",
     "02_tiling_decompose.py",
     "                tbs = tuple(mi.instruction_shape[i] * m[i] for i in range(3))",
     "                tbs = tuple(mi.instruction_shape[i] + m[i] for i in range(3))  # DEFECT",
     ["02_tiling_decompose.py"]),

    # ★★ 方向相反的两条：不是「改错」，是**改成另一个候选语义**。
    ("D8（方向相反）`our_swizzle` 的 YYY/ZZZ 位置互换 —— 另一个合法语义",
     "03_swizzle_probe.py",
     "    y = (offset >> yyy_pos(bits, base, shift)) & mask\n"
     "    return offset ^ (y << zzz_pos(bits, base, shift))",
     "    y = (offset >> zzz_pos(bits, base, shift)) & mask   # DEFECT: 位置互换\n"
     "    return offset ^ (y << yyy_pos(bits, base, shift))",
     ["03_swizzle_probe.py"]),

    ("D9（方向相反）`our_identity_bound` 改成用 ZZZ 位置 —— 另一个候选答案",
     "03_swizzle_probe.py",
     "    return 1 << yyy_pos(bits, base, shift)",
     "    return 1 << zzz_pos(bits, base, shift)   # DEFECT: 换成另一个候选答案",
     ["03_swizzle_probe.py"]),
]


# ===========================================================================
# ★★ 外部工件路径：**探测到的绝对路径**必须写进子进程的 env
# ===========================================================================
#
# 反面教材（本文件的第一版就是这么写的，控制者实测到了后果）：
#
#     env = dict(os.environ)          # ★ `Makefile` 的 `CUTLASS_PATH ?=` 默认**留空**
#
# `Makefile` 里 `export CUTLASS_PATH` 导出的是**空字符串** ⇒ `dict(os.environ)` 里
# `CUTLASS_PATH=""` —— **空值原样透传等于没传**。而本文件在**临时副本**里跑编号程序，
# 副本里那些**相对**候选（`../cutlass` 等）一个都解析不到 ⇒ 三个程序各自以
# 「找不到外部工件」rc=2 退出 ⇒ 基线被判「有程序失败」⇒ `make falsify` **rc=2**。
# 那不是「统计造假」（基线不通过时本文件拒绝给结论），是「**不能开箱即用**」。
#
# ⇒ 正确做法：**先探测**，把探测到的**绝对路径**写进子进程 env（`child_env()`）。
class ProbeError(Exception):
    """外部工件探测失败。**不静默回落** —— 由 `main()` 报出是哪一步失败后以 rc=2 退出。"""


def probe_cutlass():
    """按 `specs.CUTLASS_CANDIDATES` 探测 CUTLASS 源码树，返回绝对路径；找不到返回 None。

    ★ 候选列表里的**相对**项以**本目录**为基准（不是调用者的 cwd）——
      这一条**现在由 `specs` 自己保证**（`CUTLASS_CANDIDATE_BASE` = `specs.py` 所在目录，
      `find_cutlass()` 不碰 `os.getcwd()`；步骤 3/5 的 ⑤ / ⑥ 就是它的判据）。
      下面的 `chdir` 是**冗余**的（过去它是唯一的保障），保留只为少改一处。
    ★ `verbose=True` 让 `specs` 自己把「每个候选解析到哪儿」列出来（不重复实现解析逻辑）。
    """
    old = os.getcwd()
    try:
        os.chdir(HERE)
        return specs.find_cutlass(verbose=True)
    finally:
        os.chdir(old)


def child_env():
    """返回**写入了探测到的绝对路径**的子进程 env（探测失败即抛 `ProbeError`）。"""
    root = probe_cutlass()
    if root is None:
        raise ProbeError()
    env = dict(os.environ)
    env[specs.CUTLASS_ENV] = root           # ★★ 覆盖空值 / 相对值：空值透传等于没传
    return env


# ★ 纯探测的一行程序：**把候选列表换成一个不存在的路径** ⇒ 唯一的变量只剩
#   `CUTLASS_PATH`。这一行测的是「**透传**这件事本身」，不是「本机碰巧找得到」。
_PROBE_CHILD = ("import sys; sys.path.insert(0, %r); import specs; "
                "specs.CUTLASS_CANDIDATES = ('/__no_such_cutlass__/',); "
                "r = specs.find_cutlass(); print(r or ''); "
                "sys.exit(0 if r else 3)")


def _probe_child(cwd, env):
    """在 `cwd` 里、用 `env` 跑一次纯探测，返回 `(rc, 选中路径或 '')`。"""
    r = subprocess.run([sys.executable, "-c", _PROBE_CHILD % HERE], cwd=cwd,
                       capture_output=True, text=True, env=env, timeout=60)
    return r.returncode, (r.stdout or "").strip()


def env_selftest(env, tmpdir):
    """★★ 自证：「**不设 `CUTLASS_PATH`、且不在原位**」这个场景**是可被程序检出**的。

    在**临时目录**里（脱离原位）跑两次纯探测：

      · **不给** `CUTLASS_PATH` ⇒ 探测**必须失败**（rc=3）—— 这就是修复前的形态：
        `dict(os.environ)` 里那个空值把三个编号程序一起打成了 rc≠0，
        而报出来的是「三个程序各自炸一遍」，**说不清是哪一步失败**；
      · **给**探测到的绝对路径 ⇒ 探测**必须命中同一棵树**。

    ⇒ 第 ① 行若变红 = 「透传探测到的路径」没生效（把 `child_env()` 改回
      `dict(os.environ)` 就必然红）；第 ② 行若变红 = 本自检自己失去了区分力
      （那它就不是判据，只是一句声明）。两行都是**可证伪**的。
    """
    no_path = dict(env)
    no_path.pop(specs.CUTLASS_ENV, None)
    rc_off, out_off = _probe_child(tmpdir, no_path)
    rc_on, out_on = _probe_child(tmpdir, env)
    good = env.get(specs.CUTLASS_ENV, "")
    return [
        ("① 给了绝对路径 ⇒ 探测命中同一棵树", rc_on == 0 and out_on == good, True),
        ("② 不给路径（修复前的形态）⇒ 探测必须失败", rc_off != 0 and not out_off, True),
        ("③ env 里的路径是绝对路径且是一棵合格的树",
         bool(good) and os.path.isabs(good) and specs.is_cutlass_root(good), True),
    ]


# ===========================================================================
# ★★ 步骤 3/5：候选列表的性质自证
#    「`specs.CUTLASS_CANDIDATES` 的正确性」必须有断言钉住
# ===========================================================================
#
# 为什么单列一步（控制者上一轮在 `make mutate` 的读数里指出来的缺口）：
#
#   `CUTLASS_PATH` 是候选列表的**第一顺位** ⇒ 只要那个环境变量在（本机就是），
#   候选列表**根本不会被问到**。于是 `make mutate` 的叶子级变异里，
#   `CUTLASS_CANDIDATES[*]` 的**六个叶子全在漏网清单**：把单引号里那个字符串
#   换成任何别的东西，`make run` 照样 97/97。**「改坏了没有任何东西会叫」=
#   那张表没有任何断言钉住** —— 而它恰恰是 `CONTRIBUTING` 第 140 行
#   （「不得写死构造者本机的目录布局」）唯一约束的那张表。
#
# ⇒ 这里补的**不是变异体，是断言**（比造变异体便宜得多）：
#   把「那张表必须满足什么」写成**可机械判定**的性质（`specs.invariant_rows()`），
#   再拿**合成的坏表**去撞它 —— 恒真的检查不是判据，只是一句声明，
#   所以「这些性质抓得住坏表」这件事本身也要**当场证明**。
#
# 三组行：
#   ① / ② / ③ 静态性质（真表必须全过）；
#   ④          区分力（三张合成坏表必须各自被对应的一条抓住）；
#   ⑤ / ⑥      cwd 无关性（`os.getcwd` 桩 + 合成假树 —— **不依赖本机有没有 CUTLASS**）。
_SYNTH_PROBE = ("import sys; sys.path.insert(0, %r); import specs; "
                "specs.CUTLASS_CANDIDATES = ('../../cutlass',); "
                "assert specs.CUTLASS_CANDIDATE_BASE == %r, specs.CUTLASS_CANDIDATE_BASE; "
                "r = specs.find_cutlass(); print(r or ''); "
                "sys.exit(0 if r else 3)")


def synth_tree_probe(parent, cwds, env, timeout=60):
    """造一棵**合成假树**，只用**一条相对候选**探测它；返回 `(树, 副本目录, {cwd: (rc, 答案)})`。

    ★ 为的是**不依赖本机有没有 CUTLASS**：假树就是 5 个空文件，摆在本函数**自己**
      指定的位置（`<parent>/synth/cutlass`，候选 `../../cutlass` 从副本目录上溯两级
      正好落到它）。⇒ 无论这台机器上有没有真源码树，这个自检的区分力都一样。
      修复前那一版（用 `os.path.abspath()`）在这个自检上必然红：
      从副本目录调用会命中假树，从别的 cwd 调用会落到 `/cutlass` 那种位置 ⇒ 答案不一致。
    ★ 候选列表在子进程里**被换成只剩一条相对项**：其余候选（`~/cutlass` 等）
      在别人机器上**可能真的命中**，那会让「三个 cwd 的答案是否一致」失去意义。
    ★ 子进程的 env **摘掉了 `CUTLASS_PATH`**：它与候选列表是两条独立的通道，
      第一顺位若被设上，候选列表就又被绕过去了（正是本步骤要治的病）。
    """
    base = os.path.join(parent, "synth")
    tree = os.path.join(base, "cutlass")
    code = os.path.join(tree, "code")
    os.makedirs(code)
    shutil.copy(os.path.join(HERE, "specs.py"), os.path.join(code, "specs.py"))
    for f in specs.CUTLASS_REQUIRED:
        p = os.path.join(tree, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
    out = {}
    for cwd in cwds:
        r = subprocess.run([sys.executable, "-c", _SYNTH_PROBE % (code, code)],
                           cwd=cwd, capture_output=True, text=True, env=env,
                           timeout=timeout)
        out[cwd] = (r.returncode, (r.stdout or "").strip())
    return tree, code, out


def cwd_blind_probe():
    """把 `os.getcwd` 换成**会抛异常的桩**之后调 `find_cutlass()`：不得有异常逃出来。

    ★ 这是「相对候选的解析基准**不是 cwd**」的**直接**判据，而且**机器无关**：
      修复前那一版用 `os.path.abspath()`，而它**无论如何都会先调 `os.getcwd()`**
      ⇒ 换掉 `os.getcwd` 之后必抛。所以这一行**不依赖本机有没有 CUTLASS**。
    ★ 「只说明没读 cwd，不说明读对了哪棵树」—— 后者由步骤 1 的「打印选中路径」
      与上面的合成假树负责。
    ★ `CUTLASS_PATH` **临时摘掉**：它是**用户输入**，相对值按 cwd 解析是**有意的**
      （见 `specs.resolve_env_path()` 的说明），不适用本行要测的那条性质。
    """
    real_cwd = os.getcwd
    saved = os.environ.pop(specs.CUTLASS_ENV, None)

    def _poison():
        raise RuntimeError("find_cutlass() 读了 cwd —— 相对候选的基准必须是本文件所在目录")

    os.getcwd = _poison
    try:
        specs.find_cutlass()
    except Exception:
        return False
    finally:
        os.getcwd = real_cwd
        if saved is not None:
            os.environ[specs.CUTLASS_ENV] = saved
    return True


def candidate_selftest(env, tmpdir):
    """★★ 自证：候选列表的正确性**有断言钉住**，且这些断言**有区分力**。

    返回 `[(标签, 观测值, 期望值), …]`，与 `env_selftest()` 同形。
    每一行的反面都是可命名的：
      · ① 红 = 候选里出现了绝对路径（＝写死了构造者本机的布局）；
      · ② 红 = 必需文件表里出现了绝对路径（⇒ `is_cutlass_root()` 与那棵树无关）；
      · ③ 红 = 候选里只剩家目录项（表达不出「本仓库旁边那棵树」这个位置）；
      · ④ 红 = ①②③ 里有任意一条**抓不住**对应的合成坏表 ⇒ 那一条不是判据，只是一句声明；
      · ⑤ 红 = `find_cutlass()` 还在读 cwd；
      · ⑥ 红 = 「换个 cwd 就换一棵树（或从命中变不命中）」—— 正是本轮修掉的那个形态。
    """
    cands = tuple(specs.CUTLASS_CANDIDATES)
    required = tuple(specs.CUTLASS_REQUIRED)
    rows = list(specs.invariant_rows())
    # ④ 区分力：三张**合成坏表**，各自只违反一条静态性质（其余条目保持合法）。
    bad_tables = (
        ("①", dict(cands=cands + ("/abs/path/to/cutlass",))),
        ("②", dict(required=required + ("/etc/hosts",))),
        ("③", dict(cands=("~/cutlass", "~/src/cutlass"))),
    )
    missed = []
    for want_label, kw in bad_tables:
        # ★ 「这一条被违反了」的判据是 `got != want`，**不是**「`got` 非空」：
        #   ①② 的 got 是被抓到的坏条目**列表**，③ 的 got 是**布尔**（`False` 就是被违反）。
        #   写成「非空即命中」会把 ③ 的违反**漏掉**（本条自检第一次跑就是 2/3）。
        hit = [lbl for (lbl, got, want) in specs.invariant_rows(**kw) if got != want]
        if not any(lbl.startswith(want_label) for lbl in hit):
            missed.append(want_label)
    rows.append(("④ 区分力：三张合成坏表都必须被对应的一条抓住",
                 "%d/3 被抓住" % (len(bad_tables) - len(missed)),
                 "3/3 被抓住"))
    rows.append(("⑤ find_cutlass() 不读 cwd（把 os.getcwd 换成抛异常的桩）",
                 cwd_blind_probe(), True))

    env_off = dict(env)
    env_off.pop(specs.CUTLASS_ENV, None)
    cwds = (tmpdir, os.path.join(tmpdir, "synth", "cutlass", "code"), os.path.dirname(tmpdir))
    tree, _code, out = synth_tree_probe(tmpdir, cwds, env_off)
    answers = sorted({a for (_rc, a) in out.values()})
    same = len(answers) == 1 and answers[0] == tree
    rows.append(("⑥ 合成假树 + 相对候选：三个 cwd 必须选中同一棵（且非空）",
                 tree if same else "**不一致** %r" % (out,), tree))
    return rows


def run_case(tmp, which, patches, env):
    """把全部支撑模块复制到 `tmp`，按 `patches` 打补丁，逐个程序跑。

    `patches` 是 `[(文件名, 原文, 新文), …]`。
    """
    for f in AUX + PROGRAMS:
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tmp, f))
    for (fn, old, new) in patches:
        path = os.path.join(tmp, fn)
        text = open(path, encoding="utf-8").read()
        assert text.count(old) == 1, "锚点在 %s 里出现 %d 次" % (fn, text.count(old))
        open(path, "w", encoding="utf-8").write(text.replace(old, new))

    # ★ `env` **由 `main()` 传进来**：里面已经有**探测到的绝对路径**。
    #   副本的 cwd 是 `tmp`，**相对候选必然解析不到** —— 靠的只有这一条透传。
    out = {}
    for p in which:
        r = subprocess.run([sys.executable, p], cwd=tmp, capture_output=True,
                           text=True, env=env)
        out[p] = (r.returncode, r.stdout + r.stderr)
    return out


def main():
    print("=" * 74)
    print("元检验：把**本仓库自己的参照实现**改坏，断言编号程序必须失败")
    print("=" * 74)
    print("  ★ 本主题的 A 类断言全是「本仓库实现 vs 外部工件」——")
    print("    本文件回答的质疑是：**如果那份实现本身写错了，对拍还会成功吗？**")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本目录自己的输出形态）" % ASSERT_FAIL_MARK)
    print("  ★ 外部工件：本文件**先探测**（`specs.find_cutlass()`），再把**探测到的"
          "绝对路径**写进子进程 env。")
    print("    ★ **不依赖 `CUTLASS_PATH` 非空**：`Makefile` 的 `CUTLASS_PATH ?=` 默认为空，"
          "而**空值原样透传等于没传**。")
    print()

    # ---- 步骤 1/5：外部工件路径探测 -----------------------------------------
    print("[步骤 1/5] 外部工件路径探测（候选列表：specs.CUTLASS_CANDIDATES，相对本目录）")
    try:
        env = child_env()
    except ProbeError:
        print()
        print("!! [步骤 1/5] **探测失败** —— `%s` 为空，且候选列表里一个都没命中。"
              % specs.CUTLASS_ENV)
        print("   ⇒ 本文件**在这里就停**：三个编号程序**一次都没有运行**，")
        print("     所以这里没有「元检验失败」，只有「跑不起来」。")
        print("     （修复前那种「三个程序各自炸一遍、再汇总成一句含糊的话」不会再出现。）")
        print()
        print("   本主题的真值来源是 CUTLASS 4.8.0 自带的两个 Python 参考实现：")
        print("     python/pycute/          (CuTe layout 代数)")
        print("     python/cutlass_library/ (CUTLASS 自己的生成器)")
        print("   需要下列文件存在（相对 CUTLASS 根）：")
        for f in specs.CUTLASS_REQUIRED:
            print("     %s" % f)
        print()
        print("   请显式指定（两种方式任选）：")
        print("     make falsify %s=/path/to/cutlass" % specs.CUTLASS_ENV)
        print("     export %s=/path/to/cutlass" % specs.CUTLASS_ENV)
        print("   或把 CUTLASS 源码树放到下列任一位置（相对本目录）：")
        for c in specs.CUTLASS_CANDIDATES:
            print("     %s" % c)
        print()
        print("   ★ 本库**不复刻** CUTLASS 的源码树，也不内置它的副本。")
        return 2
    print("    透传        : 上面这个**绝对路径**已写进子进程 env 的 `%s`"
          "（副本的 cwd 是临时目录，靠的就是它）" % specs.CUTLASS_ENV)
    print()

    # ---- 步骤 2/5：子进程 env 自证 -------------------------------------------
    print("[步骤 2/5] 子进程 env 自证（★ 「不设 CUTLASS_PATH 且不在原位」是可检出的）")
    print("    ★ 做法：在**临时目录**里跑纯探测，并把候选列表换成一个不存在的路径 ——"
          " 唯一的变量只剩 `CUTLASS_PATH`。")
    with tempfile.TemporaryDirectory(prefix="cl_selfcheck_") as sc_tmp:
        rows = env_selftest(env, sc_tmp)
    sc_bad = 0
    for label, got, want in rows:
        ok = got == want
        sc_bad += 0 if ok else 1
        print("    %-40s = %-5s 期望 %-5s %s"
              % (label, got, want, "OK" if ok else "**异常**"))
    if sc_bad:
        print()
        print("!! [步骤 2/5] env 自证**失败 %d 项** ⇒ **在这里就停**："
              "不跑基线、不跑那 9 种缺陷。" % sc_bad)
        print("   第 ① 行红 = 「探测到的路径被写进子进程 env」这件事没生效"
              "（`child_env()` 被改回 `dict(os.environ)` 时必然如此）；")
        print("   第 ② 行红 = 本自检自己失去了区分力。")
        return 2
    print()

    # ---- 步骤 3/5：候选列表的性质自证 ---------------------------------------
    print("[步骤 3/5] 候选列表的性质自证（★ 「候选列表的正确性」有断言钉住、且钉得住）")
    print("    ★ 缺口：`%s` 是候选列表的**第一顺位**，它一命中，候选列表**不会被问到** ⇒"
          % specs.CUTLASS_ENV)
    print("      `make mutate` 里 `CUTLASS_CANDIDATES[*]` 的六个叶子**全在漏网清单**里 ——"
          "那张表没有任何断言钉住。")
    print("    ★ 这里补的是**断言**（不是变异体）：把「那张表必须满足什么」写成可机械判定的"
          "性质（`specs.invariant_rows()`），")
    print("      并拿**合成的坏表 / 合成假树**证明这些性质**有区分力**"
          "（恒真的检查不是判据，只是一句声明）。")
    with tempfile.TemporaryDirectory(prefix="cl_cand_") as cd_tmp:
        cand_rows = candidate_selftest(env, cd_tmp)
    cd_bad = 0
    for label, got, want in cand_rows:
        ok = got == want
        cd_bad += 0 if ok else 1
        print("    %-46s = %-5s 期望 %-5s %s"
              % (label, got, want, "OK" if ok else "**异常**"))
    if cd_bad:
        print()
        print("!! [步骤 3/5] 候选列表自证**失败 %d 项** ⇒ **在这里就停**："
              "不跑基线、不跑那 9 种缺陷。" % cd_bad)
        print("   ① / ② 红 = 表里出现了绝对路径（**那正是 `CONTRIBUTING` 第 140 行禁止的"
              "「写死构造者本机的目录布局」**）；")
        print("   ③ 红 = 候选里只剩家目录项（表达不出「本仓库旁边那棵树」这个位置）；")
        print("   ④ 红 = 上面某一条**抓不住**对应的合成坏表 ⇒ 它不是判据，只是一句声明；")
        print("   ⑤ 红 = `find_cutlass()` 还在读 cwd；"
              "⑥ 红 = 「换个 cwd 就换一棵树」——正是残留的 cwd 依赖。")
        return 2
    print()

    # ---- 步骤 4/5：基线 ------------------------------------------------------
    print("[步骤 4/5] 基线（不打补丁）")
    with tempfile.TemporaryDirectory(prefix="cl_base_") as tmp:
        base = run_case(tmp, PROGRAMS, [], env)
    bad = [p for p, (rc, _) in base.items() if rc != 0]
    print()
    if bad:
        print("[基线] 不打补丁：**有程序失败** → %r" % bad)
        print("      ★ 步骤 1 已经成功（外部工件 = %s），所以这**不是**「找不到工件」"
              "那一类失败。" % env[specs.CUTLASS_ENV])
        for p in bad:
            print("      %s 的尾部输出：" % p)
            for line in base[p][1].strip().splitlines()[-5:]:
                print("        %s" % line[:110])
    else:
        print("[基线] 不打补丁：三个编号程序全部退出 0 ✓")
    n_fail = 1 if bad else 0

    # ---- 步骤 5/5：9 种缺陷 --------------------------------------------------
    print()
    print("[步骤 5/5] %d 种缺陷（每个都在自己的副本里跑）" % len(DEFECTS))
    for (name, fn, old, new, progs) in DEFECTS:
        src = open(os.path.join(HERE, fn), encoding="utf-8").read()
        if src.count(old) != 1:
            print()
            print("[%s]" % name)
            print("    ★ 锚点在 %s 里出现 %d 次（必须恰好 1 次）→ **补丁无效**，记一次失败"
                  % (fn, src.count(old)))
            n_fail += 1
            continue
        with tempfile.TemporaryDirectory(prefix="cl_def_") as tmp:
            res = run_case(tmp, PROGRAMS, [(fn, old, new)], env)
        print()
        print("[%s]" % name)
        print("    改的是：%s" % fn)
        for p in PROGRAMS:
            rc, txt = res[p]
            if p in progs:
                ok = rc != 0
                first = next((l.strip() for l in txt.splitlines()
                              if ASSERT_FAIL_MARK in l), None)
                print("    %-26s 退出 %-4d 期望失败 → %s"
                      % (p, rc, "OK" if ok else "**没失败，这条 A 类断言无效**"))
                if first:
                    print("        证据：%s" % first[:110])
                if not ok:
                    n_fail += 1
            else:
                print("    %-26s 退出 %-4d （未注入本缺陷）" % (p, rc))

    print()
    print("=" * 74)
    if n_fail:
        print("元检验失败 %d 项：至少有一条 A 类对拍是「只演示、不检查」" % n_fail)
    else:
        print("元检验通过：**%d 种缺陷全部被断言抓住**，且不打补丁时三个程序全过。"
              % len(DEFECTS))
        print("→ 「本仓库那份参照实现自己写错了怎么办」这个质疑，被这 %d 条反证正面回答。"
              % len(DEFECTS))
        print("★ 其中 D8 / D9 是**方向相反**的两条：它们改成的是**另一个看起来同样合理的"
              "写法**，")
        print("  也必须被抓住 —— 它证明「本实现选的是哪一个」是**被断言钉住的**，"
              "而不是一句声明。")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
