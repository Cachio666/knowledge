#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
★ 本文件**从 `100-parallelism/expert-parallel/code/mutate.py` 逐字复制**（**工具逻辑本身是对的**，
  且与库内其余主题一致）。
★★★ **本主题对正文的改动**（**只有四处**，逐条列在这里，便于复核 —— 这是本库
  「抬头必须与事实相符」的落法）：
  ① 顶部标题行：`100-parallelism/expert-parallel 叶子级变异测试`
     → `110-communication/collectives 叶子级变异测试`；
  ② 临时目录前缀：`mut100ep_` → `mut110co_`（★ 防止两个主题的临时目录撞名）；
  ③ 空分母分支里的一句：`（本主题在 **Chunk 5** 已非空，走不到这一支）`
     → `（本主题在 **Chunk 1** 已非空，走不到这一支）`；
  ④ 本抬头（整段换成**本主题**的陈述）。
  ★★ 除这四处与**本抬头**之外，正文**逐字相同**。
  ★★★ **为什么必须重写抬头**：`expert-parallel` 的抬头里那一整套陈述（分母 `74`、
    `specs.py` 的 `56` 个叶子、它的三个程序名、它的常量名、它的 `D.4`/`C.1`/`C.2`、
    「本主题的标记含 `{`」、「引文常量 14 个」……）**在本主题全部为假** ——
    照抄会制造一条**抬头与事实不符**的资产（L2 级错误）。
    本文件里所有关于「本主题」的数，都是**用本文件自己的 `enumerate_leaves()` 实测出来的**
    （见下面「本主题的实测口径」一节）。

===============================================================================
★ 本主题的状态：**分母非空**
===============================================================================

本主题的三个编号程序（`01_algorithm_family` / `02_stages_and_costs` /
`03_all_to_all_and_p2p`）随 **Chunk 1** 落盘（提交日 **2026-10-08**），所以本工具
**不走空分母分支**（那条分支仍然留着：`EXIT_NO_TARGET = 3`）。

★ 本主题**特别需要这道闸的四条理由**（每一条都对应程序里的一处真实风险）：
  · ★★★ **本主题的 A 类是「消息序列模拟 vs 闭式」** —— 判据的**参照是一段代码**
    （模拟器），被测是**另一段代码**（闭式函数）。★ 本工具**看不见**「哪条判据是恒真的」，
    它只变异**字面量**。⇒ 本主题把这件事**显式写进每个程序的抬头与 `A.1g` 守卫**：
    「把块索引改错 ⇒ 结果必须变、而搬运量不变」。
  · ★★★ **规模型常量成批存在**（`DEV_GRID` / `SAT_GRID` / `GATHER_GRID` / `A2A_GRID` /
    `STAGE_GRID` 等元组）⇒ ★★ **每个元组都必须被【逐元素】`==` 钉住**，
    否则元素级变异会整批漏网（本库实测：`pipeline-parallel` 那一轮一次报出 **58 条**）。
  · ★★ **字符串常量一批**（`specs.py` 的 MPICH 引文）—— 它们的变异形态是**末尾追加一个 `X`**
    ⇒ 凡用 `startswith` / `count` / `in`（对子串）的钉住**都抓不到它**（盲区③）。
    ★★ 本主题的关键引文**一律有 `==` 逐字钉住**（`01` 的 `D.5`/`D.6`，`02` 的 `D.5`/`D.6`，
    `03` 的 `D.5`/`D.6`）。
  · ★★ **一条【否定性】记录**（`specs.UNSOURCED`）—— ★ 它**留空**（`()`，**0 个叶子**），
    而**空元组本身有消费者**（三个程序的 `D.3`/`D.9` 各钉一次）⇒ 它**不产生变异体**、
    **也不构成漏网候选**（★ 这是本主题与 `expert-parallel` 的**一处形态差异**：
    那边另有一条非空的 `EP_NOT_FOUND_NOTE`）。

★★ **本主题的实测口径**（2026-10-08，用本文件自己的 `enumerate_leaves` 数出来的；
   ★ **重跑本文件即可复现**，不写死）：
  分母 = **本目录全部编号程序 + `specs.py`** 的**模块级常量叶子**。
  ★★ **读数**见本文件运行时的输出（`测 N / 捕获 M / 漏网 0`）——
    此处**不写死任何数字**（★ 写了就会随维护过期，本库有先例）。
  ★ 本主题的**跨行字面量 `0` 个**、**幂次字面量 `0` 处**（`specs.py` 的常量全是十进制字面量）、
    **模块级常量里没有 `dict`**、**没有 `UnaryOp`** ⇒ 盲区②④在本主题**不构成漏网来源**。
  ★★ 三个程序的模块级常量**逐字同形**：`ASSERT_FAIL_MARK`（1 个叶子）、
    `OTHER_THEME_MARKS`（3 个叶子）、`FAILS = []`（**空 list**，0 个叶子）、
    `MARKS_USED = set()` / `_TOTAL` / `_PASS`（`Call`，0 个叶子）、
    `TOL_MACHINE = specs.TOL_MACHINE`（`Attribute`，0 个叶子）、
    `MSG = collections.namedtuple(...)`（`Call`，0 个叶子）。
    ⇒ ★ 每个程序只贡献 **4 个叶子**（1 + 3），**全部有 `==` 钉住**。

⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
  不是本脚本的（1 / 2 / 3）⇒ **要看退出码就直接 `python3 mutate.py`**。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对每一个叶子级常量逐个问一遍，
答「不会」的（**漏网**）如实列出来。★ 本主题的断言多是
「**计数相等**」（步数、消息条数、元素数 —— 全是**整数**）与「**逐位对拍**」
（消息序列模拟 vs 单进程参照）⇒ **极易写成无条件成立的伪断言**；
而且 `specs.py` 里凡**没有消费者**的常量（「照抄但不使用」）都会在这里现形。

★★ 本主题还有一处**本工具【看不见】的**风险，必须在这里写明：
  · 三个程序的 `A.1g`/`C.1`/`C.2` 等断言的**不是「一致」而是「错」**
    （「块索引改成 `(d+k)%N`」/「发错块」/「累加到错目标」/「漏掉一段」/「只发一份」/
    「先 recv 后 send」）——
    ★★ 那些**错法本身是一段代码**（`corrupt="send"` / `corrupt="accum"` /
    `sim_a2a_pairwise(..., drop_one=True)` / `sim_sendrecv(..., mode="recv_then_send")`），
    **不是模块级常量** ⇒ 本工具**不会**看见它们。
    ★ 这是**对的**：那一类判据的价值不在「能失败」，而在**它把「这个错法真的会错」钉住了**；
      而它们的**可证伪性由它们自己的守卫/对照保证**。

===============================================================================
★★ 四条硬纪律（前两条照搬 `50-performance/profiling-tools/code/mutate.py` 的教训）
===============================================================================

1. **只在副本上做，绝不动仓库里的原文件。**
   每个变异体在**自己的临时目录**里执行；仓库目录从头到尾**只被读**，
   跑完再校验一次 sha256，变了就以非零码退出（那时读数已经不可信）。

2. **必须区分「断言捕获」与「崩溃捕获」。**
   · **断言捕获**：程序打印了**本主题自己的**失败标记，且 rc≠0。
   · **崩溃捕获**：rc≠0 但**没有**失败标记 —— 异常 traceback，或环境类退出。
   ⇒ 崩溃捕获**不算**真捕获：它证明「代码炸了」，不是「判据抓住了错」。两类**分开数**。

3. **`ASSERT_FAIL_MARK` 必须用 AST 从编号程序里读出来，不得写死。**
   ★ 本主题四个主题标记是 `&&` / `><` / `,,` / `+=` ⇒ **两两不互为前缀**，
     跨主题照抄会被编号程序自己的 `D.2b` 当场抓住。
   ★ 读法两条（来自 `50-performance` 的实测教训）：
     · 那个常量必须是**普通字符串字面量**（不是 `re.compile(...)`）；
     · 判定必须用「**行首**锚定」，不能用 `MARK in output` ——
       程序**成功时也会打印这个字面量**（标记自检那一条），`in` 在 rc=0 时也成立。
   ★★ 本主题的标记是 `&& FAIL: collectives`；★★ 那个 `&&` 在 **`Makefile` 的注释里**
     写成 `&&`（★ `&` **不是** shell 的元字符分隔符意义上的麻烦，但它是 **shell 的
     控制操作符**）⇒ ★ 本文件与编号程序里**都未转义**，且它**只出现在 `print()` 里**。

4. **对照组必须有。**把某条常量**改回它自己**，harness **不得**报告「捕获」。

===============================================================================
★ 口径（分母）
===============================================================================

  分母 = **本目录全部编号程序**（`NN_*.py`）**+ `specs.py`** 的**模块级常量叶子**。
  **不含**断言调用里的字面量：那些是**期望值**（测试自己的一部分），
  改它们必然「失败」，把这种必然算进分子等于给分母灌水。
  ★★ **本主题的期望值确实住在断言里**（例如 `B.1` 的饱和值、`D.7` 的档位）——
  ★ 这是**故意的**：它们是**推导值 / 本机实测值**，不是从任何来源抄来的规格值
    ⇒ 按 `CONTRIBUTING` §2「只放从来源抄来的数，**不放推导值**」，它们**不应该**进 `specs.py`。

★★ **driver 规则（本主题的 `specs.py` 依赖它）**：变异体若在**辅助模块**（`specs.py`）里，
  就**遍历全部编号程序、任一捕获即算捕获**（见下面 `main()` 里的 **R17** 注释）。
  ★ 若写成「只由第一个编号程序检验」，`specs.py` 的变异会**整体免检**。
  ★★ **本主题三个程序都 `import specs`** ⇒ 这条规则在本主题是**必需的**，不是保险。

★★ **本工具的已知盲区**（本主题会碰到哪几条，逐条写明）：
  ① **只变异字面量、不变异代码**：本主题的**四条最重的判据**（块索引、累加目标、
     漏一段、只发一份、先 recv 后 send）**全是代码级** ⇒ 本工具**看不见**它们
     （★ 见上面「本工具看不见的风险」）。
  ② **负字面量不进分母**：★ 本主题 `specs.py` 的模块级常量值里 **`UnaryOp` 计 0 处**
     ⇒ 不构成漏网来源。
  ③ **前缀/包含式检查会漏网**：★ 本主题的关键常量**一律 `==` 逐字钉住**，
     不用 `startswith`/`count`/`in`（对子串）。
  ④ **`_walk()` 不下钻 `BinOp`**：★ 本主题 `specs.py` 里**一处幂次字面量都没有**
     ⇒ 不构成漏网来源。
  ⑦ **跨行字符串被整条跳过**：★ 本主题**没有**这个问题（实测跨行字面量 **0** 个）。

===============================================================================
★★ 一条【本主题特有】的引文纪律
===============================================================================

1. **载体只有一个**：Thakur / Rabenseifner / Gropp 2005（IJHPCA 19(1):49–66）。
   ★★★ 它的公式在 `pdftotext -layout` 下有三种形态：
   **`dlg pe` = `⌈lg p⌉`** · **`blg pc` = `⌊lg p⌋`** · **裸 `lg p` = 无取整** ——
   ★★ 本库的 `specs.py` 存**排版形态**，并在每条的 `# 来源：` 里记**提取形态**。
   ★★★ **误判取整符号不会有任何报错**（`grep "lg p"` 会同时命中三种）⇒
   凡改这一批常量，**先看它属于哪一态**。

用法：`make mutate`（= `python3 mutate.py`）。
"""

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PER_RUN_TIMEOUT = 600            # 单个程序一次运行的秒数上限
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".last_out")
MARK_NAME = "ASSERT_FAIL_MARK"   # 本域约定：每个主题在编号程序里定义自己的失败标记
EXIT_NO_TARGET = 3               # ★ 空分母的**说明性非零码**（1 留给「有漏网」）
EXIT_ESCAPED = 1
EXIT_UNTRUSTED = 2


# ===========================================================================
# 1. 目标发现与 `ASSERT_FAIL_MARK` 的 AST 读取
# ===========================================================================
def discover_programs():
    """编号程序 = `NN_*.py`（`CONTRIBUTING` §2 的外键形态）。"""
    return sorted(f for f in os.listdir(CODE_DIR)
                  if len(f) > 3 and f[:2].isdigit() and f[2] == "_" and f.endswith(".py"))


def top_consts(text, path):
    """模块级 `NAME = <字面量>`（`Assign` 与 `AnnAssign`）-> [(名字, 值节点)]。"""
    out = []
    for n in ast.parse(text, path).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            out.append((n.targets[0].id, n.value))
        elif isinstance(n, ast.AnnAssign) and n.value is not None \
                and isinstance(n.target, ast.Name):
            out.append((n.target.id, n.value))
    return out


def read_fail_mark(path):
    """从**编号程序**里用 AST 读出它自己的 `ASSERT_FAIL_MARK`。

    ★ **不得写死**：本库纪律是每个主题自定义失败标记。
    ★ 找不到 / 不是字符串字面量 / 一个文件里出现多次 —— 都**以可操作的错误退出**。
    """
    text = open(path, encoding="utf-8").read()
    hits = [(name, node) for name, node in top_consts(text, path) if name == MARK_NAME]
    if not hits:
        raise ValueError("%s 里没有模块级常量 %s —— 本域约定每个编号程序都要定义它"
                         % (os.path.basename(path), MARK_NAME))
    if len(hits) > 1:
        raise ValueError("%s 里 %s 出现了 %d 次 —— 约定是恰好一次"
                         % (os.path.basename(path), MARK_NAME, len(hits)))
    node = hits[0][1]
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        raise ValueError("%s 的 %s 不是**字符串字面量**（读出来是 %s）—— "
                         "写成 `re.compile(...)` 之类的表达式会让读取方拿到 None"
                         % (os.path.basename(path), MARK_NAME, type(node).__name__))
    return node.value


# ===========================================================================
# 2. 叶子枚举（分母）
# ===========================================================================
def _walk(node, suffix=""):
    """递归收集字面量叶子 -> [(节点, 后缀名)]。后缀名形如 `[0]` / `['k']`。"""
    out = []
    if isinstance(node, ast.Constant):
        out.append((node, suffix))
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for i, e in enumerate(node.elts):
            out += _walk(e, "[%d]%s" % (i, suffix))
    elif isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                out += _walk(k, "{%r}%s" % (getattr(k, "value", "?"), suffix))
            out += _walk(v, "[%r]%s" % (getattr(k, "value", "?"), suffix))
    return out


def enumerate_leaves(files):
    """-> [(文件, 常量名, 后缀, 节点)]。**只取模块级常量的叶子**（口径见模块头）。"""
    out = []
    for f in files:
        text = open(os.path.join(CODE_DIR, f), encoding="utf-8").read()
        for name, value in top_consts(text, f):
            for node, suffix in _walk(value):
                out.append((f, name, suffix, node))
    return out


def _mutate_value(v):
    """给出「另一个值」。返回 `None` 表示**这一种叶子没有有意义的另一个值**。"""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 1.0
    if isinstance(v, str):
        return v + "X"
    return None


def _patch(lines, node, new_text):
    """把某一行上的那个字面量替换掉。

    ★ `ast` 的 `col_offset` 是**这一行 UTF-8 编码后的字节偏移**（本文件里满是中文注释，
      按字符切会切错位）⇒ 先 `encode` 再切，最后 `decode` 回来。
    """
    if node.end_lineno != node.lineno:
        raise ValueError("跨行字面量，本骨架不处理")
    li = node.lineno - 1
    raw = lines[li].encode("utf-8")
    new_line = (raw[:node.col_offset] + new_text.encode("utf-8")
                + raw[node.end_col_offset:]).decode("utf-8")
    out = list(lines)
    out[li] = new_line
    return out


def _text_of(lines):
    return "".join(lines)


# ===========================================================================
# 3. 跑一次（副本 + 分类）
# ===========================================================================
def run_once(program, mutated=None, timeout=PER_RUN_TIMEOUT):
    """把本目录**全部** `.py` 复制到临时目录后跑 `program`。

    ★ 必须**整目录**复制：编号程序 `import specs`（`code/` 自包含，`CONTRIBUTING:36`）。
    `mutated` 是 `(文件名, 改写后的行列表)`：在副本上覆盖那一份。
    ★ **仓库里的文件只被读**，一切改写都发生在临时目录。
    """
    tmp = tempfile.mkdtemp(prefix="mut110co_")
    try:
        for f in sorted(os.listdir(CODE_DIR)):
            if f.endswith(".py"):
                shutil.copy2(os.path.join(CODE_DIR, f), os.path.join(tmp, f))
        if mutated is not None:
            name, lines = mutated
            open(os.path.join(tmp, name), "w", encoding="utf-8").write(_text_of(lines))
        r = subprocess.run([PY, program], cwd=tmp, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def classify(rc, out, mark):
    """★ 行首锚定的判定（不能用 `mark in out`，见模块头纪律 3）。"""
    if rc == 0:
        return None
    for line in out.splitlines():
        if line.lstrip().startswith(mark):
            return "assert"
    return "crash"


# ===========================================================================
# 4. 主流程
# ===========================================================================
def main():
    programs = discover_programs()
    targets = programs + (["specs.py"] if os.path.exists(os.path.join(CODE_DIR, "specs.py")) else [])
    leaves = enumerate_leaves(targets)

    print("=" * 74)
    print("110-communication/collectives 叶子级变异测试")
    print("  编号程序 %d 个：%s" % (len(programs), ", ".join(programs) if programs else "（无）"))
    print("  参与统计的文件 %d 个：%s" % (len(targets), ", ".join(targets) if targets else "（无）"))
    print("=" * 74)

    # ── ★★ 空分母自检 ──────────────────────────────────────────────────────
    if not programs:
        print()
        print("测 0 / 捕获 0 / 漏网 0")
        print()
        print("!! 分母为 0：**本主题此刻一个编号程序都没有**。")
        print("   ⇒ 上面的「捕获 0 / 漏网 0」**不含任何信息**，它既不是通过也不是失败。")
        print("   ⇒ 空分母自检：本工具在**没有对象时**")
        print("      会诚实地报 0 并给出非零退出码（本主题在 **Chunk 1** 已非空，走不到这一支）。")
        print("   ⇒ 本工具以退出码 %d 退出（**说明性非零码**，不是故障）：" % EXIT_NO_TARGET)
        print("      空跑一次「绿」的 make mutate 会被误读成「变异测试通过了」。")
        if leaves:
            print("   （附带记录：即使没有编号程序，本目录仍有 %d 个模块级常量叶子"
                  "—— 它们**不计入**本工具的分母。）" % len(leaves))
        return EXIT_NO_TARGET

    # ── 标记读取（每个程序各读一次，跨主题照抄会被这里挡住）────────────────
    marks = {}
    for p in programs:
        try:
            marks[p] = read_fail_mark(os.path.join(CODE_DIR, p))
        except ValueError as e:
            print("!! %s" % e)
            return EXIT_UNTRUSTED
        print("  %-32s ASSERT_FAIL_MARK = %r" % (p, marks[p]))
    # ★★★ 本域四个主题标记**两两不互为前缀** —— 这里再做一次**跨程序**的核对：
    #   ★ 同一主题的三个程序**必须**用**同一个**标记（本工具不禁止「同一主题多程序同标记」）。
    uniq_marks = sorted(set(marks.values()))
    print("  本主题用到的失败标记取值（去重）：%s" % (uniq_marks,))

    # ── 基线（未变异时必须全清）───────────────────────────────────────────
    for p in programs:
        rc, out = run_once(p)
        kind = classify(rc, out, marks[p])
        if kind is not None:
            print("!! 基线（未变异）不是干净的：%s rc=%d，捕获种类=%s" % (p, rc, kind))
            print(out[-2000:])
            return EXIT_UNTRUSTED
    print("  基线：%d 个编号程序在未变异时全部 rc=0 且无失败标记 ✓" % len(programs))

    # ── 逐叶子变异 ────────────────────────────────────────────────────────
    total = caught_a = caught_c = escaped = skipped = 0
    escaped_list = []
    for f, name, suffix, node in leaves:
        v = getattr(node, "value", None)
        label = "%s::%s%s" % (f, name, suffix)
        new = _mutate_value(v)
        if new is None:
            skipped += 1
            continue
        lines = open(os.path.join(CODE_DIR, f), encoding="utf-8").readlines()
        try:
            patched = _patch(lines, node, repr(new))
        except ValueError:                 # 跨行字面量：本骨架不处理，计入跳过
            skipped += 1
            continue
        total += 1
        # ★★ R17 修复（2026-09-22）—— 原来的规则是
        #       owner = f if f in marks else programs[0]
        #   它让**整个 `specs.py` 只被第一个编号程序检验**。
        #   新语义 = 「**这条变异体被本主题的测试集捕获了吗**」：
        #   遍历全部编号程序，**任一捕获即算捕获**（短路）。
        if f in marks:
            drivers = [f]                    # 变异体就在某个编号程序里 ⇒ 只它自己
        else:
            drivers = list(programs)         # 变异体在辅助模块里 ⇒ 全部编号程序都有资格
        kind = None
        for cand in drivers:
            rc, out = run_once(cand, mutated=(f, patched))
            k = classify(rc, out, marks[cand])
            if k is not None:                # 捕获了（None = 未捕获）
                kind = k
                break
        if kind == "assert":
            caught_a += 1
        elif kind == "crash":
            caught_c += 1
        else:
            escaped += 1
            escaped_list.append(label)

    print()
    print("测 %d / 捕获 %d / 漏网 %d" % (total, caught_a + caught_c, escaped))
    print("    断言捕获 %d · 崩溃捕获 %d（★ 崩溃捕获不算真捕获，两类分开数）"
          % (caught_a, caught_c))
    print("    跳过 %d（没有有意义的另一个值）" % skipped)
    if escaped_list:
        print("  漏网：")
        for x in escaped_list:
            print("    · %s" % x)
    print("=" * 74)
    return EXIT_ESCAPED if escaped else 0


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    before = {f: _sha256(os.path.join(CODE_DIR, f))
              for f in os.listdir(CODE_DIR) if f.endswith(".py")}
    rc = main()
    after = {f: _sha256(os.path.join(CODE_DIR, f))
             for f in os.listdir(CODE_DIR) if f.endswith(".py")}
    if before != after:
        print("!! 仓库里的 .py 在本次运行中被改动了 —— 读数**不可信**，请检查！！")
        sys.exit(EXIT_UNTRUSTED)
    sys.exit(rc)
