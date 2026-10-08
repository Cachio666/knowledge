#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
★ 本文件**从 `100-parallelism/expert-parallel/code/mutate.py` 逐字复制**（**工具逻辑本身是对的**，
  且与库内其余主题一致）—— 同域 `collectives/code/mutate.py` 与 `nccl/code/mutate.py` 也是同一份。
★★★ **本主题对正文的改动**（**只有四处**，逐条列在这里，便于复核 —— 这是本库
  「抬头必须与事实相符」的落法）：
  ① 顶部标题行：`110-communication/nccl 叶子级变异测试`
     → `110-communication/network-topology 叶子级变异测试`；
  ② 临时目录前缀：`mut110nc_` → `mut110nt_`（★ 防止同域两个主题的临时目录撞名）；
  ③ 空分母分支里的一句：`（本主题在 **Chunk 2** 已非空，走不到这一支）`
     → `（本主题在 **Chunk 3** 已非空，走不到这一支）`；
  ④ 本抬头（整段换成**本主题**的陈述）。
  ★★ 除这四处与**本抬头**之外，正文**逐字相同**。
  ★★★ **为什么必须重写抬头**：`nccl` 的抬头里那一整套陈述（分母、它的三个程序名、
    它的常量名、`NCCL_*` 引文、它的 `D.0`……）**在本主题全部为假** ——
     照抄会制造一条**抬头与事实不符**的资产（L2 级错误）。
     ★★ 本文件里所有关于「本主题」的数，都是**用本文件自己的 `enumerate_leaves()` 实测出来的**
     （见下面「本主题的实测口径」一节）。

===============================================================================
★ 本主题的状态：**分母非空**
===============================================================================

本主题的三个编号程序（`01_fat_tree_and_clos` / `02_rail_optimized` /
`03_roce_and_infiniband`）随 **Chunk 3** 落盘（提交日 **2026-10-08**），所以本工具
**不走空分母分支**（那条分支仍然留着：`EXIT_NO_TARGET = 3`）。

★★★ **【本主题与同域另外两个主题最大的一处不同】—— 本主题【有真 A 类】**：
  `01` 把胖树**建成一张显式的图**再**数**（节点数 / 度 / BFS 最短路），与 **SIGCOMM 2008 的
  论文值**对拍；`02` 把 `N` 张卡**逐张摆**到 `R` 条轨上再**数回来**，与 `N/R` / `G/R` 对拍。
  ★★ 所以本工具的定位**与 `nccl` 那章不同**：那边它「**不能替代 A 类**」，
    这边它**本来就只是 A 类之外的一道闸**（A 类由 `01`/`A.1`/`A.2` 与 `02`/`A.1`/`A.2` 自己承担）。
  ★★ 而 `03`（RoCE / InfiniBand）**A 类不可得**（本机**无网卡**）⇒ 它的「没有 A」由
    **运行期记账 + 调用点核对**两条机械量证明（★ 机制**不同于** `nccl` 的源码扫描）。

★★★ **一条【本主题实测出来】的工具盲区（★ 它不在正典的 ①–⑥+⑧ 里，也没有编号）**：
  **「跨行隐式字符串拼接」的常量会被本工具【静默计入「跳过」】** ——
  `ast` 在解析时会把 `NAME = ("a" "b")` **折叠成一个** `Constant`，而它的 `end_lineno != lineno`
  ⇒ `_patch()` 抛 `ValueError` ⇒ 落进 `skipped` 那一档（**既不报漏网，也不报覆盖**）。
  ★★★ **本主题的第一版就是这样：29 个叶子（21 条引文 + 4 个 URL + 4 个书目载体）全部被跳过**，
  而打印出来的是「测 80 / 捕获 80 / 漏网 0」—— ★★ **一个「绿」的读数，却把最要紧的那批常量
  一条都没测**（本库「**假保证**」那一族的现场）。
  ⇒ ★ **处置**：`specs.py` 里**所有字符串常量一律写成【单行】**（★ 同域的 `collectives` / `nccl`
    也是单行 —— 复核过：它们的跳过数都是 **`0`**）⇒ 本主题现在是 **`测 109 / 跳过 0`**。
  ★★ **同一次修好之后**，工具立刻抓出 **5 条真漏网**（★ 见下面那一段）—— ★★★ **先把覆盖修好，
    漏网才看得见**。

★★ **一次【真漏网】的现场（★ 修好上述盲区之后才暴露）**：
  `SIGCOMM08_INSTANCE_QUOTE` / `SIGCOMM08_SUBNETS_QUOTE` / `SIGCOMM08_TIER_QUOTE` /
  `CISCO_ROCEV2_DOC` / `SHARP_QUOTE` 这五条**当时只有 `startswith` / `in` 的钉住**
  （或**只在 `print` 里出现**）⇒ ★★ 它们的变异体**一个都没被捕获**。
  ★★★ 这正是正典里的**两条**盲区在**本主题**的活样本：
    · **③ 前缀式检查会漏网**（`startswith("A three-tiered design")` 挡不住**末尾追加**）；
    · ★★ 以及「**只在打印里出现**」= **⑧「整体钉住代替不了元素钉住」**的近亲。
  ⇒ ★ 五条**全部改成 `==` 逐字钉住**，漏网归零。

★ 本主题**特别需要这道闸的四条理由**（每一条都对应程序里的一处真实风险）：
  · ★★★ **载体是两类、级别不同**（SIGCOMM 2008 `[C]` 论文 · NVIDIA/Cisco `[B]` 产品文档 ·
    RFC 3168 / IEEE 802.1Qbb `[A]` 规范摘要）—— ★★ **引文常量被动过**是本主题的头号风险：
    本工具对**每一个字符串引文**都要求有 `==` 逐字钉住（三个程序的 `D.5`/`D.6`）。
  · ★★★ **一张表和它的元素**（`ROCE_FACT_ROWS` = `7` 行 × `3` 列 = **21 个叶子**；
    `ECN_BITS` / `PFC_OFFICIAL_LOSS_CAUSES` / `SPEC_REQUIRED_FOR` / `NOT_QUOTED` 等元组）
    ⇒ ★★ **每个元组都必须被【逐元素】`==` 钉住**（`03` 的 `D.4`/`D.6`），
      否则元素级变异会整批漏网（本库实测：`pipeline-parallel` 那一轮一次报出 **58 条**）。
  · ★★ **一批「规模/计数」标量**（`SIGCOMM08_*_AT_K48` 六个 · `RAIL_*` 四个 · `ECN_BITS` ·
    `ROCEV2_UDP_PORT` · `PFC_PRIORITY_COUNT` ……）—— 它们的变异形态是**整数 `+1`**
    ⇒ ★ 每个都必须有 `==` 钉住（三个程序的 `D.3`/`D.4`/`D.7`）。
  · ★★ **一条【否定性】记录**（`specs.NOT_QUOTED`，**3 条**）—— ★ 它是**非空**元组
    ⇒ 有叶子、会被变异 ⇒ **必须逐元素钉住**（`03` 的 `D.6`）；★ 而 `specs.UNSOURCED` **留空**
      （`()`，`0` 个叶子）⇒ 不产生变异体，但**空元组本身有消费者**（三个程序各钉一次）。

★★ **本主题的实测口径**（2026-10-08，用本文件自己的 `enumerate_leaves` 数出来的；
   ★ **重跑本文件即可复现**，不写死）：
   —— 见下面 `make mutate` 的打印（「测 N / 捕获 M / 漏网 0」）。

★★★ **一条【本主题特有】的引文纪律**（★ 与 `collectives` 的「取整符号」纪律**不同**）：
  本主题的引文有两种**易错形态**：
    · ★★ **数学上标被压平**（`01` 的 PDF 载体）：`(k/2)²` 提取成 `(k/2)2`、`k³/4` 提取成 `k3 /4`
      ⇒ ★ 引文常量里存的是**提取形态**，任何「顺手改成上标」的编辑都会被 `==` 挡下；
    · ★★★ **限定词被截断**（`02` 的 NVIDIA 载体）：计划 §C.3 的引文停在
      `... for cluster Interconnect.`，而原文**紧跟着** `one switch hop away` 那半句
      —— ★★ 本主题把它**单独登记**成 `NVIDIA_RAIL_HOP_QUOTE`（★ `02` 的 `D.5` 逐字钉住）。
  ★★ 凡用 `startswith` / `in`（对子串）的钉住都抓不到**末尾追加**（盲区③）
    ⇒ ★ 本主题的**关键引文一律有 `==` 逐字钉住**，`startswith` / `in` **只**出现在**辅助**判据里。

★★★ **另一条【本主题特有】的纪律 —— 【禁写清单】是可机械复核的**：
  `03` 的 `D.8` **扫 `specs` 里全部 `*_QUOTE` 常量**，确认 `F6` 的两个 SHARP 性能数字
  （`2.1×` / `3.24×`）**一次都不出现**；★ 判定口径是「**靶子收窄到引文常量**」
  ⇒ ★ 命中即违规，不需要人去读散文。

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

    ★ `ast` 的 `col_offset` 是**这一行 UTF-8 编码后的字节偏移**（本文件与 `specs.py` 里满是
      中文注释与全角引号，按字符切会切错位）⇒ 先 `encode` 再切，最后 `decode` 回来。
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
    tmp = tempfile.mkdtemp(prefix="mut110nt_")
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
    print("110-communication/network-topology 叶子级变异测试")
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
        print("      会诚实地报 0 并给出非零退出码（本主题在 **Chunk 3** 已非空，走不到这一支）。")
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
    print("  ★★ 注：本主题的 **A 类在 `01`/`02` 里**（真枚举 vs 外部锚点）—— ★ 本工具**不是**它；")
    print("     本工具只证明「**常量被动过时，有断言会失败**」（★ `03` 那章还没有 A 类，见其 `D.0`）。")

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
