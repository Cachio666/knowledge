#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
★ 本文件**从 `100-parallelism/data-parallel/code/mutate.py` 逐字复制**（而那一个是
  `90-training/optimizers/code/mutate.py` 的逐字副本），只改**四处**：
  **主题名**、**临时目录前缀**（`mut100pp_`）、**抬头里本主题特有的那一节**、
  以及**抬头里若干条只对 data-parallel 成立的断言**（★ 那几条**必须**改 ——
  留着它们就是**与事实不符的陈述**，比缺一条说明更糟）。

===============================================================================
★ Chunk 3 的状态：**分母非空**
===============================================================================

本主题的三个编号程序（`01_bubbles` / `02_schedules` / `03_balance`）已随
Chunk 2 落盘（2026-09-23），所以本工具**不再走空分母分支**（那条分支仍然留着：
`EXIT_NO_TARGET = 3`，它的验收在 Chunk 0 已完成 —— 「工具在还没有对象时它会诚实地报 0」）。

★ 本主题**特别需要这道闸的四条理由**（每一条都对应三个程序里的一处真实风险）：
  · ★★★ **本域的头号错误模式是「模拟器同时是被测对象」**（计划 §1.3.3）：三个编号程序
    **全是模拟器**（`all_reduce` 就是 `sum()`、`all_gather` 就是 `concat`；`03` 更是**纯公式**）
    ⇒ 断言**极易写成自证**。本主题的活法是「**每条 A 类都配一个把参照故意改错的目标**」——
    而那些「改错的目标」**全部是模块级常量**（阈值 `TOL_*`、形状 `B_SIZE/S_LEN/D_*`、
    `TP_GRID`、`BS_GRID`、`SHARE_LO/SHARE_HI`、`GAIN_8_16/GAIN_16_32`、`VOCAB*`）
    ⇒ ★★ **它们必须真的被断言消费**，否则「守卫」本身会在变异测试里现形。
  · ★★★ **`specs.py` 里有一批「同值但不同项」的常量**（★ 本主题的头号陷阱）：
    `RING_ALLREDUCE_VOLUME_FACTOR` 与 `DP_COMM_SATURATION_MULTIPLE` **都是 `2.0`**
    ⇒ 它们极易被当成重复项合并掉，而**合并就会抹掉「一个是单次 all-reduce 的系数、
    一个是 DP 的饱和倍数」这个区别本身**。★ 另有一批**成对存放**的常量：
    `MEGATRON_VOCAB_PAD_SIZE`（`51200`）与 `MEGATRON_VOCAB_PAD_RESULT_QUOTE`
    （同一句引文里的那个数）—— **两处都留**（一个供整除判据、一个供逐字引用）。
  · ★★ **字符串常量一大批**（逐字引文，来自 **2 篇论文**：`1909.08053` 十三条 +
    `2104.04473` 六条）：它们的变异形态是**末尾追加一个 `X`** ⇒ 凡用 `startswith` /
    `count` / `in`（对子串）的钉住**都抓不到它**（盲区③）。
    ★★ 本主题的**关键常量一律有 `==` 钉住**（★ 三个程序里另有一批 `D.5b` 用的是
    `in` / `startswith` / `count`，但**常量放在左端**、且同一批常量另有 `D.5` 的 `==` 钉住）
    —— ★ 而 `03` 的 `D.10` 还多加一条：
    **两篇论文的引文常量集合必须不相交**（防止「把 `2104` 的句子挂到 `1909` 名下」）。
  · ★★ **盲区④在本主题的危险面在【幂次】**：本主题的 `specs.py` **一处幂次字面量都没有**
    （`2^53`、`2^32` 这类写法会让整个 `BinOp` **不进变异分母**）⇒ 三个程序的
    `FLOAT_EXACT_LIMIT = 9007199254740992` **是十进制字面量**，它**进分母**。
    ★★ **本主题的实测口径**（2026-09-23，用本文件自己的 `enumerate_leaves` 数出来的）：
    分母（模块级常量叶子）**`120`** 个；其中**跨行字面量 `0`** 个（⇒ 没有因为盲区而被跳过、
    跑不进分母的）；★ **模块级常量的值里 `UnaryOp` `0`** 处（⇒ 盲区②在本主题
    **不构成漏网来源** —— `−1` 这类写法只出现在**函数体内**（切片 `[:-1]` 之类），
    而 `_walk()` **只看模块级常量的值**）。

⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
  不是本脚本的（1 / 2 / 3）⇒ **要看退出码就直接 `python3 mutate.py`**（计划 §4 的 T11）。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对每一个叶子级常量逐个问一遍，
答「不会」的（**漏网**）如实列出来。这是本域最需要的一道闸：
本主题的断言多是「**数值不变量**」（与闭式对拍、机器精度档的相等、**差恒为 `0.0`**），
**极易写成无条件成立的伪断言**；而且 `specs.py` 里凡**没有消费者**的常量
（「照抄但不使用」）都会在这里现形 —— 计划 §4 **T11 的盲区①** 说的大就是这件事。
★★ 本主题还有一处**本工具【看不见】的**风险，必须在这里写明：三个程序里有若干条
**「恒真检查」**（`01`/`02` 的 `D.10`）—— 它们**故意**断言某个差是 `0.0`。
本工具**不会**把它们判成漏网（那些 `0.0` 是**断言里的字面量**、不在分母里，
而 `specs.TP_BOUNDARY_REF` 之类的常量另有 `D.5` 钉住）—— 这是**对的**：
恒真检查的价值不在「能失败」，而在**它把「这条判据没有信息量」这件事本身钉住了**。

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
   本库纪律是**每个主题自定义**失败标记（`30` 的坑之一：不跨主题照抄）。
   写死一个值，等于把「跨主题照抄」从程序里搬到了工具里。
   ★ 读法有两条来自 `50-performance` 的实测教训：
     · 那个常量必须是**普通字符串字面量**（不是 `re.compile(...)`），否则读出来是 `None`；
     · 判定必须用「**行首**锚定」的形态，不能用 `MARK in output` ——
       程序**成功时也会打印这个字面量**（标记自检那一条会把它打出来），
       于是 `in` 在 rc=0 时也成立，整个变异测试会**拒绝给结论**。
   ★★ **本主题的标记里含 `^`**（`^` 是 shell / make 的元字符；本域六个主题的标记
     **各含一个**这样的字符，计划 §1.4 的两条纪律）
     ⇒ ★ **它只允许出现在编号程序的 `print()` 字符串里**；本文件的 `read_fail_mark()`
     是把它从编号程序里**读出来比较**（AST 读取），**本文件不写它的字面量、
     也不把它写进任何命令行**。

4. **对照组必须有。**把某条常量**改回它自己**，harness **不得**报告「捕获」。

===============================================================================
★ 口径（分母）
===============================================================================

  分母 = **本目录全部编号程序**（`NN_*.py`）**+ `specs.py`** 的**模块级常量叶子**。
  **不含**断言调用里的字面量：那些是**期望值**（测试自己的一部分），
  改它们必然「失败」，把这种必然算进分子等于给分母灌水。

★★ **driver 规则（本主题的 `specs.py` 依赖它）**：变异体若在**辅助模块**（`specs.py`）里，
  就**遍历全部编号程序、任一捕获即算捕获**（见下面 `main()` 里的 **R17** 注释）。
  ★ 若写成「只由第一个编号程序检验」，`specs.py` 的变异会**整体免检** ——
  `80` 实测过 **44/44 全部漏网**。★★ **本主题的 `specs.py` 有 `41` 个模块级赋值**
  （实测：`enumerate_leaves` 数出 `120` 个叶子，其中 `specs.py` 占绝大部分），
  且**三个程序都 `import specs`** ⇒ 这条规则在本主题是**必需的**，不是保险。

★★ **本工具的已知盲区**（计划 §3 开头列了三条，本主题三条都会碰到）：
  ① **只变异字面量、不变异代码**：翻转一个**结构选择**（例如把 `mlp_pair` 里的
     `first == "col"` 改成 `"row"` —— 那会让**配对约束的整张表**换个样子；
     或者把 `softmax_masked` 里的 `-np.inf` 去掉）不会改任何常量的值
     ⇒ 本工具**看不见**它。⇒ **本主题另做了一次代码级抽查**（2026-09-23，在 `/tmp` 的
     整目录副本上做、跑完逐字还原并复核基线仍为 `rc=0`），**实测结果**：
       · `01` 把 `mlp_pair` 里 `(col,row)` 那一格的 `comm_total += 1` 改成 `+= 2`
         ⇒ **断言捕获 `A.3` 与 `B.2`**（配对表与每层次数都不再成立）、**没有崩溃**；
       · `01` 把 `all_reduce_sum` 改成 `sum(parts[:-1])`（漏掉最后一片）
         ⇒ **断言捕获 `A.1` / `A.3` / `A.4`**、**没有崩溃**；
       · `02` 把 `softmax_masked` 的屏蔽去掉 ⇒ **断言捕获 `C.3`**；
       · `03` 把 `structural_bytes` 的 `TP_COMM_OPS_PER_LAYER` 换成
         `2 * MEGATRON_TP_COMM_OPS_PER_LAYER` ⇒ **断言捕获 `A.1`**；
       · `01` 把 `A.2` 的「行并行先做」那一格改成与参照**同构**（即把错法**改对**）
         ⇒ **断言捕获 `A.2`**、**没有崩溃**；
       · `02` 把 `embed_column_split_lookup` 的区间改成**闭区间**（`<=` 上界）
         ⇒ 一个 token 会落进**两片** ⇒ **断言捕获 `A.3`**（★ 这正是 `A.3` 那条「成因」判据的守卫）、
         **没有崩溃**；
       · `03` 把 `saturation_factor` 的 `(t − 1) / t` 换成 `1.0`（丢掉饱和因子）
         ⇒ **断言捕获 `A.1` / `A.2` / `B.1` / `C.2` / `D.7d`**、**没有崩溃**；
       · `02` 把 `padded_vocab_logits` 的返回值错位成 `[..., 1:]`（「取前 V 列」索引错位）
         ⇒ **断言捕获 `A.1` / `A.2` / `B.1`**（★ 且**同时**触发了一个下游崩溃 ——
         两类捕获都出现了，按纪律 2 分开记）。
     ★★ **一处【没有捕获、而是崩溃】的样本（也要如实记下）**：把 `mlp_pair` 的
     `if first == "col"` 改成 `if first == "row"` ⇒ `rc=1` 但**没有任何失败标记**，
     是 `ValueError`（形状不匹配）—— 按本文件纪律 2，它算**崩溃捕获**，**不算**真捕获。
     ⇒ ★ **这条结论要写进报告**：**代码级变异里有一部分会先崩，断言根本没机会跑**
     （★ 八个代码级样本里，**六个是纯断言捕获、一个是纯崩溃、一个是两者都有**）。
     ★★★ **而这八个样本顺带把「每条 A 类都有守卫」变成可复核的**：
     `01` 的 `A.1`/`A.2`/`A.3`/`A.4`、`02` 的 `A.1`/`A.2`/`A.3`、`03` 的 `A.1`/`A.2` ——
     **每一条**都有一个**代码级**的改错样本能让它失败（不只是常量变异）。
     ★★★ **一处由这次抽查【逼出来】的修正（本库纪律：把现场留住）**：
     `01` 的 `A.2` **第一版**的「线性对照」是**自证的** ——
     `sum(parts)` 与 `sum([p for p in parts])` 是**同一个表达式**，差**恒为 `0.0`**；
     ⇒ 现行版本改成 **① 每格与独立参照比**（不两格互比）、
     **② 线性对照换成非平凡的 `φ = 3u`**（两格互差 `8.882e-16`，**不是 `0`**）。
     ⇒ ★★★ **这条**：**「代码级抽查」不只是补漏，它能把【自证的判据】逼出来** ——
     常量变异测试**看不见**这种事（那几行里没有模块级常量）。
  ② **负字面量不进分母**：`-1` 是 `UnaryOp(USub, Constant(1))`、
     `_mutate_value()` 面对它返回 `None`（计入「跳过」）。★ **本主题的模块级常量值里
     `UnaryOp` 计 `0` 处**（实测）—— `−1` 这类写法只出现在**函数体内**（切片 `[:-1]` 之类），
     而 `_walk()` **只看模块级常量的值** ⇒ 这一条在本主题**不构成漏网来源**。
  ③ **前缀/包含式检查会漏网**：变异是**在字符串末尾追加一个 `X`** ⇒
     凡用 `startswith` / `count` / `in`（对子串）的钉住都抓不到它 ⇒ **关键常量一律有 `==` 钉住**。
     ★★ **本主题最容易犯这一条的地方**：`D.5b` 那几条「引文的作用域」检查 ——
     ★ 它们的落法是「**把整串与【固定字面量】比对、且【常量放在左端】**」
     （`CONST in "完整原文"`、`CONST.endswith("...")`）：常量当的是**被查的针**，
     不是**被搜的干草堆** ⇒ 末尾追加 `X` 会立刻让比较失败；★ 而**同一批常量另有 `D.5` 的
     `==` 逐位钉住** ⇒ 变异一条也漏不掉（本机实测 `120/120`）。

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
    ★ 找不到 / 不是字符串字面量 / 一个文件里出现多次 —— 都**以可操作的错误退出**，
      而不是回落到某个默认值（回落会把「程序没按约定写」掩盖成「工具跑通了」）。
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
                         "写成 `re.compile(...)` 之类的表达式会让读取方拿到 None，"
                         "本库已有先例（50-performance 的实施者踩过）"
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
                # ★ T29 修复（2026-09-23）：原来写的是字面量 "{key}"，**不含键名**
                #   ⇒ 一个 dict 的多个键得到**完全相同的标签**（★ 这是 `data-parallel` 在 **Chunk 1** 实测到的
                #   `_REC{key}` 在漏网清单里出现 3 次一模一样，分不出是哪个键）。
                #   与下面 `v` 的处理保持一致，用键的**实际值**。
                out += _walk(k, "{%r}%s" % (getattr(k, "value", "?"), suffix))
            out += _walk(v, "[%r]%s" % (getattr(k, "value", "?"), suffix))
    return out


def enumerate_leaves(files):
    """-> [(文件, 常量名, 后缀, 节点)]。**只取模块级常量的叶子**（口径见模块头）。"""
    out = []
    for f in files:
        text = open(os.path.join(CODE_DIR, f), encoding="utf-8").read()
        for name, value in top_consts(text, f):
            # ★ `_walk` 的初始后缀是**空串**，标签由 `name + suffix` 拼成 ——
            #   后缀里已经含层级（`[0]` / `['k']`）。把 `name` 传进 `_walk` 会让
            #   标签变成 `NAME[0]NAME`（本骨架第一次烟测就把它打出来了）。
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

    ★ 必须**整目录**复制：编号程序 `import specs`（`code/` 自包含，`CONTRIBUTING:36`），
      只复制驱动文件会让它在临时目录里 `ModuleNotFoundError` ——
      那会被 `classify()` 记成**崩溃捕获**，于是基线就「不干净」了（本骨架第一次烟测踩到）。
    `mutated` 是 `(文件名, 改写后的行列表)`：在副本上覆盖那一份。
    返回 `(rc, 合并后的输出)`。★ **仓库里的文件只被读**，一切改写都发生在临时目录。
    """
    tmp = tempfile.mkdtemp(prefix="mut100pp_")
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
    print("100-parallelism/pipeline-parallel 叶子级变异测试")
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
        print("      会诚实地报 0 并给出非零退出码（本主题在 **Chunk 2** 已非空，走不到这一支）。")
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
        #   它让**整个 `specs.py` 只被第一个编号程序检验**：若那个程序不 import specs
        #   （骨架篇就是这样），改 specs 里任何值，它的输出都**逐字节不变** ⇒
        #   `data-parallel` 在 **Chunk 1** 实测 **44/44 全部漏网**（全部在 `specs.py`）。
        #   新语义 = 「**这条变异体被本主题的测试集捕获了吗**」：
        #   遍历全部编号程序，**任一捕获即算捕获**（短路，平均只多跑一点点）。
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
