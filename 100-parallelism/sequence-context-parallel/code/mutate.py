#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
★ 本文件**从 `100-parallelism/pipeline-parallel/code/mutate.py` 逐字复制**（**工具逻辑本身是对的**，
  且与域内其余主题一致）。
★★★ **本主题对正文的改动**（**只有三处**，逐条列在这里，便于复核 —— 这是本库
  「抬头必须与事实相符」的落法）：
  ① 顶部标题行：`100-parallelism/pipeline-parallel 叶子级变异测试`
     → `100-parallelism/sequence-context-parallel 叶子级变异测试`；
  ② 临时目录前缀：`mut100pp_` → `mut100scp_`（★ 防止两个主题的临时目录撞名）；
  ③ 空分母分支里的一句：`（本主题在 **Chunk 3** 已非空，走不到这一支）`
     → `（本主题在 **Chunk 4** 已非空，走不到这一支）`。
  ★★ 除这三处与**本抬头**之外，正文**逐字相同**。
  ★★★ **为什么必须重写抬头**：`pipeline-parallel` 的抬头里那一整套陈述（分母 `85`、
    `specs.py` 有 `28` 个模块级赋值、它的三个程序名、它的常量名、它的 `D.13`/`D.10`、
    「本主题的标记含 `/`」、「本主题连 `numpy` 都不用」……）**在本主题全部为假** ——
    ★ 上一轮（`pipeline-parallel`）就是因为抬头里**整段 TP 的陈述留下**而被判「与事实不符」。
    本文件里所有关于「本主题」的数，都是**用本文件自己的 `enumerate_leaves()` 实测出来的**
    （见下面「本主题的实测口径」一节）。

===============================================================================
★ Chunk 4 的状态：**分母非空**
===============================================================================

本主题的三个编号程序（`01_activation_sharding` / `02_ring_attention` / `03_constraints`）
随 **Chunk 4** 落盘（提交日 **2026-10-08**），所以本工具**不走空分母分支**
（那条分支仍然留着：`EXIT_NO_TARGET = 3`，它的验收在 Chunk 0 已完成 ——
「工具在还没有对象时它会诚实地报 0」）。

★ 本主题**特别需要这道闸的四条理由**（每一条都对应三个程序里的一处真实风险）：
  · ★★★ **`01` 的 A 类是「恒等式核对」，不是「实现 vs 参照」**：它的三式（式(1)/(2)/(4)）
    **都是从 `specs.py` 的引文抄来的闭式** ⇒ 判据**对转写错误敏感、对物理错误不敏感**。
    ⇒ ★ 于是「**把常量改一格，程序会不会报错**」这件事在本主题**格外要紧**：
    `specs.py` 里的每一条引文与每个系数都必须**真的被断言消费**，否则那三条 A 类
    只是**印在纸上的公式**。★ 本机实测 `81/81` 一条也漏不掉（见下）。
  · ★★ **`02`/`03` 是「分块实现 vs 全量参照」**：判据是**逐位对拍**（差落在 1e-16 档）⇒
    ★★ 只要 `full_attention()` / `causal_full()` 与分块版**共用任何一段代码**，
    对拍就退化成**自证**。本工具**看不见**这件事（它只变异字面量）⇒
    它由两个程序的 `D.8`（AST 调用表）与 `A.1g`（抽掉重缩放 ⇒ 判据失败）兜住。
  · ★★★ **`specs.py` 里有一对「同值不同指」的常量**（★ 本主题的头号陷阱）：
    `SP_SPLIT_REPLICATED = 10`（式(2) 里**不除以 `t`** 的那一项）与
    `SP_EQ2_TP_NUM = 10`（式(4) 里**被切开之后**的 `10/t`）—— **数值相同、位置不同**。
    ★★ 把它们合并成一个常量，就抹掉了「一个是复制区、一个是切开后的复制区」这个区别本身
    ⇒ `01` 的 `D.4b`/`D.6` **分别**钉住两者。
    ★ 同理 `SP_EQ1_ATTN_COEF = 5` 与 `SP_EQ2_TP_SHARD = 24`、`SP_EQ1_SBH = 34` 也是**不同判据的锚**。
  · ★★ **字符串常量一大批**（逐字引文，来自 **三条来源**：`2205.05198` **10 条** +
    `2310.01889` **7 条** + Megatron `context_parallel.md` **11 条**，外加 `SP_MLSYS_INDEX_TITLE` 1 条
    ⇒ ★ 总数 **29** 条 —— **分项按 `specs.py` 每条常量的 `# 来源：` 逐条数出**，
    ★ 不是凭印象分派；★ 上一版把 `2205` 记成 11、把 CP 文档记成 10，**两个分项都错了一格**）：
    它们的变异形态是**末尾追加一个 `X`** ⇒ 凡用 `startswith` / `count` / `in`（对子串）的钉住
    **都抓不到它**（盲区③）。★★ 本主题的关键引文常量**一律有 `==` 逐字钉住**
    （`01` 的 `D.5`/`D.6`/`D.9`/`D.10`，`02` 的 `D.5`/`D.6`/`D.7`，`03` 的 `D.5`/`D.6`/`D.7`）。

★★ **本主题的实测口径**（2026-10-08，用本文件自己的 `enumerate_leaves` 数出来的）：
  分母（模块级常量叶子）**`81`** 个 —— `specs.py` 占 **`63`**，三个编号程序各占 **`6`**
  （`01` 的 `7` 个模块级赋值里，`ASSERT_FAIL_MARK` 与 `OTHER_THEME_MARKS` 的 **5** 个元素是叶子；
  `02` 是 `11` 个赋值 / `6` 个叶子；`03` 是 `10` 个赋值 / `6` 个叶子）；
  按类型 **`str` 47 / `int` 32 / `float` 2**。
  ★★ **跨行字面量 `0`** 个 —— 本主题**特意**把全部引文写成**单行字符串**
  （★ 本主题的第一版有 **14** 条用隐式拼接跨了行，实测被 `_patch()` 整条跳过 ⇒ 已逐条改成单行）。
  ★ **模块级常量的值里 `UnaryOp` `0`** 处、**`BinOp` `0`** 处（⇒ 盲区②④在本主题不构成漏网来源）。
  ★★ **本主题的模块级常量里一个 `list` / `dict` 都没有**（全是 `str` / `int` / `float` / `tuple`），
  且 `specs.UNSOURCED` **留空**（`()`）⇒ `_walk()` 的 `Dict` 分支（本库 T29 修过的那一支）
  **在本主题没有对象**。

⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
  不是本脚本的（1 / 2 / 3）⇒ **要看退出码就直接 `python3 mutate.py`**（计划 §4 的 T11）。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对每一个叶子级常量逐个问一遍，
答「不会」的（**漏网**）如实列出来。★ 本主题的断言多是
「**数值恒等式**」（`01` 的三式）与「**逐位对拍**」（`02`/`03` 的分块 vs 全量）
⇒ **极易写成无条件成立的伪断言**；而且 `specs.py` 里凡**没有消费者**的常量
（「照抄但不使用」）都会在这里现形 —— 计划 §4 **T11 的盲区①** 说的大就是这件事。
★★ 本主题还有一处**本工具【看不见】的**风险，必须在这里写明：
  · `01` 的 `C.1`/`C.2` 与 `02` 的 `C.1`、`03` 的 `C.1`/`C.2`/`C.3` 断言的**不是「一致」而是「错」**
    （「只切一半」/「切错轴」/「不做重缩放给 `nan`」/「局部位置」）——
    ★★ 那些**错法本身是一段代码**（`eq4_partial_ln()` / `eq4_hidden_split()` /
    `ring_attention(rescale=False)` / `causal_blockwise(local_pos=True)`），
    **不是模块级常量** ⇒ 本工具**不会**看见它们。
    ★ 这是**对的**：那一类判据的价值不在「能失败」，而在**它把「这个错法真的会错」钉住了**；
      而它们的**可证伪性由它们自己的守卫/对照保证**（见下面「代码级抽查」）。

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
   ★★ **本主题的标记是 `{{ FAIL: seq-context-parallel`**（★ 本域六个标记**两两不互为前缀**：
     `%%` / `^^` / `==` / `{{` / `||` / `??` ⇒ 跨主题照抄会被 `D.1`/`D.2`/`D.2b` 当场抓住）。
     ★ 它**只允许出现在编号程序的 `print()` 字符串里**；本文件的 `read_fail_mark()`
     是把它从编号程序里**读出来比较**（AST 读取），**本文件不写它的字面量**。

4. **对照组必须有。**把某条常量**改回它自己**，harness **不得**报告「捕获」。

===============================================================================
★ 口径（分母）
===============================================================================

  分母 = **本目录全部编号程序**（`NN_*.py`）**+ `specs.py`** 的**模块级常量叶子**。
  **不含**断言调用里的字面量：那些是**期望值**（测试自己的一部分），
  改它们必然「失败」，把这种必然算进分子等于给分母灌水。
  ★★ **本主题的期望值确实住在断言里**（例如 `01` 的 `B.1` 里那四个收益比
  `1.1039` / `1.2226` / `1.0533` / `1.1142`、`03` 的 `B.1` 里那个 `104`）——
  ★ 这是**故意的**：它们是**推导值**（由闭式算出），不是从任何来源抄来的规格值
  ⇒ 按 `CONTRIBUTING` §2「只放从来源抄来的数，**不放推导值**」，它们**不应该**进 `specs.py`。

★★ **driver 规则（本主题的 `specs.py` 依赖它）**：变异体若在**辅助模块**（`specs.py`）里，
  就**遍历全部编号程序、任一捕获即算捕获**（见下面 `main()` 里的 **R17** 注释）。
  ★ 若写成「只由第一个编号程序检验」，`specs.py` 的变异会**整体免检** ——
  `80` 实测过 **44/44 全部漏网**。★★ **本主题的 `specs.py` 有 `46` 个模块级赋值**
  （实测：`enumerate_leaves` 数出 `81` 个叶子，其中 `specs.py` 占 `63`），
  且**三个程序都 `import specs`** ⇒ 这条规则在本主题是**必需的**，不是保险。

★★ **本工具的已知盲区**（计划 §3 开头列了三条，本主题三条都会碰到）：
  ① **只变异字面量、不变异代码**：翻转一个**结构选择**不会改任何常量的值 ⇒ 本工具**看不见**它。
     ⇒ **本主题做了一次代码级抽查**（2026-10-08，在 `/tmp` 的整目录副本上做、跑完逐字还原），
     **实测结果**（**七个样本**；★ 这些落点全部是**本主题**的函数）：
       · `01` 把式(2) 的 `10.0 + 24.0/t` 改成 `10.0/t + 24.0/t`（即「把复制项也切了」）
         ⇒ **断言捕获** `A.2`/`B.1`/`B.2`、**没有崩溃**；
       · `01` 把式(1) 的 `34.0` 改成 `35.0` ⇒ **断言捕获** `A.1`/`A.3`/`C.2`、**没有崩溃**；
       · `02` 把 `acc * alpha + e @ vb` 里的 `acc * alpha` 改成 `acc`（不缩回累积量）
         ⇒ **断言捕获** `A.1`/`C.1g`、**没有崩溃**；
       · `02` 把参照 `full_attention()` 的 `scores.max(axis=1)` 改成 `axis=0`
         ⇒ **断言捕获** `A.1`/`A.2`/`C.1`/`C.1g`、**没有崩溃**；
       · `03` 把 `causal_blockwise()` 的默认 `local_pos=False` 改成 `True`
         ⇒ **断言捕获** `A.1`/`C.3`、**没有崩溃**；
       · `03` 把 `_step(..., guard)` 里的 `guard` 写死成 `False`（不兜底 `-inf`）
         ⇒ **断言捕获** `C.3`、**没有崩溃**；
       · `03` 把掩码条件 `kpos <= qpos` 改成 `kpos < qpos`（把对角线也挡掉）
         ⇒ **断言捕获** `A.1`/`A.2`/`C.1`/`C.3`、**没有崩溃**
         （★ `A.1` 也在这里被捕获 —— **前提是 `A.1` 带那条 `isfinite` 护栏**：
           四档的差全变成 `nan`，而 `max(0.0, nan) == 0.0` 会把 `nan` **吞掉**）。
     ⇒ ★★★ **这七个样本顺带把「每条 A 类都有守卫」变成可复核的**：
       `01` 的 `A.1`/`A.3`、`02` 的 `A.1`、`03` 的 `A.1` —— **每一条**都有一个**代码级**改错样本
       能让它失败（不只是常量变异）。★ 且**本轮七个样本一个挂死都没有**（与
       `pipeline-parallel` 那一轮不同：那一轮有一个样本**先挂死**、断言根本没机会跑）。
  ② **负字面量不进分母**：`-1` 是 `UnaryOp(USub, Constant(1))`、
     `_mutate_value()` 面对它返回 `None`（计入「跳过」）。★ **本主题的模块级常量值里
     `UnaryOp` 计 `0` 处**（实测）—— `−inf` 这类写法只出现在**函数体内**
     （`np.full((blk, 1), -np.inf)` 之类），而 `_walk()` **只看模块级常量的值**
     ⇒ 这一条在本主题**不构成漏网来源**（★ 它由 `03` 的 `C.3` 从**行为**上钉住）。
  ③ **前缀/包含式检查会漏网**：变异是**在字符串末尾追加一个 `X`** ⇒
     凡用 `startswith` / `count` / `in`（对子串）的钉住都抓不到它 ⇒ **关键常量一律有 `==` 钉住**。
     ★★ 本主题用 `in`（对子串）的地方恰有**两处**：`02` 的 `D.6` 与 `03` 的 `D.9`，
     都是 `"permutation invariance" in specs.RING_PERMUTATION_QUOTE`
     （★ `02` 的 `D.6` 另有 `"rescaling" in specs.RING_RESCALE_QUOTE`）——
     但**这两个常量各自另有 `==` 逐字钉住**（★ 都在 `02` 的 `D.5`：它把
     `RING_PERMUTATION_QUOTE` / `RING_RESCALE_QUOTE` / `RING_ORDER_INVARIANT_QUOTE`
     三条**逐字**钉死）⇒ 末尾追加 `X` 会立刻让它们失败（本机实测 `81/81`）。
     ★★ **订正说明**：上一版把这两处写成「`02` 的 `D.6` 与 `01` 的 `D.9`」、
     并把 `==` 钉住写成「`02` 的 `D.5`、`01` 的 `D.9`」—— ★ **`01` 的 `D.9` 用的全是 `==`**
     （它钉的是 `SP_NO_OVERHEAD_QUOTE` 那三条 SP 引文），**根本没有 `in`**
     ⇒ 两处都改成本主题真正的用 `in` 的两处（`02` 的 `D.6` / `03` 的 `D.9`），
     而 `==` 钉住的落点只有 `02` 的 `D.5`。
  ④ **`_walk()` 不下钻 `BinOp`**（本主题**不构成**漏网来源）：本主题 `specs.py` 里
     **一处幂次字面量都没有**（两个 `float` 常量 `TOL_MACHINE` / `RING_AMPLIFY` 都是十进制字面量），
     `03` 的补齐算的是 `-(-seq // cp)`（在**函数体里**）⇒ 模块级常量的值里 `BinOp` 计 **`0`** 处。
  ⑦ **跨行字符串被整条跳过**（★ 本主题**已消掉**）：本主题的第一版有 **14** 条引文用
     隐式字符串拼接**跨了行** ⇒ `_patch()` 抛 `ValueError`、整条计入「跳过」。
     ⇒ 已**逐条改成单行字符串**（行最长的一条 `SP_NO_OVERHEAD_QUOTE` 是 196 个字符）。
     ★★ 现在实测**跨行字面量 `0`** 个、**跳过 `0`**。

===============================================================================
★★ 五条【本主题特有】的引文纪律（照 `specs.py` 的抬头，写在这里便于复核）
===============================================================================

1. **载体必须写明**：SP 与 Ring Attention 的引文载体是 **`ar5iv.labs.arxiv.org` 的正文页**，
   Megatron CP 的载体是 **GitHub raw 上的 `docs/user-guide/features/context_parallel.md`**；
   ★ 全部抓取日 **2026-10-08**。
2. **只取不含公式、不跨标记的散文**：ar5iv 把 `sbh` / `10sbh` / `t` 包在 `<math …>` 里、
   把式号包在 `<a …>` 里 ⇒ 本主题的引言**一律以 `… to:` / `… from Equation` / `… with the` 收尾**。
3. **每一条都在【声明的那个载体】上 `count()` 到 `1`**（用 `raw.count()`，不先转成文本）。
4. **不写「原文是 `X`」除非真的数到它**（本域第三代的控制者在这里出过事）。
5. **Megatron CP 是【活文档】**（无版本号、随 `main` 变动）⇒ 引用写明抓取日、**不做行号锚点**；
   ★★ 它的来源级别是 **[B]（某实现的官方文档）**，**不是 [A]（规范）**。

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
    tmp = tempfile.mkdtemp(prefix="mut100scp_")
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
    print("100-parallelism/sequence-context-parallel 叶子级变异测试")
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
        print("      会诚实地报 0 并给出非零退出码（本主题在 **Chunk 4** 已非空，走不到这一支）。")
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
