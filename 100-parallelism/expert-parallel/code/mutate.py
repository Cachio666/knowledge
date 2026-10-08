#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
★ 本文件**从 `100-parallelism/sequence-context-parallel/code/mutate.py` 逐字复制**（**工具逻辑本身是对的**，
  且与域内其余主题一致）。
★★★ **本主题对正文的改动**（**只有三处**，逐条列在这里，便于复核 —— 这是本库
  「抬头必须与事实相符」的落法）：
  ① 顶部标题行：`100-parallelism/sequence-context-parallel 叶子级变异测试`
     → `100-parallelism/expert-parallel 叶子级变异测试`；
  ② 临时目录前缀：`mut100scp_` → `mut100ep_`（★ 防止两个主题的临时目录撞名）；
  ③ 空分母分支里的一句：`（本主题在 **Chunk 4** 已非空，走不到这一支）`
     → `（本主题在 **Chunk 5** 已非空，走不到这一支）`。
  ★★ 除这三处与**本抬头**之外，正文**逐字相同**。
  ★★★ **为什么必须重写抬头**：`sequence-context-parallel` 的抬头里那一整套陈述（分母 `81`、
    `specs.py` 有 `46` 个模块级赋值、它的三个程序名、它的常量名、它的 `D.13`/`D.10`、
    「本主题的标记含 `{`」、「本主题只有 `01` 不用 numpy」……）**在本主题全部为假** ——
    ★ 该主题的上一轮（`pipeline-parallel`）就是因为抬头里**整段 TP 的陈述留下**而被判「与事实不符」。
    本文件里所有关于「本主题」的数，都是**用本文件自己的 `enumerate_leaves()` 实测出来的**
    （见下面「本主题的实测口径」一节）。

===============================================================================
★ Chunk 5 的状态：**分母非空**
===============================================================================

本主题的三个编号程序（`01_expert_sharding` / `02_load_imbalance` / `03_ep_and_other_axes`）
随 **Chunk 5** 落盘（提交日 **2026-10-08**），所以本工具**不走空分母分支**
（那条分支仍然留着：`EXIT_NO_TARGET = 3`，它的验收在 Chunk 0 已完成 ——
「工具在还没有对象时它会诚实地报 0」）。

★ 本主题**特别需要这道闸的五条理由**（每一条都对应三个程序里的一处真实风险）：
  · ★★★ **`01` 的「恒真性自检」是本主题的头号陷阱**：本域最自然的 A 类判据
    「EP 切分 == 单卡」在本组形状下 `max|diff|` **恰为 `0.000e+00`**（纯索引搬运）⇒
    **恒真**。程序把它降级成 `D.4` 的**报告**（★ 与 `C.1`/`C.2` 的 `2.47`/`2.81` **并列**），
    而把 A 类钉在**整数划分**（`A.1`/`A.2`）上。★ 本工具**看不见**「哪条判据是恒真的」
    （它只变异字面量）⇒ 这件事由 `01` 的 `D.4` 与 `C.1`/`C.2` 的**对照**兜住。
  · ★★ **`02` 的两类判据写法不同（恒等式 vs 统计量）**：`A` 类（发出量）是**恒等式**、
    `B` 类（接收量）是**统计量**（报八个种子的均值与范围）⇒ ★★ 于是
    **`specs.EP_IMB_SEEDS` 那八个种子必须被 `==` 逐元素钉住** —— 否则「换掉种子」
    将没有任何断言能察觉（★★ 这与同域 `sequence-context-parallel`【不把种子放进 `specs.py`】
    的处置**相反**，两边都对，理由不同：那边的种子对任何断言都没有可观察后果）。
  · ★★★ **`03` 的 A 类把两个条件写进【同一条】**（`tp = 1` 恰为 `0.0` **且** `tp > 1` 严格 `> 0`）——
    ★★ 少了后半条，前半条就是**恒真**的伪断言。★ 这条**不是**常量级的事，
    本工具看不见 ⇒ 由 `03` 的 `A.1g`（TP 少加几片 ⇒ 差跳到 `1e+0` 档）**行为级**兜底。
  · ★★ **字符串常量一大批**（逐字引文，来自 **三条来源**）—— ★★★ **本节在 Chunk 5 修过一次，
    而这是【同一缺陷的第二次复发】**（★ Chunk 4 已被判过「引文的分项计数错一格」）：
    ★ **实测（用本文件自己的 `top_consts()`，2026-10-08）**：
      · **字符串**口径 = `2101.03961` **7 条** + `2006.16668` **2 条** + Megatron README **4 条**
        ⇒ **13 条**；★ 另加 `EP_NOT_FOUND_NOTE`（**[E] 本库的否定性记录**）**1** 条
        ⇒ ★★ `specs.py` 的**字符串**常量共 **14** 个模块级名字（★ 与本行的 `14` **自洽**）；
      · **全部常量**（把整数计数也算进去）口径 = Switch **9** + GShard **4** + Megatron **4** ⇒ **17**。
    ★★★ **原文写的是「`8 条` + `4 条` + `4 条` ⇒ 总数 `16` 条」、再「另加 `1` 条 ⇒ 共 `12` 个名字」——
      那一组数【自相矛盾】**（`16 + 1 ≠ 12`），且分项也数错了（把四个**整数**计数常量
      `SWITCH_ALLTOALL_COUNT` / `SWITCH_HYPHENATED_VERB_COUNT` / `GSHARD_ALLTOALL_COUNT` /
      `GSHARD_HYPHEN_COUNT` 当成了字符串）。★ **上面那两行才是逐条数出来的**。
    它们的变异形态是**末尾追加一个 `X`** ⇒ 凡用 `startswith` / `count` / `in`（对子串）的钉住
    **都抓不到它**（盲区③）。★★ 本主题的关键引文常量**一律有 `==` 逐字钉住**
    （`01` 的 `D.5`/`D.6`/`D.9`/`D.10`，`02` 的 `D.5`/`D.6`，`03` 的 `D.5`/`D.8`/`D.9`）。
  · ★★ **本主题有一条【否定性】记录**（`EP_NOT_FOUND_NOTE`）：本库在任何载体上**没有取到**
    「专家数与 EP 度之间的大小关系」这类规则。★ 它**有一个真消费者**（`03` 的 `D.9`）⇒
    它**不是**一条无消费者的漏网候选 —— ★★ 这一点**必须写明**，否则它看起来就像
    「照抄但不使用」那种典型漏网（计划 §4 **T11 的盲区①**）。

★★ **本主题的实测口径**（2026-10-08，用本文件自己的 `enumerate_leaves` 数出来的）：
  分母（模块级常量叶子）**`74`** 个 —— `specs.py` 占 **`56`**，三个编号程序各占 **`6`**
  （三个程序都是 `7` 个模块级赋值 / `6` 个叶子：`ASSERT_FAIL_MARK` 是 1 个叶子、
  `OTHER_THEME_MARKS` 的 5 个元素是 5 个叶子，而 `FAILS = []` 是**空 list**（0 个叶子）、
  `MARKS_USED = set()` / `_TOTAL` / `_PASS` 是 **`Call`**（0 个叶子）、
  `TOL_MACHINE = specs.TOL_MACHINE` 是 **`Attribute`**（0 个叶子））；
  按类型 **`str` 32 / `int` 41 / `float` 1**。
  ★★ **Chunk 5 的变化**：分母 `72 → 74`（`specs.py` `54 → 56`）、`str` `30 → 32` ——
    原因是新登记了两条《Switch §2.2 丢弃率》常量（`SWITCH_DROPPED_RATE_QUOTE`/`_ALT`），
    修的是本主题原来那条**假否定**（见本文件「引文纪律」与 `specs.py` 抬头）。
  ★★ **跨行字面量 `0`** 个 —— 本主题**特意**把全部引文写成**单行字符串**。
  ★ **模块级常量的值里 `UnaryOp` `0`** 处、**`BinOp` `0`** 处（⇒ 盲区②④在本主题不构成漏网来源）。
  ★★ **本主题的模块级常量里一个 `dict` 都没有**；`list` 有 **3** 个，
  但**全是空的**（`FAILS = []`，三个程序各一）⇒ **贡献 0 个叶子**；
  `specs.UNSOURCED` **留空**（`()`，0 个叶子）⇒ `_walk()` 的 `Dict` 分支
  （本库 T29 修过的那一支）**在本主题没有对象**。

⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
  不是本脚本的（1 / 2 / 3）⇒ **要看退出码就直接 `python3 mutate.py`**（计划 §4 的 T11）。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对每一个叶子级常量逐个问一遍，
答「不会」的（**漏网**）如实列出来。★ 本主题的断言多是
「**整数划分算术**」（`01` 的 `A.1`/`A.2`）、「**计数恒等式**」（`02` 的 `A.1`）与
「**逐位对拍**」（`03` 的 `A.1`：`tp = 1` 恰好 `0.0`、`tp > 1` 几 ulp）
⇒ **极易写成无条件成立的伪断言**；而且 `specs.py` 里凡**没有消费者**的常量
（「照抄但不使用」）都会在这里现形 —— 计划 §4 **T11 的盲区①** 说的大就是这件事。
★★ 本主题还有一处**本工具【看不见】的**风险，必须在这里写明：
  · `01` 的 `C.1`/`C.2`、`02` 的 `A.1g`/`A.2`、`03` 的 `A.1g`/`C.1` 断言的**不是「一致」而是「错」**
    （「专家放错」/「只搬一半」/「把发出量当接收量」/「设备编号换成跨步」/「TP 少加几片」/
    「组划分错位」）——
    ★★ 那些**错法本身是一段代码**（`sharded(weight_shift=1)` / `sharded(combine=False)` /
    `device_of_expert()` 换个算式 / `ep_tp(tp_parts=1)` / `ep_group_strided()`），
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
   ★★ **本主题的标记是 `|| FAIL: expert-parallel`**（★ 本域六个标记**两两不互为前缀**：
     `%%` / `^^` / `==` / `{{` / `||` / `??` ⇒ 跨主题照抄会被 `D.1`/`D.2`/`D.2b` 当场抓住）。
     ★ 它**只允许出现在编号程序的 `print()` 字符串里**；本文件的 `read_fail_mark()`
     是把它从编号程序里**读出来比较**（AST 读取），**本文件不写它的字面量**。
     ★★ 注意那个 `||` 在 **`Makefile` 的注释里**被转义写成 `\|\|`（★ 域 README 同）——
     本文件与编号程序里都是**未转义**的两根竖线。

4. **对照组必须有。**把某条常量**改回它自己**，harness **不得**报告「捕获」。

===============================================================================
★ 口径（分母）
===============================================================================

  分母 = **本目录全部编号程序**（`NN_*.py`）**+ `specs.py`** 的**模块级常量叶子**。
  **不含**断言调用里的字面量：那些是**期望值**（测试自己的一部分），
  改它们必然「失败」，把这种必然算进分子等于给分母灌水。
  ★★ **本主题的期望值确实住在断言里**（例如 `01` 的 C 类那两条 `2.472807` / `2.811856`、
  `03` 的 B 类那几个条目数 `64` / `128` / `256`）——
  ★ 这是**故意的**：它们是**推导值 / 本机实测值**，不是从任何来源抄来的规格值
  ⇒ 按 `CONTRIBUTING` §2「只放从来源抄来的数，**不放推导值**」，它们**不应该**进 `specs.py`。

★★ **driver 规则（本主题的 `specs.py` 依赖它）**：变异体若在**辅助模块**（`specs.py`）里，
  就**遍历全部编号程序、任一捕获即算捕获**（见下面 `main()` 里的 **R17** 注释）。
  ★ 若写成「只由第一个编号程序检验」，`specs.py` 的变异会**整体免检** ——
  `80` 实测过 **44/44 全部漏网**。★★ **本主题的 `specs.py` 有 `31` 个模块级赋值**
  （实测：`enumerate_leaves` 数出 `74` 个叶子，其中 `specs.py` 占 `56`），
  且**三个程序都 `import specs`** ⇒ 这条规则在本主题是**必需的**，不是保险。

★★ **本工具的已知盲区**（计划 §3 开头列了三条，本主题三条都会碰到）：
  ① **只变异字面量、不变异代码**：翻转一个**结构选择**不会改任何常量的值 ⇒ 本工具**看不见**它。
     ⇒ **本主题做了一次代码级抽查**（2026-10-08，在 `/tmp` 的整目录副本上做、跑完逐字还原），
     **实测结果**（**九个样本**，★ 每一条都是**改一处、真跑一次**记下来的；
     ★★ 这些落点全部是**本主题**的函数）：
       · `01` 把 `part_C` 里 `C.1` 那一次的 `weight_shift=1` 改成 `0`（即「不错位」）
         ⇒ **断言捕获** `C.1`、**没有崩溃**；
       · `01` 把 `part_C` 里 `C.2` 那一次的 `combine=False` 改成 `True`（即「搬全」）
         ⇒ **断言捕获** `C.2`、**没有崩溃**；
       · `01` 把 `experts_per_device()` 的 `E // EP` 改成 `-(-E // EP)`（换成 `ceil` 口径）
         ⇒ **断言捕获** `A.1`、**没有崩溃**；
       · `02` 把 `part_A` 里 `A.1` 的 `m.sum(axis=1)`（**发出量**）改成 `axis=0`（**接收量**）
         ⇒ **断言捕获** `A.1`、**没有崩溃**；
       · `02` 把 `route_topk()` 的 `[:, :top_k]` 改成 `[:, :1]`（每 token 只选 1 个专家）
         ⇒ **断言捕获** `A.1`、**没有崩溃**；
       · ★ `02` 把 `device_of_expert()` 的 `e * EP // E` 改成 `e % EP`（**跨步**编号，
         每设备专家数不变、`A.1` 仍过）⇒ **断言捕获** `A.2`、**没有崩溃**
         （★★ 这是 Chunk 5 新加的那条判据 `A.2` 的样本；★ 改前它**全部通过**：实测 15 条断言全过）；
       · `03` 把 `ep_tp()` 的 `tp_parts`（默认 `None` ⇒ 用**全部** `TP` 片）改成硬写 `1`
         ⇒ **断言捕获** `A.1`、**没有崩溃**；
       · `03` 把 `data_shard()` 的 `rank // EP` 改成 `rank % EP`（**错法的等价写法**：
         组内分片 id 由「相同」变成「互不相同」）
         ⇒ **断言捕获** `C.1`、**没有崩溃**；
       · `03` 把 `ep_group_contiguous()` 的 `g * EP + j` 改成 `g + j * EP`（**跨步**划分）
         ⇒ **断言捕获** `C.1`、**没有崩溃**。
     ⇒ ★★★ **这九个样本顺带把「每条 A 类都有一个【代码级】改错样本能让它失败」变成可复核的**：
       `01` 的 `A.1`、`02` 的 `A.1`/`A.2`、`03` 的 `A.1` —— **每一条**都有一个代码级样本
       （★ 不只是常量变异）。★ 且**本轮九个样本一个挂死都没有**
       （★★ 与 `pipeline-parallel` 那一轮不同：那一轮有一个样本**先挂死**、断言根本没机会跑）。
     ★★ **一处【试过但没用】的样本，记在这里免得后人重踩**：把
       **`01` 的** `device_of_expert()` 的 `e * EP // E` 直接改成 `e // (E // EP)`（换成朴素分块口径）
       **不是**断言捕获 —— 它会在 `01` 的 `part_A` 实摆那一步抛 `IndexError`
       （设备号越界）⇒ 那是**崩溃捕获**，**不算真捕获**。
       ★★ 本节采用的等价样本是上面第 3 条（改 `01` 的 `experts_per_device()` 的**取整口径**），
       它**真的**让 `01` 的 `A.1` 断言失败。
       ★★★ **别把这一条与上面那条 `02` 的样本混了**：**两个程序各有一个同名函数
       `device_of_expert()`**（本库的「同名不同指」），但**改法不同、后果也不同** ——
       `01` 改成**朴素分块** ⇒ `IndexError`（崩溃）；`02` 改成**跨步** ⇒ `A.2` 断言捕获。
  ② **负字面量不进分母**：`-1` 是 `UnaryOp(USub, Constant(1))`、
     `_mutate_value()` 面对它返回 `None`（计入「跳过」）。★ **本主题的模块级常量值里
     `UnaryOp` 计 `0` 处**（实测）—— `-(-ep // e_num)`（`ceil`）这类写法只出现在**函数体内**
     （`part_B`/`part_C` 之内），而 `_walk()` **只看模块级常量的值**
     ⇒ 这一条在本主题**不构成漏网来源**（★ 它由 `03` 的 `B.2` 从**行为**上钉住）。
  ③ **前缀/包含式检查会漏网**：变异是**在字符串末尾追加一个 `X`** ⇒
     凡用 `startswith` / `count` / `in`（对子串）的钉住都抓不到它 ⇒ **关键常量一律有 `==` 钉住**。
     ★★ 本主题用 `in`（对子串）的地方**一处都没有** —— 全部关键引文常量都是 `==` 逐字钉住
     （★ 与同域 `sequence-context-parallel` **不同**：那边有两处 `in`，靠「另有 `==` 钉住」兜底）。
     ★★ **唯一一处非 `==` 的检查**是 `01` 的 `D.10` 里 `specs.EP_NOT_FOUND_NOTE != ""`
     —— ★ 它**抓不到**末尾追加 `X`；★ 但**同一个常量另有 `==` 逐字钉住在 `03` 的 `D.9`**
     （★ 而 `specs.py` 的变异会**遍历全部编号程序**）⇒ 末尾追加 `X` 会立刻让 `03` 的 `D.9` 失败。
  ④ **`_walk()` 不下钻 `BinOp`**（本主题**不构成**漏网来源）：本主题 `specs.py` 里
     **一处幂次字面量都没有**（唯一的 `float` 常量 `TOL_MACHINE` 是十进制字面量），
     模块级常量的值里 `BinOp` 计 **`0`** 处。
  ⑦ **跨行字符串被整条跳过**（★ 本主题**没有**这个问题）：实测**跨行字面量 `0`** 个、**跳过 `0`**。

===============================================================================
★★ 六条【本主题特有】的引文纪律（照 `specs.py` 的抬头，写在这里便于复核）
===============================================================================

1. **载体必须写明**：Switch Transformer 与 GShard 的引文载体是
   **`ar5iv.labs.arxiv.org` 的正文页**；同行评审的依据载体是 **`arxiv.org/abs/<id>` 的 abs 页**；
   Megatron MoE 的载体是 **GitHub raw 上的 `megatron/core/transformer/moe/README.md`**
   （★★ **不是** `docs/user-guide/features/moe.md` —— 那一份 **663 字节**、只 `{include}` 这一份）；
   ★ 全部抓取日 **2026-10-08**。
2. **只取不含公式、不跨标记的散文**：ar5iv 把 `[E, C, d_model]` 包在 `<math …>` 里；
   ★★ **附录 F 的伪码更狠 —— 它把每一个词都切成独立的 `<span>`** ⇒
   `All-to-All communication.` 这个串在原始字节上**计到 `0`** ⇒ 本模块**只登记 `All-to-All` 这个词本身**。
3. **每一条都在【声明的那个载体】上 `count()` 到它写的次数**（用 `raw.count()`，不先转成文本）。
4. **不写「原文是 `X`」除非真的数到它**（本域第三代的控制者在这里出过事）。
5. ★★★ **「数不到」与「不存在」是两件事**（Chunk 5 补的一条**反向**纪律）——
   `count() == 0` **只能**得出「**在这个载体上、用这个片段**数不到」；
   ★★ **要断言「不存在」，必须换锚点、换窗口、换载体再试**。
   ★ 本主题踩过：用 `capacity` 当锚点在两篇正文里数 `%` 各得 0 次 ⇒ 据此写下
   「本库在任何载体上都没有取到任何丢弃率数字」—— ★★★ **那句话是假否定**：
   Switch **§2.2** 逐字就有 `… the number of tokens dropped (typically <1%)`
   （★ 锚点应是 **`dropped tokens`**；`capacity` 距它附近的 `%` **694 字节**）。
   ⇒ 现在登记为 `SWITCH_DROPPED_RATE_QUOTE` / `SWITCH_DROPPED_RATE_ALT`（`02` 的 `D.5` 钉住）。
6. **Megatron MoE 是【活文档】**（无版本号、随 `main` 变动）⇒ 引用写明抓取日、**不做行号锚点**；
   ★★ 它的来源级别是 **[B]（某实现的官方文档）**，**不是 [A]（规范）**；
   ★★★ 而且它讲的是 **`EP` 与 `DP`** 的关系，**不是**「专家数与 EP 度之间的大小关系」——
   后者**在任何载体上都没取到**（`specs.EP_NOT_FOUND_NOTE`）。

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
    tmp = tempfile.mkdtemp(prefix="mut100ep_")
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
    print("100-parallelism/expert-parallel 叶子级变异测试")
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
        print("      会诚实地报 0 并给出非零退出码（本主题在 **Chunk 5** 已非空，走不到这一支）。")
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
