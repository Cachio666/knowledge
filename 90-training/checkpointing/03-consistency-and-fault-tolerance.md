---
title: 一致性恢复：RNG 内部状态、dataloader 位置与优化器状态
slug: consistency-and-fault-tolerance
domain: 90-training
topic: checkpointing
level: 3
difficulty: advanced
prerequisites:
  - 90-training/checkpointing/01-state-and-bytes
  - 90-training/optimizers/02-adam-and-adamw
tags: [checkpoint, determinism, rng-state, dataloader, fault-tolerance, reproducibility]
related:
  - 90-training/checkpointing/02-sharded-save-and-load
  - 90-training/mixed-precision/02-amp-execution
code:
  - code/03_consistency.py
status: stable
updated: 2026-09-23
---

# 一致性恢复：RNG 内部状态、dataloader 位置与优化器状态

> **「恢复之后轨迹逐位一致」这句话有一个必须写死的前提：恢复时只允许从文件读状态。**
> ★★★ 判据 ④ 因此是**三条路径**：
> **正确** → `True`（本机 `37` 步每一步都逐位相同，RNG 的内部状态也相同）；
> **阴性对照**（存档里**不含** RNG 状态）→ **`False`**（★ **第 1 个恢复步就分叉**）；
> **反例**（重新 `seed()` + 快进）→ **也 `True`**（★★ **它一致是因为它重建了同一份状态**，
> 不是因为它检查了存档 —— **P17**）。
> ⇒ ★★ **阴性对照那一条才是可证伪的那一半**；而「写死从文件读」必须配一条**源码级结构钉住**，
> 否则数值判据抓不住第三条。

## 0. 背景与动机

### ★★★ 先说清楚：判据 ④ 为什么必须是三条路径

计划 §3 表 A 的**原稿**写的是「**（在同一 seed 下）**恢复后与原轨迹逐位一致」——
★★ **那个括号会把实现者推向「恢复时重新 `seed(k)`」**，而那样轨迹**当然一致、
却什么都没检查**（结构层审查 **M6** 指出的 **P17**）。
⇒ 改写后的判据是：

| 路径 | 实现 | 预期 | 它证明了什么 |
|---|---|---|---|
| **1 正确** | 存档含 RNG 的**内部状态**（`bit_generator.state`）+ dataloader 位置；恢复时**只从文件读**、**不调用任何 `seed()`** | **`True`** | 「恢复」这件事**能做对** |
| **2 阴性对照** | 存档里**不含** RNG 状态 | **`False`** | ★★ **少了那 190 字节真的会改变轨迹** ⇒ 判据**不是恒真** |
| **3 反例** | 重新 `seed()` + 把存档点之前的消耗**快进**一遍 | **`True`** | ★★★ **「一致」可以由「重建」获得** ⇒ 判据必须**禁止**它（**P17**） |

★★ **本机实测（本程序 `A.1`/`A.2`/`A.3`）**：`True` / `False` / `True` ——
与控制者的**预演**（计划 §3 的预演表）逐位一致。

### ★★★ 路径 3 的工程解读：它**不是错的**（[C] PaLM 的逐字支持）

[C] PaLM（`arXiv:2204.02311`，**不确认同行评审**）§5 的 `Bitwise determinism` 一节逐字给了
**两条**做法，其中第二条是：
`a deterministic dataset pipeline where the shuffled data is written out in a random-access
format`，即 `the contents of a given training batch is only a function of the step number`。
⇒ ★★ **路径 3 就是这条做法的玩具版**（把随机性**重写成步号的函数**）。
★★ **它成立的条件是「消耗模式可重放」**（本机实测：每步恰好 `DRAWS_PER_STEP = 2` 次、
与数据无关）；★ **它不检查「存档里有没有那份状态」** —— 两件事，本文把它们分开：

| 问题 | 谁来回答 |
|---|---|
| 「恢复之后轨迹一致吗」 | 路径 1（**也**被路径 3 回答，但那不是它的功劳） |
| 「这份存档**完整**吗」 | 路径 2 + **缺键的 `strict` 检查**（§1.5） |

### ★★ 同名不同指与范围

- ★★ **本文的 `checkpoint` = 权重检查点**；「**重算换显存**」是另一件事
  （正文在 `90-training/autograd-and-backward/02-gradient-checkpointing.md`）。
  **中文词「检查点」不得用于重算**（域 README 的消歧节）。
- **本文不写**（回链目标）：**导出 / 推理格式的选择** →
  `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；
  **异步落盘 / 通信-计算重叠** → `110-communication/README.md:12`；
  **优化器状态的分片与通信量** → `100-parallelism/zero-fsdp/`；
  **「溢出检测 + 跳步」** → `90-training/mixed-precision/02-amp-execution.md`。
- ★ **一处必须写明的让步**：本文的一切数值都是**本机玩具规模的真跑**
  （`python3 3.13.5` + `numpy 2.4.4`、macOS arm64、**无 GPU**、不训练任何真模型）；
  ★ **本机没有 torch** ⇒ 涉及框架内部时序的那一条（§1.6）**只作引用判据**（**P3**）。

## 1. 原理

### 1.1 ★★★ 判据 ④ 的实验形状（照 PaLM 那句做的）

[C] PaLM 逐字：`fully bitwise reproducible from any checkpoint`、
`if the model has been trained up to step 17,000 in a single run` ——
完整句还给了机制（`we restart from checkpoint 15,000, then the training framework is
guaranteed to produce identical results in both runs from checkpoint 15,000 to 17,000`）。

本程序的同构版本（`code/03_consistency.py`）：

| PaLM | 本程序 |
|---|---|
| 一口气训到 step 17,000 | `run_reference(60)`：**一口气**跑 60 步，记下**每一步**的 `w` 字节 |
| 从 checkpoint 15,000 重启 | 在第 `SAVE_AT = 23` 步存档，从存档点继续到 60 |
| 比较 15,000 → 17,000 的每一步 | 比较 `23 → 60`（`37` 步）的每一步（**逐位**：比 `w.tobytes()`） |

★ 玩具：`D = 6`、数据集 `32` 条、批 `8`、每步注 `NOISE = 1e-3` 的随机噪声（代表 dropout / 增强）
⇒ **轨迹与随机流绑死**，所以「RNG 状态对不对」这件事**在轨迹上看得见**。

### 1.2 ★★ 三条路径的实测结果

| 路径 | 实现要点 | 与参照逐位一致 | RNG 内部状态 |
|---|---|---|---|
| **1 正确** | 从**文件**读存档；`restore_rng` 把存档里的状态**整体赋给**一个新的 `PCG64`（**不 seed**） | **`True`**（`37` 步全同） | **逐位相同** |
| **2 阴性对照** | 存档里没有 `rng_state` ⇒ 用一个**新的**发生器继续 | **`False`**（★ **第 1 个恢复步**、即第 `24` 步就分叉） | —— |
| **3 反例** | `default_rng(SEED)` + 快进 `SAVE_AT` 步的抽样 | **`True`** | ★★ **快进之后**与**存档点那一刻**逐位相同；跑完之后也与参照相同 |

★★★ **路径 3 那两列就是 P17 的量**：它**重建了同一份状态**，所以轨迹一致 ——
⇒ **数值判据无法把它与路径 1 区分开**（两条路径的**结果**一样、**来源**不同）。
★★ **这是本程序 `A.4` 存在的唯一理由**：一条**源码级结构钉住** ——
`restore_rng` 的调用表里**不得出现** `default_rng` / `seed`，且**必须**出现 `PCG64`、
状态**从存档取**（`.state = archive["rng_state"]`）。

★★ **一处必须写清的区分（本程序实测出来的）**：「重新 `seed()`」有**两个变体**：

| 变体 | 是否快进 | 数值判据抓得到吗 |
|---|---|---|
| (a) `default_rng(SEED)`，**不快进** | 否 | ★ **抓得到**（轨迹与参照**不同** ⇒ `A.1` 失败） |
| (b) `default_rng(SEED)`，**快进到存档点** | 是 | ★★ **抓不到**（轨迹**相同**）⇒ **只有结构钉住能抓** |

⇒ ★★ 「写死从文件读」这一条**必要且不可省**：
数值判据能挡住 (a)，**挡不住 (b)**。

### 1.3 ★★ RNG 的内部状态：载体与它的完整性

★★ 内部状态是**四个字段**（本机 NumPy 2.4.4 实测）——
`{'bit_generator', 'state', 'has_uint32', 'uinteger'}`，序列化后 `∈ [165, 176]` 字节。
★★★ 其中 `has_uint32`（**缓冲区里的半个 32 位字**）有两个坏法，**一个报错、一个静默**：

| 情形 | 结果 |
|---|---|
| 缺这个键 | **`KeyError: 'has_uint32'`**（NumPy 自己守卫） |
| 把它**归一化成 `0`** | ★★ **静默不一致**，**只在 32 位抽样路径上**（`integers`）；64 位路径（`random()`）照样一致 |

（三态的实验在 `90-training/checkpointing/01-state-and-bytes.md` 的 `C.1`；
★ 本文的玩具循环因此**必须**含真正用到的那个抽样路径 ——
`dataloader` 取下标走的是 **32 位**那条。）

### 1.4 ★★ 三条字节账（判据 ④ 的 B 类）

| 路线 | 它把状态放在哪 | 字节 | 随训练长度 |
|---|---|---|---|
| **A：存 RNG 的内部状态** | **检查点**里 | `190 B`（本机实测：状态本体 `175` + 键名与标点 `15`） | ★ **不变** |
| **B：把随机性重写成步号的函数**（PaLM 的第二条做法） | **数据集侧**（物化下标） | `步数 × 批大小 × 8` = `60 × 8 × 8` = **`3840 B`** | ★ **线性**（翻倍 ⇒ `7680 B`） |

★★★ **两笔账记在不同的科目上** ⇒ 说「RNG 一致性很便宜」是**跨科目的**：
路 A 便宜（常数级），路 B 的账**随训练长度增长**（而且属**数据集管线**，不属检查点大小）。
★ 路 B 的字节数是**本程序的推导值**（PaLM 只说 `random-access format`，**没给字节数**）——
已登记 `UNSOURCED`：`palm_pipeline_materialized_bytes`。

### 1.5 ★★ 缺一项：`strict` 拦住 / 不 `strict` 静默顶上

[B] PyTorch 2.14 的 DCP 文档逐字：
`allow_partial_load: If False, will raise a runtime error if a key is present in state_dict,
but not in the checkpoint.`（★ `DefaultLoadPlanner` 的默认值就是 `False`）。

本机实测两个方向：

| 读法 | 结果 |
|---|---|
| `strict=True`（等价于默认） | **报错**：`存档缺键 ['m']` |
| `strict=False` | ★ 读回**成功**，缺的 `m` 被**默认值**顶上 ⇒ **与真值不同** |

⇒ ★★ **「读回来没报错」不等于「读回来是对的」**。
★ 另有一条 [B] 逐字（`load` 的说明）：`Mismatched keys may result in hangs or errors.` ——
★★ **`hang` 比 `error` 更坏**（没有异常、没有栈）。

### 1.6 ★★ 「就地恢复」的代价：半新半旧

[B] 逐字：`It operates in place, meaning that the model should allocate its data first and
DCP uses that storage instead.` ⇒ ★★ **「写了一半」是这套接口下有可能的状态**。
本机实测（逐个字段就地写入、在第 `v` 个字段失败）：对象变成 **`w` 已经是新的、`step` 还是旧的**
—— ★★ 而它**看起来很正常**（没有异常、类型对、形状对）。

⇒ ★ **恢复必须先装配到临时对象、成功后一次性替换**（这也解释了为什么写侧要先写**临时目录**、
再靠元数据标记成功：`Write the metadata and marks the current checkpoint as successful.`）。

### 1.7 ★ 一条只在原文成立的话（**引用判据**）

[B] 逐字：`WARN: set_optimizer_state_dict can only be called before backward()`
… `or after step() is called on the optimizers.`
（原文紧跟着给的理由里 `won’t` 是**弯引号** ⇒ 按本域 §0.3 的形态 (6) 不取下半句。）
★★ **本主题不重推它**：那是**框架内部**的调用契约，**本机没有 torch、复现不了** ⇒
按 **P3**（不要求实现者复现跑不了的东西）它**只作引用判据**，逐字钉在 `C.3` 里。

## 2. 执行映射

### 2.1 存档的键清单（本程序的 `ARCHIVE_KEYS`）

| 键 | 是什么 | 少了它会怎样 |
|---|---|---|
| `w` / `m` / `v` | 参数与优化器状态（张量） | ★ 轨迹不同（`strict` 会拦住） |
| `step` | 步号 —— **Adam 的偏置校正要用它** | ★ 恢复后的更新量偏（`1 − β^t` 项不对） |
| `rng_state` | RNG 的**内部状态**（四个字段） | ★★ **第 1 个恢复步就分叉**（`A.2`） |
| （dataloader 位置） | epoch / 游标 / 顺序 | ★ 同上（本文的玩具用「每步重新取下标」的等价形式） |
| `meta` | **集合级完整性标记** | ★ 读侧必须拒绝（对应 [B] 的 `finish()` 语义） |

★ **键名要在 `preflight()` 里钉住**（本程序的 `D.2c`）：改名会让 `A.1`/`C.1`/`C.2`
**静默失去载荷**（判据还在跑、但它检查的东西已经不是原来那个了）。

### 2.2 恢复侧的清单

| 步骤 | 做什么 | 依据 / 后果 |
|---|---|---|
| 1 | **校验集合级标记**（`meta` / 全局元数据） | §1.6；[B] 的 `finish()` |
| 2 | **键集完整性**：缺项**报错**（不要默认值顶上） | §1.5 |
| 3 | **RNG 状态**：从存档**读**（**不得** `seed()`） | §1.2 的三条路径 |
| 4 | **先装配、再一次性替换**（不要就地逐个字段写） | §1.6 |
| 5 | **优化器状态**：按框架要求的**时序**恢复 | §1.7（[B] 的 `WARN`） |

### 2.3 本机实测的一组典型值

| 量 | 值（`STEPS = 60`、`SAVE_AT = 23`、`D = 6`、`NOISE = 1e-3`） |
|---|---|
| 参照与路径 1 的逐位一致 | `37/37` 步全同（且 RNG 内部状态相同） |
| 路径 2 第一次分叉 | **第 `24` 步**（= 第 1 个恢复步） |
| 路径 3 快进后的 RNG 状态 | 与**存档点那一刻**逐位相同 |
| 档案（完整 / 去掉 RNG 项） | `642 B` / `452 B` ⇒ 差 **`190 B`**（本体 `175` + 键名与标点 `15`） |
| 参数部分的字节（对照） | `12Ψ = 2064 B` ⇒ 与 RNG 项差 **`11` 倍** |
| 路 B 的数据集侧字节 | `3840 B`（翻倍训练长度 ⇒ `7680 B`） |

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| **只存参数** | 只 `state_dict()` 里的权重 | 推理够用；★ 训练**不能**从这里续（`m`/`v`/步号全丢） |
| **存参数 + 优化器状态** | 加上 `m`/`v`/步号 | 数值上能续；★★ **轨迹仍会分叉**（随机流没了） |
| **加上 RNG 的内部状态** | 存 `bit_generator.state`（~170 B） | ★★ **这才叫「一致性恢复」**（判据 ④ 路径 1）；★ 代价是**常数级的字节** + 恢复时**必须禁止 `seed()`** |
| **把随机性重写成步号的函数**（PaLM 第二条做法） | 数据按步号索引、物化 `random-access` | ★ 检查点里**不需要** RNG 状态；★★ 代价记在**数据集侧**（`步数 × 批大小 × 8`，线性） |
| **集合级标记**（元数据 / `finish()`） | 写成功之后才写元数据 | ★ 「这份检查点完整」有了**唯一**的判据；★ 代价是**多一次写**（且协调者成了单点） |

## 4. 踩坑与失败模式

**① ★★★ 用「同一 seed」当恢复的判据。**
- **现象**：判据写成「（在同一 seed 下）恢复后一致」。
- **成因**：那个括号把实现推向「重新 `seed(k)`」，而那样轨迹**当然一致**
  （本机实测路径 3 = `True`），**却什么都没检查**（**P17**）。
- **规避**：判据写死「**只允许从文件读、不得重新调用任何 `seed()`**」，
  ★ 并配一条**源码级结构钉住**（数值判据挡不住「快进版」的 `seed`）。

**② ★★ 以为「RNG 状态很便宜 ⇒ 可以省」。**
- **现象**：只存参数与优化器状态。
- **成因**：它确实只有 `190 B`（参数的 `12Ψ` 是它的 `11` 倍），
  ★ 但少了它**第 1 个恢复步就分叉**（`A.2`）。
- **规避**：把「字节大小」与「必须性」分开写。

**③ ★★ 把「重写随机性」与「恢复随机性」混成一句。**
- **现象**：说「我们的数据是确定性的，所以检查点不用存 RNG」。
- **成因**：那是**另一条路线**（PaLM 的第二条做法），它的账记在**数据集侧**（§1.4）。
- **规避**：两条路线分开写，并写明**各自的科目**。

**④ ★★ 读侧的默认值是「用默认值顶上」。**
- **现象**：缺一项照跑，结果悄悄不对。
- **成因**：`strict=False` 的语义（§1.5）。
- **规避**：缺项 ⇒ **报错**（[B] 的默认就是如此）。

**⑤ ★★ 就地恢复、中途失败。**
- **现象**：对象变成半新半旧（本机实测：`w` 新的、`step` 旧的），而它**看起来很正常**。
- **成因**：「就地」是这套接口的语义（§1.6）。
- **规避**：先装配到临时对象、成功后一次性替换。

**⑥ ★★ 不管优化器状态的恢复时序。**
- **现象**：`m`/`v` 恢复了，但后续更新不对。
- **成因**：[B] 逐字给了时序约束（§1.7）。
- **规避**：按框架要求的时序恢复；★ 本机**复现不了**这条 ⇒ 它只作**引用判据**。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 恢复后与原轨迹一致吗 | **一致**（本机 `37/37` 步逐位相同），**且 RNG 的内部状态也相同** |
| 怎么知道这条判据不是恒真 | **阴性对照**：存档里去掉 RNG 状态 ⇒ **第 1 个恢复步就分叉**（第 `24` 步） |
| 「重新 `seed()` + 快进」能通过吗 | ★★ **能**（它也 `True`，且快进后的状态与存档点逐位相同）⇒ **判据必须禁止它** |
| 怎么禁止 | **源码级钉住**：恢复函数的调用表里没有 `default_rng` / `seed`，且有 `PCG64`、状态**从存档取** |
| 数值判据能替代这条结构钉住吗 | ★★ **不能**：它挡得住「`seed` 不快进」，**挡不住「`seed` + 快进」** |
| 存档要带哪些非张量项 | **RNG 的内部状态** + 步号 + dataloader 位置（+ 调度器 / scaler）；★ 键名要在 `preflight()` 里钉住 |
| RNG 状态多大 | `190 B`（本体 `175` + 键名与标点 `15`）；★ 参数部分是它的 `11` 倍 |
| 「把随机性重写成步号的函数」花多少 | 账在**数据集侧**：`步数 × 批大小 × 8`（本机 `3840 B`，**随训练长度线性**） |
| 缺一项会怎样 | `strict`（默认）⇒ **报错**；`strict=False` ⇒ **静默**用默认值顶上、结果不同 |
| 键集不一致会怎样 | [B] 逐字 `Mismatched keys may result in hangs or errors.` —— ★ **`hang` 比 `error` 更坏** |
| 就地恢复失败会怎样 | 对象**半新半旧**（本机实测：`w` 新的、`step` 旧的），而它**看起来很正常** |
| 优化器状态能随时恢复吗 | **不能**：[B] 逐字要求 `before backward()` 或 `after step()`（★ 本机**复现不了**，只作引用判据） |
| 判据 ④ 的外部目标是什么 | [C] PaLM 逐字：`fully bitwise reproducible from any checkpoint`、`Bitwise determinism` |

> ★ 阈值与口径（`code/03_consistency.py` 的 `D.3`/`D.4` 逐位钉住）：
> `SEED = 20260955`、`D = 6`、`N_DATA = 32`、`BATCH = 8`、`STEPS = 60`、`SAVE_AT = 23`、
> `LR = 0.05`、`NOISE = 0.001`、`DRAWS_PER_STEP = 2`、`TRUE_W`（**全正数存放** ——
> `-1.0` 是 `UnaryOp`、不进变异分母）、`RNG_STATE_BYTES_MAX = 200`（实测 `175`）、
> `RNG_KEY_OVERHEAD_MIN/MAX = 8/32`（实测 `15`）、`INDEX_BYTES_PER_ID = 8`、
> `ARCHIVE_KEYS = ("m", "rng_state", "step", "v", "w")`、`META_KEY = "meta"`。
> ★ **`β1/β2/ε` 从 `specs.py` 读**（[C] 1412.6980，**已确认 ICLR 2015**）。

## 6. 小结与延伸

**一句话**：**「一致」是一个可以被伪造的结论** —— 判据必须**三条路径**一起写
（正确 / 阴性对照 / 反例），并且**把「只允许从文件读」写成源码级的结构钉住**，
因为数值判据抓得住「重新 `seed()` 不快进」，**抓不住「重新 `seed()` + 快进」**。
★ 而 **RNG 的内部状态只要 ~170 字节**（参数的 `1/11`），但**少了它，恢复从第 1 步就分叉**。
★★ 另一条路线（把随机性重写成步号的函数，PaLM 的做法）**不是错的** ——
它只是把账记到了**数据集侧**（线性于步数）。

- **状态的组成、RNG 状态的结构、`has_uint32` 的三态** ⇒ `90-training/checkpointing/01-state-and-bytes.md`。
- **分片的格式、字节账、重分片与它的一致性层次** ⇒ `90-training/checkpointing/02-sharded-save-and-load.md`。
- **Adam 的 `m` / `v` 与偏置校正** ⇒ `90-training/optimizers/02-adam-and-adamw.md`。
- **溢出的处置（检测 + 跳步）** ⇒ `90-training/mixed-precision/02-amp-execution.md`。
- **「重算换显存」** ⇒ `90-training/autograd-and-backward/02-gradient-checkpointing.md`
  （★ 与本文的「检查点」**同名不同指**）。
- **异步落盘 / 通信-计算重叠** ⇒ `110-communication/README.md:12`（**正文在彼**）。
- **优化器状态的分片与通信量** ⇒ `100-parallelism/zero-fsdp/`（**正文在彼**）。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU**、不训练任何真模型）⇒
  **不得支撑关于真实训练系统的结论**；★★ 尤其：**本机没有 torch** ⇒
  §1.7 那条框架内部的时序约束**只作引用判据**（**P3**）。

## 编译与运行

```bash
cd code
make run                     # 三个编号程序依次跑；本程序是 03_consistency.py
python3 03_consistency.py    # 只跑本程序
```

- 配套程序：**`code/03_consistency.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
  ★ 它会在**系统临时目录**里写两份存档（要「从文件读」就要真的有文件），跑完清理。
- 失败标记：`;; FAIL: checkpointing`（本主题自定义，**不跨主题照抄**）；`D.1`/`D.2` 守着它。
- 程序的输出分四段：`A` **判据 ④ 的三条路径** + 结构钉住 + 反例的前提 + PaLM 的逐字目标、
  `B` 代价模型（RNG 项的字节账 / 两条路线的科目）、
  `C` 边界与反例（缺一项的两种读法 / **就地恢复的半新半旧** / 引用判据）、
  `D` 标记自检、口径与阈值**与键名**钉住、specs 常量钉住、独立性证据。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。
  ★ 本程序为此把数据集构造做成了**惰性**的：`D` 被变异后，`X @ TRUE_W` 会**形状不匹配** ——
  那是**导入期**的异常，若不在 `preflight()` 之后发生，它会被记成**崩溃捕获**而不是断言捕获。
- ★★ **`A.4` 是「判据写死的那一句」的可执行形态**：源码级的结构钉住（调用表 + 状态来源）。

## 参考

**本机实测值 / 推导值**

- `code/03_consistency.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
  **无 GPU**。口径：`seed = 20260955`、`D = 6`、`N_DATA = 32`、`BATCH = 8`、`STEPS = 60`、
  `SAVE_AT = 23`、`NOISE = 1e-3`；三条路径各跑一遍恢复段 `37` 步。**全部是推导值。**
- ★ 与计划 §3 表 A 的对应：**判据④**（恢复后与原轨迹逐位一致；★ **实现口径写死** ——
  存档含 RNG 的**内部状态**与 dataloader 位置、恢复时**只允许从文件读、不得 `seed()`**；
  ★★ **阴性对照**：去掉 RNG 状态后**必须不一致** ✓）。
  ★ 控制者的**预演**（`True` / `False` / `True`）在本机**逐位复现**。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.4`（**结构钉住**）、
  `A.5`（反例的**前提**：消耗模式可重放）、`A.6`（PaLM 的逐字目标）、
  `B.1`（RNG 项的字节分解）、`B.2`（**两条路线的科目**）、
  `C.1`（缺项的两个方向）、`C.2`（**就地恢复的半新半旧**）、`C.3`（引用判据 + `hang` vs `error`）、
  `D.9d`（**T41 的同源钉住**：`train_step` 自己调 `draw` 且用 `NOISE * noise` ⇒
  `A.2` 的阴性对照**不会退化成 no-op**）。
  ★ 它们的必要性是**测出来的**：`A.4` 是「路径 3 无法用数值判据区分」的**唯一**对策；
  `B.2` 把「RNG 一致性很便宜」这句话拆成两笔**不同科目**的账。

**外部来源**

- **[C]** PaLM（`arXiv:2204.02311`，**不确认同行评审** —— abs 页 `Comments:` 字段不存在），
  §5 的 `Bitwise determinism` 一节。抓取 `https://ar5iv.labs.arxiv.org/html/2204.02311`，
  2026-09-23（**1372842 B**）。逐字（各原始字节 1 次 / 剥标签后 1 次）：
  `Bitwise determinism`、
  `fully bitwise reproducible from any checkpoint`、
  `if the model has been trained up to step 17,000 in a single run`、
  `a deterministic dataset pipeline where the shuffled data is written out in a random-access format`、
  `the contents of a given training batch is only a function of the step number`。
  ★★ 其中**最后两条**是本文路径 3 的**正面依据**（计划 §3 表 B 要求逐字钉住前三句，
  本文把这两句一并收进来 —— 因为它们是路径 3 **性质**的唯一来源）。
- **[C]** Adam（`arXiv:1412.6980`，**已确认 ICLR 2015**）的默认超参（**PDF 文本层**逐字；
  方法学让步见计划 **T5/T45**）—— 玩具更新式用它们。
- **[B]** **PyTorch 2.14** 的 `Distributed Checkpoint` 页，
  `https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`，抓取 **2026-09-23**，
  **967566 B**，`sha256[:16] = 1e78400bd3be2148`。本文用到五条逐字：
  `allow_partial_load: If False, will raise a runtime error if a key is present in state_dict,
  but not in the checkpoint.`、
  `Mismatched keys may result in hangs or errors.`、
  `WARN: set_optimizer_state_dict can only be called before backward()`、
  `It operates in place, meaning that the model should allocate its data first and DCP uses that
  storage instead.`、
  `Write the metadata and marks the current checkpoint as successful.`
  （★ 最后两条与 `File creation is atomic` 一起在 `02` 的 §1.6 里也用到了 ——
  一边是**集合级**、一边是**对象级**的完整性）。

**上游契约与边界声明**

- ★★ **「同名不同指」**：本文的 `checkpoint` = **权重检查点**；
  「重算换显存」是另一件事（正文在 `autograd-and-backward/02`）。
- **禁区**：导出/推理格式 → `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；
  异步落盘 → `110-communication/README.md:12`；状态分片与通信量 → `100-parallelism/zero-fsdp/`。
  本文**一行都不重推**。
- ★ **本主题在全库零让渡**（计划 §0.7-B 的边界表）⇒ 本文的知识点由本域**自己立**。
