---
title: 训练状态的组成与字节账（参数 / 梯度 / 优化器状态 / RNG）
slug: state-and-bytes
domain: 90-training
topic: checkpointing
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/optimizers/02-adam-and-adamw
tags: [checkpoint, optimizer-state, byte-account, rng-state, zero, mixed-precision]
related:
  - 90-training/checkpointing/02-sharded-save-and-load
  - 90-training/mixed-precision/01-type-selection
code:
  - code/01_state_bytes.py
status: stable
updated: 2026-09-23
---

# 训练状态的组成与字节账（参数 / 梯度 / 优化器状态 / RNG）

> **检查点存的是「训练状态」，它的字节数是 `Ψ` 的函数，不是训练时长的函数。**
> ★★★ 而**「训练显存」与「检查点文件」是两笔账**：ZeRO 那个著名的 `16Ψ` 算的是**前者**
> （原文的词是 `memory requirement`），它**含梯度**；检查点通常**不含梯度**
> ⇒ 两笔账**恰好差一个梯度项 `2Ψ`**（本文 §1.5 把这个差量出来）。
> ★★ 而 **RNG 的内部状态只有 ~170 字节**（常数级），但**少了它，恢复后的轨迹
> 从第 1 步就分叉**（正文在 `03-consistency-and-fault-tolerance`）。

## 0. 背景与动机

### ★ 先说清楚「检查点」这个词在本库的两种意思（同名不同指）

**本文的 `checkpoint` 一律指「权重 / 训练状态的存档」**（磁盘上的那件事）。
★ **另一个意思（「用重算换显存」）在本库一律写作「重算换显存」**，
正文在 `90-training/autograd-and-backward/02-gradient-checkpointing.md`；
★ **中文词「检查点」不得用于重算**（域 README 的消歧节、计划 §2 末尾的 **M9/M10**）。
★ 唯一的例外是**文件名**：`gradient checkpointing` 是 Chen 2016 原文的术语，改了就对不上原文。

### ★★ 本文的范围（与禁区的界线）

| 本文**写** | 本文**不写**（回链目标） |
|---|---|
| 状态的**组成**（哪些张量、哪些非张量）、每一块的**字节来源** | **导出 / 推理格式的选择**（`.keras` vs SavedModel）→ `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188` |
| 优化器状态的**精度**与字节（`m` / `v` 各 `4Ψ`） | **「为什么主权重必须 fp32」的数值理由** → `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266` |
| RNG 状态的**结构**与它的一致性**要求** | **一致性恢复的实验与判据** → `90-training/checkpointing/03-consistency-and-fault-tolerance.md` |
| 分片格式与它的**头开销**（只回链，不重推） | **分片的字节账与重分片** → `90-training/checkpointing/02-sharded-save-and-load.md` |
| —— | **优化器状态的分片策略与通信量**（ZeRO 三阶段 / FSDP） → `100-parallelism/README.md:14` |
| —— | **异步落盘 / 通信-计算重叠** → `110-communication/README.md:12` |

★ **一处必须写明的让步**：本文的一切数值都是**本机玩具规模的真跑**
（`python3 3.13.5` + `numpy 2.4.4`、macOS arm64、**无 GPU**、不训练任何真模型）
⇒ **不得支撑关于真实训练系统的结论**。

## 1. 原理

### 1.1 训练状态里有什么

| 项 | 它在哪 | 进不进检查点 | 恒等式（`Ψ` = **参数个数**） |
|---|---|---|---|
| **参数**（fp16/bf16 的一份） | 显存 | ★ **看你的 `state_dict`** | `2Ψ` 字节 |
| **参数的 fp32 副本**（主权重） | 显存 | ★ 同上 | `4Ψ` 字节 |
| **梯度** | 显存 | ★★ **通常不进**（下一步就被覆盖） | `2Ψ` 字节 |
| **优化器状态**：动量 `m` | 显存 | **进** | `4Ψ` 字节（fp32） |
| **优化器状态**：方差 `v` | 显存 | **进** | `4Ψ` 字节（fp32） |
| **步号 / 学习率调度器 / scaler** | 显存 | **进**（非张量） | 常数级 |
| **RNG 的内部状态** | 进程内 | ★★ **进**（否则恢复不一致） | 常数级（~170 B） |
| **dataloader 位置**（epoch / 游标 / 顺序） | 进程内 | ★★ **进**（同上） | 与数据集同阶 |

★★ **「进不进」这一列的判据是「你的 `state_dict` 里放了什么」**，不是某个规范的规定 ——
[B] PyTorch 2.14 的 DCP 文档逐字：`For each Stateful object (having both a state_dict and a
load_state_dict ), save will call state_dict before serialization.`
⇒ 检查点 = `state_dict()` 的返回值的序列化；**它是你写的代码决定的**（已登记 `UNSOURCED`：
`dcp_checkpoint_excludes_gradients` —— 没有一句文档说「梯度不入检查点」，那是**推理**）。

### 1.2 ★★ 判据 ①：优化器状态的字节账**按精度算**

**来源**（[C] ZeRO，arXiv 1910.02054，**不确认同行评审** —— abs 页 `Comments:` 字段不存在）
§5.1 逐字：`Mixed-precision Adam has K = 12`、`In total, this results in 2` …
`bytes of memory requirement`，定义处 `i.e., the additional memory required to store them is K`
（★ `KΨ` 里夹了公式标记，故切在 `is K`）。
★★ **`K = 12` 是 `ZeRO 的举例口径`** —— Figure 1 的说明逐字给了条件：
`based on mixed-precision training with Adam optimizer.`（计划 **Z4**）。

| 项 | 每参数字节 | 本机实测（`Ψ = 172`，`(16, 8, 4)`） |
|---|---|---|
| 参数的 fp32 主权重 | `4` | `688 B` |
| `m`（动量） | `4` | `688 B` |
| `v`（方差） | `4` | `688 B` |
| **优化器状态合计** | **`K = 12`** | **`2064 B` = `12Ψ`** |

★ **判据 ① 本身**：fp32 的 `m` + `v` = **`8Ψ` 字节**（本机 `1376 B`）。
★★ **这个判据不是恒真**：左边来自 NumPy 的 `size × itemsize`（由**形状与 dtype** 决定）、
右边来自 `specs.py` 里的 **[C] 常量**（`4 + 4`）—— 两条**来源不同**的路径给出同一个数。
★ 把 `m`/`v` 按 **fp16** 存就是 `2Ψ + 2Ψ = 4Ψ`（本机 `688 B`，**少一半**）而**不报错**
⇒ **「两个长度 Ψ 的数组」这种说法不带载荷，字节账必须写明精度**（`C.2`）。

### 1.3 ★★★ 两笔账：`16Ψ` 是**显存**，不是**检查点大小**

| 口径 | 式子 | 本机（`Ψ = 172`） | 它含什么 |
|---|---|---|---|
| **训练显存**（[C] ZeRO §5.1 的举例口径） | `2Ψ + 2Ψ + KΨ = 16Ψ` | `2752 B` | 参数（fp16）、**梯度**、优化器状态 |
| **检查点**（`state_dict` 里的 `2Ψ` 参数 + `KΨ` 优化器状态） | `2Ψ + 12Ψ = 14Ψ` | `2408 B` | 参数、优化器状态 —— **不含梯度** |

⇒ **差 = `344 B` = `2Ψ` = 恰好是梯度那一项**（`C.3` 逐位断言这个恒等式）。
★★ 原文用的词是 `memory requirement`（**显存需求**），Figure 1 的标题是
`Comparing the per-device memory consumption of model states` ⇒
**把 `16Ψ` 当检查点大小，会恰好多算一份梯度**。

### 1.4 ★★ `Ψ` 的口径是「参数个数」

[C] ZeRO Figure 1 逐字：`denotes model size (number of parameters), K`
⇒ 同一个模型有三种数法，**两两不同**（本机实测）：参数个数 `172`、张量个数 `3`、
`w` 的字节数 `688`。★ 用张量个数当 `Ψ` 会让账**差 57 倍**，而它**不会报错**。

### 1.5 ★★ 检查点的张量字节数**与步数无关**

本机实测（`Ψ = 10336`）：第 1 步 `Σ nbytes` = **`124032 B`**、第 200 步 = **`124032 B`**
（**逐位相等**），而**内容不同**（`w` 的字节不同）。
⇒ ★ **它是 `Ψ` 的函数，不是训练时长的函数** ⇒ 「存得越勤 ⇒ 单次越贵」是**错的**；
贵的是**次数**（累计 `= 次数 × W`）。

### 1.6 ★ RNG 的内部状态：四个字段、~170 字节、**常数级**

`np.random.Generator.bit_generator.state` 在本机（NumPy 2.4.4）是**四个字段**：

```
{'bit_generator': 'PCG64', 'state': {'state': …, 'inc': …}, 'has_uint32': 0, 'uinteger': 0}
```

| 量 | 本机实测 |
|---|---|
| 序列化（`json.dumps(..., sort_keys=True)`）字节数 | 跨 400 个状态 ∈ **`[165, 176]`** |
| 与 `Ψ` 的关系 | **无关**（`Ψ = 172` 时 `166 B`、`Ψ = 10336` 时 `165 B`） |
| 参数字节与它的比值 | `Ψ = 172`：**12.4**；`Ψ = 10336`：**751.7** |
| **非张量部分占总文件的比例** | `Ψ = 172`：**7.44%**；`Ψ = 10336`：**0.13%** |

★★ **最后一行是一条「关于规模的话」**：[B] DCP 逐字说非张量数据**走另一条路**
（`All non-tensor data is loaded using torch.load()`）——
★ 而它在**玩具规模上是可观的占比**、在真实规模上才可以忽略 ⇒
**「检查点几乎全是张量」这句话在玩具规模上不成立**。

★★★ **`has_uint32` 是这条状态里最阴的一个字段**（`C.1` 把三态都量出来）：

| 情形 | 结果 |
|---|---|
| **完整**（四字段） | 后续抽样**逐位**一致 |
| **缺** `has_uint32` 键 | ★ **`KeyError: 'has_uint32'`** —— NumPy **自己守卫**（不是静默） |
| `has_uint32` **被归一化成 `0`** | ★★ **静默不一致**，而且**只在 32 位抽样路径上**（`integers`）；64 位路径（`random()`）**照样一致** |

⇒ ★★ 一条推论：**「用 `random()` 做的恢复测试」抓不到这一类错**。
判据 ④ 的玩具循环因此必须含**真正用到的那个抽样路径**（`dataloader` 取下标走的是 32 位那条）。

## 2. 执行映射

### 2.1 存一份检查点要做对的三件事

| 步骤 | 做什么 | 依据 / 后果 |
|---|---|---|
| 1 | **列清单**：哪些张量、哪些非张量（步号 / 调度器 / scaler / RNG / 游标） | §1.1；★ 漏掉非张量项 ⇒ 轨迹从第 1 个恢复步就分叉 |
| 2 | **写明精度**：`m`/`v` 是 fp32 还是 fp16（差一倍字节，**不报错**） | §1.2 的 `C.2` |
| 3 | **键集一致**：每个 rank 的 `state_dict` 键集相同 | [B] DCP 逐字 `Mismatched keys may result in hangs or errors.`（★ **`hang` 比 `error` 更坏**） |

### 2.2 什么时候存

| 判据 | 结论 |
|---|---|
| 单次存档的大小随训练时间变吗 | **不变**（张量部分是 `12Ψ`，与步数无关，§1.5） |
| 那什么随训练时间变 | **累计落盘字节** = `次数 × W`（§1.5 的推论） |
| 因此「该多久存一次」的账 | 记在**时间与 I/O** 上，不在**单次大小**上（异步落盘的正文在 `110-communication/README.md:12`） |

### 2.3 本机实测的一组典型值（口径写在 `[ ]` 里）

| 量 | `(16, 8, 4)`（`Ψ = 172`） | `(128, 64, 32)`（`Ψ = 10336`） |
|---|---|---|
| 张量部分（`w` + `m` + `v`，fp32） | `2064 B` = `12Ψ` | `124032 B` = `12Ψ` |
| 其中 `m` + `v` | `1376 B` = `8Ψ` | —— |
| RNG 内部状态（JSON 口径） | `166 B` | `165 B` |
| **训练显存**（ZeRO 的 `16Ψ` 口径） | `2752 B` | —— |
| **检查点**（`14Ψ` 口径） | `2408 B` | —— |

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| **`torch.save` 整份** | 一个文件、一次 pickle | 简单；★ 单卡内存要装得下**全部**状态，且**跨版本没有保证**（[B] 逐字 `There are no guarantees of Backwards Compatibility across PyTorch versions`） |
| **分片保存** | 每卡只写自己的分片 | ★ **省的是单卡峰值**（`≈ W/N`），**不是总字节**（反而多 `N × 每片头开销`）⇒ 正文在 `02-sharded-save-and-load` |
| **分开存「参数」与「优化器状态」** | 两套键 | ★ 推理只需要参数那一半（推理口径的正文在 `45-kernel-lifecycle/requirement-discovery/01:234,301`） |
| **只存 `state_dict` 里声明过的东西** | 由调用方决定内容 | ★ 这是 [B] 的实际语义；★ 代价是**没有东西替你检查你漏了什么**（除了 `allow_partial_load`，正文在 `03`） |

## 4. 踩坑与失败模式

**① ★★★ 把 ZeRO 的 `16Ψ` 当检查点大小。**
- **现象**：按 `16Ψ` 估存档大小。
- **成因**：那是**训练显存**口径（原文 `memory requirement`），它**含梯度 `2Ψ`**（§1.3）。
- **规避**：两笔账分开算；差的**恰好**是 `ZERO_GRAD_FP16_MULTIPLE × Ψ`（`C.3` 断言）。

**② ★★ 不写精度。**
- **现象**：说「优化器状态是两个长度 `Ψ` 的数组」。
- **成因**：`m`/`v` 各 `4Ψ`（fp32）还是 `2Ψ`（fp16）**差一倍**，而两种都**不报错**（§1.2）。
- **规避**：字节账一律写成 `多个 × Ψ` 且**注明精度**。

**③ ★★ 把 `Ψ` 数成张量个数。**
- **现象**：账差几十倍。
- **成因**：`Ψ` 的逐字定义是 `number of parameters`（§1.4）。
- **规避**：`Ψ` 只在**参数个数**这一种口径上使用；三种数法两两不同（`C.4`）。

**④ ★★ 以为「存得越勤，单次越贵」。**
- **现象**：为了省空间而拉长存档间隔。
- **成因**：单次大小与步数**无关**（§1.5）；随次数增长的是**累计**字节。
- **规避**：把「单次大小」与「累计字节」分开写。

**⑤ ★★★ 忘记非张量项（或只存了 `state_dict` 里的张量）。**
- **现象**：恢复后 loss 曲线对不上。
- **成因**：RNG 的内部状态与 dataloader 位置**不在张量里**；少一项就分叉（`03` 的 `A.2`：
  **第 1 个恢复步就分叉**）。
- **规避**：把非张量项与张量项**放在同一份清单里**（`01` 的 `ARCHIVE_KEYS` 就是那份清单）。

**⑥ ★★ 用「缺键不报错」的读法。**
- **现象**：读回成功、结果不对。
- **成因**：`strict=False` 时缺的项被默认值顶上（`03` 的 `C.1` 量了两个方向）。
- **规避**：读侧的默认必须是**报错**（[B] 的 `allow_partial_load` 默认就是 `False`）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| fp32 的 `m` + `v` 是多少字节 | **`8Ψ`**（本机 `1376 B`）；★ 判据的两侧**来源不同**（NumPy 的 dtype vs `specs.py` 的 [C] 常量）⇒ **不是恒真** |
| 优化器状态的合计是多少 | **`K = 12`** 的 `4 + 4 + 4`（主权重 + `m` + `v`）；★★ 但 `K = 12` 是 **ZeRO 的举例口径**（条件：`based on mixed-precision training with Adam optimizer.`） |
| 训练显存的账 | `2Ψ + 2Ψ + KΨ = 16Ψ`（本机 `2752 B`）；★ **含梯度**，且原文的词是 `memory requirement` |
| 检查点的账 | `2Ψ + 12Ψ = 14Ψ`（本机 `2408 B`）；★★ **与 `16Ψ` 差 `2Ψ` = 梯度那一项** |
| 把 `m`/`v` 按 fp16 存会怎样 | `4Ψ`（**少一半**）而**不报错** ⇒ 精度必须写明 |
| `Ψ` 怎么数 | **参数个数**（逐字：`number of parameters`）；张量个数 / 字节数会让账差几十倍 |
| 单次存档大小随步数变吗 | **不变**（第 1 步与第 200 步逐位相等）；变的是**内容**与**累计次数** |
| RNG 内部状态多大 | 常数级：跨 400 个状态 ∈ `[165, 176]` B；**与 `Ψ` 无关** |
| 非张量数据怎么读 | [B] 逐字 `All non-tensor data is loaded using torch.load()`（**另一条路**）；★ 它的占比在玩具规模上 7.44%、`Ψ = 10336` 上 0.13% ⇒ **是「关于规模的一句话」** |
| RNG 状态少一个字段会怎样 | 缺 `has_uint32` ⇒ **`KeyError`**（有守卫）；把它**归一化成 `0`** ⇒ **静默不一致**，且**只在 32 位路径上** |
| 每个 rank 的键集可以不同吗 | **不可以**：[B] 逐字 `Mismatched keys may result in hangs or errors.` |

> ★ 阈值与口径（`code/01_state_bytes.py` 的 `D.3`/`D.4` 逐位钉住）：
> `SEED = 20260953`、`SMALL_SHAPE = (16, 8, 4)`、`LARGE_SHAPE = (128, 64, 32)`、
> `STEPS_LATE = 200`、`RNG_STATE_DRAWS = 1000`、`RNG_STATE_SWEEP = 400`、
> `UINT32_BOUND = 4294967296`、`RNG_STATE_BYTES_MIN/MAX = 160/200`（实测 `165/176`）、
> `SMALL_RATIO_MIN = 8.0`、`LARGE_RATIO_MIN = 500.0`（实测 `12.4` / `751.7`）、
> `TENSOR_FRACTION_MIN = 0.99`、`SMALL_FRACTION_MAX = 0.95`。
> ★ **`β1/β2/ε` 从 `specs.py` 读**（[C] 1412.6980，**已确认 ICLR 2015**）—— 本主题只消费、不重定义。

## 6. 小结与延伸

**一句话**：检查点的字节账**从 `Ψ` 出发**（`12Ψ` 的张量 + 常数级的非张量），
**与训练时长无关**；而 ZeRO 的 `16Ψ` 是**显存**口径、含梯度 ——
**两笔账差的恰好是那一个梯度项**。
★ 而「RNG 状态只有 ~170 字节」这件事**不等于它可省**：
少了它，轨迹从第 1 个恢复步就分叉（`03`）。

- **分片保存与恢复（含 `N × 每片头开销` 的字节账）** ⇒ `90-training/checkpointing/02-sharded-save-and-load.md`。
- **一致性恢复（RNG / dataloader / 优化器状态的三条路径）** ⇒ `90-training/checkpointing/03-consistency-and-fault-tolerance.md`。
- **Adam 的 `m` / `v` 与它们的更新式** ⇒ `90-training/optimizers/02-adam-and-adamw.md`。
- **`m`/`v` 该用什么精度** ⇒ `90-training/mixed-precision/01-type-selection.md`。
- **「重算换显存」**（`O(√n)` 是**内存**、`≈ 2n` 是**前向次数**） ⇒
  `90-training/autograd-and-backward/02-gradient-checkpointing.md`（**正文在彼**；★ 它与本文的
  「检查点」是**同名不同指**，见 §0）。
- **导出 / 推理格式的选择** ⇒ `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`（**正文在彼**）。
- **优化器状态的分片与通信量** ⇒ `100-parallelism/README.md:14`（**正文在彼**）—— 本文**一行都不写**。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU**、不训练任何真模型）⇒
  **不得支撑关于真实训练系统的结论**。

## 编译与运行

```bash
cd code
make run                     # 三个编号程序依次跑；本程序是 01_state_bytes.py
python3 01_state_bytes.py    # 只跑本程序
```

- 配套程序：**`code/01_state_bytes.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`;; FAIL: checkpointing`（本主题自定义，**不跨主题照抄**）；`D.1`/`D.2` 守着它。
- 程序的输出分四段：`A` 字节账（判据 ①）/ RNG 状态的往返 / PaLM 的逐字目标、
  `B` 代价模型（与步数无关 / RNG 是常数级 / 占比随规模变）、
  `C` 边界与反例（`has_uint32` 三态 / fp16 静默减半 / 两笔账的差 / `Ψ` 的三种数法）、
  `D` 标记自检、口径与阈值钉住、specs 常量钉住、独立性证据。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。
- ★★ **`C.3` 是本文头号口径区分的可执行落法**：两笔账的差**逐位**等于梯度那一项。

## 参考

**本机实测值 / 推导值**

- `code/01_state_bytes.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
  **无 GPU**。口径：`seed = 20260953`；两个规模 `(16, 8, 4)` 与 `(128, 64, 32)`；
  RNG 状态的字节数扫 `400` 个状态；`A.4` 比 `1000` 次抽样。**全部是推导值。**
- ★ 与计划 §3 表 A 的对应：**判据①**（优化器状态的字节账按精度算：fp32 的 `m` + `v` = `8Ψ` ✓，
  且与 `K = 12` 口径对齐 ✓）。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.3`（`16Ψ` 的算术复现）、
  `A.4`（RNG 状态的往返 + **反面对照**）、`B.1`（与步数无关）、`B.2`（RNG 字节数的带子）、
  `B.3`（参数字节 ∝ `Ψ` 而 RNG 不随 `Ψ` 变）、`B.4`（非张量占比随规模变）、
  `C.1`（`has_uint32` 的三态）、`C.2`（fp16 的静默减半）、`C.3`（**两笔账的差**）、
  `C.4`（`Ψ` 的三种数法）。
  ★ 它们的必要性是**测出来的**：`C.3` 与 `C.1` 各把一句**容易被说错的话**变成了一个数。

**外部来源**

- **[C]** ZeRO（`arXiv:1910.02054`，**不确认同行评审** —— abs 页 `Comments:` 字段不存在），
  §5.1。抓取 `https://ar5iv.labs.arxiv.org/html/1910.02054`，2026-09-23（**279500 B**）。
  逐字（各原始字节 1 次 / 剥标签后 1 次）：`Mixed-precision Adam has K = 12`、
  `to denote the memory multiplier of the optimizer states`、
  `i.e., the additional memory required to store them is K`、
  `In total, this results in 2` … `bytes of memory requirement`、
  `based on mixed-precision training with Adam optimizer.`、
  `denotes model size (number of parameters), K`。
  ★★ **一处量法口径**：ar5iv 把 `Ψ` 这类符号**渲染成两份**（`4 \Psi+K\Psi` 后紧跟 `4\Psi+K\Psi`）
  ⇒ 本文的引文**一律不跨公式**，且每一条都在**剥标签 + 折叠空白**后核过次数。
- **[C]** Adam（`arXiv:1412.6980`，**已确认 ICLR 2015**）的默认超参
  `β1 = 0.9`、`β2 = 0.999`、`ε = 10⁻⁸`（**PDF 文本层**逐字；★ 方法学让步见计划 **T5/T45**：
  该文的 ar5iv 页只有 226 字符，且 PDF 里 `ε` 是控制字符）——
  本程序的玩具更新式用它（同值位置：`optimizers/code/specs.py`）。
- **[C]** PaLM（`arXiv:2204.02311`，**不确认同行评审**），§5。抓取 ar5iv（**1372842 B**）。
  逐字：`Bitwise determinism`、`fully bitwise reproducible from any checkpoint`、
  `if the model has been trained up to step 17,000 in a single run`、
  `a deterministic dataset pipeline where the shuffled data is written out in a random-access format`、
  `the contents of a given training batch is only a function of the step number`。
  ★ 它是**判据 ④ 的外部目标**（本文只登记，实验在 `03`）。
- **[B]** **PyTorch 2.14** 的 `Distributed Checkpoint` 页，
  `https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`，抓取 **2026-09-23**，
  **967566 B**，`sha256[:16] = 1e78400bd3be2148`（与计划 §C.2b 的 **T3** 逐位一致）。
  本文用到两条逐字：`For each Stateful object (having both a state_dict and a load_state_dict ),
  save will call state_dict before serialization.` 与
  `All non-tensor data is loaded using torch.load()`。
  ★★ **一处量法口径**：同一页上 `torch.distributed.checkpoint` 在**原始字节**上计 **547** 次、
  在**剥标签 + 折叠空白后**计 **71** 次 —— **引用计数必须写明载体**（Y15 / T34 的现场）。

**上游契约与边界声明**

- ★★ **「同名不同指」的两条**（域 README）：本文的 `checkpoint` = **权重检查点**；
  「重算换显存」是另一件事（正文在 `autograd-and-backward/02`）。
- **禁区**：导出/推理格式 → `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；
  主权重的数值理由 → `10-foundations/numerical-linear-algebra/04:229-266`；
  状态分片与通信量 → `100-parallelism/README.md:14`；异步落盘 → `110-communication/README.md:12`。
  本文**一行都不重推**。
- ★ **本主题在全库零让渡**（计划 §0.7-B 的边界表）⇒ 本文的知识点由本域**自己立**，
  因此更要把上面这张界线表守住。
