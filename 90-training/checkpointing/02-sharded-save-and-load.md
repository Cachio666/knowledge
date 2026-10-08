---
title: 分片保存与恢复：格式、字节账与重分片
slug: sharded-save-and-load
domain: 90-training
topic: checkpointing
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/checkpointing/01-state-and-bytes
tags: [checkpoint, sharded, distributed-checkpoint, metadata, resharding, byte-account]
related:
  - 90-training/checkpointing/03-consistency-and-fault-tolerance
  - 100-parallelism/zero-fsdp
code:
  - code/02_sharded.py
status: stable
updated: 2026-09-23
---

# 分片保存与恢复：格式、字节账与重分片

> **分片省的是「单卡峰值」，不是「总字节」。**
> ★★ 恰恰相反：`N` 卡分片之后**总字节会变多** —— 实测总字节**逐位等于**
> `W + N × 每片头开销`（本文 §1.3 的判据 ②，本机 `W = 2064`、头 `64`、
> `N = 16` 时总字节 `3088`，头开销占 **33.161%**）。
> ★★★ 而**同一个 `W`** 在真实规模上（`Ψ = 7.5×10⁹`、`N = 1024`）头开销只占 **5.46×10⁻⁵ %**
> ⇒ **「元数据开销可以忽略」是【关于规模的一句话】**。

## 0. 背景与动机

### ★★ 本文的来源是「某实现的文档」，**不是**规范（**P19** 的反面做法）

| 项 | 值 |
|---|---|
| 载体 | **[B] PyTorch 2.14** 的 `Distributed Checkpoint` 页 |
| URL | `https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html` |
| 抓取日 / 字节数 / 指纹 | **2026-09-23** / **967566 B** / `sha256[:16] = 1e78400bd3be2148`（与计划 §C.2b 的 **T3** 逐位一致） |
| 它是什么 | **一个实现的契约**（`torch.distributed.checkpoint`），**不是 A 级规范** |

★ **本文因此把「引的哪一版」写成了载体指纹**（URL + 字节数 + `sha256` 前缀）——
因为对 [B] 级来源来说，「版本」不是修辞。
★★ **本程序一行 torch 都不 import**：把那份文档的**逐字**当作 `specs.py` 里的 [B] 常量，
按它写的规则在 numpy 上重演一遍 ⇒ 由此得到的数一律标为**推导值**（不是「PyTorch 跑出来的」）。
★ **一处量法口径的现场**：同一页上 `torch.distributed.checkpoint` 在**原始字节**上计 **547** 次、
在**剥标签 + 折叠空白后**计 **71** 次（差 **7.7 倍**，因为侧边导航栏把这个词重复了几百次）
⇒ **引用计数必须写明载体**（Y15 / T34）。

### ★★ 本文的范围（与禁区的界线）

| 本文**写** | 本文**不写**（回链目标） |
|---|---|
| 分片的**格式**（每片一个文件 + 元数据）与**头部的组成** | **导出 / 推理格式的选择**（`.keras` vs SavedModel）→ `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188` |
| 分片的**字节账**（判据 ②）与**每卡峰值 / 每卡读量** | **优化器状态的分片策略与通信量**（ZeRO 三阶段、`4x/8x`）→ `100-parallelism/zero-fsdp/` |
| **重分片**（写 `N` 片、读 `M` 片）与它的代价 | **异步落盘 / 通信-计算重叠** → `110-communication/README.md:12` |
| 版本字段、原子性、覆盖率（**完整性**的三个层次） | **数据并行 / DDP / 梯度规约** → `100-parallelism/data-parallel/` |

★ **一处必须写明的让步**：本文的一切数值都是**本机玩具规模的真跑**
（`python3 3.13.5` + `numpy 2.4.4`、macOS arm64、**无 GPU**）⇒
**不得支撑关于真实训练系统的结论**。★ 尤其：本文量的是「字节与文件数」，**不是吞吐**。

## 1. 原理

### 1.1 ★★ 判据 ②：分片的字节账（**实测 vs 常量**）

**来源**（[B] PyTorch 2.14 的 DCP 页，逐字）：
- `It produces multiple files per checkpoint, with at least one per rank.`
- `a global .metadata file with the serialized metadata if rank coordination is enabled.`
- `a rank local __{rank}.metadata file with the serialized metadata if rank coordination is NOT enabled.`

⇒ ★ 由这两条**推出**「**每个分片文件都要有一个自己的头**」（无协调时，头就是那份 rank-local 元数据）；
★★ **而头有多少字节是设计** —— DCP 的文档**不给**这个数（已登记 `UNSOURCED`：
`dcp_per_shard_header_bytes`）。本文的玩具格式取 **`64` 字节**：

```
RRRRRRRR|VVVVVVVV|SSSSSSSS|LLLLLLLL|TTTTTTTT|DDDDDDDDDDDD
                                          └─ 12 位十六进制摘要
 5 个 8 位十进制字段（rank / version / start / length / total）+ 5 个分隔符 = 45
 + 摘要 12 = 57（文本）  + 填充 7 个 `.` = 64（定长）
```

★★★ **判据 ② 的两侧来自不同的地方**：

| 侧 | 怎么来的 |
|---|---|
| 左（**实测**） | `os.path.getsize(每个分片文件)` 求和 |
| 右（**常量**） | `W + N × specs.SHARD_HEADER_BYTES` |

本机实测（`W = 2064` = `12Ψ`，`Ψ = 172`）：

| `N` | 实测总字节 | `W + N × 64` | 差 | 头开销占比 |
|---|---|---|---|---|
| 1 | `2128` | `2128` | **0** | `3.008%` |
| 4 | `2320` | `2320` | **0** | `8.511%` |
| 8 | `2560` | `2560` | **0** | `16.000%` |
| 16 | `3088` | `3088` | **0** | **`33.161%`** |

★ 而「**实测的每片头开销**」（`文件大小 − 负载长度`）跨全部 7 个分片数都**恰好是 `64`**
⇒ 与 `specs.SHARD_HEADER_BYTES` 逐位一致（**把那个常量改 1，这条立刻失败**）。
★★ **反面**：原稿的「分片总字节 == 单卡总字节」是**按定义恒真**的写法（结构层审查 **M5**）
⇒ 它不承载任何工程结论；本文改成上面这个**两侧来源不同**的等式。

### 1.2 ★★ 省的是单卡峰值（[B] 逐字 `having each rank only save their local shards`）

| `N` | 最大负载 | `W / N`（上取整） | 最大负载 / `W` |
|---|---|---|---|
| 1 | `2064` | `2064` | `1.00000` |
| 3 | `688` | `688` | `0.33333` |
| 7 | `295` | `295` | `0.14293` |
| 16 | `129` | `129` | **`0.06250`** |

⇒ ★ **「够不够放」是每卡峰值的问题，「贵不贵」是总字节的问题** —— 两笔账。
★ 读侧同理，[B] 逐字：`read the least amount of data necessary`（`load` 的说明）。

### 1.3 ★★ 重分片：**写 `N` 片、读 `M` 片**

[B] 逐字：`It handles load-time resharding which enables saving in one cluster topology and
loading into another.` ⇒ **分片的边界不是格式的一部分，元数据才是**。
本机实测（写 `3` 片，`Σ` 读量恒为 `W = 2064`）：

| 读侧分片数 | `Σ` 读字节 | 每卡最多打开几个源文件 |
|---|---|---|
| 5 / 7 / 11 / 23 | `2064` | **2** |
| **1**（比写侧**少**） | `2064` | **3**（= 写侧的分片数） |

★★ **`≤ 2` 不是无条件成立的**：它的条件是「**读侧的分片数不比写侧更少**」；
读侧只有 1 片时，那一张卡要打开**全部**源分片。

### 1.4 ★★ 「怎么切」决定每卡打开几个文件

同一个 `W`、同一份负载、**总读量相同**，但「连续切分」与「交错切分」的**每卡文件数差 4 倍**
（本机：`N_save = 8`、`N_load = 4`）：

| 切法 | 每卡读字节 | 每卡打开文件数 |
|---|---|---|
| **连续**（每卡一段连续区间） | `[516, 516, 516, 516]` | `[2, 2, 2, 2]` |
| **交错**（每卡取 `i, i+n, i+2n, …`） | `[516, 516, 516, 516]` | **`[8, 8, 8, 8]`** |

⇒ ★★ **「怎么切」不是实现细节**：交错切分把读放大了 `N_save` 倍（**打开的文件数**）；
★ 但交错切分在**负载均衡**上更均匀 —— 两笔账都要算。

### 1.5 ★★ `K = 12` 的两条外部后果（判据 ③）

[C] ZeRO（`arXiv:1910.02054`，**不确认同行评审**）§5.1 的举例：`Ψ = 7.5B`、`N_d = 64`。
★ **`K = 12` 是 `ZeRO 的举例口径`** —— Figure 1 的说明逐字给了条件：
`based on mixed-precision training with Adam optimizer.`（计划 **Z4**）。
本机从这条算术里**取回了那条小节里的两个数**：

| 式子 | 本机算出 | ZeRO 逐字 |
|---|---|---|
| `(4 + 12/64) × 7.5×10⁹ / 10⁹` | **`31.40625 GB`** | `parameter model requires 31.4GB of memory using P`（`31.4`） |
| `16 × 7.5×10⁹ / 10⁹` | **`120.00000 GB`** | `while requiring 120 GB with standard DP.`（`120`） |

★★ **口径是推导出来的**：只有用**十进制 GB**（`10⁹` 字节）这两个数才同时对上；
用 GiB（`1.0737×10⁹`）会得到 `29.25` 与 `111.76`（都不是原数）。
★ 两条引文的**写法不同**：`31.4GB` **无空格**、`120 GB` **有一个空格**（照抄时不能统一）。
★★ **别把 `4x` 与 `8x` 混成一个数**（计划 **Y14**）：`4x` 属 `P_os`（只分片优化器状态）、
`8x` 属 `P_os+g`（再加梯度分片）—— 逐字见 `specs.py`。

### 1.6 ★ 完整性的三个层次（**都是 [B] 的逐字支持**）

| 层次 | 手段 | [B] 的逐字依据 | 本文的实现 |
|---|---|---|---|
| **文件**级 | 写文件是原子的 | `File creation is atomic` | 每片带 `version` + `digest` |
| **分片**级 | 内容摘要（**长度不够**） | ——（摘要属本文的设计，已登记 `UNSOURCED`） | 翻转一个字节 ⇒ 摘要不匹配 ⇒ **报错** |
| **集合**级 | **元数据 = 成功标记** | `Write the metadata and marks the current checkpoint as successful.` | 覆盖率检查（`[0, W)` 恰好被覆盖一次） |

★★ **「每一片都对」推不出「一套都对」**：本机实测（3 片里少 1 片）——
留下的两片**各自的摘要都过**，只有**集合级**的检查发现「有 `688` 个字节没被覆盖」。
★★ 这正是 [B] 里 `finish()` 那句的语义：**元数据是集合级的完整性标记**，
而 `File creation is atomic` **只说文件级**。

### 1.7 ★ 版本字段

[B] 逐字：`There are no guarantees of Backwards Compatibility across PyTorch versions`
⇒ ★ **格式必须自报版本**，读侧才能**拒绝**而不是**猜**。
★ 本机实测的反面（`C.4`）：把版本写成 `99`（本程序是 `3`）之后，
**不做版本检查时那份数据照样拼得回来**（**静默地读错**）⇒
**版本检查不是正确性检查，它是可用性检查**。

## 2. 执行映射

### 2.1 一份分片检查点的写侧清单

| 步骤 | 做什么 | 依据 |
|---|---|---|
| 1 | 先写**临时目录**、全部写完再改名 | `The checkpoint path is an empty or non-existing directory.` + 「原子的粒度是文件」 |
| 2 | 每片写 `头 + 负载`，头里带 `rank / version / start / length / total / 摘要` | §1.1、§1.6 |
| 3 | 有协调者时**最后**写全局 `.metadata`（它标记「这份检查点成功」） | [B] 的 `finish()` 次序：写侧是 `set_up_storage_writer` → `prepare_local_plan` → `prepare_global_plan` → `write_data` → `finish` |
| 4 | 键集在每个 rank 上一致 | [B] 逐字 `Mismatched keys may result in hangs or errors.`（★ **`hang` 比 `error` 更坏**） |

### 2.2 读侧清单

| 步骤 | 做什么 | 依据 |
|---|---|---|
| 1 | **先读元数据**（每片的头或全局 `.metadata`） | [B] 的读侧次序：`read_metadata()` → `set_up_storage_reader()` → `prepare_local_plan()` → `prepare_global_plan()` → `read_data()` |
| 2 | 校验**版本**、**摘要**、**覆盖率** | §1.6、§1.7 |
| 3 | 只读自己那一段（`read the least amount of data necessary`） | §1.2、§1.3 |
| 4 | 写入时**遵守 `start`**（不是文件名、不是列举顺序） | `C.1`：忽略头部 ⇒ 结果**依赖列表顺序** |

### 2.3 本机实测的一组典型值

| 量 | 值（`W = 2064` = `12Ψ`，`Ψ = 172`，头 `64`） |
|---|---|
| 单卡（`N = 1`）总字节 | `2128`（= `W + 64`） |
| `N = 16` 总字节 / 每卡峰值 | `3088` / `129`（`6.25%` 的 `W`） |
| 头开销占比（玩具 / `Ψ = 7.5×10⁹`） | **`33.161%`** / **`5.461×10⁻⁵ %`**（差 `6×10⁵` 倍） |
| `20` 次落盘、`N = 7` 的元数据总量 | `8960 B`（负载 `41280 B`，元数据占 **`17.834%`**） |
| 重分片（写 3 读 7） | `Σ` 读 `2064`、每卡 ≤ `2` 个源文件 |
| 交错切分（`N_save = 8`、`N_load = 4`） | 每卡打开 **`8`** 个文件（连续切分是 `2`） |

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| **单文件 `torch.save`** | 一个文件 | 简单；★ 单卡要装下**全部**状态，且跨版本**没有保证** |
| **每卡一个文件** | 各写各的分片 | ★ 单卡峰值降到 `≈ W/N`；★ 代价是**总字节变多**（`N × 头`）、且**必须**有元数据才能拼回 |
| **`single_file_per_rank = True`**（DCP 的默认构造参数） | 一片一个文件 | 文件数 = `N`；★ 换成「一个 write request 一个文件」会更多（[B] 逐字：`one file per write request plus a global .metadata file`） |
| **有协调者（全局 `.metadata`）** | `N + 1` 个文件 | ★ 集合级的成功标记**只有一个地方**；代价是**协调者**成了单点 |
| **无协调者（每片元数据）** | `N` 个文件 | ★ 没有单点；代价是**每片都要带自己的元数据**（判据 ② 的 `N × 每片头开销` 在**这一支**上是字面的） |
| **重分片读** | 读侧的分片数与写侧无关 | ★ 换集群拓扑不用重存；代价是**读侧要重算映射**（读侧只有 1 片时要打开全部源文件） |

## 4. 踩坑与失败模式

**① ★★ 以为「分片能省总字节」。**
- **现象**：以为 `N` 卡分片后总字节不变。
- **成因**：每片都有自己的头 ⇒ 总字节 = `W + N × 头`（本文 §1.1）。
- **规避**：把「单卡峰值」与「总字节」**分开写**；判据 ② 的两侧**来源不同**才算判据。

**② ★★★ 把「元数据开销可以忽略」当成普适结论。**
- **现象**：在玩具规模上按真实经验忽略头开销。
- **成因**：它是**关于规模的一句话**（玩具 `33.161%` vs `Ψ = 7.5×10⁹` 上 `5.46×10⁻⁵ %`）。
- **规避**：凡说「可以忽略」，同时写出**它的规模条件**。

**③ ★★ 按文件名 / 列举顺序拼分片。**
- **现象**：换个 `ls` 顺序结果就变了。
- **成因**：分区边界由**头里的 `start`** 决定，不是由文件名决定（`C.1`：忽略头部 ⇒
  结果**依赖列表顺序**；遵守头部 ⇒ **排列无关**）。
- **规避**：读侧的第一个动作是**读元数据**（[B] 的次序也是这么定的）。

**④ ★★ 用「长度相等」当完整性。**
- **现象**：翻转一个字节后长度完全相同（本机实测），检测不出来。
- **成因**：完整性是**内容级**的要求（`C.2`）。
- **规避**：摘要（或任何内容级校验）+ **覆盖率**检查（集合级，§1.6）。

**⑤ ★★ 少一片时按「每片都对」放过。**
- **现象**：`N` 片里少了一片，程序照跑。
- **成因**：单片摘要看不出来（`C.3`：留下的两片都过校验）。
- **规避**：覆盖率检查 / 全局元数据（[B] 的 `finish()` 语义）。

**⑥ ★★ 不做版本检查。**
- **现象**：跨版本读回来「成功」了，但数据是错的。
- **成因**：不做检查时**静默**读错（`C.4`）。
- **规避**：格式自报版本、读侧**拒绝**不认识的版本。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 分片后总字节怎么算 | **`W + N × SHARD_HEADER_BYTES`**（本机逐位相等：`2128 / 2320 / 2560 / 3088`）⇒ ★「总字节 == 单卡字节」是**错的** |
| 每片头有多少字节 | **实测**（`文件大小 − 负载长度`）恒为 **`64`**，且**等于 `specs.py` 里的常量** ⇒ 改常量立刻失败 |
| 分片省的是什么 | **单卡峰值**（`N = 16` 时 `129 B` = `6.25%` 的 `W`），**不是**总字节 |
| 头开销的占比 | 玩具 **`33.161%`** / `Ψ = 7.5×10⁹`、`N = 1024` 上 **`5.461×10⁻⁵ %`** ⇒ **关于规模的一句话** |
| 元数据的账随什么变 | 随**次数 × 分片数**（`S × N × 头`）；负载随 `Ψ` —— 两笔账 |
| 换分片数还能读吗 | **能**（重分片）：写 3 片、用 5/7/11/23 片读都取回同一份字节；`Σ` 读恒为 `W` |
| 每卡要打开几个源文件 | 读侧 ≥ 写侧：**≤ 2**；★ 读侧只有 1 片：**= 写侧的分片数** ⇒ `≤ 2` **有条件** |
| 交错切分的代价 | 总读量相同，但每卡打开 **`N_save`** 个文件（连续切分是 `≤ 2`） |
| `K = 12` 能取回 ZeRO 的两个数吗 | **能**：`31.40625 → 31.4`、`120.00000 → 120`；★ 口径是**十进制 GB**（GiB 对不上） |
| `4x` 与 `8x` 是同一个阶段吗 | **不是**：`4x` 属 `P_os`、`8x` 属 `P_os+g`（逐字见 `specs.py`） |
| 怎么判断一份分片集合是完整的 | **三层**：文件级（原子创建）/ 分片级（摘要）/ **集合级（覆盖率或全局元数据）** —— ★ 前两层都推不出第三层 |
| 版本不匹配会怎样 | 必须**拒绝**（[B]：跨版本没有向后兼容保证）；★ 不做检查时**静默读错** |
| 每个 rank 的键集可以不同吗 | **不可以**（[B] 逐字 `Mismatched keys may result in hangs or errors.`） |

> ★ 阈值与口径（`code/02_sharded.py` 的 `D.3`/`D.4` 逐位钉住）：
> `SEED = 20260954`、`MODEL_SHAPE = (16, 8, 4)`（`Ψ = 172`，`W = 2064`）、
> `SHARD_COUNTS = (1, 2, 3, 4, 7, 8, 16)`、`RESHARD_SAVE_N = 3`、
> `RESHARD_LOAD_NS = (5, 7, 11, 23)`、`RESHARD_LOAD_FEW = 1`、`MAX_SOURCE_FILES = 2`、
> `SAVES = 20`、`PSI_REAL = 7.5×10⁹`、`ZERO_ND_EXAMPLE = 64`、`GB_DECIMAL = 10⁹`、
> `TOL_ROUND = 0.05`、`HEADER_FRACTION_MAX = 0.5`、`HEADER_FRACTION_REAL_MAX = 0.001`、
> `FORMAT_VERSION_BAD = 99`。
> ★ 格式常量（**E 级设计**）：`SHARD_HEADER_BYTES = 64 = 57 (文本) + 7 (填充)`、`FRAME_VERSION = 3`。

## 6. 小结与延伸

**一句话**：分片**不省总字节**（多出 `N × 每片头开销`），它省的是**单卡峰值**；
★★ 而「拼得回来」这件事**完全依赖元数据** —— 头里的 `start`（排列无关）、
摘要（内容级）、覆盖率或全局元数据（**集合级**），三层缺一不可。
★★★ 判据 ② 的两侧**来自不同的地方**（实测文件大小 vs `specs.py` 的常量）——
这是本主题把「字节账」写成**可证伪**判据的做法（原稿的写法按定义恒真，见结构层审查 **M5**）。

- **状态的组成与字节账** ⇒ `90-training/checkpointing/01-state-and-bytes.md`。
- **一致性恢复（RNG / dataloader / 优化器状态）** ⇒ `90-training/checkpointing/03-consistency-and-fault-tolerance.md`
  （★ 本文的 §1.6 与它的 `C.2` 是同一件事的两个层次：**集合级** vs **对象级**）。
- **优化器状态的分片策略与通信量（ZeRO 的 `4x/8x`）** ⇒ `100-parallelism/zero-fsdp/`（**正文在彼**）。
- **异步落盘 / 通信-计算重叠** ⇒ `110-communication/README.md:12`（**正文在彼**）。
- **导出 / 推理格式的选择** ⇒ `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`（**正文在彼**）。
- **「重算换显存」** ⇒ `90-training/autograd-and-backward/02-gradient-checkpointing.md`
  （★ 与本文的「检查点」**同名不同指**，见 `01` 的 §0）。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU**）⇒
  **不得支撑关于真实训练系统的结论**；★ 本文量的是**字节与文件数**，不是**吞吐**。

## 编译与运行

```bash
cd code
make run                     # 三个编号程序依次跑；本程序是 02_sharded.py
python3 02_sharded.py        # 只跑本程序
```

- 配套程序：**`code/02_sharded.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
  ★ 它会在**系统临时目录**里写分片文件（要落盘才能量字节），跑完清理。
- 失败标记：`;; FAIL: checkpointing`（本主题自定义，**不跨主题照抄**）；`D.1`/`D.2` 守着它。
- 程序的输出分四段：`A` 往返与覆盖性 / **判据 ③**（`K = 12` 的两个外部数）/ 重分片 / 文件数、
  `B` 代价模型（**判据 ②** / 单卡峰值 / 头开销占比 / 元数据的账）、
  `C` 边界与反例（头部的作用 / 摘要 vs 长度 / 少一片 / 版本 / 交错切分）、
  `D` 标记自检、口径与阈值钉住、specs 常量钉住、**判据 ③ 的正文检查**、独立性证据。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。
- ★★ **判据 ② 在本程序里**（★ 与计划表 A 的列不一致，理由：只有这里**有真正的分片文件**可量；
  放在 `01` 就成了「自己算公式再和自己比」—— 正是 **M5** 指出的恒真形态）。
- ★★ **`D.7d` 是判据 ③ 的正文侧检查**：它去读**本文**，确认 `ZeRO 的举例口径` 这个字符串
  **逐字**出现在正文里（★ `mutate.py` 的副本只带走 `*.py` ⇒ 在副本里它会**打印跳过**、
  而不是崩 —— 崩溃会被记成「崩溃捕获」，还会让基线不干净）。

## 参考

**本机实测值 / 推导值**

- `code/02_sharded.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
  **无 GPU**。口径：`seed = 20260954`、`MODEL_SHAPE = (16, 8, 4)`、`W = 2064`、
  分片数扫 7 档、重分片写 3 片读 5/7/11/23 片、交错切分对照 `N_save = 8` / `N_load = 4`。
  **全部是推导值。**
- ★ 与计划 §3 表 A 的对应：**判据②**（分片的元数据开销：`N 卡分片后总字节 = 单卡字节 +
  N × 每片头开销`，且**头开销 == `specs.py` 里的常量`** ✓ —— 控制者预演过「把 `HEADER` 改 1
  它就会失败」，本机实测**成立** —— ★ 本机是在 **`preflight()` 的 `D.3c`** 里被抓住的
  （不是 `B.1`：口径型常量先被钉住 ⇒ **P23** 的预期行为）；★ 而把**实现侧**的头长度写错
  （写 65 字节而规格说 64）会被 `encode_header` 的长度自检拦住（`B.1` 的实测侧仍逐位相等）；
  **判据③**（`K = 12` 只用于 ZeRO 的举例 ✓）。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.2`（覆盖性）、`A.4`/`A.4b`（重分片及其**方向**）、
  `A.5`（文件数 `N` vs `N+1`）、`B.2`（单卡峰值）、`B.3`（头开销占比随规模变）、
  `B.4`（元数据的账随次数 × 分片数）、`C.1`（头部的作用）、`C.2`（摘要 vs 长度）、
  `C.3`（少一片）、`C.4`（版本）、`C.5`（**交错切分**）。
  ★ 它们的必要性是**测出来的**：`C.5` 推翻了「每卡 ≤ 2 个文件」的无条件版本，
  `A.4b` 又把它的**条件**写清楚了。

**外部来源**

- **[B]** **PyTorch 2.14** 的 `Distributed Checkpoint` 页，
  `https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`，抓取 **2026-09-23**，
  **967566 B**，`sha256[:16] = 1e78400bd3be2148`。逐字（**剥标签 + 折叠空白后**计数；
  ★ 同一页在**原始字节**上计数会得到 ~7 倍的值，见 §0）：
  `It produces multiple files per checkpoint, with at least one per rank.`（1 次）、
  `It handles load-time resharding which enables saving in one cluster topology and loading into another.`（1 次）、
  `having each rank only save their local shards`（1 次）、
  `read the least amount of data necessary`（1 次）、
  `There are no guarantees of Backwards Compatibility across PyTorch versions`（1 次）、
  `a global .metadata file with the serialized metadata if rank coordination is enabled.`（1 次）、
  `a rank local __{rank}.metadata file with the serialized metadata if rank coordination is NOT enabled.`（1 次）、
  `File creation is atomic`（1 次）、
  `The checkpoint path is an empty or non-existing directory.`（1 次）、
  `Write the metadata and marks the current checkpoint as successful.`（1 次）、
  `Mismatched keys may result in hangs or errors.`（1 次）。
  ★★ **载体指纹**：`DCP_DOC_BYTES = 967566`、`DCP_DOC_SHA256_16 = "1e78400bd3be2148"`
  （与计划 §C.2b 的 **T3** 逐位一致）。
- **[C]** ZeRO（`arXiv:1910.02054`，**不确认同行评审**），§5.1 与 Figure 1。
  抓取 ar5iv（**279500 B**）。本文用到：`Mixed-precision Adam has K = 12`、
  `i.e., the additional memory required to store them is K`、
  `based on mixed-precision training with Adam optimizer.`、
  `parameter model requires 31.4GB of memory using P`、
  `while requiring 120 GB with standard DP.`（★ 后一条在页面上出现 **3** 次 ——
  三处小节复用了同一句）、`4x memory reduction, same communication volume as DP`、
  `8x memory reduction, same communication volume as DP`。
  ★★ **`K = 12` 是 `ZeRO 的举例口径`**（条件逐字：mixed-precision + Adam）—— 计划 **Z4**。

**上游契约与边界声明**

- ★★ **「同名不同指」**：本文的 `checkpoint` = **权重检查点**；
  「重算换显存」是另一件事（正文在 `autograd-and-backward/02`）。
- **禁区**：导出/推理格式 → `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；
  优化器状态的分片与通信量 → `100-parallelism/zero-fsdp/`；异步落盘 → `110-communication/README.md:12`。
  本文**一行都不重推**。
- ★ **本主题在全库零让渡**（计划 §0.7-B 的边界表）⇒ 本文的知识点由本域**自己立**。
