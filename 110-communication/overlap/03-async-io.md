---
title: 异步落盘：它与训练计算重叠的那一半
slug: communication-overlap-async-io
domain: 110-communication
topic: overlap
level: 3
difficulty: intermediate
prerequisites:
  - 110-communication/overlap/02-how-much-can-hide
  - 90-training/checkpointing/02-sharded-save-and-load
tags: [communication-overlap, async-checkpoint, pipeline-model, time-layer]
related:
  - 110-communication/overlap/01-mechanisms
  - 110-communication/overlap/02-how-much-can-hide
  - 90-training/checkpointing/02-sharded-save-and-load
code:
  - code/03_async_io.py
status: stable
updated: 2026-10-08
---

# 异步落盘：它与训练计算重叠的那一半

> **存一个 checkpoint，为什么可以不拖慢训练？又为什么【总归要】拖慢一点？**
> 本文讲**异步落盘**与**训练计算**重叠的那一半时间账 ——
> ★ **两段（de-stage 到 staging storage + 另起线程 save）**、能藏多少、以及**为什么它不占带宽这个说法是错的**。
> ★★★ **本层是【时间层】**。
> ★★★ **判据的来源与级别**：本文的模型是 **`[E]` 级（本库推导）**（★ 与 [02](02-how-much-can-hide.md) 同一个）；
> 关于 PyTorch 异步落盘的**机制描述是 `[B]` 级产品文档**，★ **随版本变**，且官方自己标它是**实验性的**。

## 0. 背景与动机

★★★ **先说一件【必须点明】的事**（★ 本域 `§2.2` 的「极易被合并」那一类 ——
它**不是**「同名不同指」，而是**同一件事的另一面**）。
`90-training/README.md:42-43` 逐字：

> 「**「异步落盘」**（原属 `checkpointing`）——
> **它与 `110-communication` 的 `overlap` 主题（通信-计算重叠 / 双缓冲 / 分块流水）
> 是同一件事**，正文在 `110-communication/overlap/`，**此处回链**。」

★★★ **⇒ 两个域把同一件事【各写了一半】**（★ 这是本库的分工，不是重复）：

| 谁 | 写哪一半 |
|---|---|
| **本文（`110-communication/overlap`）** | ★★ **时间账** —— 两条时间轴怎么叠、能藏多少 |
| **`90-training/checkpointing/{01,02,03}`** | ★★ **存储侧的机制** —— 写盘流水、元数据、重分片 |

★ 上游那三篇里各有 **3 处**把「**异步落盘 / 通信-计算重叠**」让渡到本域（逐字：
`→ 110-communication/overlap/`）—— ★ 本文**只接住时间那一半**。

## 1. 原理

### 1.1 异步落盘的两个阶段

★★★ **官方逐字**（PyTorch `distributed.checkpoint`，`[B]`）：

> `Asynchronous version of save. This code first de-stages the state_dict on to the staging storage (defaults to CPU memory), and then calls the save in a separate thread.`

★★★ **读法（三个词都不能漏）**：
1. **`first de-stages … on to the staging storage`** —— **第一段**：把 `state_dict`
   **先搬到 staging storage**（默认是 **CPU 内存**）；
2. **`and then calls the save in a separate thread`** —— **第二段**：在**另一个线程**里 `save`；
3. ★★ **`Asynchronous`** 说的是**第二段在另一个线程** —— ★★★ **它【没有】说这两段不花时间**。

★★ ★★ **两个句柄坐实了「两段」**：官方的返回值是
`AsyncSaveResponse(staging_completion, upload_completion)` ——
★ `staging_completion` 对应**第一段**、`upload_completion` 对应**第二段**。
★★ 默认实现是 `AsyncCheckpointerType.THREAD`（★ **线程，不是进程**）。

### 1.2 时间账（★ 与 [02](02-how-much-can-hide.md) 同一个模型）

**记号**：`C` = 一段**计算**的时长、`S` = **落盘**的时长、`K` = 分块数。
★★ **这与通信那一篇完全同构** —— 只是把「通信引擎」换成「落盘引擎」：

```
T(K) = (C + S)/K + (K − 1) · max(C/K, S/K)        （★ [E] 级：本库推导）
```

★★★ **一条依赖**：**第 `i` 段的落盘，必须在第 `i` 段的计算做完之后才能开始** ——
★ 它的实物是「**第一段计算没算完，没有任何东西可落**」。

★★ **两条免费守卫**在这里的读法：
- `K = 1`（**整份一次落盘**）⇒ 时间 = `C + S`（**无重叠**）；
- `K → ∞` ⇒ 时间 = `max(C, S)`（★ 落盘与计算**完全重叠**的极限）。

★★★ **能藏住的上限 = `min(C, S)`**（★ 由**较短的那条时间轴**决定，与 [02](02-how-much-can-hide.md) 同一条结论）。

### 1.3 ★★★ 异步 ≠ 不占带宽

★★★ **这是本文最要紧的一句**：

> **`separate thread` 说的是「它【在哪】跑」，不是「它【跑多久】」。**

★★ **落盘照样占存储 / 网络带宽** ⇒ **照样花时间** ⇒
★ 它**能藏住的上限仍然是 `min(C, S)`** —— ★ **藏不完的那部分（`S − C`，当 `S > C`）仍然暴露**。
★★ 而且**它不是免费的**：`K` 段落盘要和 `K` 段计算**争**同一台机器的资源（`01` 里说的「两条引擎」）。

★★ **两条官方限定语**（★ 引这条特性**必须带上**）：
1. **它是实验性的**（逐字）：`This feature is experimental and subject to change.`
2. **两段的顺序与落点**（逐字）：见 §1.1 的定义句。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/03_async_io.py` 的 A 类是**三条**（★ **同一个模型、第三组参数** —— `SAVE_C_S_GRID`）：

| 断言 | 它判什么 |
|---|---|
| `A.1` | ★★★ **时隙模拟 == 闭式**（`SAVE_C_S_GRID` × `SAVE_K_GRID`，**12 档**） |
| `A.2` | ★★ **三条免费守卫**（`K = 1` ⇒ `C + S`；中间值 ≥ 下界；`K → ∞` ⇒ 下界） |
| `A.3` | ★★ **落盘能藏的上限 = `min(C, S)`** |

★★★ **为什么是「换一组参数」而不是「照抄 `02`」**：★ 计划 `Chunk 4` 明确要求
「**与 `02` 同模型，但换一组参数，别照抄**」——
★★ **免得两章看起来是同一组数的复述**（本库 `P1` 家族：订正/结论不扩散到该扩散的地方，
也不该**无谓地重复**）。★ 三章的参数网格**两两不重叠**。

### 2.2 ★★★ 一条必须写明的限定语

> ★★ 本机**无 GPU / 无 NCCL / 无 InfiniBand / RoCE 网卡 / 无 NVLink**，★ 也**没有任何真实存储**。
> ★★ 本文的 `C` / `S` 是**无量纲时间单位** —— **不是**任何真实系统上的秒数。
> ★★ 载体 = PyTorch `distributed.checkpoint`（`https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`，
> **`[B]` 级，产品文档**），抓取日 **2026-10-08**；★★ **不写抓取日的引文不算引文**（`P19`）。
> ★★★ **级别纪律**：机制描述是 **`[B]`**、时间模型是 **`[E]`** —— ★ **两者都不是规范**。

★★ **本文的「形态」**：编号程序**跑在默认 `python3` 下、只用标准库**
⇒ `CONTRIBUTING` §2 表的**第二种情形**（程序内断言）—— ★ **A/B/C 齐全**。

### 2.3 ★★ 覆盖边界

- **守**：**落盘与计算叠起来要多久**，与「**把异步落盘当成不占带宽会不会被拒**」；
- **不守**：**存储侧的机制**（那是 `90-training/checkpointing`），任何**真实存储 / 网络**的吞吐。

## 3. 实现与演进

### 3.1 落盘的重叠账

`code/` 的 `B.1`（`SAVE_C_S_GRID` × `SAVE_K_GRID`，★ 每一格都是模拟器排出来的）：

| `(C,S)` | 性态 | `K=1` 总时间 | `K=2` 藏住 / 暴露 | `K=20` 藏住 / 暴露 |
|---|---|---|---|---|
| (20,5) | 计算主导 | 25 | 2.5 / 2.5 | **4.75 / 0.25** |
| (5,20) | 落盘主导 | 25 | 2.5 / 17.5 | **4.75 / 15.25** |
| (9,9) | 平衡 | 18 | 4.5 / 4.5 | **8.55 / 0.45** |

★★★ **读法**：**落盘主导档（`5,20`）怎么分块都藏不掉 `S − C = 15`** ——
★★ 这正是 [02](02-how-much-can-hide.md) 的「`T > C` 时暴露 ≥ `T − C`」在落盘场景上的同一句话。

### 3.2 与相邻主题的分工（★ 本文最要守住的一条边界）

| 话题 | 在哪 |
|---|---|
| ★★ **异步落盘的【存储侧】机制**（写盘流水 / 元数据 / 重分片） | `90-training/checkpointing/01,02,03`（★ **本文一行都不重写，只回链**） |
| **状态分片与通信量** | `100-parallelism/zero-fsdp/` |
| ★★ **重叠的时间账** | **本文** + [01](01-mechanisms.md) + [02](02-how-much-can-hide.md) |

## 4. 踩坑与失败模式

### 4.1 ★★★ 把「异步落盘」当成「不占带宽」

- **现象**：算训练时间时**把落盘当成 0 开销**（「反正它在另一个线程」）；
- **成因**：★★★ 官方文档里满是 **`asynchronous`** / **`in a separate thread`** ——
  ★ 读快了就会以为**它不花时间**；
- **本机实测**（`code/` 的 `C.1`）：把「`S` 完全不进账」（= 总时间 `C`）喂进守卫，
  **三档上一律被拒** —— ★ 它给出的时间（`C`）**低于物理下界 `max(C,S)`**
  （实测 `5 < 20`）；
- **规避**：★★★ **记住 `separate thread` 说的是【在哪跑】，不是【跑多久】** ——
  ★ **落盘照样占带宽、照样花时间**；★ 能藏住的上限**仍然是 `min(C,S)`**。

### 4.2 ★★ 把「两段」读成「一段」

- **现象**：以为「异步保存 = 后台线程里的一次 `save`」；
- **成因**：★★ 官方那句是 **`first de-stages …`** **`and then calls the save in a separate thread`**
  —— ★ **第一段（de-stage 到 staging storage）不在那个线程里**；
- **规避**：★★ 记住**两个句柄**（`staging_completion` / `upload_completion`）——
  ★ 官方的返回值**本身就分了两段**（`D.5` 逐字钉住）。

### 4.3 ★★ 把「实验性特性」当稳定契约

- **现象**：把这条特性写成一个**稳定的、可依赖的**接口；
- **成因**：★★★ 官方**在同一页**逐字标了
  `This feature is experimental and subject to change.`；
- **规避**：★★ 引它**必须带上这句**（`B.2` 逐字钉住）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到「异步落盘不占带宽 / 不花时间」 | ★★★ **不成立** —— `separate thread` 只说**在哪跑**（`C.1`） |
| 见到「异步保存的机制」 | ★★ 官方逐字是**两段**：`first de-stages … and then calls the save in a separate thread`（`D.5`） |
| 见到「异步保存的返回值」 | ★★ 两个句柄 `staging_completion` / `upload_completion`（★ 对应两段，`D.5`） |
| 见到「整份一次落盘 vs 分块落盘」 | ★ `K = 1` ⇒ `C + S`（无重叠）；`K → ∞` ⇒ `max(C,S)`（`A.2`） |
| 见到「落盘能藏多少」 | ★★★ **上限 = `min(C,S)`**（`A.3`）—— ★ 与 [02](02-how-much-can-hide.md) 同一条结论 |
| 见到「落盘主导时也能藏住大半」 | ★★★ **不成立** —— 暴露 **≥ `S − C`**（`B.1` 的 `(5,20)` 档） |
| 见到「PyTorch 的异步保存是稳定接口」 | ★★ **官方自己说是实验性的**（`This feature is experimental and subject to change.`，`B.2`） |
| 见到「`overlap` 与异步落盘是两件事」 | ★★★ **是同一件事的另一面** —— 本域写**时间账**、`90-training` 写**存储侧**（§0） |

## 6. 小结与延伸

**一句话**：**异步落盘把「落盘」和「计算」放到两条时间轴上，于是能藏住 `min(C, S)`；
但它照样占带宽 —— `separate thread` 说的是「在哪跑」，不是「不花时间」。**

**三条可独立复用的结论**：

1. ★★★ **异步 ≠ 不占带宽** —— `S` 照样进账，能藏的上限仍是 `min(C, S)`；
2. ★★★ **它有两段**（de-stage 到 staging storage + 另起线程 save），官方用**两个句柄**把它暴露出来；
3. ★★ **它与 [02](02-how-much-can-hide.md) 是同一个模型** —— 只是「通信引擎」换成「落盘引擎」。

**延伸**：

- ★★ **能藏多少的判据**（可藏上限 `min(C,T)`、三条免费守卫）→ [02-how-much-can-hide](02-how-much-can-hide.md)；
- ★★★ **机制那一半**（双缓冲 / 通信分块 / 异步发起）→ [01-mechanisms](01-mechanisms.md)；
- **本层不写、只回链的正文**：
  · ★★★ **异步落盘的存储侧机制**（写盘流水 / 元数据 / 重分片）→ `90-training/checkpointing/01,02,03`；
  · **状态分片与通信量** → `100-parallelism/zero-fsdp/`。

## 参考

- **PyTorch `distributed.checkpoint`**（`https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`）——
  ★★★ **`[B]` 级（产品文档，不是规范）**。本轮 `curl --max-time 25` 实测 **`200`**（`967566` 字节），
  抓取日 **2026-10-08**。★★ 三条逐字（定义句 · 实验性警告 · 两个句柄名）注册在 `code/specs.py` 的 `# 来源：` 里。
- **本库的 `[E]` 级时间模型**（`code/03_async_io.py` 的 `cf_save_pipeline`）——
  **没有**外部出处（★ 与 [02](02-how-much-can-hide.md) 的模型同一个）。
- `90-training/checkpointing/01-state-and-bytes.md` · `02-sharded-save-and-load.md` ·
  `03-consistency-and-fault-tolerance.md` —— ★★★ **异步落盘的【存储侧】正文在彼**（★ 本文只写重叠的那一半）。
- ★ **一条术语说明**：本文与 `90-training` 的「异步落盘」**是同一件事的另一面**
  （★ **不是**同名不同指；★ `90-training/README.md:42-43` 逐字说明了这件事）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 03_async_io.py              # 只跑本文的配套程序
make 03_async_io                    # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5`，macOS arm64，**无 GPU、无 NCCL、无网卡、无真实存储**；
★★ 本目录的编号程序**只用标准库**（不需要 `numpy`）。
★★★ **本文的程序【有】A 类**：`A.1`（模拟 vs 闭式，12 档）、`A.2`（三条守卫）、`A.3`（可藏上限）——
★★ 它守的是「**落盘与计算叠起来要多久**」与「**把异步落盘当不占带宽真的会被拒**」。
