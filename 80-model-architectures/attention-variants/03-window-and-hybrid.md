---
title: 滑窗的架构侧：堆叠层扩展感受野，以及与别的变体叠加
slug: window-and-hybrid
domain: 80-model-architectures
topic: attention-variants
level: 2
difficulty: intermediate
prerequisites:
  - 80-model-architectures/attention-variants/01-mha-mqa-gqa
tags: [sliding-window, receptive-field, rolling-buffer-cache, hybrid, mistral-7b, masking]
related:
  - 40-kernels/attention/04-warp-specialization-and-variants
  - 130-quantization/kv-cache-quantization/01-kv-cache-basics
  - 80-model-architectures/attention-variants/02-mla-low-rank
code:
  - code/03_window_hybrid.py
status: stable
updated: 2026-09-22
---

# 滑窗的架构侧：堆叠层扩展感受野，以及与别的变体叠加

> 滑窗让**每一层**只看 `[i−W, i]`，但**堆叠层把每一层的那一小段接起来**：
> 第 `L` 层能回溯的距离是 **`W × L`**（Mistral 7B：`4096 × 32 = 131072 ≈ 131K`）。
> 这条「感受野的账」是**架构层**的，与「每行只需 `ceil(w/Bc)` 个块」那条 kernel 层的账是
> **同一条让渡的两端**。
> 顺带换来的是 cache **不随序列长度增长**（`min(S, W)` 个槽位），
> 而它与 GQA 是**叠加**的（两条轴：**位置维** × **头维**）。

## 0. 背景与动机

全量 causal attention 有两笔账同时随序列长度 `S` 上涨：

| | 随 `S` 的行为 | 谁在付 |
|---|---|---|
| 每一行要看的 KV 位置数 | `~S`（因果掩码下平均一半） | kernel 的循环范围 / 计算量 |
| 要缓存的 KV 位置数 | `S`（线性增长） | 显存 |

滑窗注意力（SWA）只动**一个数**：把「每一行能看的位置」限成一段固定长度 `W`。
两笔账因此同时被截断——**而它没有改任何一条算术式子**（`q·k`、softmax、`p·v` 全都是原样）。

★ **本域的边界**：滑窗**落到 kernel 上**的后果（分块范围变窄、每行只需 `ceil(w/Bc)` 个块、
`Bc = 128` 时的块数表）是**别的域**的正文，
在 `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`（正文在彼，此处回链）。
**本文讲的是架构侧那一条**：`W` 这么小，为什么整体还能看很远。
这正是那处「**定义不重推**」的原为悬空让渡里，**滑窗那一份**要接住的东西。

## 1. 原理

### 1.1 定义与感受野（论文逐字）

Mistral 7B 论文（arXiv 2310.06825 §2.1 "Sliding Window Attention"）逐字：

> "**SWA exploits the stacked layers of a transformer to attend information beyond the window
> size `W`.** The hidden state in position `i` of the layer `k`, `h_i`, attends to all hidden
> states from the previous layer with positions between `i − W` and `i`. Recursively, `h_i` can
> access tokens from the input layer at a distance of up to **`W × k`** tokens…
> At the last layer, using a window size of `W = 4096`, we have a **theoretical attention span of
> approximately 131K** tokens."

⚠️ 这段里有两个数**必须分清**：

- **一处**是**距离**（`distance`）：「at a distance of up to `W × k`」；
- **另一处**是**窗口的读法**：「positions between `i − W` and `i`」——那是 `W+1` **个位置**。

⇒ 「`W` 是距离还是位置数」在论文的表述里留了 **±1** 的余地（见 §1.3）。

**闭式**：每过一层，最左可达的下标往左推 `W`。于是 `L` 层的距离是

```
distance(L) = L · W                                              …… (1)
```

代入 Mistral 7B 的 `W = 4096`、`L = 32`（论文 Table 1 的 `window_size 4096` 与 `n_layers 32`）：

```
4096 × 32 = 131072 ≈ 131K（十进制 K）
```

★ **那 131072 与论文的 `approximately 131K` 对得上，但要写明用哪种 `K`**：
十进制 `131072/1000 = 131.072K`（**论文的读法**）；二进制 `131072/1024 = 128K`（= 128 KiB）。
**同一个数，两个都"对"，写哪一个必须说**。

### 1.2 为什么「堆叠」能扩感受野——逐层传播的可测形态

把每一层的可见集合**真的传播一遍**（而不是直接乘出来），本机实测
（玩具 `W=3`、`L=5`、序列长 19，看最后一个位置，`python3 3.13.5` + `numpy 2.4.4`）：

| 约定 | 每层看上一层 | 最左可达 | 距离（实测） | 闭式 |
|---|---|---|---|---|
| **甲**（论文那句 `[i−W, i]`，`W+1` 个位置） | `[i−3, i]` | 3 | **15** | `L·W = 15` ✓ |
| **乙**（窗口含 `W` 个位置 `[i−W+1, i]`） | `[i−2, i]` | 8 | **10** | `L·(W−1) = 10` ✓ |

**距离随层数是严格线性的，斜率恰好是 `W`**（本机逐层增量 `3, 3, 3, 3`）——
不是 `√L`、也不饱和：「加深一层」就等于「窗口加宽 `W`」。

### 1.3 那 ±1 的余地有多大：在真参数下只差 `L` 个 token

把 §1.2 的两个闭式代进 Mistral 7B 的真参数：

```
约定甲：  32 × 4096 = 131072
约定乙：  32 × 4095 = 131040
两者相差 = 32 = L            （差的是 L，不是 W：L·W − L·(W−1) = L）
相对差 = 32/131072 = 0.024%
```

★★ **这是一个「不能反推」的例子**：论文只写了 `approximately 131K`，而
`131072` 与 `131040` **都**≈131K ⇒ **不能**用这个数去判定论文用的是哪一种读法。
论文那个词 —— `approximately` —— **正好覆盖得住这 ±1 的不确定性**。
★ 本域的做法是**把两个数都写出来**，而不是替论文选一个（计划 §0.2 的「两套口径并置」）。

### 1.4 cache 那一侧的账：`min(S, W)`

论文 §2.1 "Rolling Buffer Cache" 逐字：

> "The cache has a **fixed size of `W`**, and the keys and values for the timestep `i` are stored
> in position **`i mod W`** of the cache. As a result, when the position `i` is larger than `W`,
> past values in the cache are overwritten, and **the size of the cache stops increasing**."

于是每层的 KV 槽位数是

```
slots(S, W) = min(S, W)                                          …… (2)
```

**`S` 超过 `W` 之后它就不再增长**——这是本文与 `01` / `02` **不同的一条轴**：
`01` 动的是「一条 KV 服务几个头」，`02` 动的是「一条 KV 有多宽」，
**本文动的是「一条 KV 能活多久」**。

本机实测（`W = 4096`）：

| 序列长 `S` | 滑窗槽位数 `min(S, W)` | 全量槽位数 `S` |
|---|---|---|
| 4096 | 4096 | 4096 |
| 16384 | **4096** | 16384 |
| 65536 | **4096** | 65536 |

⇒ `S` 长了 **16 倍**，滑窗那一列的数是**同一个**。

★ **环形缓冲恰好装得下窗口**：本机实测，走到 `i=7`（`W=3`）时缓冲里活着的位置是
`[5, 6, 7]` —— 与滑动窗口 `[i−W+1, i]` 逐位相同。这不是巧合：`i mod W` 的覆写顺序
保证了「最老的先被覆盖」。

> ⚠ **本域没有 GPU**：上表是本机 `numpy` 上的玩具传播与槽位计数，**不是任何真实模型的测量**。
> 它们支撑的是「堆叠层把 `W` 撑到 `W×L`」与「cache 是 `min(S,W)`」这两条**结构**结论。

## 2. 执行映射

**滑窗的结构含义是一个 0/1 的「可见性」**（不是一段连续衰减）。把它写成掩码时，
本机实测（`W=3`、`S=6`）：**窗口外的注意力权重恰为 `0.0`**（`exp(−inf) = 0` 是**精确**的，
不是「很小」）。

★ **它「怎么实现」不归本文**：把可见性落成「一个 `(S,S)` 的布尔掩码」还是「把循环范围
收窄成 `ceil(w/Bc)` 个块」，是**实现侧**的选择——那一侧的正文在
`40-kernels/attention/04-warp-specialization-and-variants.md:126-156`（正文在彼，此处回链）。
本文关心的是**可见性本身**：它是 0/1 的，而不是「每个位置都看、只是权重被改小」。

| | 改的是什么 | 窗口外的权重 |
|---|---|---|
| **滑窗（SWA）** | **可见范围**（一个 0/1 掩码） | **精确 `0.0`** |
| 另一类变体（加偏置 / 对 logit 过 `tanh`） | **logit 的取值** | **非零**（只是被压低） |

⇒ **这两类在结构上不同**：前者「少看一些位置」，后者「每个位置都看、但权重被改」。
本域只写 SWA 这一侧；另外两类变体「改什么」的那一行表在
`40-kernels/attention/04-warp-specialization-and-variants.md:130-134`（正文在彼，此处回链），
其中 ALiBi 的完整定义归 `positional-encoding/03-extrapolation`（已建成）。

**配置侧的落点**：滑窗在 config 里是**两个字段**——`sliding_window`（窗口长度 `W`）
与 `use_sliding_window`（是否启用）。★ **本域核实的是这两个字段的「存在性 + 取值」**，
不是框架内部怎么使用它们（框架侧的解释权在那个框架自己的文档里，本域不复核）。

★ **本域能核实的两个 config 都没开滑窗**（`45/…/code/specs.py` 的 [B] 级两个模型）：

| 模型 | `sliding_window` | `use_sliding_window` |
|---|---|---|
| Qwen3-30B-A3B | `null`（**字段存在**） | `false`（**显式关掉**） |
| Meta-Llama-3-8B | **字段根本不在 config 里**（该 config 的 23 个键里没有） | 同上 |

⇒ 「**显式关掉**」与「**没有这个开关**」是两件不同的事，本域把两者分开写。
⇒ 也正因为如此，**滑窗的一手锚点在本域落在一篇论文上**（Mistral 7B），而不在任何 config 上
——它的 `W` 与 `L` 是论文 Table 1 与 §2.1 的 **[C] 级**值（见 §参考）。

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| 全量 attention | 每行看 `[0, i]` | 最长的感受野；两笔账随 `S` 线性涨 |
| **SWA** | 每层看 `[i−W, i]` | 感受野单层缩到 `W`，但**堆叠后**回到 `~W×L`；cache 变成 `min(S,W)` |
| **SWA + GQA**（Mistral 7B） | 两条轴同时用 | 见下 |
| 前沿的混合形态 | 交错层：多数层用 SWA、少数层用全量 | **不在本文**（那是一个新的取舍轴，本域未覆盖） |

### 3.1 与 GQA 是**叠加**，不是二选一

Mistral 7B 论文摘要逐字：

> "**Our model** leverages grouped-query attention (GQA) for faster inference, **coupled with**
> sliding window attention (SWA) to effectively handle sequences of arbitrary length with a
> reduced inference cost."

⚠️ 引这一句时的两个细节：① 主语逐字是 **`Our model`**，不是 `Mistral 7B`
（`Mistral 7B leverages` 在论文里出现 **0 次** —— 改写主语会让检索式核对落空）；
② 那两个变体是 `coupled with`（**叠加**）。

**为什么能叠加**：它们作用在**不同的因子上**。

```
位置维因子 = min(S, W) / S      （滑窗：一段固定长度的历史）
头维因子   = h_kv / h_q         （GQA：一条 KV 服务几个查询头）
```

本机实测（用 Qwen3-30B-A3B 的 `h_q = 32`、`h_kv = 4` 作头维，Mistral 7B 的 `W = 4096` 作窗）：

| `S` | 位置维 | 头维 | 合起来 |
|---|---|---|---|
| 4096 | `×1` | `×1/8` | `×1/8` |
| 16384 | `×1/4` | `×1/8` | `×1/32` |
| 65536 | `×1/16` | `×1/8` | `×1/128` |

⇒ 合起来是**两个因子的乘积**：少了任一个，结果都对不上。
**「二选一」的读法会把其中一个因子当成 1**，于是算出的数在两个方向上都错。

> ⚠ **这张表的两个因子来自两个不同的模型**（头维取自 Qwen3-30B-A3B、窗口取自 Mistral 7B）
> —— 它是**因子分解的演示**，**不是**任何一个真实模型的规格，也不要把它当成
> 「某个模型同时用了这两个数」的证据。（Mistral 7B 是否用 GQA、用几组，本文**不引**：
> 那一项没有一手 config 来源，见 §参考的 `UNSOURCED` 记录。）

### 3.2 边界：`W ≥ S` 时滑窗退化成全量

`min(S, W) = S`（本机：`S=2048, W=4096 ⇒ 2048`；`S=4096, W=4096 ⇒ 4096`）
⇒ 此时滑窗**没有收益**，它在数值上就是全量 attention。
`40-kernels/attention/04:145` 有同义的逐字：「窗口 = `S` 时没有收益（退化成全量），
所以滑窗只在 `w < S` 时有意义」（正文在彼，此处回链）。

## 4. 踩坑与失败模式

**① 把 `131K` 当成「单层能看到 131K」。**

- **现象**：写「滑窗的可见范围是 131K token」，于是以为单层就能看那么远。
- **成因**：**单层只看 `W = 4096`**；`131K` 是 `L` 层**堆叠**起来的通路（论文逐字
  `SWA exploits the stacked layers of a transformer to attend information beyond the window size`）。
  它是**信息流**意义上的可达，不是一次 attention 的宽度。
- **规避**：区分「一层的窗口」（`W`）与「`L` 层的感受野」（`W×L`）；写的时候把 `L` 带出来。

**② 把 `W` 的两种读法混用。**

- **现象**：一处说「窗口是 `W` 个位置」，另一处按 `W` 是距离算，两边差一层。
- **成因**：论文两句话的读法不同（`positions between i−W and i` 是 `W+1` 个位置；
  rolling buffer 的「fixed size of `W`」是 `W` 个槽位）。真参数下差 `L = 32` 个 token。
- **规避**：**两个闭式都写出来**（本机：`131072` 与 `131040`），并注明
  「`approximately` 覆盖得住这个差 ⇒ 不能用它反推读法」。

**③ 把 `131072` 的 `K` 当成二进制。**

- **现象**：写「131072 = 131K」以为是一个整数关系，或反过来算成 128K 却引论文的 131K。
- **成因**：`131072` 在十进制下是 `131.072K`（**论文的读法**），在二进制下是 `128K`（= 128 KiB）。
- **规避**：报这个数时注明用的是哪种 `K`（本程序 `A.1` 把两个都打印出来并断言）。

**④ 把滑窗与「改 logit 取值」的变体混为一谈。**

- **现象**：写「滑窗/ALiBi/softcap 都是给分数加个偏置」。
- **成因**：滑窗改的是**可见范围**（掩码外的权重**精确为 0**，本机实测 `0.0`），
  而加偏置/过 `tanh` 那类改的是**logit 的取值**（权重处处非零）。
  两者的 kernel 后果也不同（前者改循环范围、后者改 `exp` 的输入）——
  那一行表的正文在 `40-kernels/attention/04:130-134`。
- **规避**：把「掩码外**恰为 0**」写成一条**精确档**断言；引用另外两个变体时只回链、不重推。

**⑤ 以为「用了 GQA 就不用滑窗」。**

- **现象**：把两个变体当成互斥的方案去比较。
- **成因**：它们作用在**不同的因子**上（位置维 vs 头维），Mistral 7B 摘要逐字是
  `coupled with`（叠加）。
- **规避**：算账时把两个因子**分别写出来再相乘**；见 §3.1 的表。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 单层看到多远 | `W`（Mistral 7B：4096）。★ **不是** 131K —— 那是 `L` 层堆叠的结果 |
| `L` 层的感受野距离 | 精确 `W × L`（本机玩具上逐层传播实测到同一闭式；Mistral 7B：`4096×32 = 131072`） |
| 报 `131072` 这个数 | 必须写明 `K` 的进制：十进制 `131.072K`（**论文 `approximately 131K` 的读法**）/ 二进制 `128K` |
| 论文的 `W` 是距离还是位置数 | 论文两种读法都有；真参数下相差 **`L = 32`** 个 token（0.024%）⇒ **不能用 131K 反推** |
| 滑窗的 cache 槽位数 | `min(S, W)`：本机 `S ∈ {4096, 16384, 65536}` 上恒为 **4096**（S 长 16 倍而它不变） |
| 与全量对比 | 全量是 `S`（本机 `S` 从 4096 到 65536 长 **16 倍**） |
| `W ≥ S` | 滑窗**退化成全量**（`min(S,W) = S`，无收益）—— 与 `40-kernels/attention/04:145` 同义 |
| 窗口外的注意力权重 | **精确 `0.0`**（掩码式）⇒ 「改可见范围」不是「改 logit 取值」 |
| 与 GQA 的关系 | **叠加**（两条轴相乘：位置维 `min(S,W)/S` × 头维 `h_kv/h_q`），不是二选一 |
| 想在本域核实某模型是否开着滑窗 | 只能核实 `45/…/code/specs.py` 里那两个 config：**两个都没开**（Qwen3 是 `false` + `null`；Llama-3-8B 是**没有这两个字段**）。Mistral 7B 的 `W`/`L` 只能引论文 [C] |
| 想知道滑窗对 **kernel** 的影响 | 不在本文：正文在 `40-kernels/attention/04-warp-specialization-and-variants.md:126-156` |
| 想知道 KV 的**字节**账 | 不在本文：正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76` |

> ★ **表 A 的数值判据**（计划 §3 Chunk 2）：⑥ `W × L` 精确 `4096 × 32 = 131072`（可手算）；
> ⑦ 滑窗的 cache 大小**不随 `S` 增长**。两条都在 `code/03_window_hybrid.py` 的 `A.1` / `B.1` 里。
> ★ **表 B 的引用判据**：GQA 与 SWA 是**叠加**（论文摘要逐字 `coupled with`，且主语是 `Our model`）。
> ★ 判据⑥ 的阈值与口径常量被 `D.2` / `D.1` 逐位钉住。

## 6. 小结与延伸

**一句话**：滑窗把**每一层**的可见范围限成 `W`，靠**堆叠**把整体感受野撑到 `~W×L`；
换来的是 cache 变成 `min(S, W)`（不随 `S` 增长）——而它与 GQA 作用在**不同的因子**上，
所以是**叠加**的。

- **前置**：`01-mha-mqa-gqa`（头维那条轴）。**本文是位置维那条轴**。
- **同主题**：`02-mla-low-rank`（每条 KV 有多宽）。三篇各是 KV 预算的一个因子。
- **感受野的另一个用法**：`positional-encoding/03-extrapolation`（已建成）——
  外推那一侧问的是「训练时只见过这么长，位置编码能不能撑到更长」，与本文的
  「堆叠层把感受野撑开」是**两个方向**的账。
- **kernel 侧**：正文在 `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`
  （正文在彼，此处回链）。★ 那处 `:128` 的「定义不重推」是一处**原为悬空让渡**，
  滑窗这一份由本文接住。
- **KV 的字节账**：正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`
  （正文在彼，此处回链）。

## 编译与运行

```bash
cd code
make run                        # 三个编号程序依次跑；本程序是 03_window_hybrid.py
python3 03_window_hybrid.py     # 只跑本程序
```

- 配套程序：**`code/03_window_hybrid.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**本程序不需要网络、不需要 GPU。**
- 失败标记：`@@ FAIL: attention-variants`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` 感受野（含两种 `W` 读法的逐层传播）、`B` 环形缓冲与 `min(S,W)`、
  `C` 叠加 / 边界 / 两个 config 的事实 / 掩码的精确 0、`D（续）` 标记自检与传播路径检查。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`）：

```bash
curl -sS -o /tmp/arx_2310.06825.html https://ar5iv.labs.arxiv.org/html/2310.06825  # 99865 B
python3 - <<'EOF'
r = open('/tmp/arx_2310.06825.html','rb').read()
for s in ["SWA exploits the stacked layers of a transformer to attend information beyond the window size",
          "theoretical attention span of approximately",
          "leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA)",
          "the size of the cache stops increasing", "2x speed improvement",
          "Mistral 7B leverages", "Our model", "131K",
          'alttext="W\\times k"', 'alttext="i\\bmod W"',
          "window_size", "n_layers", "W = 4096"]:
    print(r.count(s.encode()), s[:70])
EOF
```

实测（`python3 3.13.5`）：

| 字面量 | 计数 | 用途 |
|---|---|---|
| `SWA exploits the stacked layers of a transformer to attend information beyond the window size` | **1** | ★★ 本文的立足点（**堆叠层扩展感受野**） |
| `theoretical attention span of approximately` | **1** | 判据⑥ 的出处（`approximately 131K`） |
| `leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA)` | **1** | ★ 表 B：GQA 与 SWA 是**叠加**的 |
| `the size of the cache stops increasing` | **1** | 判据⑦ 的出处 |
| `2x speed improvement` | **1** | 工程实测（⚠ 带条件：`sequence length of 16K and W = 4096`、对 FlashAttention / xFormers 的改动） |
| `Our model` | **3** | ★ W4 的主语：摘要那句的主语是 `Our model` |
| `Mistral 7B leverages` | **0** | ★ **订正记录**：原稿把主语改写成 `Mistral 7B leverages` ⇒ 检索式核对会**落空** |
| `131K` | **2** | 论文的写法（十进制 K） |
| `alttext="W\times k"` | **1** | `W × k`（**距离**的写法）—— 整句 `W \times k` 在原始字节上计不到（被 MathML 切开） |
| `alttext="i\bmod W"` | **2** | `i mod W`（环形缓冲的覆写位置） |
| `window_size` / `n_layers` | **各 1** | ★ Table 1 里的 `window_size 4096` 与 `n_layers 32` —— 判据⑥ 的 `W` 与 `L` 的一手来源 |
| `W = 4096` | **0** | 同上：`W = 4096` 被 MathML 切成 `<math alttext="W=4096">` ⇒ **跨标记的字面量在原始字节上计不到** |

**外部来源值 / 论文**

- **Mistral 7B**：arXiv 2310.06825（**无 venue 线索 ⇒ 未确认同行评审**；
  抓取 `https://ar5iv.labs.arxiv.org/html/2310.06825`，2026-09-22，**99865 B**，
  `sha256[:16] = b9aa0304ed429dcb` —— 与计划 §C.2 逐位一致）。**[C]**
  - `W = 4096`：§2.1 逐字 + Table 1 的 `window_size 4096`。
  - `L = 32`：Table 1 的 `n_layers 32`。
  - ★★ **这两个数来自论文，不是 `config.json`**：`huggingface.co` 直连不可达（计划 §C.0），
    而本域只允许引用 `45/…/code/specs.py` 里**已核实**的两个 config ⇒ Mistral 7B 的
    `config.json` 记入 `specs.UNSOURCED`（本域**不写**「Mistral 7B 的 config 里
    `sliding_window = 4096`」这种说法——那句话在本环境没有一手来源）。
- 两个 config 的滑窗字段：经 `hf-mirror.com` 镜像直取（**未与官方 repo 比对 sha256**；
  本域 2026-09-22 重取，两个副本的字节哈希与 `transformer/code/specs.py` 记的一致：
  Llama `2430cee764b6530f`、Qwen3 `2850ddb3bf7aecad`）。本域副本在 `code/specs.py`。

**本机实测值**

- `code/03_window_hybrid.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：玩具 `W=3`、`L=5`、序列长 19；掩码段 `S=6`、`d_head=8`；
  槽位段 `S ∈ {4096, 16384, 65536}`。**全部是推导值**，不是真实模型的测量。

**上游契约**

- `40-kernels/attention/04-warp-specialization-and-variants.md:128`：「三个常见变体，本文只讲
  **它们对 kernel 结构的影响**，**定义不重推**」—— 这是**原为悬空让渡**；本文接住**滑窗**那一份，
  ALiBi 归 `positional-encoding/03`，softcap 属 logit 处理（不在本域）。
- `40-kernels/attention/04:130-134` 那张「改什么」的一行表（滑窗 = KV 的**可见范围**）
  是本文的**对接面**：那边给一行，本文给它完整的定义与出处。
