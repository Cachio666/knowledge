---
title: 两个延迟：TTFT、inter-token latency 与它们的冲突
slug: inference-serving-latency-and-slo
domain: 120-inference
topic: serving-infra
level: 3
difficulty: intermediate
prerequisites:
  - 120-inference/batching-and-scheduling/02-chunked-prefill
tags: [ttft, inter-token-latency, slo, batching, latency-throughput, serving]
related:
  - 120-inference/serving-infra/02-inference-parallelism-and-pd
  - 120-inference/serving-infra/03-scaling-and-observability
  - 120-inference/batching-and-scheduling/01-iteration-level-scheduling
  - 45-kernel-lifecycle/requirement-discovery/03-optimization-priority
code:
  - code/01_latency_and_slo.py
status: stable
updated: 2026-10-09
---

# 两个延迟：TTFT、inter-token latency 与它们的冲突

> **推理的延迟不止一个。** 用户先等「第一个字」出来（**TTFT**），然后一个字一个字地等下去
> （**inter-token latency**）。**这两个延迟由【同一个旋钮】控制，而且方向相反**：
> 把在飞请求数调大，**吞吐上去、inter-token latency 变差**。
> **没有一个 `B` 能让两头同时最优** —— 这就是服务侧一切取舍的起点。
>
> ★★★ **本主题【有真 A 类】**：本文程序用一个**离散事件模拟器**（按步推进、量事件时间戳）
> 与一套**闭式**（纯代数）**两条独立路径**互相对拍。★★ **如实标注（S1）**：`A.1`/`A.2`/`A.3`
> **共用同一对**被测/参照（**不是三条独立参照** —— 是同一个被测量了**三个不同的量**）。
> ★★ **本文只写这两个延迟的【算术与它们的关系】** —— 别处的正文一律回链：
> **chunked prefill 的分块数权衡** → [02-chunked-prefill](../batching-and-scheduling/02-chunked-prefill.md)；
> **continuous batching 的收益来源** → [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)；
> **KV 缓存的账** → `130-quantization/kv-cache-quantization/01`。
> ★★★ **限定语**：**不是任何真实引擎，也不是真机** —— 本机**无 GPU、无 vLLM / SGLang / TensorRT-LLM**。

## 0. 背景与动机

一个请求的一生，在时间轴上分成**三段**（这个划分是本文的全部基础）：

| 段 | 从哪到哪 | 谁在等 |
|---|---|---|
| **排队** | 到达 → 被调度器收进某一轮 | 用户（**完全在等**） |
| **prefill** | 收进来 → **第一个 token** | 用户（还没看到任何字） |
| **自回归解码** | 第一个 token → 最后一个 token | 用户（**一个字一个字地看**） |

前两段之和就是 **TTFT**；第三段被切成 `(输出长度 − 1)` 个间隔，每个间隔就是
**inter-token latency**（★ **vLLM 的官方词**；见 §4.1）。

**一个具体的对照场景**（本文程序的真算例，`code/01_latency_and_slo.py`）：

`60` 条请求、每条解码 `12` 个 token、**同时到达**、在飞上限 `B`。一步（一次 iteration）的
耗时模型是 `2.0 + 0.5 × 在飞请求数`。把 `B` 从 `1` 调到 `60`：

| `B` | inter-token latency | 端到端吞吐 `N·D/makespan`（token / 单位时间） |
|---|---|---|
| `1` | **`2.5`** | `0.3692` |
| `60` | **`32.0`** | `2.0168` |

⇒ ★★★ **同一个旋钮，两个走向**：`B` 变大，**inter-token latency 涨 `12.8` 倍**（流不流畅**变差**），
**吞吐涨 `5.46` 倍**（服务能力**变好**）。这不是「调参没调好」，而是**两个延迟与吞吐在同一个批里的必然交换**。

## 1. 原理

### 第一步：把「一步」的耗时写成 `n` 的函数

本文唯一的一条模型假设（★ **写明的简化，不是任何实测**）：

```
一步耗时  S(n) = α + β·n          α = 2.0（与批无关的固定开销）· β = 0.5（每请求项）
```

- `α` 是**固定项**：一步不管装几个请求都要付的开销（kernel launch、同步、调度元数据）；
- `β·n` 是**随批增长**的那部分。

★ **prefill 在本文里记为一次【独立的 prefill 步】**（固定耗时 `prefill = 5.0`），
**不占 decode 步的时间** —— ★★★ **这是一条必须写明的简化**：真实里 prefill 与 decode
**共享算力**，那套「取较慢者」的耦合模型正文在
[02-chunked-prefill](../batching-and-scheduling/02-chunked-prefill.md)，**本文不重推**（`P37`）。

### 第二步：两个延迟各自是什么

- **inter-token latency = 一步的耗时 `S(n)`** —— 一个 decode 步恰好产出 `n` 个 token（每请求一个），
  所以**相邻两个 token 之间的时间就是这一步的耗时**；
- ★★ **步吞吐 = `n / S(n)`** —— 一步产出 `n` 个 token、花 `S(n)` 时间；
- ★★ **端到端吞吐 = `N·D / makespan`** —— 整批请求的 token 数除以总时间（§0 与 §2 的表用的是**这一个**）。

⇒ ★★★ **两个吞吐口径不是同一个量**：`B=N` 时**步吞吐的渐近上界**是 `1/β = 2.0`，
而**端到端吞吐**是 `2.0168 > 2.0`（★ 一个按「一步」算、一个按「整批」算）。
★★ **本文一律标注用的是哪一个**；把两者混叫「吞吐」正是 §4 的失败模式之一。

⇒ **两者都由 `n` 决定，且一个变差、一个变好**（ITL 与吞吐的**数值都升**，但一个是延迟、一个是速率）。
这是本文全部结论的来源。

### 第三步：TTFT = 排队 + prefill（+ 首 token）

★ 在本文模型里，**首个 token 在 prefill 步末产生**，所以

```
TTFT = (等到自己那一相位开始) + prefill
```

★★ **注意它【不含】任何 decode 步** —— TTFT 只取决于「相位起点」，与**相位内的步耗时无关**。
这一点在整除批（`LAT_N_REQ` 能被每个 `B` 整除）下尤其清楚：相位数 = `⌈N/B⌉`。

### 第四步：三个量的闭式

同时到达 ⇒ 引擎按**相位**满批推进（相位 `k` 的在飞数 `n_k = min(B, N − Σ_{j<k} n_j)`）：

```text
相位 k 的耗时 = prefill + (D − 1)·S(n_k)          D = 每条请求的 token 数
makespan      = Σ_k [ prefill + (D − 1)·S(n_k) ]
TTFT(相位 k)  = (相位 k 的起点) + prefill
ITL(相位 k)   = S(n_k)
```

★ 这四条式子是**参照实现**（`closed_form()`）；**模拟器**（`simulate()`）**一个字都不用它** ——
它按步推进、量事件时间戳。`A.1`/`A.2`/`A.3` 断言两条路径**逐位一致**（容差 `1e-12`）。

## 2. 执行映射

引擎层（一个**请求调度器** + 一个前向函数）。本文的每个「量」都是**调度层的账**，
不是任何硬件数字。

**（1）一次「步」的形状**：`{ 准入（队首最多 B 条）→ 跑一个 prefill 步或一个 decode 步 }`。
★ 模拟器把这条写成一行循环，`A.2` 断言**调度步数 = 相位数 × D**（本文算例：`15 × 12 = 180`）。
★ 一旦步数不对，说明推进逻辑变了 —— 这是最便宜的一条哨兵。

**（2）`n` 是引擎里唯一的「旋钮」**：它同时定 **ITL（`S(n)`）**、
**步吞吐（`n/S(n)`）**、**TTFT（经由相位数）**。§3 的表把这三条一起列出来。

**（3）★★ 两个「延迟」的名字必须分清楚**：
- **TTFT**：只发生**一次**（每个请求一次），它是「**多久开始看到东西**」；
- **inter-token latency**：发生 `(输出长度 − 1)` 次，它是「**看到东西之后流不流畅**」。

★★★ 一个用户对「慢」的抱怨，可能来自其中**任何一个** —— 而**优化它们的旋钮是同一个、方向相反**。

**（4）本文算例的实际数字**（`make run`，`code/01_latency_and_slo.py`）：

| `B` | ITL | 端到端吞吐 `N·D/makespan` | makespan |
|---|---|---|---|
| `1` | `2.5000` | `0.3692` | `1950.00` |
| `2` | `3.0000` | `0.6316` | `1140.00` |
| `4` | `4.0000` | `0.9796` | `735.00` |
| `10` | `7.0000` | `1.4634` | `492.00` |
| `20` | `12.0000` | `1.7518` | `411.00` |
| `30` | `17.0000` | `1.8750` | `384.00` |
| `60` | `32.0000` | `2.0168` | `357.00` |

**（5）★ 这条链的产物是「机制与上界」，不是「真实系统的性能」。** 本机无 GPU、无引擎
⇒ 上表**没有一个数**来自真实部署。★★★ **限定语**：「不是任何真实引擎，也不是真机」。

## 3. 实现与演进

**第一步：一个前向函数。** 只回答「给定这一批输入，输出是什么」。**没有「一步」这个概念** ——
★ 于是**没有** inter-token latency 可谈：请求**要么全出来，要么都没出来**。

**第二步：把前向切成「步」。** 一旦每一步只推进一个 token，就**多出**一个可观测的量 ——
**inter-token latency（= 步耗时）**。★★ 这是「两个延迟」出现的那一刻。

**第三步：把 `n` 参数化。** 步耗时从常量变成 `S(n) = α + β·n`，于是**吞吐与 ITL 被同一个 `n` 绑在一起**。
★ 这就是 §0 那张表的由来 —— ★★ **代价是**「追求吞吐」与「追求流畅」变成**同一根绳的两端**。

**第四步：把 TTFT 拉进来。** TTFT 由**相位数**决定（`⌈N/B⌉`），
★★ **而相位数由 `B` 决定** ⇒ 三个量被**同一个旋钮**串起来。

★★★ **一个必须写清的点（本节最后一段）**：**在本文的模型里，`B` 越大 TTFT 反而越小**
（相位更少、排队轮数更少）。★★ **这与「批越大 TTFT 越差」的通常说法相反** ——
★ 成因是**本文把 prefill 记成独立的一步**（不与 decode 争算力）。
★★ **TTFT 变差需要 prefill 与 decode 【争算力】**，那正是
[02-chunked-prefill](../batching-and-scheduling/02-chunked-prefill.md) 的耦合模型（「取较慢者」）。
⇒ ★ **本文如实报告这一点**，并把两种到达过程下的实际走向都算出来（程序 `B.3`）。

## 4. 踩坑与失败模式

### 4.1 把社区缩写当成官方词

- **现象**：写「vLLM 用 `TPOT` 衡量每 token 延迟」；
- **成因**：★★ vLLM 的官方词是 **inter-token latency**（`mean_itl_ms` /
  `request_time_per_output_token_seconds`）；**该缩写在 vLLM 的 metrics 页面上出现 `0` 次**；
- **本库判据**（`code/01_latency_and_slo.py` 的 `C.1`）：把它当**官方词**会被**拒**（反例 → 被拒）；
  该缩写计数 = **`0`**；
- **规避**：★★ **写社区缩写时注明「这是社区常用词」**，并给出 vLLM 的官方词。

### 4.2 把两个 TTFT 口径混起来

- **现象**：拿 vLLM 的 TTFT 指标直接当「用户等了多久」；
- **成因**：★★★ vLLM 的 TTFT **从 `request was scheduled` 起算**（逐字：「Time from when the request
  was **scheduled** until the first output token was generated (TTFT).」）—— **不含排队**；
  而本库口径是「**从到达**到首 token」，**含排队**；
- **本库算例**（`C.2`）：两条口径的引文**逐字钉住**；vLLM 的 `generation_time` 还**显式排除**
  「queue wait and prefill/TTFT」；
- **规避**：★★★ **报一个延迟前先问「它的起算点是什么」** —— 起算点不同，量就不同。

### 4.3 「批越大越好」或「批越小越好」

- **现象**：为了吞吐把 `B` 拉满，用户说「卡顿」；为了流畅把 `B` 调到 1，吞吐塌了；
- **成因**：★★★ ITL 与吞吐是 `n` 的**同一个单调函数的两个读数**（§2 表：两列**同向**）；
- **本库算例**（`B.1`）：ITL `2.5 → 32.0` 与吞吐 `0.37 → 2.02` **同时上升** ⇒
  **没有一个 `B` 同时最优**（ITL 最小在 `B=1`，吞吐最大在 `B=60`）；
- **规避**：★★ **先写 SLO，再选 `B`** —— `B` 是**在 SLO 内**最大化吞吐的那一点，不是「越大越好」。

### 4.4 用「一步的耗时」当 TTFT

- **现象**：拿引擎打点里「一次 iteration 的耗时」当 TTFT 报出去；
- **成因**：★★ TTFT 里**包含排队**，而 iteration 耗时**不含** —— 两者在不同口径上；
- **本文判据**（`A.3`/`B.3`）：TTFT `= (相位起点 + prefill)`，**与相位内的步耗时无关**；
  ★ 免费守卫：**`TTFT ≥ prefill`** 逐条成立（首 token 不可能早于 prefill 走完）；
- **规避**：★★ **TTFT 是一个「含排队的端到端」量**，用它时必须说明起点。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到「inter-token latency」 | ★ 这是 **vLLM 的官方词**（`mean_itl_ms`）；★ 见到那个社区缩写时**必须注明它是社区常用词**（`C.1`） |
| 见到「vLLM 的 TTFT」 | ★★★ **起算点是 `scheduled`**（**不含排队**）⇒ 它不是用户的端到端首 token 延迟（`C.2`） |
| 见到「`generation_time`」 | ★★ 它 **排除排队与 prefill**（首 token → 末 token）—— 与 TTFT **不重叠**（`C.2`） |
| 见到「批越大越好」 | ★★★ **错**：ITL 与吞吐**同向上升** ⇒ 没有一个 `B` 两头最优（`B.1`） |
| 见到「批越大 TTFT 越差」 | ★★ **需要 prefill 与 decode 【争算力】** 才成立；★ 本文模型（prefill 独立成步）里**方向相反**（`B.3`，如实报告） |
| 见到一个「每输出 token 时间」 | ★ 先问**分母口径**：单 token 响应上它**无定义**（vLLM 逐字 `null for single-token responses`） |
| 见到「吞吐 = n / 步耗时」 | ★ 分母是**步耗时**；★ 换一个分母（如按请求数）会得到**另一个数**（§1 第二步） |
| 见到「ITL 与吞吐的取舍曲线」 | ★ 用 `S(n) = α + β·n` 就能算出端点：`B=1` ⇒ `ITL = α+β`；`B=N` ⇒ 单相位（`C.3`/`B.2`） |

## 6. 小结与延伸

**一句话**：**推理的延迟有两个 —— TTFT（多久开始）与 inter-token latency（流不流畅）；
它们与吞吐由同一个在飞请求数 `n` 绑在一起，方向相反，所以「调 `B`」是一个取舍而不是一个优化。**

**三条可独立复用的结论**：

1. ★★★ **两个延迟、一个旋钮**：见到「延迟」先问是**TTFT** 还是 **inter-token latency**；
2. ★★ **口径即定义**：TTFT 的起算点（`scheduled` vs 到达）决定它是哪个量；
3. ★★★ **「批越大 TTFT 越差」有条件** —— 它需要 prefill 与 decode 争算力；★ 把简化写明，结论才不会越界。

**延伸**：

- ★★ **推理侧的并行形态与 PD 分离** → [02-inference-parallelism-and-pd](02-inference-parallelism-and-pd.md)；
- ★★ **扩缩容 / 路由 / 指标口径** → [03-scaling-and-observability](03-scaling-and-observability.md)；
- ★★★ **chunked prefill 的分块数权衡 `k*`** → [02-chunked-prefill](../batching-and-scheduling/02-chunked-prefill.md)（**正文在彼处**）；
- ★★ **continuous batching 的收益来源（长度方差）** → [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)（**正文在彼处**）；
- ★ **真实剖面里的 TTFT 数字**（CPU 剖面的 `1070 ms`）→ `45-kernel-lifecycle/requirement-discovery/03-optimization-priority:205,315`（**正文在彼处**）。

## 参考

- **`[B]` vLLM 官方文档** —— `usage/metrics/` 与 `features/per_request_metrics/`
  （`time_to_first_token_ms` · `mean_itl_ms` · `generation_time_ms` ·
  `vllm:inter_token_latency_seconds` · `vllm:request_time_per_output_token_seconds`）；
  文档站 = **latest developer preview**（页脚 2026-10-08），对应发布版 **0.31.0**；抓取日 **2026-10-09**。
- ★★ **一条载体纪律**：vLLM 的两个 metrics 页面上**没有**那个社区缩写（`grep` 计数 = **`0`**）。
- ★ **跨域回链**（**正文在彼处，本文只回链**）：
  · chunked prefill 的耦合模型 → [02-chunked-prefill](../batching-and-scheduling/02-chunked-prefill.md)；
  · 真实剖面里的 TTFT 数字（`1070 ms`）→ `45-kernel-lifecycle/requirement-discovery/03-optimization-priority:205,315`。
- ★ **一条术语纪律**：本文一律写「**KV 缓存**」「**请求调度器**」全称。

## 编译与运行

```sh
cd code && make run                       # 三个编号程序，任一断言失败即非零退出
python3 01_latency_and_slo.py             # 只跑本文的配套程序
make 01_latency_and_slo                   # ★★ 与上一行**不等价**：它先过 check-env（查解释器 / numpy）再跑
```

★ 本机的运行条件：`python3 3.13.5`、`numpy 2.4.4`、macOS arm64，**无 GPU、无任何引擎**。
★★★ **本文的程序【有真 A 类】**（`D.0` 用 **AST 扫自己的源码**证明 `check("A…")` 的调用数 ≥ 3）：
**离散事件模拟器 vs 闭式**，两条无共享代码的独立路径。

★ `A.1`/`A.2`/`A.3` 断言**逐请求的 ITL / makespan / TTFT** 两条路径逐位一致（★★ **三者共用**同一对
被测/参照 —— 见抬头与 `D.8`）；
★ `B.1`–`B.3` 是**两个延迟的冲突**与 TTFT 的分解；★ `C.1`–`C.3` 是**禁写的真现场**与两端边界；
★ `D.4` 是**守卫实验**（把准入上限改成 `B + 1` ⇒ 断言必须失败）；
★ `D.5` 是**六条引文逐字钉住**；★ `D.9` 是**术语纪律**（`裸写被禁 ASCII 词的计数 == 显式豁免清单覆盖数`）。
★ **本文对应 `code/01_latency_and_slo.py`**（三类断言全在程序内部，失败即非零退出）。
