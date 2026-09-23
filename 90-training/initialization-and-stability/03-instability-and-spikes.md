---
title: loss spike：现象、已被证伪的归因、以及仍是假设的归因
slug: instability-and-spikes
domain: 90-training
topic: initialization-and-stability
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/initialization-and-stability/02-clipping-and-monitoring
  - 90-training/optimizers/02-adam-and-adamw
tags: [loss-spike, training-instability, adam, beta2, gradient-norm, attribution]
related:
  - 90-training/mixed-precision/02-amp-execution
  - 90-training/optimizers/03-schedules-and-clipping
code:
  - code/03_spikes.py
status: stable
updated: 2026-09-23
---

# loss spike：现象、已被证伪的归因、以及仍是假设的归因

> **本文能写的只有三样**：**现象**（可复现的数）、**已被原文证伪的归因**、**仍是假设的归因**。
> ★★★ 这一条不是措辞偏好，是**来源本身的状态** —— 「loss spike 由梯度范数突增引起」在
> **本域取到的最好的一手来源里仍被显式标注为假设**（Spike No More 摘要第一句逐字
> `Based on the assumption that …`）。
> ★ 而最反直觉的一条**本机可量**：**spike 那一步的 Adam 更新量反而【变小】**
> （`0.702777` vs 正常 `0.999999`）—— **代价不在那一步，而在随后的几百步**。

## 0. 背景与动机

### ★★★ 先说纪律：三条归因，三种措辞

| 归因 | 状态 | 逐字依据 |
|---|---|---|
| ★★ 「**loss spike 由梯度范数突增引起**」 | **仍是假设** | Spike No More 摘要逐字：`Based on the assumption that the loss spike is caused by the sudden growth of the gradient norm` |
| 「spike 是**坏数据**造成的」 | ★★ **被原文否认** | PaLM §5.1 逐字：`We do not believe that the spikes were caused by` … `“bad data” per se`（后文给了它的替代说法：`This implies that spikes only occur due to the combination of specific data batches with a particular model parameter state.` —— ★ 那是 `This implies`，**推断**，不是机制证明） |
| ★★ 「**开了 clip 就不会 spike**」 | ★★ **错** | PaLM 同一句逐字：`spikes in the loss roughly 20 times during training, despite the fact that gradient clipping was enabled` |
| 「两个**条件**（`small sub-layers and large shortcut`）能防住」 | ★ **论文的建议，不是定理** | 原文逐字：`Our findings suggest that stabilizing the pre-training process requires two conditions: small sub-layers and large shortcut.` |

★ **最后一行还要再收窄一处**：计划 §C.2 把论文的结论转述成「**充分条件**」，
而**原文里没有 `sufficient condition` 这个词**（实测 **0** 次）——
论文的措辞是 `Our findings suggest` + `requires two conditions`。
⇒ 本文**只引原文那两句逐字**（`C.3` 把三条措辞的**逐字**都钉住）。

### ★★ 本文的范围（与禁区的界线）

**loss spike 的机制层**在本域取到的一手文献里**没有可定位的结论**（`Z2`）
⇒ 本文写的是三件事：**现象**（本机可复现的数）、**已被证伪的归因**、**仍是假设的归因**。
★ 下面这些**都在别处**，本文**只回链、不重推**：

- **`LR warmup` 的机制解释**（post-LN 的平均场理由）→
  `80-model-architectures/normalization/02-placement-and-qk-norm.md:83,93-97,100,278`；
- **残差流的范数随深度增长**（`small sub-layers and large shortcut` 里的 `shortcut` 那一半，
  它的**结构侧**账）→ `80-model-architectures/transformer/01-block-and-residual-stream.md:191,232-233,263`；
- **误差沿深度的传播** → `130-quantization/theory/03-outliers-and-propagation.md:124,205`。

★★ **本文因此不写「为什么 sub-layer 小就好」** —— 那需要 Jacobian 谱范数的账，
而那条账的**结构侧正文在 `80`**。本文只把**可测的那一半**（Adam 的更新量在 spike 前后怎么变）量出来。

## 1. 原理

### 1.1 现象：PaLM 与 Spike No More 说了什么

| 来源 | 逐字 | 它能支撑什么、不能支撑什么 |
|---|---|---|
| **PaLM**（`arXiv:2204.02311`，**不确认同行评审**） | `spikes in the loss roughly 20 times during training, despite the fact that gradient clipping was enabled` | ★ `20` 是**最大的那个模型、那一次训练**的次数（同段还说 `were not observed when training the smaller models`）⇒ **不得**写成「训练一定会 spike 约 20 次」 |
| **Spike No More**（`arXiv:2312.16903`，计划记为**已确认 COLM 2025**） | `Based on the assumption that …` + `Our findings suggest that … requires two conditions: small sub-layers and large shortcut.` | ★ 给出的是**假设 + 两个条件**；★★ **不是**「机制已被证明」（`C.3`） |
| **PaLM 的缓解手段** | `We re-started training from a checkpoint roughly 100 steps before the spike started, and skipped roughly 200–500 data batches` | ★ 这是**工程处置**（回滚 + 跳数据）；「为什么有效」原文**没有给**机制 |

### 1.2 ★★ 本机量的那一半：**spike 那一步的更新量反而变小**

构造：`g = 0.01`（正常）、第 50 步塞一个 `g = 100`（**1e4 倍**），
Adam 默认超参（`β1 = 0.9`、`β2 = 0.999`、`ε = 1e-8`，[C] 1412.6980，**已确认 ICLR 2015**）。

```
ratio_t = |m_t/(1−β1^t)| / ( √( v_t/(1−β2^t) ) + ε )        ← Adam 的「有效更新量」（不含 lr）
```

| 步 | `ratio` | 与正常步之比 |
|---|---|---|
| 第 49 步（正常） | `0.999999` | `1.0` |
| **第 50 步（spike）** | **`0.702777`** | **`0.702777`**（**变小**） |
| 第 51 / 60 / 100 / 200 步 | `0.638694` / `0.268705` / `0.006156` / `0.001452` | **单调下降**（**长期被压低**） |

★★ **为什么变小 —— 用两个跳变说，不用一句散文**（`A.1b`）：

| | 第 49 步 | 第 50 步 | **跳变倍率** |
|---|---|---|---|
| `m` | `0.009943` | `10.008948` | **×`1006.659398`** |
| `v` | `4.784221e-06` | `10.000000` | **×`2090205.268487`** |

⇒ **`v` 的跳变比 `m` 大 `2076.4` 倍**（`v` 里是 `g²`、`m` 里是 `g`）⇒ `√v` 涨得比 `m` 快
⇒ `m̂/√v̂` 变小。★ 两个跳变**各自逐位等于它的递推闭式**（`A.1b`）——
所以这句话是**量出来的**，不是**编出来的**。

★★★ **但这条判据不是恒真的**（`A.3`）：同一批梯度交给 **SGD**，
spike 那一步的 `|lr·g|` **精确地大 `10000` 倍**（`= g_spike/g_normal`）——
**方向正好相反**。⇒ 「spike 会不会让更新量爆炸」的答案是「**看优化器**」。

### 1.3 ★ 一个更强的形态：Adam 的更新量对**绝对幅值**不敏感

「从零开始的一个 spike」（前 `t−1` 步梯度为 0、第 `t` 步是一个 `G`）：`m̂ = G`、`√v̂ = |G|` ⇒

```
ratio = |G| / (|G| + ε)          ★ 精确闭式（t = 1 时）
```

| `G` | `t = 1` 的比值 | 闭式 `G/(G+ε)` | `\|比值 − 1\|` |
|---|---|---|---|
| `1e-8` | `0.50000000000000000` | `0.50000000000000000` | `5.0e-01` |
| `1e-4` | `0.99990000999899997` | `0.99990000999900019` | `1.0e-04` |
| `1e+0` | `0.99999999000000017` | 同 | `1.0e-08` |
| `1e+6` | `0.99999999999998979` | `0.99999999999999001` | `1.0e-14` |

⇒ ★★ **幅值跨了 `1e14` 倍，而比值始终由 `G/(G+ε)` 决定** ——
这是「spike 不会让更新量爆炸」的**最强形态**。
★★ **而它同时给出了自己的作用域**（`A.4b`）：只有在 `G ≫ ε` 时才能说「与幅值无关」；
`G = 1e-8`（= `ε`）时比值掉到 `0.5` —— 那时 `ε` 不是「保护项」，而是一个**参与者**。

★ **`t → ∞` 的上界**：`(1−β1)/√(1−β2) = 3.162277660`（**有限**）
⇒ Adam 的更新量给的是「有界」，**不是**「稳定」。

### 1.4 ★★ 代价的形状：先压到最低、再按 `1/(1−β2)` 的标度恢复

本机在三个 `β2` 上扫（`A.2b` / `B.1` / `B.2`）：

| `β2` | `1/(1−β2)` | 极小值 / 正常 | 极小值出现在第 `k` 步 | 回到正常 50% 需要 `k` |
|---|---|---|---|---|
| `0.9` | `10` | `0.02000` | `66` | `143` |
| `0.99` | `100` | `0.00147` | `91` | `1266` |
| `0.999` | `1000` | `0.00127` | `97` | `10410` |

★★ **极小值的位置有闭式**（`B.1`）：spike 之后**分子**里的那一项按 `β1^k` 衰减，
当它衰减到与「正常项」`g0` 同量级时比值触底 ⇒

```
k* = ln(g0 / ((1−β1)·G)) / ln β1   =   ln(0.01/(0.1·100)) / ln(0.9)   =   65.56
```

`β2 = 0.9` 上实测 **`66`**（差 `0.4` 步）✓。★ **口径**：更大 `β2` 上分母的衰减更慢、
极小值会**往后推**（实测 `91` / `97`）⇒ 这条闭式**只在 `β2 = 0.9` 上断言**。

★★ **恢复时间与 `1/(1−β2)` 同阶**（`B.2`）：`1/(1−β2)` 每涨 10 倍，恢复时间涨
**`8.85` / `8.22` 倍**（**不是** 10 倍 —— 里面还有一个 `ln` 因子）。
⇒ 「记忆长度 ≈ `1/(1−β2)`」这句话有一个**可测的标度**，而不是一个比喻。

★ **口径**：这里量的是「**更新量**回到正常 50%」所需的步数，
**不是**「**loss** 恢复」所需步数 —— loss 的恢复依赖参数轨迹（本机没有真模型）。

### 1.5 ★★ 「开了 clip 就不会 spike」：裁剪给的是**逐点**上界

裁剪**确实**给出一个硬上界：每一步 `‖Δw‖ = lr·max_norm`（逐位可测）。
但它**不约束轨迹**（`C.2`）：

```
每步   ‖Δw_t‖ = lr·max_norm = 0.01
N 步   Σ_t ‖Δw_t‖ = N·lr·max_norm = 20000 · 0.01 · 1 = 200      ← 是 max_norm 的 200 倍
```

⇒ ★★ **逐点的上界推不出轨迹的稳定性**。这正是 PaLM 的实测形态
（裁剪开着、spike 仍发生约 20 次）的**数值侧**解释 ——
★ 而它**不是**「裁剪没用」：PaLM 用的缓解手段（回滚 + 跳数据）是另一回事，
「为什么有效」原文**没有给**。

### 1.6 ★ 一个 `inf` 就够：`m`/`v` 被永久污染

`inf` 一旦进 `m` 或 `v`，**每一步都是 `nan`**（`C.1`：第 1 步塞一个溢出成 `inf` 的梯度，
之后 100 步的梯度**完全正常**，但 **`101/101` 步全是 `nan`**）。
⇒ 训练**不会自愈**。★ 这是「必须靠**溢出检测 + 跳步**」的负面证明
（接口：`90-training/mixed-precision/02-amp-execution.md`；★ 同一个现象的另一个侧面在
`90-training/optimizers/03-schedules-and-clipping.md` 的 `C.1`：那一边是从**裁剪**看的）。

## 2. 执行映射

### 2.1 spike 发生了，先做哪三件事

| 步骤 | 做什么 | 为什么（本文/别处的依据） |
|---|---|---|
| 1 | **保住证据**：`T_before` 的曲线、触发率、`‖Δw‖` | ★ **不要**只看裁剪后的范数（`02` 的 `A.2b`：它恒 ≤ `max_norm`、报警不可能响） |
| 2 | **处置溢出**：检测到 `inf`/`nan` 就**跳步** | 裁剪**替代不了它**（§1.6：`101/101` 步 `nan`） |
| 3 | **回滚 + 跳数据**（PaLM 的做法：回滚约 100 步、跳约 200–500 个 batch） | ★ PaLM 逐字给了这个做法，但**没有给机制解释** —— 转述时**只能**说「原文的做法」 |

### 2.2 不要做的两件事

- ★★ **不要**把「spike 是坏数据造成的」当成结论（PaLM **否认**了它）；
- ★★ **不要**把「spike 由梯度范数突增引起」写成「成因是…」（原文是 `Based on the assumption that`）。

### 2.3 本机实测的一组典型值

| 量 | 值（`g = 0.01`、第 50 步 `g = 100`、`β1/β2/ε = 0.9/0.999/1e-8`） |
|---|---|
| 正常步的 `ratio` | `0.999999` |
| **spike 步的 `ratio`** | **`0.702777`**（**变小**，不是爆炸） |
| spike 后 1/10/50/150 步 | `0.638694` / `0.268705` / `0.006156` / `0.001452`（**单调下降**） |
| `v` 的跳变 / `m` 的跳变 | `×2090205` / `×1006.66`（**`v` 多 `2076.4` 倍**） |
| SGD 在同一步的 `\|lr·g\|` 之比 | **`10000`**（爆炸） |
| 「从零 spike」在 `t = 1` 的比值 | `G/(G+ε)`（`G = 1e6` 时偏差 `1.0e-14`） |

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| **不管** | 让它 spike | 有时会自愈；代价是**有时不会**（`C.1` 的 `nan` 永不自愈） |
| **降 `lr`** | 手动或按规则调小 | 简单；代价是**全局**变慢（不是只针对那一步） |
| **梯度裁剪** | `coef = max_norm/(T+ε)` | 给**逐点**上界；★ 代价是它**推不出轨迹稳定**（§1.5） |
| **回滚 + 跳数据** | PaLM 的做法 | 实测有效；★ 代价是**机制未明**（原文没给） |
| **溢出检测 + 跳步** | 检测 `inf`/`nan` 就跳过那一步 | 唯一能挡住 §1.6 那种**永久污染**的手段（正文在 `mixed-precision/02`） |
| **调 `β2`** | 把 `β2` 调小 | ★ 缩短「恢复时间」（`1/(1−β2)` 的标度，`B.2`）；代价是**二阶矩的更短记忆**本身有代价（更抖） |
| **小 sub-layer + 大 shortcut** | Spike No More 的**两个条件** | ★ **论文的建议**（`Our findings suggest`）；★ 它的**结构侧**账在 `80`（本文不重推） |

## 4. 踩坑与失败模式

**① ★★★ 把假设写成结论。**

- **现象**：写「loss spike 的成因是梯度范数突增」。
- **成因**：本域**最好**的来源（Spike No More）在**摘要第一句**就写着
  `Based on the assumption that …`。
- **规避**：措辞由 `specs.SNM_ASSUMPTION_QUOTE` 决定；`C.3` 把三条逐字钉住。

**② ★★ 把「坏数据」当原因。**

- **现象**：spike 之后去查数据管线。
- **成因**：**原文否认了这条**，而且给了反证：把 spike 前后的数据拿来、
  从**更早的检查点**重训，**没有再出现 spike**。
- **规避**：引用 `specs.PALM_BAD_DATA_DENIAL_QUOTE`；★ 也不得把原文的 `This implies …` 升格成结论。

**③ ★★ 以为「开了 clip 就不会 spike」。**

- **现象**：裁剪开着，spike 照来。
- **成因**：裁剪约束的是**单步**的梯度范数（`lr·max_norm`），
  而**位移**是 `N·lr·max_norm`（随 `N` 线性增长，`C.2`）。
- **规避**：把「逐点 vs 轨迹」区分写清楚；★ PaLM 的实测就是这条的现场。

**④ ★★ 以为 spike 那一步的更新量会**变大**。**

- **现象**：一看到 spike 就去降 `lr`（因为「更新量爆炸了」）。
- **成因**：**Adam 不是这样**——`v` 被 `g²` 抬得比 `m` 更多 ⇒ `m̂/√v̂` **变小**
  （`0.702777`）。
- **代价**：真正的伤害在**随后几百步**（`0.638694 → 0.001452`，单调下降）——
  也就是说，**spike 之后的那一段「看起来平静」的训练其实在空转**。
- **规避**：同时看 `m̂/√v̂`（或 `‖Δw‖`）的**长窗口**，而不是只看 spike 那一步。
- ★★ **但这条只对 Adam 成立**：SGD 在同一步**真的爆炸 `1e4` 倍**（`A.3`）。

**⑤ ★★ 只报「极小值」不报「恢复时间」。**

- **现象**：说「spike 之后更新量被压到近乎 0」就结束了。
- **成因**：真正决定账单长度的是 `β2`（恢复 `50%` 需要 `143` / `1266` / `10410` 步，
  对应 `1/(1−β2) = 10 / 100 / 1000`）。
- **规避**：把恢复时间与 `1/(1−β2)` 的**标度**一起写（`B.2`）。

**⑥ ★★ 拿一个 `inf` 赌它会自愈。**

- **现象**：`nan` 出现后继续训，指望下一步把它冲掉。
- **成因**：`nan` 有传染性（`m`/`v` 一旦被污染，后续干净梯度也救不回来）：`101/101` 步全 `nan`。
- **规避**：溢出检测 + 跳步（`mixed-precision/02`）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| spike 那一步的更新量会怎样 | **Adam 上变小**：`m̂/√v̂` 的比值 `∈ [0.65, 0.75]`（实测 `0.702777`，正常步 `0.999999`）；★ **机制 = 两个跳变**（`v` 跳 `×2090205`、`m` 跳 `×1006.66`） |
| 这条对别的优化器成立吗 | **不成立**：同一批梯度下 SGD 的 `\|lr·g\|` **精确大 `1e4` 倍** ⇒ 判据**不是恒真的** |
| 代价出现在哪 | **不在 spike 那一步**：之后 1/10/50/150 步是 `0.638694` / `0.268705` / `0.006156` / `0.001452`（**单调下降**） |
| 极小值出现在第几步 | 闭式 `k* = ln(g0/((1−β1)G))/ln β1`（本机 `β2 = 0.9`：`65.56` vs 实测 `66`） |
| 账单有多长 | 恢复（回到正常 50%）需要 `k(50%)`；`1/(1−β2)` 每涨 10 倍，`k(50%)` 涨 **`8.85` / `8.22` 倍** |
| Adam 会不会因大梯度而有过大更新 | **`t = 1` 时比值精确 = `G/(G+ε)`**（`G` 跨 `1e14` 倍都对）；★ **作用域**：只有 `G ≫ ε`；`t → ∞` 的上界是 `(1−β1)/√(1−β2) = 3.162278`（**有限**） |
| 开了裁剪能不能保证不 spike | **不能**：裁剪给**逐点**上界 `lr·max_norm`，而位移是 `N·lr·max_norm`（线性增长）；PaLM 的实测是「裁剪开着、spike 约 20 次」 |
| `inf` / `nan` 会自愈吗 | **不会**：污染后 `101/101` 步全是 `nan`（即使梯度完全正常）⇒ 必须**跳步** |
| 「spike 的成因」能写吗 | ★★★ **只能写「假设」**（原文 `Based on the assumption that …`）；「坏数据」**已被原文否认**；`small sub-layers and large shortcut` 是**论文的建议**（`Our findings suggest`），原文里**没有** `sufficient condition` |
| `β2` 能不能用 `1.0` | **不能**（`1−β2 = 0` ⇒ `v` 永不衰减 ⇒ 一次 spike 永久留存）；程序里必须拦住（`C.4`） |

> ★ 阈值与口径（`code/03_spikes.py` 的 `D.3`/`D.4` 逐位钉住）：
> `SEED = 20260952`、`G_NORMAL = 0.01`、`G_SPIKE = 100.0`（**`1e4` 倍**）、`SPIKE_AT = 50`、
> `TOTAL_STEPS = 200`、`POST_OFFSETS = (1, 10, 50, 150)`、
> `POST_EXPECT = (0.638694, 0.268705, 0.006156, 0.001452)`（计划 §3 表 A ⑥ 的四个数）、
> `SPIKE_RATIO_LO/HI = 0.65/0.75`（表 A ⑤）、`BETA2_SWEEP = (0.9, 0.99, 0.999)`、
> `RECOVER_TARGET = 0.5`、`SWEEP_HORIZON = 200000`、`SCALE_LO/HI = 8.0/9.5`、
> `TOL_CLOSED = 1e-12`、`TOL_IDENT = 1e-15`、`CLIP_LR/CLIP_STEPS = 0.01/20000`。
> ★ **`β1/β2/ε` 从 `specs.py` 读**（[C] 1412.6980，**已确认 ICLR 2015**）——
> 本主题**只消费、不重定义**。

## 6. 小结与延伸

**一句话**：loss spike 的**机制**在本域**没有一手结论** —— 能写的是**现象**（Adam 的更新量
在 spike 那一步**变小**、随后几百步被压低）、**已被证伪的归因**（坏数据、以及「开了 clip 就不会 spike」）、
和**仍是假设的归因**（梯度范数突增）。
★ 而「spike 之后那段平静的训练在空转」这件事，**只有把长窗口量出来才看得见**。

- **裁剪与范数监控的判据** ⇒ `90-training/initialization-and-stability/02-clipping-and-monitoring.md`。
- **Adam / AdamW 与「自归一化」** ⇒ `90-training/optimizers/02-adam-and-adamw.md`
  （★ 本文 §1.2 的现象是那条「自归一化」在 spike 上的形态）。
- **溢出的处置（跳步）** ⇒ `90-training/mixed-precision/02-amp-execution.md`。
- **`LR warmup` 的机制解释** ⇒ `80-model-architectures/normalization/02-placement-and-qk-norm.md:83,93-97,100,278`（**正文在彼**）。
- **残差流/深度的账** ⇒ `80-model-architectures/transformer/01-block-and-residual-stream.md:191,232-233,263`（**正文在彼**）。
- **误差沿深度的传播** ⇒ `130-quantization/theory/03-outliers-and-propagation.md:124,205`（**正文在彼**）。
- **优化器状态的分片与通信量** ⇒ `100-parallelism/README.md:14`（`zero-fsdp`）—— 本文**一行都不写**。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU**、不训练任何真模型）⇒
  **不得支撑关于大规模训练的结论**。★ 尤其：本文量的是**更新量**的形态，**不是 loss 的形态**。

## 编译与运行

```bash
cd code
make run                     # 三个编号程序依次跑；本程序是 03_spikes.py
python3 03_spikes.py         # 只跑本程序
```

- 配套程序：**`code/03_spikes.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`// FAIL: init-stability`（本主题自定义，**不跨主题照抄**）；`D.1`/`D.2` 守着它。
- 程序的输出分四段：`A` 现象（spike 那一步**变小** / 两个跳变 / 长期压制 / SGD 对照 / 从零 spike）、
  `B` 代价模型（极小值的闭式 / 恢复时间的标度）、
  `C` 边界与反例（`inf` 永久污染 / 裁剪是逐点的 / **归因纪律 P20** / `β2 = 1` 被拦住）、
  `D` 标记自检、口径与阈值钉住、specs 常量钉住、独立性证据。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。
- ★★ **`C.3` 是 P20 的可执行落法**：三条归因的**逐字原文**都钉在程序里，
  文档的措辞由它们决定（`assumption` / `suggest` / `do not believe` 三个标记词各钉一次）。

## 参考

**本机实测值 / 推导值**

- `code/03_spikes.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
  **无 GPU**。口径：`seed = 20260952`、`g_normal = 0.01`、`g_spike = 100`（第 50 步）、
  `β1/β2/ε = 0.9/0.999/1e-8`；`A.2b`/`B.2` 扫 `β2 ∈ {0.9, 0.99, 0.999}`
  （搜索窗口 `200000` 步）；`A.4` 的幅值跨 `1e14` 倍。**全部是推导值。**
- ★ 与计划 §3 表 A 的对应：**判据⑤**（spike 那一步的 `m/√v` 反而变小：`∈ [0.65, 0.75]`，
  实测 `0.70×` ✓）、**判据⑥**（spike 后 1/10/50/150 步单调下降，实测
  `0.638694 / 0.268705 / 0.006156 / 0.001452` ✓ —— 与 `D15` 逐位一致）。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.1b`（两个跳变）、`A.3`（SGD 对照）、
  `A.4`/`A.4b`（从零 spike 的精确闭式与它的作用域）、`B.1`（极小值的闭式）、
  `B.2`（恢复时间的标度）、`C.1`（`inf` 永久污染）、`C.2`（逐点 vs 轨迹）、
  `C.3`（**P20 的三条逐字**）、`C.4`（`β2 = 1` 的边界）。
  ★ 它们的必要性是**测出来的**：`A.3` 证明 `A.1` 那句「spike 不让更新量爆炸」
  **只对 Adam 成立** —— 不写它，读者会以为那是普遍规律。

**外部来源**

- **[C]** PaLM（`arXiv:2204.02311`，**不确认同行评审** —— abs 页 `Comments:` 字段不存在），
  §5.1。抓取 `https://ar5iv.labs.arxiv.org/html/2204.02311`，2026-09-23（**1372842 B**）。
  逐字（各原始字节 1 次）：`spikes in the loss roughly 20 times during training`、
  `despite the fact that gradient clipping was enabled`、
  `We do not believe that the spikes were caused by`。
  ★ **一处量法口径**：`“bad data”` 在原页上是**弯引号**，且 `This implies that spikes only
  occur due to the combination of specific data batches with a particular model parameter
  state.` 那一段在**原始字节上计 0**（句子中间夹了标记）⇒ 本域的两条常量都只取**不含换行、
  不含弯引号的片段**。
- **[C]** Spike No More（`arXiv:2312.16903`；★ 评审状态按**计划 §C.2** 记为**已确认 COLM 2025**
  —— 本次抓取的 abs 页 `Comments:` 字段实测 0 次 ⇒ **本主题未自行升格**，已登记 `UNSOURCED`），
  **摘要**。抓取 ar5iv（**395170 B**）。逐字（各原始字节 1 次）：
  `Based on the assumption that the loss spike is caused by the sudden growth of the gradient norm`、
  `Our findings suggest that stabilizing the pre-training process requires two conditions:
  small sub-layers and large shortcut.`
  ★★ **一处收窄**：`sufficient condition` 在该页**实测 0 次** ——
  计划 §C.2 把它转述成「充分条件」，本文**不沿用那个转述**。
- **[C]** Adam（`arXiv:1412.6980`，**已确认 ICLR 2015**）的默认超参
  `β1 = 0.9`、`β2 = 0.999`、`ε = 10⁻⁸`（**PDF 文本层**逐字）——
  ★ 本主题的**全部结论**都挂在这三个数上（同值位置：`optimizers/code/specs.py`）。
- **[B]** **PyTorch 2.14** 的 `torch.nn.utils.clip_grad_norm_` 页 ——
  本文的 `C.2` 用它来量「逐点 vs 轨迹」（同一个 `max_norm` 语义的规格）。

**上游契约与边界声明**

- ★★ 「**loss spike 的机制层没有可定位的一手结论**」（计划 §C.3 的 `Z2`）是本文的**根本处境**
  ⇒ 本文只写「现象 + 已被证伪的归因 + 仍是假设的归因」，并在 `specs.UNSOURCED` 里登记
  `spike_mechanism_one_source_conclusion`。
- **禁区**：`LR warmup` 的机制解释 → `80-…/normalization/02:83-100`；
  残差流/深度的账（`small sub-layers and large shortcut` 的结构侧）→ `80-…/transformer/01`；
  误差传播 → `130-…/theory/03:124,205`；状态分片与通信量 → `100-parallelism/README.md:14`。
  本文**一行都不重推**。
