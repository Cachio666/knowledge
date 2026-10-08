# 110-communication / probes

★ **域级探测目录**：**只打印、不断言**，**不进**任何 `make run`。（★ 与 `100-parallelism/probes/` 同形态。）

## ★★★ 一条硬边界（`P36`）

本机**没有** GPU / NCCL / InfiniBand / RoCE 网卡 —— 本目录的 socket 实验走的是 **loopback**
（连一张真网卡都没过）⇒ ★★★ **这里的任何数字都不得用作带宽 / 延迟 / 吞吐的依据**，
**不得**与 `20-gpu-architecture/interconnect/01-bandwidth-conventions.md` 的规格值并列。
（★ 沿用 `100-parallelism/probes/README.md:78` 的逐字禁令。）

★ 本目录能证明的**只有一件事**：**算法在【真通信】下是对的**（数据等价、轮数、顺序）。

## 探测清单

| 脚本 | 证明什么 | 状态 |
|---|---|---|
| `probe_110a.py` | ★ **真 socket 多进程 ring allreduce**（`N=4`、6 轮通信）⇒ 四个 rank 与单进程 `sum` **逐位相同** | ✅ 连跑 **8 次全绿**，墙钟 **27–29 ms** |

（其余探测随各 chunk 落盘。）

## ★★ `probe_110a` 记下的两条实测教训（★ 都已写进脚本抬头）

1. **`buf` 必须是【完整数组】**（只有自己那一块有值）—— 第一版写成「只有每卡自己的那一块」
   ⇒ `reduce-scatter` 的累加目标**越界**（空切片）。★ 报错信息**指向形状，不指向索引**。
2. ★★★ **`connect` 失败后必须【新建 socket】再重试** —— **macOS/BSD 上同一个 socket 不能重试**：
   现象是「**只有最后启动的那个 rank 连得上**」、其余 **249 次重试全失败**、墙钟卡满 `settimeout`。
   ★ 修法实测把 **20 s 降到 ~28 ms**（★ 这是 `D1` 那个「80% 挂」的真正根因）。
