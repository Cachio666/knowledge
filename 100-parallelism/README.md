# 100-parallelism 并行策略

把模型切到多卡上。每种切法都有通信量与显存占用的代价，本域讲清楚这些账怎么算。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| data-parallel | DP/DDP 的**工程形态与正确性**：梯度为什么是平均不是求和、全局 batch 的等价性、副本一致性；★ **梯度规约的算法**（ring allreduce / `V(N)`）在 `110-communication`，此处回链 |
| tensor-parallel | TP 的**切法与代价**：列/行并行与它们的配对、头切分与 vocab 并行、通信量为什么也**饱和** |
| pipeline-parallel | PP 的**微批与气泡率**、GPipe / 1F1B / interleaved 三种调度、各 stage 的**负载均衡** |
| sequence-context-parallel | SP 与 CP 的**分工**（切的范围不同）、跨设备的**块统计量重缩放**、`cp` 的约束 |
| expert-parallel | EP 的**专家分片**与两次 all-to-all、**负载不均与 straggler**、EP 与其他轴的组合 |
| zero-fsdp | ZeRO **三个阶段的账**（**分片不是除法**）、通信量、FSDP 的策略与分片的正确性 |

## 与其他域的关系

- **上游**：`90-training`（它写**单卡算法**，并留一个接口给本域：**优化器状态的分片策略与通信量** —— `4x/8x` 与 `16Ψ` 的账，落到下面 `zero-fsdp` 那一栏）、`110-communication`（并行策略的通信开销由通信域定性）、`80-model-architectures`（MoE 引出专家并行）。
- **下游**：`120-inference`（推理的并行形态与训练不同，但共享概念）。
- **横切**：梯度规约算法（ring allreduce 等）（正文在 `110-communication`，此处回链）；集合通信在**拓扑**上的行为（bisection 下界 / all-to-all 的跨切口量 `∝ N` / 分层）正文在 `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；`all-to-all` 的通信量分析在 `110-communication/collectives`。

## 主线

> **并行不是把工作除以 `N`。**
>
> **每一刀都要指定三件事**：
> 1. **切在哪个轴** —— 位置决定通信模式：列并行（切 `D_out`）**免通信**、行并行（切 `D_in`）要 all-reduce、
>    PP（切层）是点对点、EP（切专家）是 all-to-all；
> 2. **什么时刻谁手里有哪一块** —— 状态的**时间维**：`1F1B` 的调度、
>    `ZeRO-3`/`FSDP` 的「按需 all-gather、用完即丢」、微批的进出时机；
> 3. **切完还算不算原来那个数** —— 正确性由**切分本身**引入：行并行的求和顺序、
>    跨设备的块统计量重缩放、`(Σg)/N` vs `Σ(g/N)`、容量因子造成的丢弃。

**三个落点各对应一组主题，且各挂一条本机可复现的证据**（数字见 [`probes/README.md`](probes/README.md)）：

| 落点 | 它落在哪些主题上 | 证据（本机实测，口径见 `probes/`） |
|---|---|---|
| ① **切在哪个轴** | `tensor-parallel` · `pipeline-parallel` · `expert-parallel` | **列并行 `max\|diff\| = 0.000e+00`（本组形状）vs 行并行 `8.882e-16`** —— 切在哪个轴，**连误差档位都不同**；★★ 但列并行那个 `0.0` **不是恒真的**，它是**形状依赖**的（同一份数据换个形状就可能非零，见 `tensor-parallel/01`/`02` 的 `D.10`） |
| ② **什么时刻谁手里有哪一块** | `pipeline-parallel` · `zero-fsdp` | **`P_os` 在 `N=512` 时每卡仍需 `4.0234Ψ`**（不是 `16Ψ/512`）；FSDP 的 `FULL_SHARD` 逐字「前向前 unshard、前向后 reshard、反向前 unshard、反向后 reshard」 |
| ③ **切完还算不算原来那个数** | `data-parallel` · `sequence-context-parallel` · `expert-parallel` | **`N` 是 2 的幂时 `(Σg)/N` 与 `Σ(g/N)` 逐位相同（4000/4000）**、非 2 的幂则差 1 ulp；**朴素版 ring attention 在小尺度下看起来也对（`1.4e-16`）、放大后才给 `nan`** |

**与相邻三条主线并排放**：

| 域 | 主线 | 自觉 |
|---|---|---|
| `50-performance` | »工具的每一个输出都带口径« | **测量**的自觉 |
| `80-model-architectures` | »每个选择都在别处付账« | **结构**的自觉 |
| `90-training` | »每个「省」都在别处记账« | **过程**的自觉 |
| **`100-parallelism`** | **»并行不是把工作除以 N«** | **切分**的自觉 |

**与 `90-training` 的分工**（**必须写，防止两条主线撞车**）：

> `90-training` 的账记在 **`精度 / 显存 / 时间 / 稳定性`** 四个账户上（**「省」的成本**）；
> 本域的账记在 **`哪张卡、哪一刻`** 上（**「切」的位置与时序**）。
> ⇒ `zero-fsdp` 的「**分片不是除法**」在本域**不是**「省在别处记账」的实例，
> 而是「**切了几项就只省几项**」—— 它讲的是**切分的范围**，不是**省的成本**。

## 本域另有的目录

- **`probes/`** —— **域级**探测脚本（**不是**编号程序，也不在任何主题的 `code/` 下，
  因此**不参与「文档数 = 编号程序数」的核对**，也**不在**任何 `code/Makefile` 的扫描范围内）。
  它们是建域之初的一批探测：「**切分的这些账，在本机到底能量到什么**」。
  **只打印、不断言**（可证伪的断言属于各主题 `code/` 下的编号程序）。
  种子、激活清单、判据式都写在脚本里；**`probes/README.md` 是这些数的唯一口径记录**
  （含「**不可用的数**」一栏 —— 读数字之前先读它）。
  ★ `probe_100mp.py` 是**形态探针**（探「本机能不能跑多进程 loopback」），**不是知识探针**。

## 本域不做什么（硬边界）

下表是**硬边界**：本域的主题**不写**下表『本域不写』那一列的内容，只在正文里**回链**。
回链的注记方向按 `CONTRIBUTING` §7（「正文在 X，此处回链」/「正文在本域，回链 Y」两种，不可混用）。

| 主题 | 本域**不写** | 那里的正文 |
|---|---|---|
| **全主题** | **集合通信原语的算法与复杂度**（ring / tree allreduce、`V(N) = 2(N−1)/N × S`、reduce-scatter / all-gather 的阶段划分） | `10-foundations/performance-models/02-amdahl-and-scaling.md` 与 `110-communication/collectives`（**未建成**） |
| **全主题** | **拓扑受限的集合通信**（bisection 下界、all-to-all 跨切口量 `∝ N`、分层） | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` |
| **全主题** | **链路带宽的三种口径**与 NVLink / IB 的规格值 | `20-gpu-architecture/interconnect/01-bandwidth-conventions.md` |
| **全主题** | **通信-计算重叠的机制**（双缓冲、分块流水、异步落盘） | `110-communication` 的 `overlap` 主题（**未建成**） |
| **全主题** | **Amdahl / Gustafson / `N₅₀` / 扩展效率的模型** | `10-foundations/performance-models/02-amdahl-and-scaling.md` |
| `zero-fsdp` | **`16Ψ` 的组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`） | `90-training/checkpointing/01-state-and-bytes.md`（**本域只写「分片后」的账**） |
| `data-parallel` | **梯度累积与 batch 放大的等价性**（含归一化层时的失效） | `90-training/autograd-and-backward/03-gradient-lifecycle.md` |
| `zero-fsdp` | **`4x`/`8x` 与 `P_os`/`P_os+g` 的对应关系** | `90-training/checkpointing/02-sharded-save-and-load.md` |
| `expert-parallel` | **容量因子的定义与语义**（定义式、取整口径、dropping vs No-Token-Left-Behind） | `80-model-architectures/moe/02-capacity-factor.md` |
| `expert-parallel` | **aux loss 的式子 / 路由 gating 的机制** | `80-model-architectures/moe/01`、`03` |
| `sequence-context-parallel` | **「在线 softmax」的 `(m, l)` 更新公式本身** | `40-kernels/softmax-normalization/02-online-softmax.md`（★ 本域写的是**跨设备**的块统计量重缩放） |
| `tensor-parallel` | **注意力变体的结构**（MHA / MQA / GQA 的定义与配比） | `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`（★ 本域只写「**头怎么分给设备**」） |
| `pipeline-parallel` | **重算换显存**的机制 | `90-training/autograd-and-backward/02-gradient-checkpointing.md` |
| **全主题** | **框架的分布式 API 语义**（`torch.distributed` / `DDP` / `FSDP` 的类与参数） | ★★ **框架侧当前没有正文**（`70-frameworks/pytorch/` 与 `tensorflow/` 对 `torch.distributed` / `DDP` / `FSDP` / `ProcessGroup` 实测命中 `0` 次；★ 唯一的分布式*契约*正文在 `90-training/checkpointing/01,02`，那是 `torch.distributed.checkpoint` 的保存/恢复语义，不是 API 语义）⇒ 本域只在 `zero-fsdp` 里引 `sharding_strategy` 的**取值列表**（标 **B 级 + 版本 2.14 + URL + 抓取日**），**不解释其语义** |
| **全主题** | **JAX 的自动分片 / SPMD**（`shard_map` / `pjit` / `pmap`、"谁来把切分假设写下来"） | `70-frameworks/jax/03-sharding-and-differences.md` |
| **全主题** | **推理侧的并行形态**（推理的 TP/PP/EP、PD 分离、KV cache 切分） | `120-inference/`（**未建成**） |
| **全主题** | **通信压缩 / 低精度 all-reduce / PowerSGD** | ★ **本域不做**（那是「通信」侧，不是「切分」侧） |
| `tensor-parallel` | ★★★ **`performance-models/02:223` 已经逐字写完了的三条**（「通信发生在**每一层内部**（attention 后、FFN 后各一次 all-reduce）」「通信量与**激活**成正比而不是与参数量成正比」「**只在高带宽互联上可用 / 通常限于单节点 8 卡**、TP 的规模上限由节点内互联决定」） | `10-foundations/performance-models/02-amdahl-and-scaling.md:223`（**v3 由事实层审查者实测发现**）⇒ ★★ **本域不得重写这三条**（否则是「同一个载体重复」）；**只写它没写的**：切法本身（列/行的配对）、`2 + 2` 的逐字出处、通信量**也饱和**这一层 |
| **全主题** | ★★ **`00-meta/glossary.md` 已有的本域相关词条**（`ZeRO` / `FSDP` / `all-reduce` / `all-to-all` / `容量因子` / `aux loss` / `sharded checkpoint` / `grouped GEMM`） | `00-meta/glossary.md` 的那几条 ⇒ ★★ **本域不得另立同名词条**（否则出现**同名两条**）；新词条建成时**并入**并逐条去重（Chunk 7 办） |

## 五处「同名不同指」+ 一处「极易被合并」

同一个词在本库已有另一种含义 ⇒ 写正文时必须把它们分开（不是措辞偏好）。

| 词 | 库内已有的含义 | 本域的含义与纪律 |
|---|---|---|
| `气泡`（bubble） | `00-meta/glossary.md:23`（**`TLP` 条**）的「流水线气泡」= **GPU 指令流水线**的气泡 | **流水线并行**的气泡 ⇒ 本域一律写全「**流水线并行的气泡（bubble）**」，**首次出现处**注明与 TLP 那条不是同一个量 |
| `分片`（shard） | `00-meta/glossary.md:275` 的 `sharded checkpoint` = **落盘**；另有 **JAX 的 Sharding API**（`70-frameworks/jax/03`） | **显存**分片 ⇒ 写「**显存分片**」并点明；落盘一律回链 `90-training/checkpointing/02` |
| **`stage`** | `60-compilers/tvm/01-compute-and-schedule.md` 的 TVM **计算阶段**；`30-gpu-programming/cutlass/02-hierarchical-tiling.md` 的**流水级数**（★ `cutlass/02:455` 已记为**同名不同指**） | ★★ **本域一部域内就有两个含义**：**流水线并行的 stage** 与 **ZeRO 阶段** ⇒ **一律写全称，不得只写 `stage`** |
| **`bucket`** | **形状分桶**（`70-frameworks/jax/01-jit-and-tracing.md`、`tensorflow/01`） | **DDP 的梯度桶** ⇒ 本域一律写「**梯度分桶**」 |
| `rank` | **张量秩**（`60-compilers/xla/01`、`80-model-architectures/attention-variants/02-mla-low-rank.md`）；**设备号**（`performance-models/02` 等） | 设备号 ⇒ 本域**尽量写「设备」** |

★ **另有一处不是同名不同指、而是极易被合并**：`SP`（sequence parallelism）与 `CP`（context parallelism）。
Megatron-LM 官方文档逐字把两者分开了：

> **Unlike prior SP (sequence parallelism), which only splits the sequence of Dropout and LayerNorm activations,
> CP partitions the network inputs and all activations along the sequence dimension.**

⇒ 本域主题名 `sequence-context-parallel` **涵盖两者** ⇒ 正文**必须逐篇点明讲的是哪一个**。

## 代码约定

**形态**：编号程序是 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言（A 算法正确性 / B 代价模型 /
C 边界与反例）放在编号程序内部**，失败时以**非零退出码**结束。**不设 `verify.py`**，
也**不设 `meta_falsify.py`**（后者是**测量域**的形态，见 `50-performance`）。
四类数字的定义、来源分级 A–E、「不得把推导写成实测」这些**全库性规则写在 `CONTRIBUTING` §2**
（正文在彼，此处回链）。本域的**落法**：

| 项 | 约定 |
|---|---|
| **形态** | 编号程序是 `.py`，**默认 `python3` 下可跑** ⇒ 三类断言（**A 算法正确性 / B 代价模型 / C 边界与反例**）**放在编号程序内部**，失败以**非零退出码**结束。**不设 `verify.py`**，**不设 `meta_falsify.py`** |
| **本机没有多卡 / 没有 NCCL / 没有真实网络** | ★★ 一切程序验证的是**切分的代数**（等价性、通信次数、字节账、气泡率、负载分布），**不是**任何关于真实集群性能的结论。★ 强制限定语：**「不是 NCCL，也不是真机」** |
| **编号程序用纯 numpy 单进程** | ★★ **通信被实现为一个显式的张量搬运**（`all_reduce = Σ局部`、`all_gather = concat`）。理由：本域的知识对象是**切分策略**，不是通信原语的实现；通信算法的正文在 `110-communication`。⇒ 这样「切分是否改变数学」「忘了通信会怎样」**全部本机可跑且确定性可复现** |
| **三类判据的分工（互不覆盖）** | **A** = 切分**不改变数学**（等价性）；**B** = 切分的**代价**（通信字节数 / 显存 / 气泡率 / 负载分布）；**C** = 切分**做错**时的**失败**（忘通信 / 切错轴 / 索引错位）。★ **C 类不得装「成功行为」**（`CONTRIBUTING:238`） |
| ★★★ **A 类判据只在「切分改变了求值顺序」的地方才有信息量** | ★★ 实测：**「切分 == 单卡」在两处的性质【不同】** —— ① **头切分**是**结构性恒真**（每卡做的**就是**单卡时那几个 head 的同一批运算，**不涉及 GEMM 的 `M`/`N`**）；② **vocab 维列切**（= 列并行切 `D_out`）是 **BLAS 路径依赖**（同一份数据换个形状就可能非零，**且哪些形状给 `0` 是数据/种子依赖的**）。★ 两者**都不能当 A 类，但理由不同** ⇒ 那类地方**改用 C 类**（证明「搬运做错会失败」）；★ **EP 的 token 分发**同属第 ② 类（本组形状下为 `0.0`，但差是否为零取决于数据与设备归属）。★★ **可引用的是「形状依赖」这个性质；某组具体形状给 `0` 的清单本身不可引用。** ★ **且有信息量的那些**（数字以 `probes/probe_100h.py` 的 `H3` 为准）：行并行 `8.882e-16`、MLP 列+行 `1.554e-15`、DP 的 N 卡平均 `8.882e-16`、ring 的跨块 softmax `1.943e-16`。★★ **「改变了求值顺序没有」是可引用的形态，「几 ulp」不是** —— 它随形状/规模变 |
| ★★★ **参考实现的独立性（硬要求）** | 参照实现**不得调用被测实现的任何函数**（给共享代码装调用计数器，参照跑完后计数必须恰好为 `0`）；**判定方法**：把参照实现**故意改错**，A 类断言**必须失败**（实测：把参照的输出列反转 ⇒ `2.051588`，断言失败 ✓）。★ 覆盖边界要写明：本域 A 类守的是「**切分做对了没有**」，**不守**「算子本身对不对」 |
| ★★★ **`P12`「模拟器同时是被测对象」是本域头号风险** | 本域六个主题的程序**全是模拟器** ⇒ 三类断言极易写成**自证**。**自查问句**：**如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？** 答「会」⇒ 它**不是判据** |
| **`specs.py`** | ★ **每主题一份**，只放本主题**真的会断言的常量**，每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`。**不设跨主题共享模块**（`CONTRIBUTING:36`「`code/` 必须自包含」是全库规则） |
| **数字分四类** | **规格值**（`specs.py`）/ **推导值**（程序算出）/ **引用值**（外部来源的实测，**须注明「本仓库不复核其测量口径」**）/ **本机实测值**（**须写 python3 与 numpy 版本**） |
| **`ASSERT_FAIL_MARK`** | **每个主题一个，不跨主题照抄**。本域六个：`data-parallel` = `%% FAIL: data-parallel` · `tensor-parallel` = `^^ FAIL: tensor-parallel` · `pipeline-parallel` = `== FAIL: pipeline-parallel` · `sequence-context-parallel` = `{{ FAIL: seq-context-parallel` · `expert-parallel` = `\|\| FAIL: expert-parallel` · `zero-fsdp` = `?? FAIL: zero-fsdp`。★ 标记**只允许出现在 `print()` 的字符串里**（**不得**出现在 `Makefile` 的 recipe 行或任何 shell 命令行里）。每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量比较**） |
| **`mutate.py` + `make mutate`** | 每主题一份，含**口径变异**；只在副本上做；断言捕获与崩溃捕获**分开数**；driver 规则是「**遍历全部编号程序，任一捕获即算捕获**」。★ **本域所有常量取到 `< 2^53`**（浮点变异是 `v + 1.0`，大浮点数会吞掉它） |
| **`make lint`** | **AST** 检查（不是 grep）；本域**任何第三方框架都不 import**，`numpy` 是唯一例外 |
| **§5 用「判据表」形态** | 理由两条：**本机无 GPU**（给不出「实测数据」形态要求的 GPU 型号与 CUDA 版本）；**本域的知识对象是方法与工程约定**。⇒ §5 写成「**可机械判定的条件 → 结论**」的表，**表要闭合** |
| **篇幅** | ★ 单篇 **≤ 约 500 行**（`CONTRIBUTING:451`） |

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
