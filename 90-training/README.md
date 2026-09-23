# 90-training 训练系统

把模型训起来，并让它不崩、不慢、可复现。

## 主线

> **训练的每一个「省」都在别处记账：精度、显存、时间、或者稳定性。**
> **而这些账可以在本机用 numpy 量出来 —— 不需要真的训一个模型。**

**与相邻两条主线并排放**：
`50-performance` 是 »**工具的每一个输出都带口径**«（**测量**的自觉）；
`80-model-architectures` 是 »**每个选择都在别处付账**«（**结构**的自觉）；
`90-training` 是 »**每个「省」都在别处记账**«（**过程**的自觉）。

## 五个主题：三种「省」+ 一个底座 + 一份保障

本域不按「训练阶段」分主题，而按**每一步在拿什么换什么**分：

| 层 | 主题 | 它省什么 / 它是什么 | 三篇（编号即外键） |
|---|---|---|---|
| **底座** | [`autograd-and-backward`](autograd-and-backward/) | **梯度本身怎么来**（含用**重算**换显存） | `01-backward-from-scratch`（反向传播本身 + 数值梯度对拍的方法学）· `02-gradient-checkpointing`（**重算换显存**）· `03-gradient-lifecycle`（训练循环里的梯度：累积 / 清零 / 裁剪） |
| **省精度** | `mixed-precision` | **用更窄的格式算** —— 代价记在**范围**上 | `01-type-selection`（主权重 / 梯度 / 优化器状态 / loss 四者的**类型选择表**）· `02-amp-execution`（AMP 与 unscale 的时机）· `03-fp8-training`（训练循环里怎么用 fp8） |
| **省显存** | `checkpointing` | **用更少的地方存状态** —— 代价记在**时间与一致性**上 | `01-state-and-bytes`（状态的组成与字节账）· `02-sharded-save-and-load`（分片保存与恢复）· `03-consistency-and-fault-tolerance`（一致性恢复：RNG / dataloader / 优化器状态） |
| **省不掉的那一步** | `optimizers` | **梯度怎么变成更新** —— 这里没有省，只有**选择** | `01-sgd-and-momentum`（★ 文件名由上游指名，见下）· `02-adam-and-adamw`（weight decay 的两种语义）· `03-schedules-and-clipping`（学习率调度与梯度裁剪） |
| **起点与兜底** | `initialization-and-stability` | **从哪开始**、**崩了怎么办** | `01-initialization`（He / Xavier / μP）· `02-clipping-and-monitoring`（裁剪与范数监控）· `03-instability-and-spikes`（loss spike：现象、**已证伪**的归因、**仍是假设**的归因） |

**规模**：**15 篇文档 ↔ 15 个编号程序**（三篇一主题），全部本机可跑；
依赖关系（`prerequisites` 的 DAG）以 `autograd-and-backward/01` 为根。
建成进度见 [roadmap.md](../00-meta/roadmap.md)（**进度只写在那里**，本文件不重复「现在是几篇」）。

### ★ 从上一版 README 删掉的两条声明（它们没有落点）

上一版列了五个主题的「内容」，其中两条在本域的实施计划里**没有任何篇目**，
且都不属本域的知识对象 ⇒ **删除，并写明归谁**（`CONTRIBUTING` §7：一个知识点只有一处正文）：

- **「精度对齐与复现」**（原属 `mixed-precision`）——
  「复现」那一半的正文在**本域** `checkpointing/03-consistency-and-fault-tolerance`
  （恢复后轨迹逐位一致、RNG 内部状态、阴性对照）；
  「数值精度对齐」（迷你实现 vs 真系统的数不一样）**不构成一个知识点** ——
  它是 `CONTRIBUTING` §2「本机实测值」的**限定语纪律**，不是一篇文档。
- **「异步落盘」**（原属 `checkpointing`）——
  它与 `110-communication` 的 `overlap` 主题（通信-计算重叠 / 双缓冲 / 分块流水）
  是同一件事，正文在 `110-communication/README.md:12`，**此处回链**。

## 本域不做什么（硬边界）

下表是**硬边界**：本域的主题**不写**「禁区」那一列的内容，只在正文里**回链**。
回链的注记方向按 `CONTRIBUTING` §7（「正文在 X，此处回链」/「正文在本域，回链 Y」两种，不可混用）。

| 主题 | 本域**不写** | 那里的正文（回链目标） |
|---|---|---|
| `autograd-and-backward` | 反向图结构、`is_leaf`、`saved_tensors` 逐算子表、`no_grad` vs `inference_mode`、`retain_graph` 默认值、`backward()` vs `autograd.grad()` | `70-frameworks/pytorch/02-autograd.md`（**全篇 256 行**）；术语五条在 `00-meta/glossary.md:203-211` |
| `autograd-and-backward` | 自定义反向的**实现**（`log1pexp` 的 NaN 例子） | `70-frameworks/tensorflow/03-custom-ops-and-export.md:106-150,217` |
| `autograd-and-backward` | 控制流与反向模式的可微性 | `70-frameworks/jax/03-sharding-and-differences.md:110-112,379-381` |
| `autograd-and-backward` | 不可微算子（STE）的反向 | `130-quantization/qat/01-ste.md:28,181`、`02-fake-quant-and-lsq.md:204-212` |
| `mixed-precision` | 「**为什么 fp16 要 loss scaling、bf16 为什么不用**」**整块** | `10-foundations/floating-point/04-low-precision-formats.md:24,28,41,65,189,195-201,211,219-231,303`；★ `130-quantization/numerical-formats/03-bf16-fp16-tf32.md:154` 已明写正文在那里 |
| `mixed-precision` | 三个后果的分解（范围 / 精度 / 累加宽度）；格式的位分配 | `10-foundations/floating-point/04-low-precision-formats.md:24,303`、`01-ieee754-format.md:189-226` |
| `mixed-precision` | **fp8 的位分配与四个变体**（`e4m3fn` / `e5m2` / `fnuz`、`±448` vs `±57344`） | `130-quantization/numerical-formats/01-fp8-family.md`（**§0 全节就是那张表**） |
| `mixed-precision` | **scale 的作用域与 bit 开销**（每张量 / 每通道 / 每 token / 每 32 元素的块） | `130-quantization/numerical-formats/02-e8m0-and-mx.md:276`（逐字要求「凡写 scale 必写作用域与 bit 开销」）+ `130-quantization/theory/02-granularity.md` |
| `mixed-precision` | **累加器宽度的误差界**（fp32 → fp16 使绝对误差界放大 **8192 倍**、split-K 的顺序不确定性） | `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md:118-126,304-306,326` |
| `mixed-precision` | 主权重「**为什么必须 fp32**」的**数值理由** | `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266`（尤其 `:245,259-262`） |
| `mixed-precision` | QAT 侧的「优化器更新的是谁」；STE / 伪量化 / LSQ / PACT 全主题 | `130-quantization/qat/02-fake-quant-and-lsq.md:69,420`、`01-ste.md:202` |
| `optimizers` | ★★ **优化器状态的分片策略与通信量**（ZeRO 三阶段、FSDP、`4x/8x`、`16Ψ` 的账） | `100-parallelism/README.md:14`（`zero-fsdp` 那一栏）；`10-foundations/performance-models/02-amdahl-and-scaling.md:219,221,417` 已逐字列为那边的正文。★ 本域只留**接口** |
| `optimizers` | 数据并行 / DDP / 梯度规约 | `100-parallelism/README.md:9`（`data-parallel`）、`110-communication/` |
| `optimizers` | `lr` 被编译 guard 固化 | `60-compilers/inductor/04-dynamic-shapes.md:42,163`（**编译角度，不是优化器角度**） |
| `optimizers` | 推理口径的「权重文件含优化器状态」 | `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:234,301`（那一行逐字：「权重文件里含优化器状态 ⇒ 那是**训练**侧的字节；**不得**计入推理的访存」） |
| `initialization-and-stability` | ★ **warmup 的机制解释**（post-LN 的平均场理由） | `80-model-architectures/normalization/02-placement-and-qk-norm.md:83,93-97,100,278`；`transformer/01:49,252`。★ 本域只写**工程判据与设定** |
| `initialization-and-stability` | 残差流的范数随深度增长（**结构**侧的账） | `80-model-architectures/transformer/01:191,232-233,263` |
| `initialization-and-stability` | 误差沿深度的传播 | `130-quantization/theory/03-outliers-and-propagation.md:124,205` |
| `initialization-and-stability` | MoE 的 aux loss 尺度（`×N` 漏项） | `80-model-architectures/moe/03-balance-and-granularity.md:260`；`00-meta/glossary.md:241` |
| `checkpointing` | 导出 / 推理格式的选择（`.keras` vs SavedModel） | `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；`00-meta/glossary.md:221` |
| **全主题** | 框架的**精度 API 语义**（`autocast` 的算子白名单 / 黑名单） | `70-frameworks/`（**各框架互不相同**，属框架 API 语义）。★ 本域写的是**工程约定**，不是某个框架的 API 行为 |
| **全主题** | `00-meta/glossary.md` **已有的**本域相关词条（`autograd` / `grad_fn` / `is_leaf` / `saved tensors` / `inference_mode` / FP32·FP16·BF16 / ZeRO / iterative refinement / SavedModel） | `00-meta/glossary.md:81,126,203-211,221,254` ⇒ **本域不得另立同名词条**（否则会出现**同名两条**） |

★ **反向的入口**：本域的立足点是**上游指名的让渡**，它们正是本域必须接住的正文——
`10-foundations/numerical-linear-algebra/04-iterative-methods.md:386`（**最具体的一处**：
指名要写「主权重、优化器状态、loss scaling **三者的类型选择**」）、
`10-foundations/floating-point/04-low-precision-formats.md:14`（frontmatter 的 `related`）、
`70-frameworks/pytorch/02-autograd.md:13`（**全库唯一一处「编号级」让渡**：
它逐字指定 `90-training/optimizers/01-sgd-and-momentum` ⇒ **那一篇的文件名不可改**）、
`:47` 与 `:256`（「本文**不**讨论优化器怎么用梯度」）。

★★ **`initialization-and-stability` 与 `checkpointing` 两处几乎零让渡**
（`checkpointing` 在全库**没有任何一处**指向本域）⇒ 这两块的知识点由本域**自己立**，
因此更要守住上表：**不写**那些已被别处占掉的内容。

## ★ 两处「同名不同指」——**写正文时必须把它们分开**

这两条不是措辞偏好，是**同一个词在本库已有另一种含义**（计划 §2 末尾，结构层审查 M9/M10）。

### ① `checkpoint` —— 「检查点」这个词，中文留给**权重**，重算一律叫「**重算换显存**」

- **建域前，全库（除 `docs/` 与本域）`checkpoint` 忽略大小写共 11 处；其中 10 处是「权重/模型检查点」的用例**
  （2026-09-23 逐行扫描确认）：
  `10-foundations/floating-point/01-ieee754-format.md:232`、`20-gpu-architecture/interconnect/01-bandwidth-conventions.md:43`、
  `80-model-architectures/positional-encoding/01-rope-definition.md:50,177`（及其 `code/01_rope_definition.py:409`）、
  `80-model-architectures/attention-variants/01-mha-mqa-gqa.md:192,206,212,215`、
  `40-kernels/attention/05-decoding-attention.md:354`（那是**论文标题**里的 `Checkpoints`）；
  **第 11 处**是 `00-meta/roadmap.md:71` 的**主题名**（`- [ ] checkpointing`），不是用例。
  ★ **量法口径的坑**：`40-kernels/attention/05:354` 用的是**大写** `Checkpoints` ⇒
  **大小写敏感地数会少 1 处**（10 vs 11）。引用这个数时**必须写明是不是忽略大小写**。
- **讲「重算」的那一处，全库用的是 `重算` / `recompute`**：
  `40-kernels/attention/03-flash-attention-backward.md` 里 `重算` **36** 次、`recompute` **5** 次、
  `checkpoint` **0** 次。
- ⇒ **本域纪律**：
  - **中文「检查点」不得用于重算**；重算一律写「**重算换显存**」。
  - **英文 `checkpoint` 只留给权重检查点** ⇒ 本域 `checkpointing` 主题 = **权重检查点**。
  - ★ **唯一的例外是文件名**：`autograd-and-backward/02-gradient-checkpointing.{md,py}` ——
    `gradient checkpointing` 是**论文（Chen 2016）里的术语**，改了就对不上原文。
    **正文第一次出现它时必须写明这一点**（「这里的 checkpointing 指重算，不是权重检查点」）。

### ② `warmup` —— glossary 里已有的那条指**benchmark 预热**，本域要写的是 **LR warmup**

- `00-meta/glossary.md:146` **已有 `warmup` 词条**，定义是「**预热 / 丢弃前导测量**」——
  即 **benchmark 预热**（`45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md`
  里 `warmup` **23** 次，`50-performance` 等**全是同一语义**）。
- 本域要写的是**学习率 warmup（LR warm-up）**，正文在
  `optimizers/03-schedules-and-clipping`（**这是「同一个词指两件事」，比 ① 更硬**：
  glossary 已经收了其中一件）。
- ⇒ **本域纪律**（计划 §2 末尾的处置）：
  - glossary 的那条改成 **`warmup（benchmark 预热）`**，本域新增
    **`学习率 warmup（LR warm-up）`**，两条**互相指认**。★ 这两步是**收尾 chunk** 的动作
    （术语迁入时一并做）；**在它做完之前，本域任何提到 warmup 的地方都要自己带上这句限定语**。
  - ★★ **边界声明必须跟着知识点走**（而不是只写在某一篇的抬头）：
    凡本域写 LR warmup 的地方，都要回链「机制解释（post-LN 的平均场理由）在
    `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`」。
  - ★ 顺带记一笔：**`clip`** 也是同形 —— `00-meta/glossary.md:99,:105` 的 `clip`
    是**量化意义**的（ACIQ / PACT 学的裁剪上界），与**梯度裁剪**不是一回事。

## 代码约定

**形态**：编号程序是 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言（A 算法正确性 / B 代价模型 /
C 边界与反例）放在编号程序内部**，失败时以**非零退出码**结束。**不设 `verify.py`**，
也**不设 `meta_falsify.py`**（后者是**测量域**的形态，见 `50-performance`）。
四类数字的定义、来源分级 A–E、「不得把推导写成实测」这些**全库性规则写在 `CONTRIBUTING` §2**
（正文在彼，此处回链）。本域的**落法**：

| 项 | 约定 |
|---|---|
| **真值来源（三类，标签必须分开）** | ① **外部规格值**：论文原文（**须注明是否确认同行评审**）、厂商/产品文档（**须写 A–E 级 + URL + 章节 + 抓取日**）；② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）；③ **推导值**：程序算出的量（**须可手算复核，并标为「推导值」**） |
| **本域特有的一条（P19）** | ★★ 凡引用 `O0/O1/O2/O3`、`GradScaler`、`clip_grad_norm_` 这类**「某实现的规格」**，**必须标 B 级并写明是哪个实现（含版本）**。本域的头号错误模式就是**把「某实现的文档」当成「规范的文档」** |
| **本域没有 GPU** | 一切数值都是 `python3` + `numpy` 的玩具规模真跑，**不得支撑任何关于大规模训练的结论**（「本机实测值」的强制限定语） |
| **`specs.py`** | ★ **每主题一份**（`autograd-and-backward/code/specs.py`、…），每份**只放本主题真的会断言的常量**，每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + **`# 同值位置：`**。**不设跨主题共享模块**——`CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**（全库规则不住在域 README 里）。跨主题同值的常量**各写一份**，靠 `# 同值位置：` 互指 |
| **`ASSERT_FAIL_MARK`** | **每个主题一个，不跨主题照抄**（本库纪律）。五个取值：`autograd-and-backward` = `.. FAIL: autograd` · `mixed-precision` = `:: FAIL: mixed-precision` · `optimizers` = `++ FAIL: optimizers` · `initialization-and-stability` = `// FAIL: init-stability` · `checkpointing` = `;; FAIL: checkpointing`。**每个编号程序另加一条「标记自检」断言**（失败标记**去重后只有一个**、**等于**本主题的值、且**含 `FAIL`**，**字面量比较**）。`mutate.py` 用 **AST** 从编号程序里读出该值，**不得写死** |
| **三类断言（A/B/C）落在编号程序内部** | **A 算法正确性** = 「与**独立参照** / 闭式对拍」（手写反向 vs 中心差分、Adam+L2 vs 闭式）；**B 代价模型** = 「**前向次数 / 字节账 / 精度损失**」（重算换显存的 `≈2n` 前向、优化器状态的字节账）；**C 边界与反例** = 「**失败行为真的成立**」（`w + Δw == w` 的更新被吞、fp16 下 `1e-8` 变成 `0`）。★ 缺一不可（`CONTRIBUTING:133-143`） |
| **参考实现的独立性** | 定义在 `CONTRIBUTING` §2（正文在彼，此处回链）。★ 本域为什么单列：本域的判据多长成「与闭式对拍」「机器精度档的差」的样子，**这一条最容易自欺**——两边共用一份代码时，判据照样「通过」 |
| **`mutate.py` + `make mutate`** | ★ **每主题一份**，含**口径变异**。只**在副本上**做，仓库文件只读；**断言捕获与崩溃捕获分开数**。★ driver 规则必须是「**遍历全部编号程序，任一捕获即算捕获**」，否则 `specs.py` 的变异会**整体免检**（`80` 实测 **44/44 全部漏网**）。★ **三个盲区要手工补测**：只变异字面量 / 负字面量不进分母 / 前缀式检查会漏网 |
| **`make lint`** | **AST** 检查（不是 grep：注释里出现框架名**不算违规**，真 `import` 才算）。★ 本域**任何第三方框架都不 import**，`numpy` 是唯一例外（`requirements.txt` 里只有它） |

## §5 用「判据表」形态（理由）

按 `CONTRIBUTING` §4，§5 有三种合法形态（实测数据 / **判据表** / 整节省略）。**本域一律用「判据表」**，理由两条：

1. **本机无 GPU** —— 「实测数据」形态要求注明 GPU 型号与 CUDA 版本，本域给不出；
2. **本域的知识对象是方法与工程约定**（哪些张量该是什么类型、unscale 什么时候做、
   裁剪的阈值怎么定），不是某台机器上的数。§5 因此写成
   「**可机械判定的条件 → 结论**」的表，且**表要闭合**（读完能照着做判断），
   不写「根据情况判断」。

★ **推论**：本域**别处**（编号程序的输出、`probes/` 的探测）若出现数值对照，仍按
`CONTRIBUTING` §2 标为**本机实测值**并写明测量条件，**不并入 §5**。

## 术语

- 本域相关词条的**定义正文在本域**；`00-meta/glossary.md` 只放短条目并回链。
- ★ **本域不得另立 `glossary` 已有的同名词条**（本域相关的九条在
  `00-meta/glossary.md:81,126,203-211,221,254`）—— 新词条建成时**并入**并**逐条去重**。
- ★★ **建域时已识别的两处同名不同指见上**（`checkpoint` / `warmup`）；
  新增词条时**先查 glossary**，冲突的按上面的处置办。
- 术语的检索入口：`00-meta/glossary.md`；全库进度与主题清单：`00-meta/roadmap.md`。

## 与其他域的关系

- **上游**：`70-frameworks/pytorch`（**框架的 autograd API 与语义**）、
  `10-foundations/floating-point`（格式的范围 / 精度 / 累加三个后果）、
  `10-foundations/numerical-linear-algebra`（迭代精化的 `κ·u` 判据链）。
  ★ 本域**不改这些机制**，只写**落在训练循环里的工程约定**。
- **下游**：`100-parallelism`（本域只写**单卡算法**，分布式只留**接口**：
  「与分布式优化器状态的接口（正文在 `100-parallelism`，具体主题由该域建成时定）」）、
  `110-communication`（分布式训练的通信开销；异步落盘/重叠的正文也在那里）。
- **与 `80-model-architectures` 的分工**：那边是**结构**的账（每个选择在别处付账），
  本域是**过程**的账（每个「省」在别处记账）。
- **与 `130-quantization/qat` 的分工**：那边是**定点**（int8/int4、STE、伪量化），
  本域是**浮点**（fp16/bf16/fp8/tf32）；**本域不改反向公式**。
- **横切**：优化器**未**独立成一级域，因为它与训练循环、混合精度、
  **与分布式优化器状态的接口**强耦合（那一侧的正文在 `100-parallelism`，此处回链）。

## 本域另有的目录

- **`probes/`** —— **域级**探测脚本（**不是**编号程序，也不在任何主题的 `code/` 下，
  因此不参与「文档数 = 编号程序数」的核对，也不在任何 `code/Makefile` 的扫描范围内）。
  它们是建域之初的一批探测：「**训练的账在本机到底能不能量出来、量级多少**」。
  **只打印、不断言**（可证伪的断言属于各主题 `code/` 下的编号程序）。
  种子、抽样次数、判据式都写在脚本里；**口径索引与「不可用的数」见 `probes/README.md`**。

## 进度

主题建成进度见 [roadmap.md](../00-meta/roadmap.md)。
