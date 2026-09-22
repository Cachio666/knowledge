# 80-model-architectures 模型架构

要优化的对象。所有算子、并行策略、推理优化的收益最终都要落在这个结构上。

## 主线

> **架构是一组取舍：每一个选择都在别处付账——参数、字节、数值条件、位置容量、或者训练稳定性。**
> **而这些账住在结构里，不需要训练模型就能看见。**

**与 `50-performance` 的分工**（两条主线并排放）：
`50` 是 »**工具的每一个输出都带口径**«（**测量**的自觉）；`80` 是 »**每个选择都在别处付账**«（**设计**的自觉）。

## 五个主题：骨架 + 元件 + 取舍

本域不按「模型种类」分主题，而按**一个 block 的构成**分——每个主题回答一个固定的问题：

| 层 | 主题 | 它回答 |
|---|---|---|
| **骨架** | `transformer` | 一个 block 由什么构成？**残差流**是什么？FFN 取什么形态？深度与宽度怎么换？ |
| **元件** | `normalization` | 归一化用哪种、放在哪、为什么？ |
| **元件** | `positional-encoding` | 位置信息怎么编码？能编多长？为什么外推会崩？ |
| **取舍** | `attention-variants` | KV 用什么结构换？ |
| **取舍** | `moe` | 参数用什么方式换激活？ |

五个主题的依赖关系（`prerequisites` 的 DAG）以 `transformer/01` 为根，写作时逐个建立。

## 本域不做什么（硬边界）

下表是**硬边界**：本域的主题**不写**「禁区」那一列的内容，只在正文里**回链**。
回链的注记方向按 `CONTRIBUTING` §7（「正文在 X，此处回链」/「正文在本域，回链 Y」两种，不可混用）。

| 主题 | 本域**不写** | 那里的正文（回链目标） |
|---|---|---|
| `transformer` | 算子清单、每 token FLOPs、`2ND` vs `6ND`、参数量的**反算核对** | `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md`（全篇）；逐位断言在 **`…/code/01_operator_inventory.py:549`**（⚠ 是 `code/` 下的程序，**不是**那篇同名文档） |
| `transformer` | `head_dim` **不得**用 `hidden_size ÷ num_attention_heads` 代替 | `45-…/requirement-discovery/01-model-to-operator-inventory.md:71-93` 与 `:270-276`（判据在 `:288`）。★ 本域只做**「逐字段标注 config 原生 / 推导」**这一新增部分，**不重推那个例子** |
| `transformer` | FFN 的**投影数**（系数 4 vs 6，即两层 vs SwiGLU） | `40-kernels/moe-kernels/01-moe-compute-pattern.md:72,278`；`40-kernels/moe-kernels/code/verify.py:443` |
| `transformer` | 「残差连接把深度方向的误差松弛为加性」——⚠ 那条结论的作用域**只到浮点舍入误差** | `130-quantization/theory/03-outliers-and-propagation.md:128-130,298,314` |
| `attention-variants` | `4S²d` 的来源、算术强度、decode 的访存账、split-KV、paged KV | `40-kernels/attention/01`、`40-kernels/attention/05` |
| `attention-variants` | 「复制 KV 后与 MHA 逐位相同」与「KV 元素倍率 `1/r`」的**实现与比较** | `40-kernels/attention/05-decoding-attention.md:60,66-94`；`40-kernels/attention/code/verify.py` 的 A4（`:509-529`）与 B8（`:1008-1040`）。★ 本域只写**共享图景与 uptraining** 这一新增部分 |
| `attention-variants` | KV cache 的六项乘积与 GQA 的字节坑 | `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76` |
| `attention-variants` | 变体对 **kernel 结构**的影响（滑窗改循环范围、ALiBi/softcap 改 `exp` 的输入） | `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`。★ 那边讲**每行只需 `ceil(w/Bc)` 个块**，本域讲**堆叠层把感受野从 `W` 扩到 ~`W×L`**——同一条让渡的两端 |
| `attention-variants` | RoPE 的「**成对通道旋转**」这句结构陈述 | `130-quantization/kv-cache-quantization/02-kivi-asymmetry.md:184`（该行**逐字**含 `jointly rotates pairs of channels by different angles depending on the position`；`03-low-bit-and-engines.md:163` 是同一主题的邻近段落，但**不含**这条逐字引文）。★ 本域建成后应回链到 `positional-encoding/01` |
| `moe` | MoE 的 FLOPs/权重流量/算术强度、routing 的两步形态、permute、grouped GEMM、slot mask | `40-kernels/moe-kernels/01-moe-compute-pattern.md`、`02-grouped-gemm.md`、`03-permute-and-unpermute.md` |
| `moe` | 三种 gating 归一化口径（甲/乙/丙）的**实现与比较** | `40-kernels/moe-kernels/code/verify.py` 的 A1（`:273-300`）。★ 本域只写「**哪些真实模型用哪种口径**」这一架构层新增部分 |
| `normalization` | LayerNorm / RMSNorm 的**公式**、Welford 递推、`Σx²−nμ²` 灾难、`eps` 位置带来的数值差异 | `40-kernels/softmax-normalization/03-normalization-fusion.md:53-345`（公式在 `:55-71`）。★ 本域只写**位置之争、QK-norm、不变性的精确形态** |
| `normalization` | 归约的访存账、融合形态 | `40-kernels/softmax-normalization/03-normalization-fusion.md:213-263`、`40-kernels/elementwise-fusion/01-memory-bound-limit.md` |
| `positional-encoding` | —（★ **本域唯一没有禁区的主题**；**变体的 kernel 后果**那一类已在别处） | `40-kernels/attention/04-warp-specialization-and-variants.md:130-134`（那里只给出三个变体「**改什么**」的一行表，**定义与出处**正是本域要补的） |
| **全主题** | `00-meta/glossary.md` **已有的**模型架构词条（GQA/MQA、MoE、容量因子） | `00-meta/glossary.md:236-239`。★ **本域不得另立同名词条**（否则会出现**同名两条**）；词条的定义正文在**本域**，glossary 只放短条目并回链（正文在本域，回链 `00-meta/glossary.md`） |
| **全主题** | roofline / 算术强度 / 逐算子判定 | `10-foundations/performance-models/01`、`02`、`03` |
| **全主题** | 浮点误差模型与舍入（本域只写**不变性**，不写误差模型） | `10-foundations/floating-point/02-rounding-and-error-model.md:14`（该文件的 `related` 指向本域 `normalization`；本域**不写公式、不写误差模型**） |

★ **反向的入口**：`45-kernel-lifecycle` 与 `40-kernels` 已留下多处**指名的让渡**
（「正文在 `80-model-architectures/…`」）——`moe` 最多，其次是 `transformer`、`attention-variants`。
这些让渡正是本域**必须接住**的正文，位置见 `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:323`、
`40-kernels/attention/04-warp-specialization-and-variants.md:128`、`40-kernels/moe-kernels/03-permute-and-unpermute.md:47` 等处。

## 代码约定

**形态**：编号程序是 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）
放在编号程序内部**，失败时以**非零退出码**结束。**不设 `verify.py`**，也**不设 `meta_falsify.py`**
（后者是**测量域**的形态，见 `50-performance`）。四类数字的定义、来源分级 A–E、
「不得把推导写成实测」这些**全库性规则写在 `CONTRIBUTING` §2**（正文在彼，此处回链）。

本域的**落法**：

| 项 | 约定 |
|---|---|
| **真值来源** | ① **外部规格值**：真实模型的 `config.json` 与论文原文（**须写来源等级 + URL/arXiv 号 + 抓取日**；论文**须注明是否确认同行评审**）；② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）；③ **推导值**：程序算出的量（**须可手算复核，并标为「推导值」**）。★ 三类**标签必须分开**，不得互相顶替 |
| **本域没有 GPU** | 一切数值都是 `python3` + `numpy` 的推导与本机实测，**不得支撑任何关于 GPU 的结论** |
| **`specs.py`** | ★ **每主题一份**（`transformer/code/specs.py`、`moe/code/specs.py`、…），每份**只放本主题真的会断言的常量**。**不设跨主题共享模块**——`CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**（全库规则不住在域 README 里）。跨主题同值的常量**各写一份**，靠 `# 同值位置：` 互指（先例与格式：`45-kernel-lifecycle/requirement-discovery/code/specs.py:27,344`） |
| **`specs.py` 的来源分两类** | ★ 逐字来自 `45-kernel-lifecycle/requirement-discovery/code/specs.py` 的字段标 **[B]** 并给 `# 同值位置：`；**不在那份文件里**的字段**必须**带限定语「**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**」。★ **未核实的字段保留「未核实」标注**，不得静默升格 |
| **`ASSERT_FAIL_MARK`** | **每个主题一个，不跨主题照抄**（本库纪律）。五个取值：`transformer` = `!! FAIL: transformer` · `attention-variants` = `@@ FAIL: attention-variants` · `moe` = `## FAIL: moe` · `normalization` = `~~ FAIL: normalization` · `positional-encoding` = `$$ FAIL: positional-encoding`。**每个编号程序另加一条「标记自检」断言**（失败标记去重后**只有一个**、**等于**本主题的值、且**含 `FAIL`**）。`mutate.py` 用 **AST** 从编号程序里读出该值，**不得写死** |
| **`mutate.py`** | ★ **每主题一份**，含**口径变异**（去掉 `1/√d_h`、把 `eps` 挪到分母外、把 `n−m` 写成 `m−n`）。只**在副本上**做，仓库文件只读；**断言捕获与崩溃捕获分开数** |
| **`make lint`** | **AST** 检查（不是 grep：注释里出现框架名**不算违规**，真 `import` 才算）。★ 本域**任何第三方框架都不 import**，`numpy` 是唯一例外（`requirements.txt` 里只有它） |
| **参考实现的独立性** | A 类断言（`diff == 0` / 机器精度档）**必须配「调用计数器 = 0」的证据**：参考实现**不得调用被测实现的任何函数**，且**不得比被测实现更复杂**（判据与两种相反含义见 `00-meta/glossary.md:240`） |

## §5 用「判据表」形态（理由）

按 `CONTRIBUTING` §4，§5 有三种合法形态（实测数据 / **判据表** / 整节省略）。**本域一律用「判据表」**，理由两条：

1. **本机无 GPU**——「实测数据」形态要求注明 GPU 型号与 CUDA 版本，本域给不出；
2. **本域的知识对象是结构与取舍**，不是某台机器上的数。§5 因此写成
   「**可机械判定的条件 → 结论**」的表，且**表要闭合**（读完能照着做判断），
   不写「根据情况判断」。

★ **推论**：本域**别处**（编号程序的输出）若出现数值对照，仍按 `CONTRIBUTING` §2 标为
**本机实测值**并写明测量条件，**不并入 §5**。

## 术语

- 模型架构词条的**定义正文在本域**；`00-meta/glossary.md` 只放短条目并回链。
  ★ **本域不得另立 `glossary` 已有的同名词条**（`00-meta/glossary.md:236`（GQA / MQA）、
  `:237`（MoE）、`:239`（容量因子））——新词条建成时**并入**该节，**逐条去重**。
- 术语的检索入口：`00-meta/glossary.md`；全库进度与主题清单：`00-meta/roadmap.md`。

## 与其他域的关系

- **上游**：`70-frameworks`（模型用框架实现）。
- **下游**：`40-kernels`（注意力变体直接决定 kernel 设计）、`100-parallelism`（MoE 引出专家并行）、
  `120-inference`（KV cache 的大小由架构决定）。
- **与 `45-kernel-lifecycle` 的分工**：那一边从 config **反算**算子清单、FLOPs、参数量
  （正文在该域，此处回链）；本域讲**结构为什么这样选、每个选择的账在哪里**。
- **横切**：本域只讲**架构层面的取舍**；这些结构**落成 kernel 要用的语言与抽象层**
  （CUDA / Triton / PTX-SASS / CuTe）正文在 `30-gpu-programming`，此处回链。

## 本域另有的目录

- **`probes/`** —— **域级**探测脚本（**不是**编号程序，也不在任何主题的 `code/` 下，
  因此不参与「文档数 = 编号程序数」的核对）。它们是建域之初的一批探测：
  「**架构层的不变量在本机到底能不能测出来、量级多少**」。**只打印、不断言**
  （可证伪的断言属于各主题 `code/` 下的编号程序）。
  种子、初始化分布、重复次数都写在脚本里；口径索引见 `probes/README.md`。

## 进度

主题建成进度见 [roadmap.md](../00-meta/roadmap.md)。
