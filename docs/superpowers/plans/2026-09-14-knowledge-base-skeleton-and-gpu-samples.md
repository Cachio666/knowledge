# 大模型全栈知识库 骨架与 GPU 硬件样板 实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立知识库的目录骨架与写作规范，并完成 `20-gpu-architecture/` 下 execution-model 与 memory-system 共六篇样板文档与六个可编译的 CUDA 程序。

**Architecture:** 仓库是纯 Markdown 知识库，无构建链。骨架层（规范、模板、索引、进度表）先建，为内容层提供约束；内容层按「一篇文档 ↔ 一个代码文件」的编号对应关系成对产出，文档负责原理推导与执行映射，代码负责可验证的观测。

**Tech Stack:** Markdown + YAML frontmatter；CUDA C++（`nvcc`，目标 CUDA ≥ 12.0 / sm_80+）；GNU Make。

**Spec:** [docs/superpowers/specs/2026-09-14-llm-fullstack-knowledge-base-design.md](../specs/2026-09-14-llm-fullstack-knowledge-base-design.md)

**重要前提（来自 spec §1.3）：** 本机为 macOS，**无 CUDA 工具链、无 `nvcc`**。代码**不在本机编译**。编写阶段对代码只做**静态审查**；同步到远程机器编译运行由维护者本人执行。因此计划中所有代码任务的验证步骤都是**静态审查清单**，不是 `nvcc` 调用。

---

## 文件结构

本计划创建 36 个文件，分五组：

### 骨架组（22 个）

| 路径 | 职责 |
|---|---|
| `README.md` | 仓库总入口：定位、16 个一级域说明、学习路径图 |
| `CONTRIBUTING.md` | 写作规范：命名、frontmatter、深度标准、链接约定、核对清单 |
| `_template.md` | 单篇文档模板，可直接复制 |
| `00-meta/README.md` | `00-meta` 域说明 |
| `00-meta/glossary.md` | 术语表中英对照，≥30 条 |
| `00-meta/index.md` | 全库索引，按域组织 |
| `00-meta/roadmap.md` | 全库进度表，61 个二级主题 |
| `10-foundations/README.md` | 域说明（`30`~`130` 共 11 个域 README 结构相同；`140`/`150` 结构不同，见 Task 6） |
| `30-gpu-programming/README.md` | 〃 |
| `40-kernels/README.md` | 〃 |
| `50-performance/README.md` | 〃 |
| `60-compilers/README.md` | 〃 |
| `70-frameworks/README.md` | 〃 |
| `80-model-architectures/README.md` | 〃 |
| `90-training/README.md` | 〃 |
| `100-parallelism/README.md` | 〃 |
| `110-communication/README.md` | 〃 |
| `120-inference/README.md` | 〃 |
| `130-quantization/README.md` | 〃 |
| `140-papers/README.md` | 〃（无固定二级结构） |
| `150-playground/README.md` | 〃（无固定二级结构） |
| `20-gpu-architecture/README.md` | 域说明 + 两个首期二级主题的入口 |

### execution-model 组（7 个）

| 路径 | 职责 |
|---|---|
| `20-gpu-architecture/execution-model/01-sm-internals.md` | SM 结构、寄存器预算 |
| `20-gpu-architecture/execution-model/02-simt-and-divergence.md` | SIMT 与分支发散 |
| `20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md` | 占用率与延迟隐藏 |
| `20-gpu-architecture/execution-model/code/01_sm_probe.cu` | 查询 SM 规格与寄存器预算 |
| `20-gpu-architecture/execution-model/code/02_divergence_demo.cu` | 发散 vs 无分支计时对比 |
| `20-gpu-architecture/execution-model/code/03_occupancy_probe.cu` | 占用率与吞吐的关系 |
| `20-gpu-architecture/execution-model/code/Makefile` | 构建三个目标 |

### memory-system 组（7 个）

| 路径 | 职责 |
|---|---|
| `20-gpu-architecture/memory-system/01-hierarchy-overview.md` | 存储层次与量级对比 |
| `20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md` | 访存合并与事务数 |
| `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md` | bank 冲突与 padding |
| `20-gpu-architecture/memory-system/code/01_bandwidth_probe.cu` | 各级带宽实测 |
| `20-gpu-architecture/memory-system/code/02_coalescing_demo.cu` | 连续 vs 跨步访存 |
| `20-gpu-architecture/memory-system/code/03_bank_conflict_demo.cu` | 有/无 padding 对比 |
| `20-gpu-architecture/memory-system/code/Makefile` | 构建三个目标 |

**不创建**：其余全部二级目录（spec §3.3）。

---

## 通用约定

**提交信息格式**（来自 CLAUDE.md）：

```
<type>: <subject>
# type: feat | fix | docs | style | refactor | test | chore
```

本计划所有提交使用 `docs:`（文档与知识库内容）。

**文档验证脚本**（每个文档任务复用，替换 `$f`）：

```bash
# 用法：把 $f 设为待检查的文档路径
f=20-gpu-architecture/execution-model/01-sm-internals.md

echo "--- 五节完整性 ---"
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done

echo "--- frontmatter 字段 ---"
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
```

**预期输出**：五节全部 `OK`；12 个字段中，`code` 字段对无代码文档显示 `MISS`（可接受，spec §6 规定 `code` 为可选），其余 11 个全部 `OK`。

**每篇文档还须包含「编译与运行」一节**（spec §10.2 验收标准 4 要求每篇末尾有可执行的编译运行命令）。放在「小结与延伸」之前：

````markdown
## 编译与运行

```bash
cd code
make <NN>_<名称>
./<NN>_<名称>
```
````

其中 `<NN>_<名称>` 填该文档对应代码文件的文件名（见各任务 frontmatter 的 `code` 字段，去掉 `code/` 前缀与扩展名），例如 `01_sm_probe`。

**关于 `status`**：六个文档任务的 frontmatter 都写 `status: stable`，因为每个任务都会在**一次提交内**把第 0–4 节写全并通过 Step 8 检查 —— 这正是 spec §6 判定表中 `stable` 的定义。**`status` 与可选的第 5 节无关**（spec §6 明写），所以省略第 5 节不影响 `stable`。

若你选择分次撰写、某次提交时第 0–4 节尚未齐全，则那一次应写成 `draft`，补齐后在同一提交里改成 `stable`。spec §10.2 验收标准 2 要求最终六篇均为 `stable`，Chunk 5 Task 17 Step 2 会断言这一点。

（`CONTRIBUTING.md` 第 3 节的 frontmatter 示例与 `_template.md` 中的 `status: draft` 是**示例文本，保持 `draft` 不动** —— 它们展示的是新建文档的起始状态。）

---

## Chunk 1: 仓库基础（三个根文件）

这三个文件定义全库的展示方式与写作标准，必须先于内容层完成。

### Task 1: 仓库总入口 `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 `README.md`**

完整内容如下：

````markdown
# 大模型全栈知识库

从 GPU 硬件底层到推理引擎上层，覆盖训练与推理全链路的学习型知识库。

## 这个知识库是什么

不是教程摘抄，而是**原理推导 + 执行映射 + 失败模式**三件套的沉淀。每篇文档回答三个问题：

1. 这东西的数学/算法原理是什么，怎么推导出来的？
2. 它在真实系统（SM、缓存、dispatch、调度器）上实际发生了什么？
3. 什么情况下会失效、退化、或给出反直觉的结果？

## 目录结构

按技术栈分层，自下而上编号。编号三位数、间隔 10，插入新领域不需重排。

| 域 | 覆盖内容 |
|---|---|
| `00-meta` | 知识库自身：术语表、索引、进度表 |
| `10-foundations` | 浮点与数值、数值线性代数、性能模型（roofline、Amdahl） |
| `20-gpu-architecture` | SM 与执行模型、存储层次、Tensor Core、互联、架构演进 |
| `30-gpu-programming` | CUDA、Triton、PTX/SASS、CUTLASS |
| `40-kernels` | GEMM、attention、reduce、softmax、融合算子、MoE kernel |
| `50-performance` | profiling 工具、瓶颈定位、调优实战 |
| `60-compilers` | MLIR、TVM、Inductor、XLA、自动调优 |
| `70-frameworks` | PyTorch、TensorFlow、JAX |
| `80-model-architectures` | Transformer、注意力变体、MoE、归一化、位置编码 |
| `90-training` | 反向传播、混合精度、优化器、训练稳定性、检查点 |
| `100-parallelism` | DP、TP、PP、SP/CP、EP、ZeRO/FSDP |
| `110-communication` | 集合通信、NCCL、网络拓扑、通信计算重叠 |
| `120-inference` | KV cache、batching 调度、解码策略、推理引擎、部署 |
| `130-quantization` | 数值格式、量化理论、PTQ、QAT、量化算子、KV cache 量化 |
| `140-papers` | 论文精读 |
| `150-playground` | 实验场 |

`00-meta/` 下有三个文件：[术语表](00-meta/glossary.md)、[索引](00-meta/index.md)、[进度表](00-meta/roadmap.md)。

## 学习路径图

自下而上，每一层都是下一层的前提。箭头表示「读懂了左边再读右边」。

```
10-foundations  浮点与数值 / 性能模型
      │
      ▼
20-gpu-architecture   SM 与执行模型 → 存储层次 → Tensor Core → 互联
      │
      ▼
30-gpu-programming    CUDA → Triton → PTX/SASS → CUTLASS
      │
      ▼
40-kernels            GEMM → attention → reduce / softmax / 融合算子
      │
      ├──────────────► 50-performance   瓶颈定位与调优（横切，随时可读）
      │
      ▼
60-compilers          MLIR → TVM → Inductor → XLA
      │
      ▼
70-frameworks         PyTorch / TensorFlow / JAX
      │
      ├──────────────► 80-model-architectures   要优化的对象
      │
      ▼
90-training           反向传播 → 混合精度 → 优化器 → 训练稳定性
      │
      ├──────────────► 100-parallelism   DP / TP / PP / EP / ZeRO
      ├──────────────► 110-communication 集合通信 / NCCL / 拓扑
      │
      ▼
120-inference         KV cache → 调度 → 解码 → 推理引擎
      │
      └──────────────► 130-quantization  量化与压缩（横切训练与推理）

140-papers      论文精读（随时可读）
150-playground  实验场（随时可写）
```

## 怎么用

**按依赖顺序学**：每篇文档头部的 frontmatter 里有 `prerequisites`（必须先读）和 `related`（建议延伸）。从 `prerequisites` 为空的文档开始，跟着链接走。

**按需查找**：`00-meta/index.md` 是全库索引；各域 README 列出该域的主题清单。

**看进度**：`00-meta/roadmap.md` 记录每个主题的完成状态。

## 文档与代码的关系

编号即外键。文档 `NN-<名称>.md` 与代码 `code/NN_<名称>.<扩展名>` 一一对应：

```
20-gpu-architecture/memory-system/
├── 01-hierarchy-overview.md          ←→  code/01_bandwidth_probe.cu
├── 02-coalescing-and-cache-line.md   ←→  code/02_coalescing_demo.cu
├── 03-shared-memory-bank-conflict.md ←→  code/03_bank_conflict_demo.cu
└── code/
    ├── 01_bandwidth_probe.cu
    ├── 02_coalescing_demo.cu
    ├── 03_bank_conflict_demo.cu
    └── Makefile
```

正文内嵌讲解所需的关键代码片段；完整可编译版本在 `code/` 下，自带 `Makefile`，可单独拷贝到有 CUDA 环境的机器运行。

## 贡献与规范

写作前请读 [CONTRIBUTING.md](CONTRIBUTING.md)。新文档可从 [_template.md](_template.md) 复制。
````

- [ ] **Step 2: 验证**

Run:
```bash
test -f README.md && echo "OK 文件存在"
grep -c '^| `' README.md
```
Expected: `OK 文件存在`，且第二行输出 `16`（表格中 16 个域行）。

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs: 添加仓库总入口 README"
```

---

### Task 2: 写作规范 `CONTRIBUTING.md`

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: 写 `CONTRIBUTING.md`**

完整内容对应 spec 第 6、7、8 节，逐节转写：

````markdown
# 写作规范

本文件是知识库的写作约束。新增或修改文档前请通读。

## 1. 命名规范

| 对象 | 规范 | 示例 |
|---|---|---|
| 一级域目录 | `NNN-kebab-case`，三位数，间隔 10 | `20-gpu-architecture` |
| 二级/三级目录 | `kebab-case`，无编号 | `memory-system` |
| 文档文件 | `NN-kebab-case.md`，两位数，域内主题内唯一 | `02-coalescing-and-cache-line.md` |
| 代码文件 | `NN_snake_case.ext`，`NN` 与对应文档相同 | `02_coalescing_demo.cu` |
| 目录索引 | 固定名 `README.md` | — |
| 非知识文档 | 前缀下划线 `_`，表示"不是知识内容" | `_template.md` |
| 设计与规划工件 | 放 `docs/`，不进 16 个一级域 | `docs/superpowers/specs/` |

**下划线前缀的含义是"非知识文档"（模板、临时文件），不是"草稿"。** 知识内容一律不加下划线前缀。

## 2. 文档与代码的对应关系

编号即外键。同一主题目录下：

- 文档 `NN-<名称>.md` ↔ 代码 `code/NN_<名称>.<扩展名>`，`NN` 相同。
- **文档数与 `code/` 中的源文件数必须相等。** 不允许出现"3 篇文档配 2 个代码文件"的错位。
- 正文只嵌入讲解所需的关键片段；完整可编译版本放 `code/`，自带 `Makefile`，可单独拷贝运行。

**没有配套代码的文档不单独存在。** 若某主题确实不需要代码，则该主题下所有文档都不建 `code/`，正文也不出现指向 `code/` 的链接。

## 3. frontmatter 规范

每篇文档头部必须包含以下 YAML frontmatter：

```yaml
---
title: 共享内存 bank 冲突
slug: shared-memory-bank-conflict
domain: 20-gpu-architecture
topic: memory-system
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
tags: [shared-memory, bank-conflict, padding]
related:
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/03_bank_conflict_demo.cu
status: draft
updated: 2026-09-14
---
```

| 字段 | 必填 | 类型 | 取值与约定 |
|---|---|---|---|
| `title` | 是 | string | 文档标题，与正文一级标题一致 |
| `slug` | 是 | string | 全局唯一，kebab-case，用于跨文档引用 |
| `domain` | 是 | string | 一级域目录名全称，含编号，如 `20-gpu-architecture` |
| `topic` | 否 | string | 二级目录名；无二级目录时**省略该字段**（不写空串、不写 null） |
| `level` | 是 | int | `1`=概念 · `2`=原理 · `3`=实现 · `4`=优化 · `5`=前沿 |
| `difficulty` | 是 | enum | `beginner` · `intermediate` · `advanced` |
| `prerequisites` | 是 | list | 前置文档的**仓库根相对路径**（不含 `.md` 后缀）；无前置时写 `[]` |
| `tags` | 是 | list | 检索标签，小写 kebab-case；无标签时写 `[]` |
| `related` | 是 | list | 相关文档路径，格式同 `prerequisites`；可跨域；无则 `[]` |
| `code` | 否 | list | **文档所在目录的相对路径**，如 `code/03_bank_conflict_demo.cu`；无代码时不写该字段 |
| `status` | 是 | enum | `draft` · `stable` · `deprecated` |
| `updated` | 是 | date | 最后更新日期，格式 `YYYY-MM-DD` |

**路径基准**：`prerequisites` 与 `related` 用**仓库根相对路径**（跨域引用必须如此）；`code` 用**文档所在目录的相对路径**。

**`prerequisites` 与 `related` 的区别**：前者是**必须**先读的（否则看不懂），后者是**建议**延伸的。

**`prerequisites` 构成有向无环图。** 允许指向尚未创建的文档路径——这是知识图谱的前向声明，不视为错误。

### status 判定

| 取值 | 判定标准 |
|---|---|
| `draft` | 第 0–4 节**未全部覆盖**，或未经一次校对 |
| `stable` | 第 0–4 节全部覆盖，且通过第 5 节的自查清单 |
| `deprecated` | 内容已过时，文中已注明替代文档 |

`status` 与第 5 节「实测数据」无关——实测数据是可选章节，缺失不影响 `stable` 判定。

## 4. 文档模板

每篇文档必须包含以下章节：

```markdown
# 标题

> 一句话结论。读完这句就知道全文要讲什么。

## 0. 背景与动机
这项技术解决什么问题？不用它的代价是什么？朴素做法的瓶颈在哪？
须给出一个**具体的对照场景**，而非泛泛而谈。

## 1. 原理
数学/算法层面的推导。公式怎么来的、复杂度如何、边界条件是什么。
应当让读者能自己重新推导出结论，而不是记住结论。

## 2. 执行映射
这段东西在真实系统上实际发生了什么。按主题层次选择切入点：
- 硬件层主题：在 SM、warp、寄存器、各级缓存、互联上发生了什么。
- 框架层主题：在 dispatch、计算图、调度器、内存分配器上发生了什么。
- 引擎层主题：在调度策略、batching、显存管理上发生了什么。
这一节回答"为什么这个写法快、那个写法慢"。

## 3. 实现与演进
朴素做法 → 主流实现 → 前沿方案。每一步说明**为什么需要下一步**。

## 4. 踩坑与失败模式
每一条写成「**现象 → 成因 → 规避**」三段式。

## 5. 实测数据
**可选章节**。只放数字与测量方法，注明 GPU 型号与 CUDA 版本。
未实测的数字须标注为「推导值」。

## 6. 小结与延伸
## 参考
```

## 5. 深度标准与自查清单

一篇文档算"写完"（`status: stable`），须通过五条自查。前四条可机械核对，第五条需一次通读。

| # | 检查项 | 通过判据 |
|---|---|---|
| 1 | **章节完整** | 第 0、1、2、3、4 节标题均存在且非空 |
| 2 | **有推导过程** | 第 1 节包含至少一处**多步推导**（含中间步骤的公式变形或复杂度分析），而非仅罗列结论公式 |
| 3 | **有系统细节** | 第 2 节至少指明 **2 个具体组件名**（如 warp scheduler、L1 sector、dispatch key）并给出**至少 1 个数值量级**（如延迟 cycles、带宽 GB/s、寄存器数） |
| 4 | **有失败模式** | 第 4 节至少 1 条，且为「现象 → 成因 → 规避」完整三段式 |
| 5 | **可独立读懂** | 通读一遍：不依赖未列在 `prerequisites` 中的文档即可理解 |

## 6. 篇幅指引

- 单个文档聚焦**一个概念**。超过约 500 行说明该拆分。
- 宁可拆成两篇并建立 `prerequisites` 关系，也不要写一篇长文。

## 7. 横切内容的处理

量化、通信、性能、并行天然跨越多个栈层。规则：

1. **一个知识点只有一处正文**，位置由"该知识的核心归属层"决定。
2. 其他位置**只放链接**，不复制内容。
3. frontmatter 的 `related` 字段承载跨域引用。

**判定依据**：如果去掉该层，这个知识点还成立吗？成立则归更基础的层。

**引用注记**有两种方向，不可混用：

| 情形 | 模板 |
|---|---|
| 本目录只是入口，正文在别处 | `（正文在 X，此处回链）` |
| 正文在本目录，另有域外引用 | `（正文在本域，回链 Y）` |

## 8. 新增文档核对清单

每次新增或修改文档，人工核对以下四项：

1. `prerequisites` 中的所有路径是否已存在，或是否为**有意的前向声明**？
2. 新增的 `prerequisites` 边是否引入**环**？（沿 `prerequisites` 向上遍历能否回到自身）
3. `code` 中的编号是否与文档编号一致，且文件确实存在？
4. `status` 是否与第 5 节的判定相符？

## 9. 提交规范

```
<type>: <subject>
```

`type` 取值：`feat` · `fix` · `docs` · `style` · `refactor` · `test` · `chore`。知识库内容变更使用 `docs`。

## 10. 进度表维护

每完成一个二级主题，更新 `00-meta/roadmap.md`：

- `- [x] <topic>` 完成
- `- [ ] <topic> (进行中)` 进行中
- `- [ ] <topic>` 未开始

同时更新顶部的 `最后更新` 与 `进度: X / Y 主题完成`，以及该主题所属一级域 README 中的主题清单。三者必须一致。
````

- [ ] **Step 2: 验证**

Run:
```bash
test -f CONTRIBUTING.md && echo "OK 文件存在"

echo "--- 10 个编号章节 ---"
n=0
for h in "## 1. 命名规范" "## 2. 文档与代码的对应关系" "## 3. frontmatter 规范" \
         "## 4. 文档模板" "## 5. 深度标准与自查清单" "## 6. 篇幅指引" \
         "## 7. 横切内容的处理" "## 8. 新增文档核对清单" "## 9. 提交规范" \
         "## 10. 进度表维护"; do
  grep -qF "$h" CONTRIBUTING.md && n=$((n+1)) || echo "MISS $h"
done
echo "命中章节数：$n"

grep -q '编号即外键' CONTRIBUTING.md && echo "OK 含编号即外键"
grep -q '有向无环图' CONTRIBUTING.md && echo "OK 含无环约定"
grep -q '正文在 X，此处回链' CONTRIBUTING.md && echo "OK 含引用注记模板"
```
Expected: `OK 文件存在`；命中章节数 `10`，无 `MISS`；三个 `OK` 全部输出。

**为什么用逐个 `grep -qF` 而不是 `grep -c`**：第 4 节「文档模板」内嵌了一个 markdown 代码块，块内是文档模板的章节标题 `## 0. 背景与动机` … `## 6. 小结与延伸`，它们**同样是编号标题**，会被 `grep -cE '^## [0-9]+\. '` 一并计入（实测得 17 而非 10）。grep 不区分代码围栏，逐个精确匹配标题文本可以完全避开这个问题。

- [ ] **Step 3: 提交**

```bash
git add CONTRIBUTING.md
git commit -m "docs: 添加写作规范 CONTRIBUTING.md"
```

---

### Task 3: 文档模板 `_template.md`

**Files:**
- Create: `_template.md`

- [ ] **Step 1: 写 `_template.md`**

````markdown
---
title: <文档标题，与下方一级标题一致>
slug: <全局唯一，kebab-case>
domain: <一级域目录名全称，含编号，如 20-gpu-architecture>
topic: <二级目录名；无二级目录时删除本行>
level: <1=概念 2=原理 3=实现 4=优化 5=前沿>
difficulty: <beginner | intermediate | advanced>
prerequisites:
  - <仓库根相对路径，不含 .md 后缀；无前置时写 []>
tags: [<小写 kebab-case 标签>]
related:
  - <仓库根相对路径；无则写 []>
code:
  - code/<NN_名称.扩展名，NN 与本文件名一致；无代码时删除本行>
status: draft
updated: <YYYY-MM-DD>
---

# <文档标题>

> <一句话结论。读完这句就知道全文要讲什么。>

## 0. 背景与动机

<这项技术解决什么问题？不用它的代价是什么？朴素做法的瓶颈在哪？>
<须给出一个具体的对照场景，而非泛泛而谈。>

## 1. 原理

<数学/算法层面的推导。公式怎么来的、复杂度如何、边界条件是什么。>
<应当让读者能自己重新推导出结论，而不是记住结论。>
<至少包含一处多步推导，保留中间步骤。>

## 2. 执行映射

<这段东西在真实系统上实际发生了什么。>
<硬件层主题写 SM/warp/寄存器/缓存/互联；框架层写 dispatch/计算图/调度器/内存分配器；引擎层写调度策略/batching/显存管理。>
<至少指明 2 个具体组件名，并给出至少 1 个数值量级。>

## 3. 实现与演进

<朴素做法 → 主流实现 → 前沿方案。每一步说明为什么需要下一步。>
<关键代码片段内嵌在正文中，完整可运行版本见 code/。>

## 4. 踩坑与失败模式

<每一条写成「现象 → 成因 → 规避」三段式。>

### 现象
### 成因
### 规避

## 5. 实测数据

<可选章节。只放数字与测量方法，注明 GPU 型号与 CUDA 版本。>
<未实测的数字须标注为「推导值」。测量方法链接到 50-performance/。>

## 6. 小结与延伸

<要点回顾 + 指向 related 中的文档。>

## 参考

- <官方文档、论文、源码位置>
````

- [ ] **Step 2: 验证**

Run:
```bash
test -f _template.md && echo "OK 文件存在"
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式" "## 5. 实测数据" "## 6. 小结与延伸" "## 参考"; do
  grep -qF "$h" _template.md && echo "OK   $h" || echo "MISS $h"
done
```
Expected: `OK 文件存在`，八个小节全部 `OK`。

- [ ] **Step 3: 提交**

```bash
git add _template.md
git commit -m "docs: 添加单篇文档模板 _template.md"
```

---

## Chunk 2: 骨架与元信息（19 个文件）

Chunk 1 定下规范后，本组创建全部一级域目录及其 README，以及 `00-meta/` 的四个文件。

**硬约束（spec §3.3）**：**只创建 16 个一级域目录。** 除 `20-gpu-architecture/` 下的 `execution-model/` 与 `memory-system/`（Chunk 3、4 创建）外，**不得创建任何其他二级目录**。

### Task 4: `00-meta/` 四个文件

**Files:**
- Create: `00-meta/README.md`
- Create: `00-meta/glossary.md`
- Create: `00-meta/index.md`
- Create: `00-meta/roadmap.md`

- [ ] **Step 1: 创建目录并写 `00-meta/README.md`**

```bash
mkdir -p 00-meta
```

```markdown
# 00-meta —— 知识库自身

存放知识库的元信息，不是学习内容。

| 文件 | 用途 |
|---|---|
| [glossary.md](glossary.md) | 术语表中英对照。全库术语以此为准，避免同一概念多种译名。 |
| [index.md](index.md) | 全库索引，按域组织。 |
| [roadmap.md](roadmap.md) | 实施进度表，粒度到二级主题。 |

写作规范见仓库根目录的 [CONTRIBUTING.md](../CONTRIBUTING.md)。
```

- [ ] **Step 2: 写 `00-meta/glossary.md`**

术语表按域分组，**不少于 30 条**。每条格式为 `| 英文 | 中文 | 说明 |`。首批收录以下 **49 条**（完整内容如下，分组为 15 + 15 + 3 + 8 + 8）：

````markdown
# 术语表

全库术语以此为准。新增术语按所属域插入对应分组。

## GPU 硬件与执行模型

| 英文 | 中文 | 说明 |
|---|---|---|
| SM (Streaming Multiprocessor) | 流式多处理器 | GPU 的基本计算单元，内含多个 CUDA core、warp 调度器、寄存器文件与共享内存。是 warp 调度的容器。 |
| warp | 线程束 | 32 个线程组成的调度单位。SM 以 warp 为最小单位调度与发射指令。 |
| SIMT (Single Instruction, Multiple Threads) | 单指令多线程 | 硬件负责将标量写法向量化的执行模型，与 SIMD 的"程序员负责向量化"相对。 |
| thread block | 线程块 | 由多个 warp 组成，是共享内存与同步的作用域，且必须整体调度到同一个 SM 上。 |
| grid | 网格 | 一次 kernel 启动的全部线程块集合。 |
| CUDA core | CUDA 核心 | SM 内执行整数/浮点运算的流水线。其数量常被误当作唯一性能指标。 |
| warp scheduler | 线程束调度器 | SM 内负责挑选就绪 warp 并发射指令的部件。每个 SM 通常有多个。 |
| register file | 寄存器文件 | SM 内寄存器总量。是分配预算，不是缓存，用满即限制占用率。 |
| occupancy | 占用率 | SM 上实际常驻 warp 数与最大可常驻 warp 数之比。受寄存器、共享内存、block 数、warp 数四项约束取最小。 |
| latency hiding | 延迟隐藏 | 通过维持足够多的并发 warp，在部分 warp 等待访存时让其他 warp 继续执行，从而掩盖延迟。 |
| SIMT divergence | 分支发散 | 同一 warp 内线程走不同控制流路径，导致路径串行执行。 |
| reconvergence | 重汇聚 | 发散的分支路径执行完毕后，warp 内线程重新同步到同一控制流点。 |
| predication | 谓词执行 | 用指令前缀谓词代替跳转的机制，可避免短分支产生发散。 |
| ILP (Instruction-Level Parallelism) | 指令级并行 | 单个线程内多条无依赖指令的重叠执行。 |
| TLP (Thread-Level Parallelism) | 线程级并行 | 通过多 warp 并发来填补流水线气泡。 |

## 存储层次

| 英文 | 中文 | 说明 |
|---|---|---|
| memory hierarchy | 存储层次 | 按容量、带宽、延迟权衡分级的存储结构。 |
| HBM (High Bandwidth Memory) | 高带宽显存 | 片外显存，容量最大、带宽最高、延迟最长。 |
| L2 cache | 二级缓存 | 片上全局共享缓存，所有 SM 可见，是 HBM 之前的最后一级。 |
| L1 cache | 一级缓存 | 每 SM 私有，与共享内存共享同一块物理存储。 |
| shared memory | 共享内存 | 片内可编程管理的存储，延迟约比 HBM 低两个数量级。软件管理的缓存。 |
| local memory | 本地内存 | 寄存器溢出时的落点，物理上位于片外，按线程私有寻址。名字有误导性。 |
| coalescing | 访存合并 | 同一 warp 内多个线程的访存请求被合并为更少的内存事务。 |
| cache line | 缓存行 | L2 的传输单位，通常 128 字节。 |
| sector | 扇区 | L1 的传输单位，通常 32 字节。每个 cache line 含 4 个 sector。 |
| bank conflict | bank 冲突 | 同一 warp 内多个线程访问共享内存的同一 bank 的不同地址，导致访问串行化。 |
| padding | 填充 | 通过在共享内存数组维度上加 1 来错开 bank 映射，消除冲突。 |
| swizzle | 重排 | 以异或等方式重映射地址，消除冲突或提升合并度，且不浪费容量。 |
| bandwidth | 带宽 | 单位时间可传输的数据量。衡量访存受限算子的核心指标。 |
| arithmetic intensity | 算术强度 | 每字节访存所对应的浮点运算次数。roofline 模型的横轴。 |
| roofline | 屋顶线 | 以算术强度为横轴、可达性能为纵轴的性能上界模型。 |

## 互联与多卡

| 英文 | 中文 | 说明 |
|---|---|---|
| NVLink | — | NVIDIA 的 GPU 间高速互联，带宽显著高于 PCIe。 |
| NVSwitch | — | 连接多张 GPU 的交换芯片，构成全互联拓扑。 |
| PCIe | — | 通用外设互联总线，GPU 与主机通信的主要通道。 |

## 精度与量化

| 英文 | 中文 | 说明 |
|---|---|---|
| FP32 / FP16 / BF16 | 单/半精度浮点 | 标准浮点格式。BF16 牺牲尾数换取与 FP32 相同的指数范围。 |
| TF32 | — | Tensor Core 上的 19 位浮点输入格式，兼顾精度与吞吐。 |
| FP8 | 8 位浮点 | 分 E4M3 与 E5M2 两种变体，用于 Hopper 及以后的低精度计算。 |
| INT8 / INT4 | 8/4 位定点 | 量化推理常用格式。 |
| PTQ (Post-Training Quantization) | 训练后量化 | 不重新训练，仅用校准数据确定量化参数。 |
| QAT (Quantization-Aware Training) | 量化感知训练 | 训练时插入伪量化节点，让模型适应量化误差。 |
| scale / zero-point | 缩放因子/零点 | 仿射量化的两个参数，决定浮点与整数域的映射。 |
| outlier | 离群值 | 数值分布中的极端值，是激活量化困难的主因。 |

## 训练与推理系统

| 英文 | 中文 | 说明 |
|---|---|---|
| autograd | 自动微分 | 依据计算图自动求导的机制。 |
| KV cache | 键值缓存 | 自回归解码时缓存历史 token 的 K/V，避免重复计算。 |
| continuous batching | 连续批处理 | 在请求级别动态进出批次，而非等待整批完成。 |
| PagedAttention | 分页注意力 | 将 KV cache 按页管理，消除显存碎片。 |
| speculative decoding | 投机解码 | 用小模型草拟、大模型验证，减少解码步数。 |
| ZeRO | — | 将优化器状态、梯度、参数分片到各卡的显存优化方案。 |
| FSDP (Fully Sharded Data Parallel) | 全分片数据并行 | ZeRO 思想在 PyTorch 中的实现。 |
| all-reduce | 全规约 | 集合通信原语：所有参与者的数据求和后广播给所有人。 |
````

- [ ] **Step 3: 写 `00-meta/index.md`**

````markdown
# 全库索引

按一级域组织。每个域下列出该域已创建的主题目录；尚未创建的主题见 [roadmap.md](roadmap.md)。

> 本文件手工维护。当文档数超过 50 时改为脚本生成（见设计文档 §11）。

## 00-meta
- [术语表](glossary.md)
- [索引](index.md)（本文件）
- [进度表](roadmap.md)

## 10-foundations
尚未开始。

## 20-gpu-architecture
<!-- 以下两行是前向声明：目录由 Chunk 3/4 创建。全部创建完成后可删除本注释。 -->
- [execution-model](../20-gpu-architecture/execution-model/) —— SM 结构 / SIMT 与发散 / 占用率与延迟隐藏
- [memory-system](../20-gpu-architecture/memory-system/) —— 存储层次 / 访存合并 / 共享内存 bank 冲突

## 30-gpu-programming
尚未开始。

## 40-kernels
尚未开始。

## 50-performance
尚未开始。

## 60-compilers
尚未开始。

## 70-frameworks
尚未开始。

## 80-model-architectures
尚未开始。

## 90-training
尚未开始。

## 100-parallelism
尚未开始。

## 110-communication
尚未开始。

## 120-inference
尚未开始。

## 130-quantization
尚未开始。

## 140-papers
尚未开始。

## 150-playground
尚未开始。
````

- [ ] **Step 4: 写 `00-meta/roadmap.md`**

**注意**：`140-papers` 与 `150-playground` 写成纯文字，不写成 `- [ ]` 任务列表项（避免被脚本统计误匹配）。分母 61 = 13 个有固定二级结构的域的二级目录总数。

````markdown
# 实施进度表

> 最后更新：2026-09-14
> 进度：0 / 61 主题完成

状态标记：`- [x]` 完成 · `- [ ] <topic> (进行中)` 进行中 · `- [ ] <topic>` 未开始

计数口径：分母 61 = 有固定二级结构的 13 个域的二级目录总数（`00-meta`、`140-papers`、`150-playground` 不计入）。新增或删除主题时须同步更新分母。

## 10-foundations
- [ ] floating-point
- [ ] numerical-linear-algebra
- [ ] performance-models

## 20-gpu-architecture
- [ ] execution-model
- [ ] memory-system
- [ ] tensor-core
- [ ] interconnect
- [ ] architecture-evolution

## 30-gpu-programming
- [ ] cuda
- [ ] triton
- [ ] ptx-sass
- [ ] cutlass

## 40-kernels
- [ ] gemm
- [ ] attention
- [ ] reduce-scan
- [ ] softmax-normalization
- [ ] elementwise-fusion
- [ ] moe-kernels

## 50-performance
- [ ] profiling-tools
- [ ] bottleneck-diagnosis
- [ ] tuning-practice

## 60-compilers
- [ ] mlir
- [ ] tvm
- [ ] inductor
- [ ] xla
- [ ] auto-tuning

## 70-frameworks
- [ ] pytorch
- [ ] tensorflow
- [ ] jax

## 80-model-architectures
- [ ] transformer
- [ ] attention-variants
- [ ] moe
- [ ] normalization
- [ ] positional-encoding

## 90-training
- [ ] autograd-and-backward
- [ ] mixed-precision
- [ ] optimizers
- [ ] initialization-and-stability
- [ ] checkpointing

## 100-parallelism
- [ ] data-parallel
- [ ] tensor-parallel
- [ ] pipeline-parallel
- [ ] sequence-context-parallel
- [ ] expert-parallel
- [ ] zero-fsdp

## 110-communication
- [ ] collectives
- [ ] nccl
- [ ] network-topology
- [ ] overlap

## 120-inference
- [ ] kv-cache
- [ ] batching-and-scheduling
- [ ] decoding-strategies
- [ ] engines
- [ ] serving-infra

## 130-quantization
- [ ] numerical-formats
- [ ] theory
- [ ] ptq
- [ ] qat
- [ ] quantized-kernels
- [ ] kv-cache-quantization
- [ ] sparsity-and-distillation

## 140-papers
（无固定二级结构，按需追加）

## 150-playground
（无固定二级结构，按需追加）
````

- [ ] **Step 5: 验证**

Run:
```bash
echo "--- 文件存在 ---"
for f in 00-meta/README.md 00-meta/glossary.md 00-meta/index.md 00-meta/roadmap.md; do
  test -f "$f" && echo "OK   $f" || echo "MISS $f"
done

echo "--- 术语条数（应为 49）---"
grep -cE '^\| [A-Za-z]' 00-meta/glossary.md

echo "--- 进度表主题数（应为 61）---"
grep -cE '^- \[ \] [a-z]' 00-meta/roadmap.md
```
Expected: 四个 `OK`；术语条数 `49`（照抄则为 49，若自行增删应 ≥ 30）；主题数恰好 `61`。

- [ ] **Step 6: 提交**

```bash
git add 00-meta/
git commit -m "docs: 添加 00-meta 术语表、索引与进度表"
```

---

### Task 5: 一级域 README 批次 A（7 个）

**Files:**
- Create: `10-foundations/README.md`
- Create: `30-gpu-programming/README.md`
- Create: `40-kernels/README.md`
- Create: `50-performance/README.md`
- Create: `60-compilers/README.md`
- Create: `70-frameworks/README.md`
- Create: `80-model-architectures/README.md`

**每个 README 的统一结构**（spec §10.1 A6 要求三部分）：

```markdown
# <域编号> <域名称>

一句话定位。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| <二级目录名> | <一句话描述，取自 spec §3.2> |

## 与其他域的关系

- **上游**：<本域依赖哪些域>
- **下游**：<哪些域依赖本域>
- **横切**：<与本域相关的横切主题及其正文位置>

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 1: 创建七个目录**

```bash
mkdir -p 10-foundations 30-gpu-programming 40-kernels 50-performance \
         60-compilers 70-frameworks 80-model-architectures
```

- [ ] **Step 2: 写 `10-foundations/README.md`**

```markdown
# 10-foundations 数学与数值基础

大模型系统里反复用到、且不理解就会算错的数学与数值前提。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| floating-point | IEEE754 格式、舍入模式、误差累积、次正规数 |
| numerical-linear-algebra | 矩阵计算、条件数、数值稳定性、迭代求解 |
| performance-models | roofline 模型、算术强度、Amdahl 定律、延迟-带宽模型 |

## 与其他域的关系

- **上游**：无。本域是全库最底层，`prerequisites` 为空。
- **下游**：`20-gpu-architecture`（延迟-带宽模型）、`40-kernels`（roofline 判断算子性质）、`90-training`（混合精度与数值稳定性）、`130-quantization`（误差分析）。
- **横切**：`performance-models` 中的 roofline 模型（正文在本域，回链 `40-kernels`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 3: 写 `30-gpu-programming/README.md`**

```markdown
# 30-gpu-programming GPU 编程模型

从 CUDA C++ 到 Triton、从源码到 SASS，指挥 GPU 的几套语言与抽象层次。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| cuda | 线程组织 / 同步原语 / stream / 内存管理 / CUDA Graph |
| triton | Triton 语言模型 / 编译流程 / 常见 kernel 模式 |
| ptx-sass | PTX 与 SASS 指令集 / 寄存器分配 / 指令调度 / 反汇编方法 |
| cutlass | CUTLASS 模板抽象 / 层次化 tiling / collective 设计 |

## 与其他域的关系

- **上游**：`20-gpu-architecture`。不理解 SM 与存储层次，写不出好 kernel。
- **下游**：`40-kernels`（用这些工具写算子）、`60-compilers`（Inductor 生成 Triton 代码，需要读懂才能调优）。
- **横切**：`ptx-sass` 是性能分析的最终依据，`50-performance` 中的瓶颈定位常需下钻到 SASS。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 4: 写 `40-kernels/README.md`**

```markdown
# 40-kernels 算子内核工程

把原理落成可跑的 kernel，并让它跑到硬件允许的上界。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| gemm | 朴素 → tiling → 寄存器分块 → tcgen05 → persistent kernel |
| attention | flash attention 系列演进 / 变体 / 解码阶段 kernel |
| reduce-scan | 归约与扫描的并行算法 / warp shuffle / 多级归约 |
| softmax-normalization | softmax 数值稳定实现 / 在线 softmax / norm 融合 |
| elementwise-fusion | 逐元素算子融合 / 访存受限算子的优化边界 |
| moe-kernels | grouped GEMM / all-to-all / 专家调度 kernel |

## 与其他域的关系

- **上游**：`20-gpu-architecture`（硬件行为）、`30-gpu-programming`（编程工具）、`10-foundations`（roofline 与数值）。
- **下游**：`60-compilers`（编译器要生成的正是这些 kernel）、`120-inference`（引擎的算力上限由这些 kernel 决定）。
- **横切**：量化算子实现（正文在 `130-quantization`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 5: 写 `50-performance/README.md`**

```markdown
# 50-performance 性能工程

跨层的性能方法论。本域不重复讲具体算子的优化手法，只讲**怎么定位瓶颈**与**怎么测量**。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| profiling-tools | nsight compute / nsight systems / CUPTI / torch profiler |
| bottleneck-diagnosis | 瓶颈定位方法论 / 常见反模式 / 计算-访存-延迟三分法 |
| tuning-practice | 调优实战 / 参数搜索 / 可复现的基准测试方法 |

## 与其他域的关系

- **上游**：`20-gpu-architecture`（硬件计数器）、`10-foundations`（roofline 作为判据）。
- **下游**：`40-kernels`、`60-compilers`、`120-inference` 的调优都引用本域的方法论。
- **横切**：本域是横切域。任何文档第 5 节「实测数据」的测量方法都链接到此处，不在各文档里重复描述测量步骤。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 6: 写 `60-compilers/README.md`**

```markdown
# 60-compilers 编译器与图优化

从计算图到 kernel 的自动化路径。理解它才能在自动生成的效果不好时知道该动哪里。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| mlir | MLIR 方言体系 / lowering 流程 / pass 编写 |
| tvm | TVM 与 TileLang / 调度原语 / 自动代码生成 |
| inductor | torch.compile 后端 / 图捕获 / codegen / 与 Triton 的关系 |
| xla | TensorFlow 与 JAX 的图编译 / HLO / 算子融合 |
| auto-tuning | 自动调优方法 / 搜索空间设计 / 代价模型 |

## 与其他域的关系

- **上游**：`70-frameworks`（编译器是框架的一部分）、`30-gpu-programming`（生成的代码是 Triton 或 CUDA）。
- **下游**：`120-inference`（引擎大量依赖编译期优化）。
- **横切**：`inductor`（正文在本域，回链 `70-frameworks/pytorch`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 7: 写 `70-frameworks/README.md`**

```markdown
# 70-frameworks 深度学习框架

写模型的地方，也是理解"一个算子从 Python 调用到 GPU 执行经过了多少层"的地方。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| pytorch | autograd / dispatch 机制 / aten / 自定义算子 / torch.compile |
| tensorflow | graph 与 eager / XLA 集成 / custom op / SavedModel |
| jax | jit / vmap / pmap / sharding / 与 PyTorch 的心智差异 |

## 与其他域的关系

- **上游**：`30-gpu-programming`（自定义算子要写 CUDA）。
- **下游**：`80-model-architectures`（模型用框架写）、`90-training`、`100-parallelism`、`120-inference`。
- **横切**：torch.compile 与 XLA 的编译器实现（正文在 `60-compilers`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 8: 写 `80-model-architectures/README.md`**

```markdown
# 80-model-architectures 模型架构

要优化的对象。所有算子、并行策略、推理优化的收益最终都要落在这个结构上。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| transformer | 原版结构 → 现代变体 / 残差设计 / 深度与宽度缩放 |
| attention-variants | MHA / MQA / GQA / MLA / 线性注意力 / 混合注意力 |
| moe | gating 机制 / 专家路由 / 负载均衡 / 容量因子 |
| normalization | LayerNorm / RMSNorm / QK-norm / pre-post 位置之争 |
| positional-encoding | RoPE / ALiBi / 相对位置 / 长度外推方法 |

## 与其他域的关系

- **上游**：`70-frameworks`（模型用框架实现）。
- **下游**：`40-kernels`（attention 变体直接决定 kernel 设计）、`100-parallelism`（MoE 引出专家并行）、`120-inference`（KV cache 的大小由架构决定）。
- **横切**：注意力变体的 kernel 实现在 `40-kernels/attention`，本域只讲架构层面的取舍。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 9: 验证**

Run:
```bash
echo "--- 文件存在 ---"
for d in 10-foundations 30-gpu-programming 40-kernels 50-performance 60-compilers 70-frameworks 80-model-architectures; do
  test -f "$d/README.md" && echo "OK   $d" || echo "MISS $d"
done

echo "--- 三要素完整性（每个应有 3 个匹配）---"
for d in 10-foundations 30-gpu-programming 40-kernels 50-performance 60-compilers 70-frameworks 80-model-architectures; do
  n=$(grep -cE '^## (本域覆盖什么|与其他域的关系|计划主题清单)' "$d/README.md")
  echo "$n  $d"
done
```
Expected: 七个 `OK`；每个域的计数均为 `3`。

- [ ] **Step 10: 提交**

```bash
git add 10-foundations 30-gpu-programming 40-kernels 50-performance 60-compilers 70-frameworks 80-model-architectures
git commit -m "docs: 添加 7 个一级域的 README（批次 A）"
```

---

### Task 6: 一级域 README 批次 B（7 个）

**Files:**
- Create: `90-training/README.md`
- Create: `100-parallelism/README.md`
- Create: `110-communication/README.md`
- Create: `120-inference/README.md`
- Create: `130-quantization/README.md`
- Create: `140-papers/README.md`
- Create: `150-playground/README.md`

结构与 Task 5 相同。`140-papers` 与 `150-playground` 无固定二级结构，「本域覆盖什么」一节改为说明收录规则。

- [ ] **Step 1: 创建七个目录**

```bash
mkdir -p 90-training 100-parallelism 110-communication 120-inference \
         130-quantization 140-papers 150-playground
```

- [ ] **Step 2: 写 `90-training/README.md`**

```markdown
# 90-training 训练系统

把模型训起来，并让它不崩、不慢、可复现。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| autograd-and-backward | 计算图构造 / 反向传播实现 / 梯度检查点 |
| mixed-precision | AMP / loss scaling / fp8 训练 / 精度对齐与复现 |
| optimizers | SGD / Adam / AdamW / Muon / 分布式优化器状态 |
| initialization-and-stability | 参数初始化 / 梯度裁剪 / loss spike 归因 |
| checkpointing | 分片保存与恢复 / 容错 / 异步落盘 |

## 与其他域的关系

- **上游**：`70-frameworks`、`10-foundations`（数值稳定性）。
- **下游**：`100-parallelism`（并行策略服务于训练）、`110-communication`（分布式训练的通信开销）。
- **横切**：优化器**未**独立成一级域，因为与训练循环、混合精度、分布式优化器状态强耦合。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 3: 写 `100-parallelism/README.md`**

```markdown
# 100-parallelism 并行策略

把模型切到多卡上。每种切法都有通信量与显存占用的代价，本域讲清楚这些账怎么算。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| data-parallel | DP / DDP / 梯度规约（正文在 110，此处回链） |
| tensor-parallel | TP / 列行并行 / 通信量推导 / 与 TP 组划分 |
| pipeline-parallel | PP / 1F1B / 气泡率分析 / 显存均衡 |
| sequence-context-parallel | SP / CP / ring attention / 长序列切分 |
| expert-parallel | EP / all-to-all 通信模式 |
| zero-fsdp | ZeRO 三阶段 / FSDP / 显存占用核算 |

## 与其他域的关系

- **上游**：`90-training`、`110-communication`（并行策略的通信开销由通信域定性）。
- **下游**：`120-inference`（推理的并行形态与训练不同，但共享概念）。
- **横切**：梯度规约算法（ring allreduce 等）（正文在 `110-communication`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 4: 写 `110-communication/README.md`**

```markdown
# 110-communication 通信与网络

分布式系统的物理约束在这里。通信算法的复杂度与网络拓扑决定了并行的可扩展性上界。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| collectives | allreduce / allgather / reducescatter 算法与复杂度 |
| nccl | NCCL 架构 / 调优 / 拓扑感知 / 常见报错归因 |
| network-topology | InfiniBand / RoCE / NVLink 域 / 胖树拓扑 |
| overlap | 通信-计算重叠 / 双缓冲 / 分块流水 |

## 与其他域的关系

- **上游**：`20-gpu-architecture/interconnect`（NVLink 的物理带宽与拓扑）。
- **下游**：`100-parallelism`（并行策略的可行性由通信决定）。
- **横切**：NVLink 与 NVSwitch 的硬件细节（正文在 `20-gpu-architecture/interconnect`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 5: 写 `120-inference/README.md`**

```markdown
# 120-inference 推理引擎与部署

从权重到服务。训练追求吞吐，推理要同时面对吞吐、延迟与显存三重约束。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| kv-cache | KV cache 原理与显存核算 / PagedAttention / 前缀共享 |
| batching-and-scheduling | continuous batching / chunked prefill / 调度策略 |
| decoding-strategies | beam search / sampling / speculative decoding / 并行解码 |
| engines | vLLM / TensorRT-LLM / SGLang / llama.cpp 架构对比 |
| serving-infra | 部署形态 / PD 分离 / 扩缩容 / 可观测性 |

## 与其他域的关系

- **上游**：`80-model-architectures`（KV cache 大小由架构决定）、`40-kernels`（算力上限由 kernel 决定）、`60-compilers`。
- **下游**：无。这是栈的最上层。
- **横切**：KV cache 量化（正文在 `130-quantization`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 6: 写 `130-quantization/README.md`**

```markdown
# 130-quantization 量化与压缩

用更少的比特表示同样的模型。横切域：既涉及硬件数值格式，也涉及算子实现与推理引擎。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| numerical-formats | fp16 / bf16 / tf32 / fp8 / int8 / int4 / nvfp4 |
| theory | 量化误差分析 / scale 与 zero-point / outlier 问题 |
| ptq | 训练后量化 / 校准方法 / GPTQ / AWQ / SmoothQuant |
| qat | 量化感知训练 / 伪量化 / 梯度近似 |
| quantized-kernels | 量化算子实现（正文在本域，回链 40-kernels） |
| kv-cache-quantization | KV cache 量化（正文在本域，回链 120-inference） |
| sparsity-and-distillation | 结构化与非结构化稀疏 / 知识蒸馏 |

## 与其他域的关系

- **上游**：`10-foundations/floating-point`（数值格式的误差来源）、`20-gpu-architecture/tensor-core`（低精度算力）。
- **下游**：`120-inference`（量化是推理加速的主要手段）。
- **横切**：量化算子实现（正文在本域，回链 `40-kernels`）；KV cache 量化（正文在本域，回链 `120-inference`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 7: 写 `140-papers/README.md`**

```markdown
# 140-papers 论文精读

论文本身的精读：动机、方法、实验、局限。

## 收录规则

- 收录**论文级别**的分析，不收录主题性知识。若某篇论文的结论已沉淀为某个主题的正文（如 flash attention 的算法在 `40-kernels/attention`），则**正文仍归该主题**，本域只保留该论文的独立分析并双向 `related` 链接。
- 无固定二级结构，一篇论文一个文件，命名 `NNN-<论文简称>.md`，如 `001-flash-attention.md`。
- 沿用全库统一模板（见 [CONTRIBUTING.md](../CONTRIBUTING.md)），但第 2 节「执行映射」改为「方法剖析」。

## 与其他域的关系

- **上游**：无固定上游，视论文而定。
- **下游**：论文的结论被各域引用，本域提供原始出处。

## 计划清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 8: 写 `150-playground/README.md`**

```markdown
# 150-playground 实验场

尚未归位的实验。

## 收录规则

- 存放**临时的、尚未沉淀为知识**的实验脚本与结论草稿。
- 一旦某实验沉淀为可复用的知识，**须迁入对应域**，不在本域长期留存。
- 无固定二级结构，按需建文件。

## 与其他域的关系

本域是**中转站**，不承担知识职责。任何内容在本域停留超过一次整理周期，都说明它该被归位或删除。

## 计划清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 9: 验证**

Run:
```bash
echo "--- 文件存在 ---"
for d in 90-training 100-parallelism 110-communication 120-inference 130-quantization 140-papers 150-playground; do
  test -f "$d/README.md" && echo "OK   $d" || echo "MISS $d"
done

echo "--- 三要素完整性（5 个常规域应各为 3）---"
for d in 90-training 100-parallelism 110-communication 120-inference 130-quantization; do
  n=$(grep -cE '^## (本域覆盖什么|与其他域的关系|计划主题清单)' "$d/README.md")
  echo "$n  $d"
done

echo "--- 引用注记（应命中 4 个文件）---"
grep -l '回链' 90-training/README.md 100-parallelism/README.md 110-communication/README.md \
  120-inference/README.md 130-quantization/README.md 2>/dev/null | sort
```
Expected: 五个常规域的计数均为 `3`；`grep -l` 命中 `100-parallelism`、`110-communication`、`120-inference`、`130-quantization` 四个文件（`90-training` 不含引用注记）。

**注**：`140-papers` 与 `150-playground` 两个域的「本域覆盖什么」一节按设计改为「收录规则」，收尾标题也从「计划主题清单见」改为「计划清单见」——这是有意为之，不参与上面的计数。

- [ ] **Step 10: 提交**

```bash
git add 90-training 100-parallelism 110-communication 120-inference 130-quantization 140-papers 150-playground
git commit -m "docs: 添加 7 个一级域的 README（批次 B）"
```

---

### Task 7: `20-gpu-architecture/README.md`

**Files:**
- Create: `20-gpu-architecture/README.md`

本域与其余 14 个域不同：它下面有两个首期二级目录，README 需要提供入口。

- [ ] **Step 1: 创建目录并写 README**

```bash
mkdir -p 20-gpu-architecture
```

```markdown
# 20-gpu-architecture GPU 体系结构

硬件的静态结构。不理解这一层，后面所有「为什么这个写法快」都没有依据。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [execution-model](execution-model/) | SM 内部结构 / warp 与 SIMT / 分支发散 / 占用率与延迟隐藏 | 未开始 |
| [memory-system](memory-system/) | 存储层次 / 访存合并 / 共享内存 bank 冲突 | 未开始 |
| tensor-core | MMA 指令族 / wmma / wgmma / tcgen05 / 稀疏 MMA | 未开始 |
| interconnect | NVLink / NVSwitch / PCIe / 多卡拓扑与带宽 | 未开始 |
| architecture-evolution | Volta→Ampere→Hopper→Blackwell 逐代规格差异 | 未开始 |

## 与其他域的关系

- **上游**：`10-foundations`（延迟-带宽模型是理解存储层次的工具）。
- **下游**：`30-gpu-programming`、`40-kernels`、`50-performance`、`110-communication`。
- **横切**：NVLink 带宽与拓扑（正文在本域，回链 `110-communication`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)
```

- [ ] **Step 2: 验证**

Run:
```bash
test -f 20-gpu-architecture/README.md && echo "OK 文件存在"
grep -c '^| \[' 20-gpu-architecture/README.md
```
Expected: `OK 文件存在`；输出 `2`（两个已建目录的主题带链接）。

**关于「状态」列**：此时两行写「未开始」，因为 Chunk 3/4 尚未产出内容。**Chunk 5 的 Task 17 会把它们改为「已完成」**，与 `00-meta/roadmap.md` 同步。不要在 Task 7 就写成「已完成」——那会与同一时刻 roadmap 的 `- [ ]` 冲突，违反 CONTRIBUTING 第 10 节的三者一致性要求。

- [ ] **Step 3: 提交**

```bash
git add 20-gpu-architecture/README.md
git commit -m "docs: 添加 20-gpu-architecture 域 README"
```

---

### Task 8: 骨架完整性核验

**Files:**
- 无新建。本任务是纯验证。

- [ ] **Step 1: 核验目录集合与 spec §3.3 完全一致**

Run:
```bash
echo "--- 实际存在的目录（排除 .git 与 docs）---"
find . -type d -not -path './.git*' -not -path './docs*' -not -name '.' | sort

echo "--- 一级域目录数（应为 16）---"
find . -maxdepth 1 -type d -name '[0-9]*-*' | wc -l

echo "--- 二级目录数（应为 0，Chunk 3/4 之前）---"
find . -mindepth 2 -maxdepth 2 -type d -not -path './.git*' -not -path './docs*' | wc -l
```
Expected: 一级域目录数为 `16`；二级目录数为 `0`。

**注意**：这里用 `-name '[0-9]*-*'` 而不是 `-regex`。两个原因：

1. `-regex '\./[0-9][0-9]-.*'` 只匹配两位数字前缀，会漏掉 `100-parallelism` 到 `150-playground` 六个三位数域（实测只输出 10）。
2. 即使写成正确的 `-regex '\./[0-9]+-.*'`，BSD `find` 与 GNU `find` 的 `-regex` 语义也不一致（BSD 用的是基本正则）。`-name` 在两个实现下行为相同，实测均得 16。

**关于「二级目录数为 0」**：这是 Chunk 3/4 **之前**的中途快照。首期骨架最终**必须包含** `20-gpu-architecture/execution-model/` 与 `memory-system/` 两个二级目录（spec §10.1 A5），它们由 Chunk 3/4 创建，并在 Chunk 5 核验。

- [ ] **Step 2: 核验 16 个域 README 齐全**

Run:
```bash
n=0
for d in $(find . -maxdepth 1 -type d -name '[0-9]*-*' | sort); do
  test -f "$d/README.md" && n=$((n+1)) || echo "MISS $d/README.md"
done
echo "有 README 的域数：$n"
```
Expected: `有 README 的域数：16`，无 `MISS` 输出。

- [ ] **Step 3: 核验进度表主题数与目录清单一致**

Run:
```bash
echo "roadmap 主题数：$(grep -cE '^- \[ \] [a-z]' 00-meta/roadmap.md)"
echo "roadmap 域标题数：$(grep -cE '^## [0-9]' 00-meta/roadmap.md)"
```
Expected: 主题数 `61`，域标题数 `15`。

**关于 15**：16 个一级域中 `00-meta` 没有 roadmap 分节（它是元信息域，无二级主题），其余 15 个都有 `## <编号>-<名称>` 标题。`140-papers` 与 `150-playground` 的标题也以数字开头，同样计入这 15 个。

- [ ] **Step 4: 核验 `.gitignore` 覆盖构建产物**

Run:
```bash
test -s .gitignore && echo "OK .gitignore 非空"
grep -q '\.out' .gitignore && echo "OK 忽略 .out"
grep -q '\.o$' .gitignore && echo "OK 忽略 .o"
```
Expected: 三个 `OK`。

- [ ] **Step 5: 核验工作区干净**

Run:
```bash
git status --short
```
Expected: 除 `docs/` 下的文件外无输出。

**注意**：`docs/` 存放的是设计与计划工件，不属于知识库交付物。若实施过程中动了 `docs/` 而未提交，这里会显示出来 —— 提交它即可，不影响本步的通过判定。

---

## Chunk 3: `20-gpu-architecture/execution-model/`（7 个文件）

三篇文档 + 三个 `.cu` + 一个 `Makefile`。编号必须一一对应：

```
01-sm-internals.md                 ←→  code/01_sm_probe.cu
02-simt-and-divergence.md          ←→  code/02_divergence_demo.cu
03-occupancy-and-latency-hiding.md ←→  code/03_occupancy_probe.cu
```

**依赖**：本块内三篇文档的 `prerequisites` 分别是 —— `01` 为空（`[]`），`02` 与 `03` 都指向 `01`。因此**必须按 01 → 02 → 03 的顺序写**。

注意 `03` 的 `prerequisites` **只列 `01`**，`memory-system/01` 是放在 `related` 里的。这是有意为之：`memory-system/01` 也把 `01-sm-internals` 列为前置，若 `03` 再把它写进 `prerequisites` 就形成了环（spec §6 要求 `prerequisites` 构成有向无环图）。不要"顺手"把它挪进 `prerequisites`。

### Task 9: `01-sm-internals.md`

**Files:**
- Create: `20-gpu-architecture/execution-model/01-sm-internals.md`

- [ ] **Step 1: 创建目录并写 frontmatter**

```bash
mkdir -p 20-gpu-architecture/execution-model
```

文档以以下内容开头（frontmatter + 标题 + 一句话结论）：

```markdown
---
title: SM 内部结构
slug: sm-internals
domain: 20-gpu-architecture
topic: execution-model
level: 2
difficulty: intermediate
prerequisites: []
tags: [sm, warp, register-file, cuda-core, scheduler]
related:
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
code:
  - code/01_sm_probe.cu
status: stable
updated: 2026-09-14
---

# SM 内部结构

> GPU 的并行度不是「核心数」，而是「每 SM 能同时挂住多少个 warp」——而这个数由寄存器文件除以每线程寄存器用量决定。
```

- [ ] **Step 2: 写第 0 节「背景与动机」**

须包含以下具体对照场景（不可泛泛而谈）：

- 反例：一个初学者看到某卡有 132 个 SM、每个 SM 有 128 个 CUDA core，算出「16896 路并行」，于是按这个数去规划数据划分 —— 结果性能远低于预期。
- 成因：CUDA core 是**流水线宽度**，不是并发线程数。真正决定「同一时刻有多少线程在飞」的是每 SM 的**常驻 warp 数**，它受寄存器文件容量约束。
- 本节结尾提出本文要回答的问题：给定一个 kernel，如何算出它的并行度上限？

- [ ] **Step 3: 写第 1 节「原理」**

必须包含**多步推导**，逐步写出中间量：

1. 寄存器是最快的存储，由编译器在编译期为每个线程静态分配。
2. 单个 warp 占用的寄存器数 = `每线程寄存器数 × 32`。
3. SM 的寄存器文件必须为**全部常驻 warp** 预留空间，因此：

   ```
   寄存器可容纳的 warp 数 = floor(寄存器文件大小 ÷ (每线程寄存器数 × 32))
   ```

4. 该值与「每 SM 最大常驻 warp 数」（架构上限）取较小者，即为寄存器约束下的并行度上限。
5. 给出一个具体算例：设寄存器文件为 65536 个 32-bit 寄存器，则每线程 32 个寄存器时上限为 `65536 / (32×32) = 64` 个 warp；每线程 64 个寄存器时降为 `32` 个 warp；每线程 255 个寄存器时降为 `8` 个 warp。
6. 推导结论：**寄存器文件是预算，不是缓存。** 它不像 cache 那样按需换入换出，而是必须为全部常驻线程静态预留，无法超额分配。

- [ ] **Step 4: 写第 2 节「执行映射」**

必须指明**至少 2 个具体组件名**并给出**至少 1 个数值量级**：

- 组件：warp scheduler、register file、CUDA core、LSU（load/store unit）、SFU（special function unit）、常数缓存。
- 说明 warp 为何是**调度最小单位**：调度器每个周期从就绪 warp 中挑选一个，发射一条指令给 32 个 lane。
- 给出数值量级：历代架构每 SM 的最大常驻 warp 数（Volta 64、Ampere 64、Hopper 64）、每 SM 寄存器文件大小（通常 256 KB，即 65536 个 32-bit 寄存器）、寄存器分配的最小粒度（通常 8 个寄存器一档，实际占用按档上取整）。
- 解释「分派（dispatch）」与「发射（issue）」的区别。

- [ ] **Step 5: 写第 3 节「实现与演进」**

按代际讲 SM 规格变化，**每步说明为什么需要下一步**：

- Volta：引入独立线程调度与 64 个常驻 warp，寄存器文件定型为 256 KB。
- Turing：SM 结构调整，分离整数与浮点流水线。
- Ampere：每 SM 的 FP32 单元数翻倍，LSU 增强。
- Hopper：引入线程块簇（thread block cluster）与分布式共享内存，SM 之间可直接访问对方共享内存。
- Blackwell：第五代 Tensor Core，进一步的低精度支持。

对每代给出可查证的规格数字（每 SM 的 CUDA core 数、最大常驻 warp 数、共享内存容量）。

- [ ] **Step 6: 写第 4 节「踩坑与失败模式」**

至少 3 条，每条写成「**现象 → 成因 → 规避**」三段式：

1. **跨架构比 SM 数量**。现象：A 卡 132 SM、B 卡 108 SM，便断言 A 卡算力更强。成因：每 SM 的规模、频率、每周期指令数都不同，SM 数不是可比的算力单位。规避：以实测吞吐或规格表上的理论峰值（TFLOPS / TB/s）为比较依据。
2. **把寄存器文件当缓存**。现象：认为「寄存器多就是缓存大，多写点临时变量没关系」。成因：寄存器是静态预算，用多了直接压低常驻 warp 数，占用率下降，延迟隐藏能力变差，反而更慢。规避：关注编译产物的寄存器用量（`--ptxas-options=-v`），必要时用 `__launch_bounds__` 设上限。
3. **`__launch_bounds__` 误用**。现象：为了让占用率最大化，把 `minBlocksPerMultiprocessor` 设得很大，结果性能反而下降。成因：编译器为满足该约束会强行压低寄存器数，导致溢出到 local memory（物理上在片外），访存代价激增。规避：把该参数当作「上限保护」而非「优化目标」，并用 profiling 验证是否发生了 spill。

- [ ] **Step 7: 写第 5、6 节与参考**

- 第 5 节「实测数据」：本节**可整体省略**（spec §7.1 规定为可选）。若省略，直接写 `## 6. 小结与延伸`。
- 第 6 节：回顾「寄存器预算」这一核心结论，指向 `related` 中的两篇文档。
- 参考：CUDA C++ Programming Guide 的「Compute Capabilities」与「Hardware Implementation」章节；NVIDIA 对应架构的 tuning guide。

- [ ] **Step 8: 验证**

Run:
```bash
f=20-gpu-architecture/execution-model/01-sm-internals.md
echo "--- 五节完整性 ---"
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done
echo "--- frontmatter 字段 ---"
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
echo "--- 关键公式与组件名 ---"
grep -q '寄存器文件大小 ÷' "$f" && echo "OK 含寄存器预算公式" || echo "MISS 公式"
grep -q 'warp scheduler' "$f" && echo "OK 含 warp scheduler" || echo "MISS warp scheduler"
grep -q '65536' "$f" && echo "OK 含数值量级" || echo "MISS 数值"
echo "--- 三段式 ---"
grep -c '成因' "$f"
```
Expected: 五节全部 `OK`；12 个 frontmatter 字段全部 `OK`；三个关键项 `OK`；`成因` 出现次数 ≥ 3。

- [ ] **Step 9: 提交**

```bash
git add 20-gpu-architecture/execution-model/01-sm-internals.md
git commit -m "docs: 添加 SM 内部结构文档"
```

---

### Task 10: `02-simt-and-divergence.md`

**Files:**
- Create: `20-gpu-architecture/execution-model/02-simt-and-divergence.md`

- [ ] **Step 1: 写 frontmatter 与结论**

```markdown
---
title: SIMT 执行模型与分支发散
slug: simt-and-divergence
domain: 20-gpu-architecture
topic: execution-model
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/execution-model/01-sm-internals
tags: [simt, warp, divergence, predication, reconvergence]
related:
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
code:
  - code/02_divergence_demo.cu
status: stable
updated: 2026-09-14
---

# SIMT 执行模型与分支发散

> 写代码时像写标量程序，执行时却是向量化的——两者之间的落差全部由硬件承担，代价在分支发散时暴露出来。
```

- [ ] **Step 2: 写第 0 节「背景与动机」**

- 对照场景：同一段带 `if/else` 的代码，把数据换成两种分布后性能差数倍，而代码一个字没改。
- 问题：为什么「发散」是免费的假象？既然 SIMT 让每个线程有自己的 PC，为什么还会串行化？

- [ ] **Step 3: 写第 1 节「原理」**

必须包含**多步推导**：

1. 厘清 SIMT 与 SIMD 的本质区别：SIMD 由**程序员**显式使用向量类型，编译器把标量代码向量化；SIMT 由**硬件**在 warp 粒度上自动向量化，程序员写的是标量代码。因此 SIMT 的向量化决策有一部分推迟到了运行时。
2. 推导有效并行度模型。设一个 warp 内 32 个线程按数据分布落在 `N` 条互斥的执行路径上，硬件逐条路径串行执行，每次只让属于当前路径的 lane 活跃：
   ```
   执行时间 ∝ N
   有效并行度 = 32 / N
   ```
3. 特例校验：`N = 1`（全 warp 同路径）→ 有效并行度 32，无损失；`N = 2`（对半分裂）→ 有效并行度 16，理论耗时翻倍；`N = 32`（每条路径 1 个线程）→ 有效并行度 1，退化为串行。
4. 说明该模型的适用边界：它给出**最坏情形**上界，实际耗时还取决于各路径各自的指令数（若分裂出的两条路径长度悬殊，代价由短路径决定）。

- [ ] **Step 4: 写第 2 节「执行映射」**

必须指明**至少 2 个具体组件名**并给出**至少 1 个数值量级**：

- 组件：warp scheduler、reconvergence stack（重汇聚栈）、active mask（活跃掩码）、predication。
- 说明 SASS 层面的实际形态：`BSSY`（压入重汇聚点）、`BRA`（跳转）、`BSYNC`（重汇聚同步）。给出这三条指令在一次发散中的执行次序。
- 说明 predication（指令前缀 `@P0` 等）与真正分支的区别：谓词执行会**两条路径都发射**但只让符合条件的 lane 写回，因此不产生控制流分歧；它适用于极短的分支，长分支用谓词反而更慢。
- 给出数值量级：一条 warp 指令的发射开销、共享内存访问延迟（约 20~30 cycles）、全局访存延迟（数百 cycles），用于说明「为什么短分支不值得分裂」。

- [ ] **Step 5: 写第 3 节「实现与演进」**

按「为什么需要下一步」组织：

- 朴素做法：直接写 `if/else`。
- 改进一：用算术选择（三元运算符、`fminf`/`fmaxf`）替代短分支 —— 编译器通常把它编译为谓词或 `SEL` 指令，无控制流分歧。
- 改进二：用 warp 级原语跨 lane 协作，把分支变成数据交换：`__ballot_sync`（投票）、`__shfl_sync`（lane 间传值）、`__any_sync` / `__all_sync`。给出一个用 `__ballot_sync` + `__shfl_sync` 消除分支的典型例子（如按条件累加到不同分量）。
- 改进三：重排数据布局，使同一 warp 内的 lane 天然处在同一路径上（如按类型分桶后再处理）。
- 演进：Volta 起引入**独立线程调度**（Independent Thread Scheduling），每个线程有自己的 PC，允许 warp 内线程在细粒度上推进不同位置。它提高了前进保证（forward progress），也意味着 `__syncwarp()` 从可选变成了必需。

- [ ] **Step 6: 写第 4 节「踩坑与失败模式」**

至少 3 条，每条「现象 → 成因 → 规避」：

1. **以为发散只来自 `if/else`**。现象：代码里没有分支语句，性能却随数据变化。成因：循环的**边界不齐**同样导致发散——若某个 warp 内部分 lane 的循环次数多于其他 lane，多出的那几轮只有部分 lane 活跃。规避：让数据量对齐 warp 边界，或在循环内用统一的迭代次数配合掩码。
2. **`__syncwarp` 语义误解**。现象：在 Volta 及以后的卡上，依赖 warp 内隐式同步的代码随机出错。成因：Volta 之前 warp 内锁步执行，`__syncwarp` 可有可无；Volta 起线程可独立推进，必须显式同步。规避：任何依赖 warp 内 lane 间数据可见性的代码，在使用 `__shfl_sync` 等原语后都要有明确的同步点；warp 级原语应使用带 `_sync` 后缀的版本并传入正确的掩码。
3. **用谓词改写长分支**。现象：把一个大 `if/else` 改成三元运算符后反而更慢。成因：谓词执行要求两条路径的指令都发射，长分支下计算量翻倍，而节省的分支开销远不足以抵消。规避：只对**极短**的分支用谓词；长分支保留真实控制流，改从数据布局上减少发散。

- [ ] **Step 7: 写第 5、6 节与参考**

- 第 5 节可整体省略。
- 第 6 节：回顾有效并行度模型（`32 / N`）与三种消除发散的手段，指向 `related` 中的 `03`。
- 参考：CUDA C++ Programming Guide 的「SIMT Architecture」章节；Volta 架构白皮书中关于独立线程调度的部分。

- [ ] **Step 8: 验证**

Run:
```bash
f=20-gpu-architecture/execution-model/02-simt-and-divergence.md
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
echo "--- 关键内容 ---"
grep -q '32 / N' "$f" && echo "OK 含有效并行度模型" || echo "MISS 并行度模型"
grep -q 'BSSY' "$f" && echo "OK 含 SASS 指令" || echo "MISS BSSY"
grep -q '__syncwarp' "$f" && echo "OK 含 syncwarp" || echo "MISS syncwarp"
echo "--- 三段式 ---"
grep -c '成因' "$f"
```
Expected: J 个字段全部 `OK`；三个关键项 `OK`；`成因` ≥ 3。

- [ ] **Step 9: 提交**

```bash
git add 20-gpu-architecture/execution-model/02-simt-and-divergence.md
git commit -m "docs: 添加 SIMT 执行模型与分支发散文档"
```

---

### Task 11: `03-occupancy-and-latency-hiding.md`

**Files:**
- Create: `20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md`

- [ ] **Step 1: 写 frontmatter 与结论**

```markdown
---
title: 占用率与延迟隐藏
slug: occupancy-and-latency-hiding
domain: 20-gpu-architecture
topic: execution-model
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/execution-model/01-sm-internals
tags: [occupancy, latency-hiding, little-law, ilp, tlp]
related:
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
code:
  - code/03_occupancy_probe.cu
status: stable
updated: 2026-09-14
---

# 占用率与延迟隐藏

> GPU 开那么多线程，主要目的不是并行计算，而是**用并发掩盖延迟**——理解了这一点，才知道占用率该追到多少、什么时候不该追。
```

**注意**：`prerequisites` 只列 `01-sm-internals`，`memory-system/01` 放在 `related` 中。这样避免与 `memory-system/01` 形成环（后者也将 `01-sm-internals` 列为前置）。

- [ ] **Step 2: 写第 0 节「背景与动机」**

- 对照场景：一个访存密集的 kernel，把线程数加倍后耗时几乎不变；再继续加倍反而变慢。
- 问题：既然更多线程能掩盖延迟，为什么收益会饱和甚至反转？

- [ ] **Step 3: 写第 1 节「原理」**

必须包含**两处多步推导**：

**推导一 —— 延迟隐藏需要多少并发（Little's Law）**

1. 系统稳定时：`在途请求数 = 到达率 × 平均驻留时间`。
2. 映射到 GPU：要让访存单元不空转，必须有足够多的独立访存请求同时在途。设访存延迟为 `L` 个周期，每条 warp 每 `I` 个周期能发起一次访存，则需要同时在途的 warp 数约为：
   ```
   所需 warp 数 ≈ L / I
   ```
3. 算例：`L = 600` cycles（典型的全局访存延迟），某 kernel 每 4 个周期可以让一个 warp 发起一次访存，则需 `600 / 4 = 150` 个 warp 才能完全掩盖。而每 SM 最多常驻 64 个 warp —— **结论：单靠线程级并行掩盖不了全局访存延迟**，必须配合指令级并行（每个线程同时发起多个独立访存）。

**推导二 —— 占用率的四项约束**

4. 占用率 = 实际常驻 warp 数 ÷ 每 SM 最大常驻 warp 数。
5. 实际常驻 warp 数由四个约束**取最小**决定：

   | 约束 | 上限 |
   |---|---|
   | 寄存器 | `floor(寄存器文件 ÷ (每线程寄存器数 × 32))` |
   | 共享内存 | `floor(每 SM 共享内存 ÷ 每 block 共享内存)` × `每 block 的 warp 数` |
   | block 数 | `每 SM 最大常驻 block 数` × `每 block 的 warp 数` |
   | 线程/warp 数 | `每 SM 最大常驻线程数 ÷ 32` |

6. 给出一个算例，四个约束分别算出 64、48、64、64，则实际取 48。

- [ ] **Step 4: 写第 2 节「执行映射」**

必须指明**至少 2 个具体组件名**并给出**至少 1 个数值量级**：

- 组件：warp scheduler、寄存器文件、LSU、指令流水线。
- 说明 warp 切换的实际机制：当一条 warp 的指令需要等待数据（如等全局访存返回），调度器把它标记为 stall，转而发射其他就绪 warp 的指令。切换本身**不保存上下文**，因为每个线程的寄存器是静态分配的——这正是 GPU 能廉价切换 warp 的原因，也是寄存器文件必须那么大的原因。
- 说明 ILP 与 TLP 的**互相替代**关系：ILP 让单个线程内多条无依赖指令同时在途，TLP 靠更多 warp 填缝。二者都消耗同一个寄存器文件预算，因此存在此消彼长。
- 给出数值量级：全局访存延迟约 400~800 cycles、共享内存访问约 20~30 cycles、L1 命中约 30 cycles，并说明为什么只有长延迟（全局访存）才值得用占用率去掩盖。

- [ ] **Step 5: 写第 3 节「实现与演进」**

- 朴素做法：凭经验设 block 大小。
- 改进一：用 API 精确计算理论占用率 —— `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 返回给定 kernel 与 block 大小下的最大常驻 block 数，乘以 block 内 warp 数即为常驻 warp 数。给出调用示例（与 `code/03_occupancy_probe.cu` 中的用法一致）。
- 改进二：用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 告诉编译器「每 SM 至少要放下几个 block」，编译器据此反推寄存器上限。**注意这是上限保护，不是优化目标**（见第 4 节）。
- 改进三：用 `-Xptxas -v` 查看编译器报告的寄存器用量与 spill 情况，作为调参依据。
- 演进：从「追高占用率」到「追求足够的延迟隐藏」——现代调优实践中，占用率达到能覆盖延迟的水平即可，继续提高往往只是在浪费寄存器预算。

- [ ] **Step 6: 写第 4 节「踩坑与失败模式」**

至少 3 条，每条「现象 → 成因 → 规避」：

1. **占用率不是越高越好**。现象：把占用率从 50% 提到 100%，性能反而下降。成因：提高占用率的手段通常是压低每线程寄存器数，这会牺牲 ILP 并可能导致寄存器溢出到 local memory；溢出后的访存代价远超多几个 warp 带来的收益。规避：把占用率当作**约束**而非**目标**——先确认瓶颈是延迟还是带宽，只有延迟受限时才需要提高占用率。
2. **把 occupancy 当唯一指标**。现象：两个 kernel 占用率相同，性能差数倍。成因：占用率只描述「有多少 warp 可用」，不描述「访存是否合并」「是否 bank 冲突」「指令是否依赖」。一个占用率满格但访存错位的 kernel 会远慢于占用率一半但访存完美的 kernel。规避：先用 profiling 确认瓶颈类型（计算受限 / 带宽受限 / 延迟受限），再决定是否调占用率。
3. **在带宽受限的 kernel 上追占用率**。现象：纯拷贝类 kernel 提高占用率后性能纹丝不动。成因：此时瓶颈是 HBM 带宽，不是延迟。硬件带宽已被打满，再多 warp 也只能排队等内存。规避：用 roofline 判断算子性质；带宽受限的算子的优化方向是**减少访存量**（融合、降低精度、复用），不是增加并发。

- [ ] **Step 7: 写第 5、6 节与参考**

- 第 5 节可整体省略。
- 第 6 节：回顾 Little's Law 推导与四项约束取最小，强调「占用率是约束不是目标」，指向 `related` 中的两篇。
- 参考：CUDA C++ Programming Guide 的「Maximize Occupancy」章节；CUDA Best Practices Guide。

- [ ] **Step 8: 验证**

Run:
```bash
f=20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
echo "--- 关键内容 ---"
grep -q '所需 warp 数' "$f" && echo "OK 含 Little's Law 推导" || echo "MISS 推导"
grep -q 'cudaOccupancyMaxActiveBlocksPerMultiprocessor' "$f" && echo "OK 含 occupancy API" || echo "MISS API"
grep -q '__launch_bounds__' "$f" && echo "OK 含 launch_bounds" || echo "MISS launch_bounds"
echo "--- 三段式 ---"
grep -c '成因' "$f"
```
Expected: J 个字段全部 `OK`；三个关键项 `OK`；`成因` ≥ 3。

- [ ] **Step 9: 提交**

```bash
git add 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md
git commit -m "docs: 添加占用率与延迟隐藏文档"
```

---

### Task 12: execution-model 的代码与 Makefile

**Files:**
- Create: `20-gpu-architecture/execution-model/code/01_sm_probe.cu`
- Create: `20-gpu-architecture/execution-model/code/02_divergence_demo.cu`
- Create: `20-gpu-architecture/execution-model/code/03_occupancy_probe.cu`
- Create: `20-gpu-architecture/execution-model/code/Makefile`

- [ ] **Step 1: 创建目录并写 `01_sm_probe.cu`**

```bash
mkdir -p 20-gpu-architecture/execution-model/code
```

完整源码：

```cuda
// 01_sm_probe.cu —— 查询 SM 规格与寄存器预算
// 对应文档：01-sm-internals.md
//
// 观测要点：
//   1. 每 SM 的寄存器文件大小与最大常驻 warp 数
//   2. 寄存器文件是「预算」而非「缓存」：每线程占用越多，
//      可常驻 warp 数越少，占用率随之下降
//
// 构建：make 01_sm_probe
// 运行：./01_sm_probe

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  std::printf("设备: %s (compute capability %d.%d)\n", p.name, p.major,
              p.minor);

  const int regs_per_sm    = p.regsPerMultiprocessor;  // 32-bit 寄存器个数
  const int threads_per_sm = p.maxThreadsPerMultiProcessor;
  const int warps_per_sm   = threads_per_sm / 32;

  std::printf("\n--- SM 规格 ---\n");
  std::printf("SM 数量                : %d\n", p.multiProcessorCount);
  std::printf("每 SM 寄存器文件       : %d 个 32-bit 寄存器 (%d KB)\n",
              regs_per_sm, regs_per_sm * 4 / 1024);
  std::printf("每 SM 最大常驻线程     : %d\n", threads_per_sm);
  std::printf("每 SM 最大常驻 warp    : %d\n", warps_per_sm);
  std::printf("每 SM 共享内存         : %zu KB\n",
              p.sharedMemPerMultiprocessor / 1024);
  std::printf("每 SM 最大常驻 block   : %d\n", p.maxBlocksPerMultiProcessor);

  // 单个 warp 占用的寄存器数 = 每线程寄存器数 x 32。
  // 寄存器文件能容纳的 warp 数 = regs_per_sm / (每线程寄存器数 x 32)。
  // 该值一旦小于架构上限 warps_per_sm，寄存器就成为占用率的限制项。
  std::printf("\n--- 寄存器预算（warp = 32 线程）---\n");
  std::printf("%-16s %-16s %-18s %s\n", "每线程寄存器", "单 warp 占用",
              "寄存器可容 warp", "是否成为瓶颈");

  for (int regs : {16, 24, 32, 40, 64, 128, 255}) {
    const int per_warp = regs * 32;
    const int capacity = regs_per_sm / per_warp;
    std::printf("%-16d %-16d %-18d %s\n", regs, per_warp, capacity,
                capacity < warps_per_sm ? "是" : "否");
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("寄存器文件是预算而非缓存：它必须为每 SM 上全部常驻线程静态预留，\n");
  std::printf("无法像 cache 那样按需换入换出。每线程多用一档寄存器，\n");
  std::printf("可常驻 warp 数就下降一档，占用率随之下降。\n");
  std::printf("当「寄存器可容 warp」小于 %d 时，降低每线程寄存器数\n",
              warps_per_sm);
  std::printf("（而非增加）才是提高占用率的正确方向。\n");

  return 0;
}
```

- [ ] **Step 2: 写 `02_divergence_demo.cu`**

完整源码：

```cuda
// 02_divergence_demo.cu —— 分支发散对有效并行度的影响
// 对应文档：02-simt-and-divergence.md
//
// 观测要点：
//   1. active = 0 或 32 时全 warp 走同一条路径，最快
//   2. active 在 1..31 之间时 warp 分裂为 2 条路径，硬件逐条串行执行，
//      耗时接近「重路径 + 轻路径」之和，而非两者中的较大者
//   3. branchless 版本必须把两条路径都算一遍，故耗时恒定且不低于 divergent
//
// 构建：make 02_divergence_demo
// 运行：./02_divergence_demo

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads = 256;
constexpr int kIters   = 32;   // 路径 A 的循环次数，用来放大发散代价
constexpr int kReps    = 20;

// 路径 A：较重的计算。刻意做得足够长，使编译器不会把
// if/else if-convert 成谓词执行——否则发散无法被观测到。
__device__ __forceinline__ float path_heavy(float v) {
  float r = v;
#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    r = fmaf(r, 1.000001f, 1e-6f);
  }
  return r;
}

// 路径 B：轻计算
__device__ __forceinline__ float path_light(float v) { return v + 1.0f; }

// kBranchless = false：真实的 if/else 控制流，会产生分支发散
// kBranchless = true ：两条路径都求值，用三元运算符取结果，无控制流分歧
//
// 控制发散的方式：warp 内前 `active` 个 lane 走路径 A，其余走路径 B。
//   active = 0 或 32 → 全 warp 同路径，路径数 1
//   active = 16      → warp 分裂为两条路径
template <bool kBranchless>
__global__ void divergence_kernel(const float* __restrict__ in,
                                  float* __restrict__ out, int n, int active) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float v    = in[i];
  const int   lane = threadIdx.x & 31;  // warp 内 lane 号

  if (kBranchless) {
    const float heavy = path_heavy(v);
    const float light = path_light(v);
    out[i]            = (lane < active) ? heavy : light;
  } else {
    float r;
    if (lane < active) {
      r = path_heavy(v);
    } else {
      r = path_light(v);
    }
    out[i] = r;
  }
}

template <bool kBranchless>
float time_variant(const float* d_in, float* d_out, int n, int active,
                   int blocks) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  // 预热
  divergence_kernel<kBranchless><<<blocks, kThreads>>>(d_in, d_out, n, active);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    divergence_kernel<kBranchless><<<blocks, kThreads>>>(d_in, d_out, n,
                                                         active);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / kReps;
}

}  // namespace

int main() {
  constexpr int n      = 1 << 22;  // 4M 元素
  constexpr int blocks = (n + kThreads - 1) / kThreads;

  float* h_in = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
  for (int i = 0; i < n; ++i) {
    h_in[i] = static_cast<float>(i % 1000) * 0.001f;
  }

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  std::printf("元素数 %d，block 数 %d，每 block %d 线程\n\n", n, blocks,
              kThreads);
  std::printf("%-8s %-10s %-16s %-16s %s\n", "active", "路径数",
              "divergent(ms)", "branchless(ms)", "div/branchless");

  float t_light   = 0.0f;  // active = 0：全 warp 走轻路径
  float t_heavy   = 0.0f;  // active = 32：全 warp 走重路径

  for (int active : {0, 4, 8, 16, 24, 32}) {
    const float t_div =
        time_variant<false>(d_in, d_out, n, active, blocks);
    const float t_bl =
        time_variant<true>(d_in, d_out, n, active, blocks);

    if (active == 0)  t_light = t_div;
    if (active == 32) t_heavy = t_div;

    const char* paths = (active == 0 || active == 32) ? "1" : "2";
    std::printf("%-8d %-10s %-16.4f %-16.4f %.2fx\n", active, paths, t_div,
                t_bl, t_div / t_bl);
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("1. active = 0 或 32 时全 warp 走同一条路径，路径数为 1，divergent 最快。\n");
  std::printf("   纯轻路径 %.4f ms，纯重路径 %.4f ms。\n", t_light, t_heavy);
  std::printf("2. active 在 1..31 之间时 warp 分裂为两条路径，硬件逐条串行执行，\n");
  std::printf("   故 divergent 的耗时上升到接近「重路径 + 轻路径」之和 %.4f ms，\n",
              t_heavy + t_light);
  std::printf("   而不是两者中的较大者。这就是有效并行度 32/N 的实际含义。\n");
  std::printf("3. branchless 版本无论 active 取何值，都必须把两条路径都算一遍，\n");
  std::printf("   因此耗时基本恒定，且不低于 divergent —— 即便在 active = 0 时也是如此。\n");
  std::printf("   这说明「用算术选择消除分支」对**长**分支并不免费：它把控制流分歧\n");
  std::printf("   换成了计算冗余。无分支写法的价值在短分支（省去 BSSY/BRA/BSYNC 的开销），\n");
  std::printf("   不在长分支。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  return 0;
}
```

- [ ] **Step 3: 写 `03_occupancy_probe.cu`**

完整源码：

```cuda
// 03_occupancy_probe.cu —— 占用率与延迟隐藏
// 对应文档：03-occupancy-and-latency-hiding.md
//
// 观测要点：
//   1. 每 block 的共享内存用量是占用率的限制项之一：用量越大，
//      每 SM 能放下的 block 越少，占用率越低
//   2. cudaOccupancyMaxActiveBlocksPerMultiprocessor 可精确算出
//      给定 kernel、block 尺寸与共享内存用量下的常驻 block 数
//   3. 占用率下降不必然带来吞吐下降 —— 若瓶颈是访存带宽而非延迟，
//      减少 warp 数并不会让带宽变小
//
// 为什么用共享内存而不是 __launch_bounds__ 来调节占用率：
// 本 kernel 只需约 10~16 个寄存器/线程，寄存器从来不是限制项，
// __launch_bounds__ 的 minBlocks 参数根本不会生效，四行输出会完全相同。
// 共享内存用量则是确定生效的限制项，且不依赖编译器行为。
// __launch_bounds__ 的用法见文档第 3 节，此处不演示。
//
// 构建：make 03_occupancy_probe
// 运行：./03_occupancy_probe

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads = 256;
constexpr int kReps    = 20;

// 流式累加的访存受限 kernel。动态共享内存的用量由启动参数给出，
// 它不参与计算，只用来人为压低每 SM 的常驻 block 数。
__global__ void accumulate_kernel(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
  extern __shared__ float tile[];

  const int stride = gridDim.x * blockDim.x;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  for (int i = tid; i < n; i += stride) {
    acc += in[i];
  }

  // 让共享内存真正参与计算，避免整块共享内存被编译器判定为无用而影响
  // 占用率计算。tile 至少要有 blockDim.x 个 float。
  tile[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = tile[0];
}

struct Result {
  size_t smem_bytes;
  int    blocks_per_sm;
  int    regs_per_thread;
  float  occupancy;  // 0..1
  float  ms;
  float  gbps;
};

Result probe(size_t smem_bytes, const float* d_in, float* d_out, int n,
             int blocks, int threads_per_sm) {
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, accumulate_kernel));

  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, accumulate_kernel, kThreads, smem_bytes));

  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  // 预热
  accumulate_kernel<<<blocks, kThreads, smem_bytes>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    accumulate_kernel<<<blocks, kThreads, smem_bytes>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  ms /= kReps;
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));

  const float resident = static_cast<float>(blocks_per_sm) * kThreads;
  return Result{smem_bytes, blocks_per_sm, attr.numRegs,
                resident / static_cast<float>(threads_per_sm), ms,
                static_cast<float>(n) * sizeof(float) / (ms * 1e-3f) / 1e9f};
}

}  // namespace

int main() {
  constexpr int n      = 1 << 24;  // 16M 元素 = 64 MB
  constexpr int blocks = 4096;

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("设备: %s\n", p.name);
  std::printf("每 SM 最大常驻线程 %d，每 SM 共享内存 %zu KB，每 block %d 线程\n\n",
              p.maxThreadsPerMultiProcessor, p.sharedMemPerMultiprocessor / 1024,
              kThreads);

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));

  // 逐个取值探测。最小取 1 KB —— 内核要用 tile[0..kThreads-1]，
  // 即 kThreads × 4 = 1024 字节，低于此值会越界。
  const size_t smems[] = {1u << 10, 8u << 10, 16u << 10, 32u << 10, 48u << 10};

  const Result first = probe(smems[0], d_in, d_out, n, blocks,
                             p.maxThreadsPerMultiProcessor);

  std::printf("寄存器用量 %d/线程（不是本 kernel 的限制项）\n", first.regs_per_thread);
  std::printf("常驻 block 数 = min(线程数限制 %d, 共享内存限制, block 数限制 %d)\n\n",
              p.maxThreadsPerMultiProcessor / kThreads, p.maxBlocksPerMultiProcessor);

  std::printf("%-14s %-16s %-12s %-12s %s\n", "共享内存/block",
              "每 SM 常驻 block", "占用率", "耗时(ms)", "带宽(GB/s)");
  std::printf("%-14zu %-16d %-12.1f %-12.4f %.1f\n", first.smem_bytes / 1024,
              first.blocks_per_sm, first.occupancy * 100.0f, first.ms,
              first.gbps);

  for (size_t i = 1; i < sizeof(smems) / sizeof(smems[0]); ++i) {
    const Result r = probe(smems[i], d_in, d_out, n, blocks,
                           p.maxThreadsPerMultiProcessor);
    std::printf("%-14zu %-16d %-12.1f %-12.4f %.1f\n", r.smem_bytes / 1024,
                r.blocks_per_sm, r.occupancy * 100.0f, r.ms, r.gbps);
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("共享内存用量从 1 KB 加到 48 KB，「每 SM 常驻 block」与「占用率」\n");
  std::printf("逐级下降。注意寄存器那一项自始至终不变 —— 它不是本 kernel 的限制项，\n");
  std::printf("真正起作用的是共享内存。四项约束取最小，谁先卡住谁决定占用率。\n\n");
  std::printf("再对比「占用率」与「带宽」两列：\n");
  std::printf("  · 若带宽基本不随占用率下降而变，说明该 kernel 的瓶颈是访存带宽\n");
  std::printf("    而非延迟 —— 硬件带宽已被打满，再多 warp 也只能排队等内存。\n");
  std::printf("  · 若带宽随占用率同步下降，说明此时仍是延迟受限，warp 数不足以\n");
  std::printf("    掩盖访存延迟。\n");
  std::printf("两种情形都指向同一结论：占用率是**约束**而非**优化目标**。\n");
  std::printf("调优时应先用 profiling 判断瓶颈类型，再决定是否值得为占用率\n");
  std::printf("牺牲寄存器或共享内存。具体数值随 GPU 型号与架构而异。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
```

- [ ] **Step 4: 写 `Makefile`**

```makefile
# execution-model 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0，GPU 算力 >= sm_80
#
# 用法：
#   make              构建全部
#   make 01_sm_probe  构建单个目标
#   make run          构建并依次运行
#   make clean        清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
LDFLAGS   ?=

TARGETS := 01_sm_probe 02_divergence_demo 03_occupancy_probe

.PHONY: all clean run
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $<

# 查看每线程寄存器用量与 spill 情况
.PHONY: regs
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(word 1,$(TARGETS)).cu -o /dev/null

run: all
	@for t in $(TARGETS); do \
		echo "=== $$t ==="; \
		./$$t || exit 1; \
		echo; \
	done

clean:
	rm -f $(TARGETS)
```

- [ ] **Step 5: 静态审查**

本机无 `nvcc`，**不编译**。逐项核对：

```bash
cd 20-gpu-architecture/execution-model/code && echo "--- 文件存在 ---" && for f in 01_sm_probe.cu 02_divergence_demo.cu 03_occupancy_probe.cu Makefile; do test -f "$f" && echo "OK   $f" || echo "MISS $f"; done && echo "--- 头文件 ---" && grep -h '^#include' *.cu | sort -u && echo "--- 每个 .cu 应有 CUDA_CHECK ---" && grep -c 'define CUDA_CHECK' *.cu && echo "--- 三对编号对应 ---" && for n in 01 02 03; do ls ${n}_*.cu >/dev/null 2>&1 && echo "OK   code/$n" || echo "MISS code/$n"; done
```

Expected: 四个 `OK 文件存在`；三个 `.cu` 各有 1 处 `define CUDA_CHECK`；三个编号 `OK`。

人工审查清单（逐条核对并在提交信息中确认）：

1. 每个 `.cu` 都包含 `#include <cuda_runtime.h>`，且错误检查宏在每次 CUDA API 调用后使用。
2. 每个 `<<<>>>` 启动序列后至少检查一次 `CUDA_CHECK_LAST()`（计时循环内不必逐次检查，但预热后与循环结束后各须检查一次）。
3. 所有 `cudaMalloc` 都有配对的 `cudaFree`，`cudaMallocHost` 有配对的 `cudaFreeHost`。
4. `03_occupancy_probe.cu` 中 `__launch_bounds__` 的第二个参数最大取 6，`6 × 256 = 1536` 不超过常见架构的每 SM 最大线程数。
5. `02_divergence_demo.cu` 的 `path_heavy` 循环次数为 32 且带 `#pragma unroll` —— 循环太短会被编译器 if-convert 成谓词执行，导致观测不到发散。
6. `Makefile` 使用 Tab 缩进（不是空格），否则 `make` 会报 "missing separator"。

- [ ] **Step 6: 提交**

```bash
git add 20-gpu-architecture/execution-model/code/
git commit -m "feat: 添加 execution-model 示例代码与 Makefile"
```

---

## Chunk 4: `20-gpu-architecture/memory-system/`（7 个文件）

结构同 Chunk 3。编号对应：

```
01-hierarchy-overview.md            ←→  code/01_bandwidth_probe.cu
02-coalescing-and-cache-line.md     ←→  code/02_coalescing_demo.cu
03-shared-memory-bank-conflict.md   ←→  code/03_bank_conflict_demo.cu
```

### Task 13: `01-hierarchy-overview.md`

**Files:**
- Create: `20-gpu-architecture/memory-system/01-hierarchy-overview.md`

- [ ] **Step 1: 创建目录并写 frontmatter**

```bash
mkdir -p 20-gpu-architecture/memory-system
```

```markdown
---
title: GPU 存储层次总览
slug: memory-hierarchy-overview
domain: 20-gpu-architecture
topic: memory-system
level: 2
difficulty: beginner
prerequisites:
  - 20-gpu-architecture/execution-model/01-sm-internals
tags: [memory-hierarchy, hbm, l2, shared-memory, bandwidth, latency]
related:
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
code:
  - code/01_bandwidth_probe.cu
status: stable
updated: 2026-09-14
---

# GPU 存储层次总览

> 存储必须分级，因为容量、带宽、延迟三者不可兼得。GPU 与 CPU 的分歧在于：CPU 靠大缓存降低平均延迟，GPU 靠大量并发线程把延迟藏起来。
```

- [ ] **Step 2: 写第 0 节「背景与动机」**

- 对照场景：一段在 CPU 上跑得飞快的代码，原样搬到 GPU 上却慢得多 —— 因为 GPU 的单次访存延迟比 CPU 还长，它靠的是并发而非低延迟。
- 问题：既然寄存器最快，为什么不全部用寄存器？既然共享内存比全局内存快两个数量级，为什么不全部用共享内存？

- [ ] **Step 3: 写第 1 节「原理」**

必须包含**多步推导**：

1. 从物理约束起步：存储单元的访问速度与它的物理尺寸、距离成反比。容量越大，寻址电路越长、信号传播越远，延迟越高。
2. 因此不可能存在「大容量 + 低延迟 + 高带宽」的单一存储。唯一出路是分级：小容量快存储靠近计算单元，大容量慢存储在外围，用**局部性**把热数据保持在快存储里。
3. 推导带宽与延迟的独立性和相互关系：
   ```
   平均访问时间 = 命中率 × 快存储延迟 + (1 − 命中率) × 慢存储延迟
   有效带宽     = 传输字节数 ÷ 时间
   ```
   两者独立：高带宽不代表单次访问快（如 HBM 靠大量并行通道堆带宽，单次延迟仍很长）。
4. 给出 GPU 各级存储的量级对比表（数值须标明是典型值，并注明会随架构变化）：

   | 层级 | 典型容量 | 典型延迟 | 谁可见 | 谁管理 |
   |---|---|---|---|---|
   | 寄存器 | 每 SM 256 KB | 1 cycle | 单线程 | 编译器 |
   | 共享内存 / L1 | 每 SM 128~256 KB | 20~30 cycles | 单 block | 程序员 / 硬件 |
   | L2 | 数十 MB | ~200 cycles | 全 GPU | 硬件 |
   | 显存 HBM | 数十 GB | 400~800 cycles | 全 GPU | 硬件 |
   | 主机内存 | — | 微秒级 | — | — |

5. 推导结论：**共享内存是"软件管理的缓存"** —— 它和 L1 使用同一块物理存储，区别在于 L1 由硬件自动替换，共享内存由程序员显式搬运。这个可控性是 GPU 编程性能优化的主要空间。

- [ ] **Step 4: 写第 2 节「执行映射」**

必须指明**至少 2 个具体组件名**并给出**至少 1 个数值量级**：

- 组件：register file、shared memory / L1（共享同一物理存储）、L2 cache、HBM 控制器、LSU。
- 说明数据路径：全局访存请求从 LSU 发出，先查 L1，未命中查 L2，再未命中才到 HBM。命中层级越靠后，延迟越长。
- 说明 **local memory 的物理归属**：它名字叫 "local"，但**不在片上**，而是显存中按线程私有寻址的一块区域。寄存器溢出时数据落到这里，每次访问都要走 L1/L2/HBM，这可能比重新计算还慢。
- 给出数值量级：L1 命中约 30 cycles、L2 命中约 200 cycles、HBM 约 400~800 cycles —— 相邻层级差 3~5 倍，跨级差数十倍。

- [ ] **Step 5: 写第 3 节「实现与演进」**

按「为什么需要下一步」组织：

- 早期：GPU 无缓存，共享内存是唯一可编程的片上存储。
- Fermi：引入统一的 L1/共享内存，容量可配置切分。
- Kepler/Maxwell：L1 与共享内存的切分比例可调。
- Pascal：引入 HBM2，带宽大幅提升。
- Volta：共享内存与 L1 合计达 128 KB，引入独立线程调度。
- Ampere/Hopper：HBM2e/HBM3，L2 容量增长，Hopper 引入分布式共享内存（DSMEM），SM 之间可直接访问对方共享内存，把"共享"的范围从 block 扩大到 cluster。
- Blackwell：HBM3e，进一步扩大的 L2 与更高的互联带宽。

每代给出可查证的容量与带宽数字。

- [ ] **Step 6: 写第 4 节「踩坑与失败模式」**

至少 3 条，每条「现象 → 成因 → 规避」：

1. **把共享内存当通用缓存**。现象：把所有数据都搬进共享内存，性能反而下降。成因：搬运本身要花全局访存时间和同步开销；若数据没有复用（每个元素只读一次），搬运纯属浪费，且共享内存容量有限会压低占用率。规避：只有当数据的**复用次数足够高**（同一块数据被多个线程读多次）时，搬运才有收益。
2. **忽略 L2 容量导致命中率崩塌**。现象：工作集稍微增大一点，性能断崖式下跌。成因：工作集超过 L2 容量后，原本每次访问都命中的模式变成全部落到 HBM，延迟跳一个数量级。规避：关注工作集大小，用分块（tiling）把工作集控制在 L2 能容纳的范围内。
3. **混淆 local memory 与 global memory**。现象：看到汇编里有 `local` 字样，以为是片上存储，忽略了它的开销。成因：local memory 物理上位于片外显存，只是按线程私有方式寻址。规避：用 `-Xptxas -v` 检查 spill 情况；避免在 kernel 中定义大数组（会被分配到 local memory）。

- [ ] **Step 7: 写第 5、6 节与参考**

- 第 5 节可整体省略。
- 第 6 节：回顾分级存储的物理必然性与「共享内存是软件管理的缓存」，指向 `related` 中的 `02`。
- 参考：CUDA C++ Programming Guide 的「Memory Hierarchy」章节；对应架构白皮书的存储规格。

- [ ] **Step 8: 验证**

Run:
```bash
f=20-gpu-architecture/memory-system/01-hierarchy-overview.md
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
grep -q '软件管理的缓存' "$f" && echo "OK 含核心论断" || echo "MISS 核心论断"
grep -q 'local memory' "$f" && echo "OK 含 local memory" || echo "MISS local memory"
grep -c '成因' "$f"
```
Expected: J 个字段全部 `OK`；两个关键项 `OK`；`成因` ≥ 3。

- [ ] **Step 9: 提交**

```bash
git add 20-gpu-architecture/memory-system/01-hierarchy-overview.md
git commit -m "docs: 添加 GPU 存储层次总览文档"
```

---

### Task 14: `02-coalescing-and-cache-line.md`

**Files:**
- Create: `20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md`

- [ ] **Step 1: 写 frontmatter 与结论**

```markdown
---
title: 访存合并与 cache line
slug: coalescing-and-cache-line
domain: 20-gpu-architecture
topic: memory-system
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
tags: [coalescing, cache-line, sector, transaction, soa, aos]
related:
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
code:
  - code/02_coalescing_demo.cu
status: stable
updated: 2026-09-14
---

# 访存合并与 cache line

> 同样的计算量，只改访存顺序就能差出数十倍。原因不在算法，而在一个 warp 的 32 次访问落在了几条 cache line 上。
```

- [ ] **Step 2: 写第 0 节「背景与动机」**

- 对照场景：两个 kernel 都对同一数组求和，元素总数相同，一个按顺序读、一个按固定间隔跳读，耗时差数十倍。
- 问题：为什么"地址连续"是性能的关键？硬件是怎么判断一次 warp 访存要发多少个内存事务的？

- [ ] **Step 3: 写第 1 节「原理」**

必须包含**多步骤事务数推导**：

1. 前提：一个 warp 的 32 个线程在同一时刻发起访存，硬件把这个 warp 的所有请求**合并处理**。
2. 情形 A —— 完全合并。32 个线程读取 32 个连续的 `float`（128 字节），全部落在同一条 128 字节的 cache line 内：
   ```
   事务数 = 1
   传输字节 = 128
   有用字节 = 32 × 4 = 128
   有效利用率 = 128 / 128 = 100%
   ```
3. 情形 B —— 跨步访问。相邻线程的地址间隔 `S` 个 `float`，则一个 warp 覆盖 `32 × S × 4` 字节，跨 `S` 条 cache line：
   ```
   事务数 = S
   传输字节 = S × 128
   有用字节 = 32 × 4 = 128
   有效利用率 = 128 / (S × 128) = 1 / S
   ```
4. 验算：`S = 1` → 利用率 100%；`S = 2` → 50%；`S = 32` → 3.125%。有效带宽按同样比例下降。
5. 推导出**有效带宽公式**：
   ```
   有效带宽 = 峰值带宽 × 有效利用率
   有效利用率 = 有用字节 ÷ 传输字节
   ```
6. 说明该模型的边界：若 warp 的访问跨越 cache line 边界（如起始地址未对齐），事务数会再多 1；若访问模式不规则（如按索引间接寻址），事务数需逐个判断。

- [ ] **Step 4: 写第 2 节「执行映射」**

必须指明**至少 2 个具体组件名**并给出**至少 1 个数值量级**：

- 组件：LSU、L1 cache、cache line（128 字节）、sector（32 字节）、L2。
- 说明两级粒度的区别：L1 以 **sector（32 字节）** 为单位向 L2 取数，L2 以 **cache line（128 字节）** 为单位向 HBM 取数。一个 cache line 含 4 个 sector。这解释了一个重要现象：**部分命中时只有需要的 sector 被取回**，因此略有不合并的代价可能小于「1/S」模型。
- 说明合并判定的实际规则：硬件按 warp 内所有活跃线程的地址所在的 cache line 集合决定事务数，与线程号顺序无关。
- 给出数值量级：128 字节 cache line、32 字节 sector、4 个 sector/line；并说明为什么 32 线程 × 4 字节恰好等于一条 cache line（这是硬件设计的对齐取值）。

- [ ] **Step 5: 写第 3 节「实现与演进」**

按「为什么需要下一步」组织：

- 朴素做法：按结构体数组（AoS）组织数据，`struct {float x, y, z;}` 的数组。读取所有 x 分量时，warp 的访问跨步 3 个 float，利用率降到 1/3。
- 改进一：改为数组结构体（SoA），把 x、y、z 分成三个独立数组。读取 x 分量时完全合并。**注意**：SoA 并非总是更优——若代码同时需要 x、y、z 三个分量，AoS 的一次访问就能拿到全部，SoA 反而要发三次访存。选择的依据是**实际访问模式**。
- 改进二：向量化访问。用 `float4` 让每个线程一次读取 16 字节，warp 一次覆盖 512 字节，事务数仍是 4 条 cache line 但指令数减少为 1/4，降低指令发射压力。
- 改进三：`__ldg` / `const __restrict__` 走只读数据缓存路径，减轻 L1 压力。
- 演进：Hopper 引入 TMA（Tensor Memory Accelerator），由专用硬件单元执行批量异步拷贝，访存模式由描述符声明而非由线程地址计算隐含。这从根本上改变了「合并」的实现方式——搬运算法的责任从程序员转移到了硬件。

- [ ] **Step 6: 写第 4 节「踩坑与失败模式」**

至少 3 条，每条「现象 → 成因 → 规避」：

1. **以为"地址连续"就等于"合并"**。现象：代码里数组下标连续递增，性能却很差。成因：合并取决于**同一 warp 内相邻 lane 的地址是否相邻**，不是取决于单个线程的地址序列。若索引写成 `in[i * S]`，每个线程自身是连续的（i 递增），但相邻 lane 相差 S，依然不合并。规避：检查的是 `in[threadIdx.x * S + base]` 这个表达式，而不是线程内部的循环。
2. **AoS 与 SoA 的选择教条化**。现象：把所有结构体都改成 SoA，某些 kernel 反而变慢。成因：SoA 优化的是"只取部分字段"的场景；若代码需要全部字段，AoS 一次访存即可拿全，SoA 要发多次。规避：按**实际访问的字段比例**决定；也可以混用（热字段抽出来单独成数组）。
3. **把只读缓存当万能加速**。现象：加上 `__ldg` 后性能没有变化，甚至下降。成因：只读缓存在数据确实只读且**有复用**时才有效；若本来就没有复用，或数据量远超缓存容量，它只是换了一条路径。规避：确认数据有复用性后再使用；用 profiling 观察 L1 命中率而非凭直觉添加修饰符。

- [ ] **Step 7: 写第 5、6 节与参考**

- 第 5 节可整体省略。
- 第 6 节：回顾事务数推导与有效利用率公式，强调「看的是 warp 内相邻 lane 的地址」，指向 `related` 中的 `03`。
- 参考：CUDA C++ Programming Guide 的「Device Memory Accesses」章节（含合并访问的图示）。

- [ ] **Step 8: 验证**

Run:
```bash
f=20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
grep -q '有效利用率' "$f" && echo "OK 含利用率公式" || echo "MISS 利用率公式"
grep -q 'sector' "$f" && echo "OK 含 sector" || echo "MISS sector"
grep -q 'AoS' "$f" && echo "OK 含 AoS/SoA" || echo "MISS AoS"
grep -c '成因' "$f"
```
Expected: J 个字段全部 `OK`；三个关键项 `OK`；`成因` ≥ 3。

- [ ] **Step 9: 提交**

```bash
git add 20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md
git commit -m "docs: 添加访存合并与 cache line 文档"
```

---

### Task 15: `03-shared-memory-bank-conflict.md`

**Files:**
- Create: `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md`

- [ ] **Step 1: 写 frontmatter 与结论**

```markdown
---
title: 共享内存 bank 冲突
slug: shared-memory-bank-conflict
domain: 20-gpu-architecture
topic: memory-system
level: 2
difficulty: advanced
prerequisites:
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
tags: [shared-memory, bank-conflict, padding, swizzle, broadcast]
related:
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/03_bank_conflict_demo.cu
status: stable
updated: 2026-09-14
---

# 共享内存 bank 冲突

> 共享内存比全局内存快近两个数量级，但用错了可以比全局内存还慢——32 个 lane 撞进同一个 bank 时，一次访问要串行 32 遍。
```

**注意**：`related` 中的 `40-kernels/gemm/02-shared-memory-tiling` 是**前向声明**，该文档尚未创建。spec §4 明确允许这种写法。

- [ ] **Step 2: 写第 0 节「背景与动机」**

- 对照场景：一个用共享内存做分块的矩阵乘法，把共享内存数组从 `[32][32]` 改成 `[32][33]`（多一列），性能提升数倍 —— 计算逻辑一行没改。
- 问题：多出的那一列明明不被使用，为什么能让程序变快？

- [ ] **Step 3: 写第 1 节「原理」**

必须包含**多步推导**：

1. 给出 bank 的划分模型。共享内存的地址空间按 4 字节为单位轮流分配到 32 个 bank：
   ```
   bank(addr) = (addr / 4) % 32
   ```
2. 一次 warp 访问的冲突判定规则：把 32 个 lane 访问的地址分别映射到 bank。
   - 若 32 个地址落在 **32 个不同的 bank** → 无冲突，一次完成。
   - 若多个地址落在**同一个 bank 的相同地址** → 硬件多播（broadcast），不额外收费，仍算无冲突。
   - 若多个地址落在**同一个 bank 的不同地址** → 冲突，按该 bank 上被请求的不同地址数串行化。
3. 推导典型冲突场景。设 `float tile[32][32]`，线程 `lane` 读取 `tile[lane][c]`（固定列 `c`，行随 lane 变化）：
   ```
   地址 = lane × 32 × 4 + c × 4  (字节)
   bank = (lane × 32 + c) % 32 = c
   ```
   32 个 lane 全部映射到 bank `c`，但地址各不相同 → **32 路冲突**，一次访问串行化为 32 次。
4. 推导 padding 的修复原理。把声明改为 `float tile[32][33]`：
   ```
   地址 = lane × 33 × 4 + c × 4
   bank = (lane × 33 + c) % 32 = (lane + c) % 32
   ```
   随 lane 递增而遍历全部 32 个 bank → **无冲突**。
5. 推导 padding 的代价：容量增加 `1/32`（`32×33` vs `32×32`），且行间不再对齐 cache line。**收益大于代价的前提是确实存在冲突**。

- [ ] **Step 4: 写第 2 节「执行映射」**

必须指明**至少 2 个具体组件名**并给出**至少 1 个数值量级**：

- 组件：shared memory（32 个 bank、每 bank 4 字节宽）、LSU、L1（与共享内存共用物理存储）。
- 说明冲突的实际代价：冲突发生时，LSU 把这次 warp 访问拆成多次，每次只服务一个地址，**耗时按冲突路数线性增长**。
- 说明 64 位访问的双字机制：当访问 `double` 或 `float2` 时，硬件把 32 个 bank 两两配对成 16 组，每个 lane 的 8 字节访问横跨两个 bank。这改变了冲突的判定方式——两个 lane 的 8 字节访问若落在同一对 bank 上，依然冲突。
- 给出数值量级：共享内存无冲突访问约 20~30 cycles；32 路冲突时相当于 32 倍；对比全局访存 400~800 cycles。**这也是为什么"共享内存冲突后比全局还慢"是可能的**：32 × 25 ≈ 800 cycles，与一次全局访存相当。

- [ ] **Step 5: 写第 3 节「实现与演进」**

按「为什么需要下一步」组织：

- 朴素做法：`float tile[32][32]`，按列访问时 32 路冲突。
- 改进一：**padding**，改为 `float tile[32][33]`。原理见第 1 节推导。代价是容量与对齐。
- 改进二：**swizzle**（重排）。用异或运算重映射地址，如 `index = i ^ (j & mask)`，在不增加容量的前提下打散 bank 映射。现代高性能 GEMM 库普遍采用，因为它不浪费存储且能同时优化向量化访问。
- 改进三：**改变数据布局以配合访问模式**，让 warp 内的访问天然落在不同 bank（如转置存储）。
- 演进：Hopper 引入**分布式共享内存（DSMEM）** —— cluster 内的 SM 可以直接访问彼此的共享内存。这把 bank 冲突问题扩展到了跨 SM 的层面，同时也让 tiling 策略有了新的可能（更大的有效 tile 不做全局访存）。

- [ ] **Step 6: 写第 4 节「踩坑与失败模式」**

至少 3 条，每条「现象 → 成因 → 规避」：

1. **padding 方向搞反**。现象：加了 padding，冲突不但没消失，反而换了个位置。成因：padding 必须加在**导致行距与 bank 数成倍数关系**的那个维度上。`[32][32]` 中行距 32 与 bank 数 32 相等是冲突根源，所以要在**列**上加 1（变成 `[32][33]`）；若加在行上变成 `[33][32]`，行距仍是 32，冲突依旧。规避：记住判据是「行距（以元素计）与 32 是否互质」——行距 33 与 32 互质，故无冲突。
2. **以为加 padding 总是有益**。现象：给一个本来没有冲突的 kernel 加 padding，性能下降。成因：padding 增加存储占用，可能把共享内存用量顶过某个档位，导致每 SM 的常驻 block 数下降；同时破坏了行对齐，可能让向量化访问失效。规避：只在 profiling 确认存在 bank 冲突时才加；加完后重新检查占用率。
3. **以为 64 位访问没有冲突**。现象：改用 `double` 或 `float2` 后，冲突"消失"了，但性能没有提升。成因：64 位访问并不是没有冲突，而是把 32 个 bank 两两配对成 16 组，冲突的判定粒度变粗了 —— 原来分散的访问可能在新粒度下重新撞在一起。规避：用 64 位访问时，按 16 组的模型重新推导 bank 映射，而不是沿用 32 组的结论。

- [ ] **Step 7: 写第 5、6 节与参考**

- 第 5 节可整体省略。
- 第 6 节：回顾 bank 寻址模型与 padding 推导，强调「行距与 32 互质」这一判据，指向 `related` 中的两篇。
- 参考：CUDA C++ Programming Guide 的「Shared Memory」章节；对应架构白皮书中关于 DSMEM 的部分。

- [ ] **Step 8: 验证**

Run:
```bash
f=20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md
for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
  grep -qF "$h" "$f" && echo "OK   $h" || echo "MISS $h"
done
for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
  grep -qE "^${k}:" "$f" && echo "OK   $k" || echo "MISS $k"
done
grep -q 'bank(addr)' "$f" && echo "OK 含 bank 寻址模型" || echo "MISS bank 模型"
grep -q '32 路冲突' "$f" && echo "OK 含冲突推导" || echo "MISS 冲突推导"
grep -q 'swizzle' "$f" && echo "OK 含 swizzle" || echo "MISS swizzle"
grep -c '成因' "$f"
```
Expected: J 个字段全部 `OK`；三个关键项 `OK`；`成因` ≥ 3。

- [ ] **Step 9: 提交**

```bash
git add 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md
git commit -m "docs: 添加共享内存 bank 冲突文档"
```

---

### Task 16: memory-system 的代码与 Makefile

**Files:**
- Create: `20-gpu-architecture/memory-system/code/01_bandwidth_probe.cu`
- Create: `20-gpu-architecture/memory-system/code/02_coalescing_demo.cu`
- Create: `20-gpu-architecture/memory-system/code/03_bank_conflict_demo.cu`
- Create: `20-gpu-architecture/memory-system/code/Makefile`

- [ ] **Step 1: 创建目录并写 `01_bandwidth_probe.cu`**

```bash
mkdir -p 20-gpu-architecture/memory-system/code
```

完整源码：

```cuda
// 01_bandwidth_probe.cu —— 各级存储带宽实测
// 对应文档：01-hierarchy-overview.md
//
// 观测要点：
//   1. 共享内存带宽远高于 HBM 带宽（典型差一个数量级）
//   2. 实测带宽通常只有规格峰值的 70%~90%
//   3. 带宽与延迟是两件事：带宽高不代表单次访问快
//
// 构建：make 01_bandwidth_probe
// 运行：./01_bandwidth_probe
//
// 说明：本程序测得的绝对值依 GPU 型号而异，请关注数量级与相对关系。

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads  = 256;
constexpr int kBlocks   = 1024;
constexpr int kReps     = 20;
constexpr int kSmemIter = 4096;  // 共享内存读取轮数，用来放大片内访问

// 全局内存拷贝：读一份写一份。带宽按「读 + 写」的总字节数计。
__global__ void copy_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = in[i];
  }
}

// 只读扫描：只读不写，测纯读带宽。
__global__ void read_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  const int stride = gridDim.x * blockDim.x;
  float acc        = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    acc += in[i];
  }
  if (acc == -1234567.0f) out[0] = acc;
}

// 共享内存读取：先把一块数据搬进共享内存，再反复读取。
// 读取模式 tile[(tid + k) & (bdim - 1)] 保证 warp 内相邻 lane 落在
// 相邻的 4 字节单元上，即映射到 32 个不同的 bank，无冲突。
__global__ void smem_read_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
  extern __shared__ float tile[];
  const int tid  = threadIdx.x;
  const int bdim = blockDim.x;

  // 装载：全局读 + 共享写，各一次
  for (int i = tid; i < bdim; i += bdim) {
    const int g = blockIdx.x * bdim + i;
    tile[i]     = (g < n) ? in[g] : 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll 8
  for (int k = 0; k < kSmemIter; ++k) {
    acc += tile[(tid + k) & (bdim - 1)];
  }
  __syncthreads();

  if (acc == -1234567.0f) out[0] = acc;
}

template <typename Kernel>
float time_kernel(Kernel k, int blocks, int threads, size_t smem, int n,
                  const float* d_in, float* d_out) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  k<<<blocks, threads, smem>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    k<<<blocks, threads, smem>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / kReps;
}

}  // namespace

int main() {
  const int n = 1 << 24;  // 16M 元素 = 64 MB

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));

  const double bytes = static_cast<double>(n) * sizeof(float);

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("设备: %s\n", p.name);
  std::printf("数组 %.0f MB，block 数 %d，每 block %d 线程\n\n", bytes / 1e6,
              kBlocks, kThreads);

  const float ms_copy =
      time_kernel(copy_kernel, kBlocks, kThreads, 0, n, d_in, d_out);
  const float ms_read =
      time_kernel(read_kernel, kBlocks, kThreads, 0, n, d_in, d_out);

  const size_t smem_bytes = kThreads * sizeof(float);
  const float  ms_smem =
      time_kernel(smem_read_kernel, kBlocks, kThreads, smem_bytes, n, d_in,
                  d_out);

  // 共享内存读取次数 = block 数 × 每 block 线程数 × 轮数
  const double smem_bytes_moved = static_cast<double>(kBlocks) * kThreads *
                                  kSmemIter * sizeof(float);

  std::printf("%-22s %-14s %s\n", "访问类型", "耗时(ms)", "带宽(GB/s)");
  std::printf("%-22s %-14.4f %.1f\n", "HBM 拷贝（读+写）", ms_copy,
              bytes * 2.0 / (ms_copy * 1e-3) / 1e9);
  std::printf("%-22s %-14.4f %.1f\n", "HBM 只读", ms_read,
              bytes / (ms_read * 1e-3) / 1e9);
  std::printf("%-22s %-14.4f %.1f\n", "共享内存读取", ms_smem,
              smem_bytes_moved / (ms_smem * 1e-3) / 1e9);

  std::printf("\n--- 解读 ---\n");
  std::printf("共享内存带宽应显著高于 HBM —— 这是各级存储分级的直接体现。\n");
  std::printf("注意本程序测的是带宽，不是延迟：HBM 靠大量并行通道堆带宽，\n");
  std::printf("单次访问的延迟仍是数百 cycles，两者并不矛盾。\n");
  std::printf("实测值通常只有规格峰值的 70%%~90%%，属正常范围。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
```

- [ ] **Step 2: 写 `02_coalescing_demo.cu`**

```cuda
// 02_coalescing_demo.cu —— 访存合并对有效带宽的影响
// 对应文档：02-coalescing-and-cache-line.md
//
// 观测要点：
//   1. kStride = 1 时，一个 warp 的 32 次访问落在同一条 128B cache line，1 个事务
//   2. kStride = S 时，一个 warp 的 32 次访问跨越 S 条 cache line，S 个事务
//   3. 四种 stride 下「有用读取次数」恒定，传输字节按 stride 增长，
//      故有效利用率 = 1 / kStride
//
// 构建：make 02_coalescing_demo
// 运行：./02_coalescing_demo
//
// 显存占用：kStride = 8 时数组为 512 MB，是本程序的最大分配。

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads = 256;
constexpr int kBlocks  = 2048;
constexpr int kIters   = 32;   // 每个线程固定读取次数
constexpr int kReps    = 20;

constexpr long long kTotalThreads = static_cast<long long>(kBlocks) * kThreads;
constexpr long long kTotalWarps   = kTotalThreads / 32;

// 全部线程共读 kTotalThreads × kIters 次。数组长度随 kStride 增长，
// 使**每个线程的读取次数恒为 kIters**，于是「有用字节」与 kStride 无关，
// 可以直接比较有效带宽。
constexpr long long kReads = kTotalThreads * kIters;

// 数组长度：total_threads × kStride × kIters。
// kStride = 8 时最大，为 1.34 亿个 float = 512 MB。
template <int kStride>
constexpr long long array_len() {
  return kTotalThreads * kStride * kIters;
}

// kStride 决定 warp 内相邻 lane 的地址间隔。
//
// 每个 warp 负责一段连续的 32 × kStride 个元素，其中第 lane 个 lane
// 读取本段内的第 lane × kStride 个元素：
//   kStride = 1 → warp 读 32 个连续元素，落在同一条 128B cache line，1 个事务
//   kStride = S → warp 的 32 次读跨越 S 条 cache line，S 个事务
//
// 全部 warp 的段拼起来恰好覆盖整个数组，且每条 cache line 只被一个 warp
// 触及一次。于是「传输字节」为 数组大小，而「有用字节」恒为 kReads × 4。
//
// 有效利用率 = 有用字节 / 传输字节 = 1 / kStride。
//
// 只取 1、2、4、8 四个取值：它们都能整除一条 cache line 含的 32 个 float，
// 因此每条被触及的 cache line 的 4 个 sector 都会被用到 —— sector 粒度
// 与整行粒度算出的利用率一致，不会出现「按 sector 算是 1/8、按整行算是
// 1/16」这种含糊结果。取 16 或 32 就会破坏这个性质。
template <int kStride>
__global__ void strided_sum_kernel(const float* __restrict__ in,
                                   float* __restrict__ out, long long n) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long warp_id = tid >> 5;
  const int       lane    = threadIdx.x & 31;

  const long long warp_span = 32LL * kStride;

  float acc = 0.0f;
  for (int j = 0; j < kIters; ++j) {
    const long long idx =
        (static_cast<long long>(j) * kTotalWarps + warp_id) * warp_span +
        static_cast<long long>(lane) * kStride;
    if (idx < n) acc += in[idx];
  }
  if (acc == -1234567.0f) out[0] = acc;
}

template <int kStride>
float time_case(const float* d_in, float* d_out, long long n) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  strided_sum_kernel<kStride><<<kBlocks, kThreads>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    strided_sum_kernel<kStride><<<kBlocks, kThreads>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / kReps;
}

struct Case {
  int       stride;
  long long n;
  float     ms;
};

}  // namespace

int main() {
  const long long n_max = array_len<8>();

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n_max * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n_max * sizeof(float)));

  const double useful_bytes = static_cast<double>(kReads) * sizeof(float);

  std::printf("每线程读取 %d 次，全部线程共读 %lld 次 = %.1f MB 有用数据\n",
              kIters, kReads, useful_bytes / 1e6);
  std::printf("数组长度随 stride 增长，最大 %.0f MB\n\n",
              static_cast<double>(n_max) * sizeof(float) / 1e6);

  const Case cases[] = {
      {1, array_len<1>(), time_case<1>(d_in, d_out, array_len<1>())},
      {2, array_len<2>(), time_case<2>(d_in, d_out, array_len<2>())},
      {4, array_len<4>(), time_case<4>(d_in, d_out, array_len<4>())},
      {8, array_len<8>(), time_case<8>(d_in, d_out, array_len<8>())},
  };

  // 以 stride = 1 为基准，其有效利用率定义为 1
  const double base_gbps = useful_bytes / (cases[0].ms * 1e-3) / 1e9;

  std::printf("%-8s %-12s %-12s %-12s %-16s %s\n", "stride", "warp 跨线数",
              "数组(MB)", "耗时(ms)", "有用带宽(GB/s)", "相对利用率");

  for (const Case& c : cases) {
    const double gbps = useful_bytes / (c.ms * 1e-3) / 1e9;
    std::printf("%-8d %-12d %-12.0f %-12.4f %-16.1f %.3f\n", c.stride, c.stride,
                static_cast<double>(c.n) * sizeof(float) / 1e6, c.ms, gbps,
                gbps / base_gbps);
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("四种 stride 下的「有用读取次数」完全相同，但数组长度按 stride 增长，\n");
  std::printf("即传输字节按 stride 增长，因此「相对利用率」一列应接近 1/stride。\n");
  std::printf("stride = 1 时 warp 的 32 次访问落在同一条 128B cache line 上，\n");
  std::printf("取回的每个字节都被用到；stride = 8 时一次访问跨越 8 条 cache line，\n");
  std::printf("每条只用到 1/8 的字节，有效带宽随之降到 1/8。\n");
  std::printf("本程序只取 1/2/4/8 四个 stride：它们都能整除一条 cache line 所含的\n");
  std::printf("32 个 float，因此每条被触及的 cache line 的 4 个 sector 都会被用到，\n");
  std::printf("sector 粒度与整行粒度给出同样的利用率。取 16 或 32 会破坏这个性质，\n");
  std::printf("使「利用率」取决于内存系统按 sector 还是按整行传输，结论会含糊。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
```

- [ ] **Step 3: 写 `03_bank_conflict_demo.cu`**

```cuda
// 03_bank_conflict_demo.cu —— 共享内存 bank 冲突与 padding
// 对应文档：03-shared-memory-bank-conflict.md
//
// 观测要点：
//   1. 行距为 32 时，warp 内 32 个 lane 读取同一列会全部撞到同一个 bank
//      —— bank = (lane × 32 + c) % 32 = c，32 路冲突
//   2. 行距改为 33 后，bank = (lane × 33 + c) % 32 = (lane + c) % 32
//      —— 随 lane 遍历全部 bank，无冲突
//   3. 两种情形的计算完全相同，只有共享内存的行距不同
//
// 构建：make 03_bank_conflict_demo
// 运行：./03_bank_conflict_demo

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads = 256;   // 8 个 warp
constexpr int kWarps   = kThreads / 32;
constexpr int kRows    = 32;
constexpr int kReps    = 200;   // 反复执行，让共享内存访问占主导

// kPad = 0 → 行距 32；kPad = 1 → 行距 33
//
// 每个 warp 独占一块 kRows × (32 + kPad) 的 tile。
// 关键访问是 tile[warp][lane][r]：固定列 r，行随 lane 变化。
//   kPad = 0: 地址 = warp×1024 + lane×32 + r，bank = r          → 32 路冲突
//   kPad = 1: 地址 = warp×1056 + lane×33 + r，bank = (lane + r) % 32 → 无冲突
// （1056 = 32 × 33，是 32 的倍数，故 warp 偏移不影响 bank 计算）
template <int kPad>
__global__ void column_read_kernel(const float* __restrict__ in,
                                   float* __restrict__ out, int n) {
  __shared__ float tile[kWarps][kRows][32 + kPad];

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  float acc = 0.0f;

  for (int rep = 0; rep < kReps; ++rep) {
    // 装载：读全局内存写入共享内存。按 [行][列] 顺序，warp 内 lane 对应列，
    // 全局侧是合并访问，共享侧无冲突。
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int g = (int)(((static_cast<long long>(rep) * kRows + r) * 32 +
                           lane) %
                          n);
      tile[warp][r][lane] = in[g];
    }
    __syncthreads();

    // 读取：固定列 r，行随 lane 变化 —— 这是产生冲突的访问模式
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      acc += tile[warp][lane][r];
    }
    __syncthreads();
  }

  if (acc == -1234567.0f) out[0] = acc;
}

template <int kPad>
float time_case(const float* d_in, float* d_out, int n, int blocks) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  column_read_kernel<kPad><<<blocks, kThreads>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < 10; ++r) {
    column_read_kernel<kPad><<<blocks, kThreads>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));

  // 返回单次执行的平均耗时。共享内存读取次数在 main 中按
  // block 数 × 线程数 × kReps × kRows 统计（每线程每轮读 kRows 次）。
  return ms / 10.0f;
}

}  // namespace

int main() {
  constexpr int n      = 1 << 22;
  constexpr int blocks = 2048;

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));

  // 每个线程每轮读取 kRows 次，故共享内存读取总量与 kPad 无关
  const double smem_reads = static_cast<double>(blocks) * kThreads * kReps *
                            kRows;
  const double smem_bytes = smem_reads * sizeof(float);

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("设备: %s\n\n", p.name);
  std::printf("共享内存 tile: %d warp × %d 行 × 32 列\n", kWarps, kRows);
  std::printf("每 block 共享内存: 无 padding %zu B，有 padding %zu B\n\n",
              static_cast<size_t>(kWarps) * kRows * 32 * sizeof(float),
              static_cast<size_t>(kWarps) * kRows * 33 * sizeof(float));

  const float ms_nopad = time_case<0>(d_in, d_out, n, blocks);
  const float ms_pad   = time_case<1>(d_in, d_out, n, blocks);

  std::printf("%-20s %-14s %s\n", "shared 布局", "耗时(ms)", "共享内存带宽(GB/s)");
  std::printf("%-20s %-14.4f %.1f\n", "tile[32][32]（冲突）", ms_nopad,
              smem_bytes / (ms_nopad * 1e-3) / 1e9);
  std::printf("%-20s %-14.4f %.1f\n", "tile[32][33]（padding）", ms_pad,
              smem_bytes / (ms_pad * 1e-3) / 1e9);
  std::printf("\n加速比: %.2fx\n", ms_nopad / ms_pad);

  std::printf("\n--- 解读 ---\n");
  std::printf("两种情形计算量完全相同，只有行距不同：32 与 33。\n");
  std::printf("行距 32 与 bank 数 32 相等，导致固定列访问时 lane 与 lane\n");
  std::printf("全部映射到同一个 bank，32 次访问被串行化为 32 轮。\n");
  std::printf("行距 33 与 32 互质，bank 随 lane 遍历 32 个位置，无冲突。\n");
  std::printf("注意：padding 会让共享内存用量增加 1/32，可能影响占用率 ——\n");
  std::printf("这也是「加 padding 不总是有益」的原因。\n\n");
  std::printf("关于上面的加速比：不必期待它恰好等于 32。kPad = 0 时行距为 32，\n");
  std::printf("编译器可能把 32 次 LDS.32 向量化成 8 次 LDS.128，冲突形态变成\n");
  std::printf("「16 路 × 2 相位」，实测加速比会小于 32 倍的直觉值。看量级即可。\n");
  std::printf("另外全局侧每次只读同一段约 800 KB 的数据（恒在 L2 命中），\n");
  std::printf("这是刻意的 —— 让共享内存访问而非全局访存主导耗时。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
```

- [ ] **Step 4: 写 `Makefile`**

```makefile
# memory-system 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0，GPU 算力 >= sm_80
#
# 用法：
#   make              构建全部
#   make 01_bandwidth_probe
#   make run          构建并依次运行
#   make clean        清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
LDFLAGS   ?=

TARGETS := 01_bandwidth_probe 02_coalescing_demo 03_bank_conflict_demo

.PHONY: all clean run regs
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $<

# 查看每线程寄存器用量、共享内存用量与 spill 情况
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(word 1,$(TARGETS)).cu -o /dev/null

run: all
	@for t in $(TARGETS); do \
		echo "=== $$t ==="; \
		./$$t || exit 1; \
		echo; \
	done

clean:
	rm -f $(TARGETS)
```

- [ ] **Step 5: 静态审查**

本机无 `nvcc`，**不编译**。逐项核对：

```bash
cd 20-gpu-architecture/memory-system/code && echo "--- 文件存在 ---" && for f in 01_bandwidth_probe.cu 02_coalescing_demo.cu 03_bank_conflict_demo.cu Makefile; do test -f "$f" && echo "OK   $f" || echo "MISS $f"; done && echo "--- 编号对应 ---" && for n in 01 02 03; do ls ${n}_*.cu >/dev/null 2>&1 && echo "OK   code/$n" || echo "MISS code/$n"; done && echo "--- Makefile 缩进检查（应为 Tab）---" && grep -q '^	' Makefile && echo "OK 使用 Tab" || echo "MISS 未使用 Tab（需检查）" && echo "--- 文档↔代码编号一致性（两侧都应输出 01 02 03）---" && ls ../*.md | sed 's|.*/||; s|\.md$||; s|-.*||' | sort | tr '\n' ' ' && echo && ls *.cu | sed 's|_.*||' | sort -u | tr '\n' ' ' && echo
```

Expected: 四个 `OK 文件存在`；三个编号 `OK`；Makefile 使用 Tab；最后一段的两行输出**都应是 `01 02 03`** —— **两者必须一致**（这是 spec §5.1 的「编号即外键」约束）。

人工审查清单：

1. 每个 `.cu` 都包含 `#include <cuda_runtime.h>`，错误检查宏在每次 CUDA API 调用后使用。
2. 每次 `<<<>>>` 启动后都有 `CUDA_CHECK_LAST()`。
3. 所有 `cudaMalloc` 都有配对的 `cudaFree`。
4. `02_coalescing_demo.cu` 中 `n = 1LL << 26` 是 32 的倍数，保证取模回绕点落在 warp 边界上；若改为其它值，需重新验证这一性质。
5. `02_coalescing_demo.cu` 在 256 MB 显存分配失败时应有清晰报错 —— `CUDA_CHECK` 宏已覆盖 `cudaMalloc` 的返回码。
6. `03_bank_conflict_demo.cu` 中 `kWarps × kRows × 33 × 4 = 33792` 字节，小于默认的 48 KB 静态共享内存上限。
7. `Makefile` 用 Tab 缩进，不是空格。

- [ ] **Step 6: 提交**

```bash
git add 20-gpu-architecture/memory-system/code/
git commit -m "feat: 添加 memory-system 示例代码与 Makefile"
```

---

## Chunk 5: 验收与收尾

### Task 17: 逐条核验 spec 的验收标准

**Files:**
- Modify: `00-meta/roadmap.md`
- Modify: `00-meta/index.md`
- Modify: `20-gpu-architecture/README.md`

- [ ] **Step 1: 核验第一组验收标准（spec §10.1）**

Run:
```bash
echo "=== 1. 目录集合与 §3.3 一致 ==="
find . -type d -not -path './.git*' -not -path './docs*' -not -name '.' | sort

echo "=== 1b. 两个首期二级目录必须存在（spec A5）==="
for d in 20-gpu-architecture/execution-model 20-gpu-architecture/memory-system; do
  test -d "$d" && echo "OK   $d" || echo "MISS $d"
done

echo "=== 1c. 除这两个外不应有其他二级目录 ==="
find . -mindepth 2 -maxdepth 2 -type d -not -path './.git*' -not -path './docs*' | sort

echo "=== 2. 16 个一级域均有 README ==="
n=0
for d in $(find . -maxdepth 1 -type d -name '[0-9]*-*' | sort); do
  test -f "$d/README.md" && n=$((n+1)) || echo "MISS $d/README.md"
done
echo "计数：$n"

echo "=== 3. roadmap 主题总数 ==="
grep -cE '^- \[ \] [a-z]' 00-meta/roadmap.md

echo "=== 4. glossary 条数 ==="
grep -cE '^\| [A-Za-z]' 00-meta/glossary.md
```
Expected: 步骤 1b 两个 `OK`；步骤 1c 只输出两行 —— `./20-gpu-architecture/execution-model` 与 `./20-gpu-architecture/memory-system`（`-maxdepth 2` 不含 depth 3 的 `code/` 子目录，后者由 Step 2 的 `ls "$d"/code/*.cu` 覆盖）；一级域计数 `16`；主题数 `61`；术语条数 ≥ 30。

- [ ] **Step 2: 核验第二组验收标准（spec §10.2）**

Run:
```bash
echo "=== 1. 文档 ↔ 代码编号对应 ==="
for d in 20-gpu-architecture/execution-model 20-gpu-architecture/memory-system; do
  echo "--- $d ---"
  doc_n=$(ls "$d"/*.md | sed 's|.*/||; s|-.*||' | sort | tr '\n' ' ')
  code_n=$(ls "$d"/code/*.cu | sed 's|.*/||; s|_.*||' | sort | tr '\n' ' ')
  echo "文档编号：$doc_n"
  echo "代码编号：$code_n"
  # 先确认两侧都非空。zsh 下 glob 无匹配会使命令替换为空串，
  # 两个空串相等会给出「OK 一致」的假阳性。
  if [ -z "$doc_n" ] || [ -z "$code_n" ]; then
    echo "MISS 文件缺失"
  elif [ "$doc_n" = "$code_n" ]; then
    echo "OK 一致"
  else
    echo "MISS 不一致"
  fi
done

echo "=== 2. 六篇文档的五节完整性、frontmatter 与 status ==="
for f in 20-gpu-architecture/*/[0-9][0-9]-*.md; do
  miss=0
  for h in "## 0. 背景与动机" "## 1. 原理" "## 2. 执行映射" "## 3. 实现与演进" "## 4. 踩坑与失败模式"; do
    grep -qF "$h" "$f" || { echo "MISS $h in $f"; miss=1; }
  done
  for k in title slug domain topic level difficulty prerequisites tags related code status updated; do
    grep -qE "^${k}:" "$f" || { echo "MISS field $k in $f"; miss=1; }
  done
  # spec §10.2 验收标准 2：status 必须为 stable
  grep -q '^status: stable$' "$f" || { echo "MISS status!=stable in $f"; miss=1; }
  test $miss -eq 0 && echo "OK   $f"
done

echo "=== 3. 六个 .cu 与两个 Makefile ==="
ls 20-gpu-architecture/*/code/*.cu | wc -l
ls 20-gpu-architecture/*/code/Makefile | wc -l

echo "=== 4. 每篇文档末尾有编译与运行小节（spec §10.2 验收标准 4）==="
for f in 20-gpu-architecture/*/[0-9][0-9]-*.md; do
  grep -qF '## 编译与运行' "$f" || { echo "MISS 编译与运行节 in $f"; continue; }
  grep -qE 'make [0-9]{2}_' "$f" || { echo "MISS 编译命令 in $f"; continue; }
  echo "OK   $f"
done

echo "=== 5. 实测数字须有来源或标注为推导值（spec §10.2 验收标准 5）==="
grep -rn '推导值\|CUDA [0-9]' 20-gpu-architecture/*/[0-9][0-9]-*.md || echo "无实测数字，无需标注"
```
Expected: 两组编号均 `OK 一致`；六篇文档全部 `OK`，无 `MISS`；`.cu` 计数为 `6`；`Makefile` 计数为 `2`；六篇文档的「编译与运行」检查全部 `OK`。

**关于检查 5**：六篇文档**可以整体省略第 5 节「实测数据」**（spec §7.1 规定为可选，§6 规定 `status` 与第 5 节无关）。若确实未写实测数字，该检查应输出「无实测数字，无需标注」——这是**通过**，不是失败。只有当文档里出现了具体性能数字时，才必须有对应的 GPU 型号 + CUDA 版本标注，或明确写成「推导值」。

**若任一文档缺「编译与运行」小节**，按通用约定中的格式补上（放在「小结与延伸」之前）。**若缺 `status: stable`**，用通用约定中给出的 `sed` 命令修正后重新提交。

**人工通读（spec §7.2 自查第 5 条）**

七条验收标准中，第 1、2、3、4 条与 §7.2 的前四条已由上面的 `grep` 覆盖。剩下两条**无法机械核验，须人工完成**：

1. **§7.2 第 5 条「可独立读懂」**：逐篇通读，确认不依赖未列在 `prerequisites` 中的文档即可理解。若发现某处理解需要额外前置，把对应路径补进该文档的 `prerequisites`，并回到 spec §8.3 的核对清单确认未引入环。
2. **§7.2 第 3 条「≥2 个组件名 + ≥1 个数值量级」**：`grep` 只能检查是否出现关键字，无法判断是否讲清楚。通读第 2 节时确认组件名与数值是**用来说明机制**的，而不是罗列。

这两条通过后，才可在 Step 7 提交并把 roadmap 标记为完成。

- [ ] **Step 3: 更新 `00-meta/roadmap.md`**

把 `20-gpu-architecture` 下的两行改为已完成，并更新顶部两行：

```markdown
> 最后更新：2026-09-14
> 进度：2 / 61 主题完成
```

```markdown
## 20-gpu-architecture
- [x] execution-model
- [x] memory-system
- [ ] tensor-core
- [ ] interconnect
- [ ] architecture-evolution
```

- [ ] **Step 4: 更新 `00-meta/index.md`**

确认 `20-gpu-architecture` 一节下列出了两个已建成的主题目录。

**同时删除该节内的前向声明注释**（Task 4 写入的 `<!-- 以下两行是前向声明… -->`），因为此刻这两个目录已真实存在：

```markdown
## 20-gpu-architecture
- [execution-model](../20-gpu-architecture/execution-model/) —— SM 结构 / SIMT 与发散 / 占用率与延迟隐藏
- [memory-system](../20-gpu-architecture/memory-system/) —— 存储层次 / 访存合并 / 共享内存 bank 冲突
```

Run:
```bash
grep -c '前向声明' 00-meta/index.md
```
Expected: `0`（注释已删除）。

- [ ] **Step 5: 更新 `20-gpu-architecture/README.md`**

确认主题表格中 `execution-model` 与 `memory-system` 两行的「状态」列为「已完成」。

- [ ] **Step 6: 三处一致性核验**

Run:
```bash
echo "--- roadmap 状态 ---"
grep -A6 '^## 20-gpu-architecture' 00-meta/roadmap.md

echo "--- roadmap 顶部进度 ---"
grep '进度：' 00-meta/roadmap.md

echo "--- 域 README 状态 ---"
grep -E 'execution-model|memory-system' 20-gpu-architecture/README.md

echo "--- 实际完成的主题数（应等于 roadmap 顶部进度分子）---"
grep -cE '^- \[x\] [a-z]' 00-meta/roadmap.md
```
Expected: roadmap 中两行为 `- [x]`；顶部进度分子与 `grep -c` 的结果一致（均为 `2`）；域 README 中两行状态为「已完成」。

- [ ] **Step 7: 提交**

```bash
git add 00-meta/roadmap.md 00-meta/index.md 20-gpu-architecture/README.md
git commit -m "docs: 更新进度表与索引，标记 GPU 硬件样板完成"
```

---

### Task 18: 最终核验

**Files:**
- 无新建。纯验证。

- [ ] **Step 1: 核验工作区干净且提交历史合理**

Run:
```bash
git status --short
echo "--- 提交历史 ---"
git log --oneline
```
Expected: `git status --short` 无输出。提交历史应有约 16 个提交，每个对应一个 Task，信息符合 `<type>: <subject>` 格式。

- [ ] **Step 2: 核验全库无 TODO 残留**

Run:
```bash
grep -rn 'TODO\|TBD\|FIXME\|待补充\|待定' --include='*.md' --include='*.cu' --include='Makefile' . 2>/dev/null | grep -v '^./docs/' || echo "无残留"
```
Expected: `无残留`。（`docs/` 下的设计与计划文档不计入。）

- [ ] **Step 3: 核验代码未提交构建产物**

Run:
```bash
git ls-files | grep -E '\.(o|out)$|^build/' || echo "无构建产物"
```
Expected: `无构建产物`。

- [ ] **Step 4: 汇总交付清单**

Run:
```bash
echo "=== 文件总数 ==="
git ls-files | grep -v '^docs/' | wc -l
echo "=== 一级域 ==="
ls -d [0-9]*/ | tr '\n' ' '
echo
echo "=== 样板文档 ==="
ls 20-gpu-architecture/*/[0-9][0-9]-*.md
echo "=== 样例代码 ==="
ls 20-gpu-architecture/*/code/*
```
Expected: 一级域列出 16 个；样板文档 6 个；样例代码 6 个 `.cu` + 2 个 `Makefile`。
