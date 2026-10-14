---
title: SGLang 的 RadixAttention：压缩前缀树、LRU 与它和 vLLM 的两处口径差
slug: inference-engines-sglang-radix
domain: 120-inference
topic: engines
level: 3
difficulty: intermediate
prerequisites:
  - 120-inference/engines/01-vllm
  - 120-inference/kv-cache/03-prefix-and-eviction
tags: [sglang, radixattention, radix-tree, lru, prefix-reuse, page-vs-block]
related:
  - 120-inference/kv-cache/02-block-management
  - 120-inference/kv-cache/03-prefix-and-eviction
  - 45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory
code:
  - code/02_sglang_and_radix.py
status: stable
updated: 2026-10-09
---

# SGLang 的 RadixAttention：压缩前缀树、LRU 与它和 vLLM 的两处口径差

> **本文只写 SGLang 的【选择与差异】** —— 它用来复用 KV 缓存的**数据结构是一个 radix tree**、
> 淘汰用 **LRU**，以及两件**必须分开写**的口径差：
> **它是 `page = 1 token`（vLLM 是 `block = 16`）**、**它是【默认行为】（vLLM 需要开关）**。
>
> ★★★ **本文是 `engines` 主题里【唯一】有 A 类的一篇** ——
> 引擎的**【行为与性能】**本机无处可验（无 GPU、无 SGLang），
> ★★ **但它的【数据结构】是可写独立参照的**：压缩前缀树 / 最长前缀查找 / LRU 淘汰。
> ★★★ **本主题的硬约束**：**「前缀共享的系统侧机制」一律回链** ——
> **`hash → 块` / 命中率 / 驱逐** → [03-prefix-and-eviction](../kv-cache/03-prefix-and-eviction.md)；
> **分页 / 块管理** → [02-block-management](../kv-cache/02-block-management.md)；
> **PagedAttention 的算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84`。
> ★★ **「不是任何真实引擎，也不是真机」**。

## 0. 背景与动机

一个请求结束之后，它的 KV 缓存**要不要丢**？

- **丢**（vLLM 的默认路线）：显存立刻回收；★ **代价**：下一个请求若与它共享一段开头，
  **那段 prefill 要重算一遍**；
- **留**（SGLang 的 RadixAttention）：把 KV 缓存**保留下来**，按**前缀结构**组织，下次直接复用。

★★★ SGLang 的官方描述逐字（`[B]` 级，LMSYS Org 博客）：

> 「Instead of discarding the KV cache after finishing a generation request, our approach **retains
> the KV cache for both prompts and generation results in a radix tree**.」

**⇒ 「留」这个选择的第一个后果，就是它【必须有一个能按前缀找东西的数据结构】。**
★★★ 本文讲的就是**那个数据结构本身** —— 它是**压缩前缀树（radix tree）**。

★ **一个具体的对照场景**（本文程序里的真算例，`code/02_sglang_and_radix.py` 的 `A.2`）：

| 结构 | 节点数（同一组 12 条序列、共 53 个 token） |
|---|---|
| **压缩前缀树（radix tree）** | **`16`** |
| **未压缩 trie**（一条边一个 token） | **`31`** |

★★★ 差别来自**压缩**：radix tree 的**一条边可以挂多个 token** —— 于是「只有单孩子」的那些**链**
被压成一个节点。★★ 而这就是官方博客那句「a **space-efficient alternative** to a trie」的**量化**：
**同一个序列集，节点数从 31 降到 16**。

## 1. 原理

### 第一步：radix tree 是什么（相对未压缩 trie 的差别）

官方博客逐字（`[B]`）：

> 「A radix tree is a data structure that serves as a **space-efficient alternative to a trie**
> (prefix tree). Unlike typical trees, the edges of a radix tree can be labeled with not just single
> elements, but also with **sequences of elements of varying lengths**.」

★★ **一句话**：**未压缩 trie 的每条边恰好一个 token；radix tree 的每条边可以是一段 token。**
⇒ 两类操作因此都要**处理边**：

| 操作 | 未压缩 trie | radix tree |
|---|---|---|
| **插入** | 每个 token 一层，遇到不存在就建节点 | ★★ **遇到「部分匹配」要【切分】一条边**（这是全部难点） |
| **最长前缀查找** | 逐 token 下降，数到哪算到哪 | ★★ 逐段比对边；★ **部分匹配时停在边中间** |
| **节点数** | `= 互异前缀数` | ★★ **更少**（链被压掉） |

### 第二步：★★★ 插入时的「切分」是全部内容

假设已有 `(1, 2)`，现在插入 `(1, 3)`：

```
未压缩 trie：                      radix tree：
   (1)                              (1,2)  ← 一条边
    ├── (2)          ⇒             (1,3)  ← 也是独立一条边（切分后）
    └── (3)
```

★★★ **关键动作**：radix tree 在插入 `(1,3)` 时，发现与已有边 `(1,2)` **前 1 个 token 相同**
（`j = 1 < len(edge) = 2`）⇒ **必须先【切分】**：把 `(1,2)` 拆成 `(1)` + `(2)`，再把 `(1,3)` 挂在 `(1)` 下。
★ 不切分（把两条序列**焊在一条边上**）会让**最长前缀查找给出错的答案** —— 这正是本文程序
`D.4` 的**守卫实验**：拆掉切分之后，查 `(1,3)` 的最长前缀**从 `2` 掉到 `1`**。

### 第三步：LRU 淘汰

官方博客逐字（`[B]`）：

> 「We implement a **Least Recently Used (LRU)** eviction policy, complemented by a cache-aware
> scheduling policy, to enhance the cache hit rate.」

★ **在本主题的模型里**（`[E]` 级，本库实现）：缓存的**单位是一条序列**，容量 = 能装几条；
访问时若命中就**刷新它的时间戳**；插入新键而超容量时，**逐出时间戳最小的那条**。
★★ 本文程序把「成员判定」与「淘汰选择」两件事都做成**两条独立路径**（见 §2）。

★★★ **与 `kv-cache/03` 的 LRU【不是同一件事】**（`S3` 互指）—— 那里的缓存单位是**块**（**系统侧块管理**，
带 `ref == 0` 过滤器与「保留区间」那一维）；这里的单位是**条序列**（**数据结构层**）。
★ 两者**不是同一个知识对象**（单位不同、角度不同）⇒ ★ **命中率的口径也不同**。

### 第四步：★★★ 两处口径差（**必须分开写**）

**（a）块粒度：`page = 1 token`（SGLang） vs `block = 16`（vLLM）**

SGLang 的存储布局逐字（`[B]`）：「These KV cache tensors are stored on the GPU in a paged layout,
where **the size of each page is equivalent to one token**.」

| 引擎 | 单位 | 术语 | 一段 `4096` token 的 KV 数出多少单位 |
|---|---|---|---|
| **SGLang** | **`1` token / 页** | `page` | **`4096` 页** |
| **vLLM** | **`16` token / 块** | `block` | **`256` 块** |

★★★ **两者差 `16` 倍** —— 这是**同名不同指**：**「块 / 页」这个词在两家指的是不同粒度**，
★ **不可互相套用**（本文程序 `C.1` 把这条算成可断言的 `16×`）。
★★ **一条纪律**：写块粒度时**必须写明是哪一家**。
★★★ **★★ 这条「同名不同指」的【唯一正文在此】**（`CONTRIBUTING` §7，**收拢已完成**）——
本文的角度是**「两家引擎的口径差」**（`C.1` 的 `16×` + `C.2` 的**默认行为**）。
★ 本域 `kv-cache` 的 `01`/`02`/`03` **已收成纯指针**（★ 那里只用**块粒度**这一个因子，`D1` 订正）。

**（b）默认行为：SGLang 默认复用 vs vLLM 需开关**

★★★ 这是**同一个能力在两家的默认值截然不同**：

| 引擎 | 前缀复用是默认吗 | 依据 |
|---|---|---|
| **SGLang** | ★ **是**（★ **从「Instead of discarding …」推得**） | 官方博客逐字（`[B]`）；★★ **「默认」是本库的推断（`[E]`，`SGLANG_PREFIX_REUSE_IS_DEFAULT`）** —— ★ 博客全文**无 `default` / 开·关字样** ⇒ **不得写成引文**（`F4`） |
| **vLLM** | ★ **需 `--enable-prefix-caching`**；★ **默认值未取到** | 文档只给开关（`[B]`）；**不写「默认开启」** |

★ **读法**：这一条**不是「谁更好」**，而是**「一家的默认值是另一家的开关」**。

## 2. 执行映射

引擎层。★ **本节的「量」全部来自本机程序，验证的是【数据结构的不变量】，不是任何引擎的性能。**

**（1）★★★ 三条 A 类判据**（`code/02_sglang_and_radix.py`，★ 被测与参照**无共享代码**）：

| 判据 | 被测 | 参照（**独立**） | 实测 |
|---|---|---|---|
| **`A.1` 最长前缀** | 压缩前缀树 | **逐条序列、逐 token 朴素比对**（不碰树） | **`85` 条查询上逐条一致**（不一致 `0`） |
| **`A.2` 空间** | 压缩前缀树 | **未压缩 trie**（一条边一个 token） | radix **`16` 节点** < trie **`31` 节点**（总 token `53`） |
| **`A.3` 淘汰** | 树判成员的缓存 | **列表线性扫描**的缓存 | **`120` 步**轨迹上**命中序列与受害序列都一致**（受害 `85` 次） |

★★★ **`A.2` 是「省空间」这句话的量化**：radix 的 `节点/token` = **`0.3019`**，
未压缩 trie = **`0.5849`**（**同一个序列集**）—— ★ **几乎减半**。

**（2）★★ 两条守卫实验**（`CONTRIBUTING:235` 的要求：**把参照 / 被测改错，断言必须失败**）：

- **`D.4`**：插入遇部分匹配时**不切分** ⇒ 查 `(1,3)` 的最长前缀**从 `2` 掉到 `1`**；
- **`D.5`**：把 LRU 改成**逐出最近用过的**（MRU）⇒ 受害序列**变了**（正确 `85` 次 vs 坏法 `96` 次）。

⇒ ★★ 这说明 `A.1`/`A.3` **不是恒真**，而是**对这两条规则敏感**。

**（3）★ 压缩比与命中率**（`B.1`/`B.2`）：
- `B.1`：radix `节点/token = 0.3019` < trie `0.5849`；
- `B.2`：命中率随容量**单调不降**（容量 `2→4→8→16` 时 `0.1417 → … → 0.5417`）；
  ★ 大容量下仍未命中的是**强制缺失**（首次出现）。
  ★★★ **★ 它与 `kv-cache/03` 的「命中率 vs 容量」【不是同一张表】**（`S3` 互指）——
  ★ 那里的单位是**块**、口径是 `Σ命中块数 / Σ总块数`；★ 这里是**条序列** ⇒ **不可互相套用**。

**（4）★★★ 这条链的产物是「数据结构的不变量」，不是「真实性能」。**
本机**无 GPU、无 SGLang** ⇒ 上面的数**没有一个**来自真实系统。
★★★ **限定语**：「不是任何真实引擎，也不是真机」。

## 3. 实现与演进

**第一步：不共享（每个请求自己一份 KV）。** ★ 失败模式：**共享前缀被重复算**。

**第二步：按 `hash → 块` 复用（vLLM 的路线）。** ★ 机制正文在
[03-prefix-and-eviction](../kv-cache/03-prefix-and-eviction.md) —— **本文不重推**（`P37`）。
★ 它的粒度是**块**（`16` token）。

**第三步：按 **radix tree** 复用（SGLang 的路线）。** ★★★ 差别**：
- **粒度是 `1 token`**（`page = 1 token`）—— 所以它**不需要「只缓存整块」那类规则**；
- **结构是压缩前缀树** —— 前缀查找是**沿边下降**，而不是查哈希表；
- ★★ **淘汰用 LRU**；
- ★★★ **它默认开启**（vLLM 需开关）。

**第四步：与 paged attention / continuous batching 兼容。** 官方博客逐字（`[B]`）：
「RadixAttention is compatible with existing techniques like continuous batching and paged attention.」
★ **读法**：SGLang **把这两项当【已有技术】**，而非自己发明的 —— 引用时**不要倒过来写**。

## 4. 踩坑与失败模式

### 4.1 把 `page` 与 `block` 当同一个东西（差 16 倍）

- **现象**：用 vLLM 的「块」数去描述 SGLang，或反之；
- **成因**：★★★ **同名不同指** —— SGLang 的 `page = 1 token`，vLLM 的 `block = 16 token`；
- **本库算例**（`code/02_sglang_and_radix.py` 的 `C.1`，★ 纯算术 `4096/1` 与 `4096/16`、**非实测**）：
  同一段 `4096` token 的 KV，SGLang 数出 **`4096` 页**、vLLM 数出 **`256` 块** ⇒ **差 `16` 倍**；
- **规避**：★★ **写块粒度时【写明是哪一家】**；★ 不可互相套用。

### 4.2 把「默认行为」与「需要开关」混为一谈

- **现象**：写「前缀复用默认开启」，而这只对 SGLang 成立；
- **成因**：★ vLLM 的 APC 需要一个显式开关，且**其默认值未取到**；
- **本文实测**（`C.2`）：SGLang 的「默认即复用」= `True`；vLLM 的默认值 = **未取到**（不得臆断）；
- **规避**：★★★ **写「默认开启」前问一句：这是【谁的】默认？** ★ 一家的默认可以是另一家的开关。

### 4.3 插入时忘了切分边 —— 而第一眼看不出来

- **现象**：radix tree 实现里「部分匹配」被直接塞进子节点，多数用例**居然还是对的**；
- **成因**：★ 只有当**两条序列共享一段前缀、又在分叉点不同**时，错误才暴露；
- **本文实测**（`D.4`）：`insert(1,2)` 之后 `insert(1,3)`，不切分时查 `(1,3)` 的最长前缀
  **从 `2` 掉到 `1`**；
- **规避**：★★ **构造守卫时必须让两条序列在【分叉点】相邻** —— 否则你得到一个「绿」的守卫，
  而它**什么都没测到**。

### 4.4 LRU 写成了 MRU（逐出最近用过的）

- **现象**：命中率诡异下降，但「缓存确实在淘汰」；
- **成因**：★ 受害者选的是 `max` 时间戳而不是 `min`；
- **本文实测**（`D.5`）：改成 MRU 后受害序列**变了**（`85` vs `96` 次）⇒ `A.3` 对这条规则敏感；
- **规避**：★ **受害者是【最小】时间戳**；★★ 用两条独立路径（树判成员 vs 列表扫描）对拍。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到「SGLang 用什么复用 KV」 | ★★ **压缩前缀树（radix tree）** + **LRU 淘汰**（`A.1`/`A.3`） |
| 见到「radix tree 省空间」 | ★★ 是**量化**的：同一序列集 radix `16` 节点 vs 未压缩 trie `31` 节点（`A.2`/`B.1`） |
| 见到「块 / 页的粒度」 | ★★★ **同名不同指** —— SGLang `page = 1 token`、vLLM `block = 16 token`（**差 16 倍**，`C.1`）⇒ **必须写明是哪一家** |
| 见到「前缀复用是不是默认」 | ★★★ **是【谁的】默认？** SGLang **是**；vLLM **需 `--enable-prefix-caching`** 且默认值**未取到**（`C.2`） |
| 见到 radix tree 的插入 | ★★★ **部分匹配必须【切分】边**（`D.4`）；不切分 ⇒ 最长前缀少算一个 token |
| 见到 radix tree 的淘汰 | ★★ **受害者 = 最小时间戳**（`D.5`）；写成 `max` 就是 MRU |
| 见到「最长前缀查找」的边界 | ★ 空查询 ⇒ `0`；恰好命中 ⇒ 全长；查询长于一切 ⇒ **停在已存的最长**（`C.3`） |
| 见到「RadixAttention 与 continuous batching 的关系」 | ★ SGLang **把后者当【已有技术】**（官方博客逐字）—— 不要倒过来写 |
| 见到「`hash → 块` / 命中率 / 遍历式驱逐」 | ★★★ **不在本文** —— 正文在**彼处** [03-prefix-and-eviction](../kv-cache/03-prefix-and-eviction.md)（★ `kv-cache` 是**兄弟主题目录**，不是本目录 ⇒ 用「**正文在彼处，此处回链**」，`D5` 订正） |
| 见到「PagedAttention 的算式」 | ★★★ **不在本文** —— `130-model-optimization/quantization/18-kv-cache-basics:68-84` |
| 见到「SGLang 的真实吞吐 / 命中率」 | ★★★ **本机测不了**（无 GPU、无 SGLang）—— 本文的数只支撑**数据结构的不变量** |

## 6. 小结与延伸

**一句话**：**SGLang 的 RadixAttention 把「复用 KV 前缀」做成了一套
【压缩前缀树 + LRU】的数据结构，并且它是默认行为 —— 而它和 vLLM 的差别里，
最容易被写错的是【块粒度差 16 倍】与【默认值截然不同】这两处。**

**三条可独立复用的结论**：

1. ★★★ **`page` 与 `block` 是同名不同指**：SGLang `1` token / 页 vs vLLM `16` token / 块 ⇒
   **引用必须写明是哪一家**；
2. ★★ **radix tree 的「省空间」可量化**：`16` vs `31` 节点（同一序列集），来自**边的压缩**；
3. ★★★ **「默认开启」是【谁的】默认**：SGLang 默认即复用，vLLM 需一个开关 —— 写成一句就没有信息量。

**延伸**：

- ★★ **vLLM 的架构选择**（引擎循环 / 参数面）→ [01-vllm](01-vllm.md)；
- ★★ **TensorRT-LLM 与 llama.cpp** → [03-tensorrt-llm-and-llamacpp](03-tensorrt-llm-and-llamacpp.md)；
- ★★★ **前缀共享的系统侧机制（`hash → 块` / 命中率 / 驱逐）** → 本域
  [03-prefix-and-eviction](../kv-cache/03-prefix-and-eviction.md)（**正文在彼处**）；
- ★★ **分页 / 块管理（引用计数 / CoW）** → 本域 [02-block-management](../kv-cache/02-block-management.md)；
- ★★★ **PagedAttention 的定义与算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84`。

## 参考

- **`[B]` LMSYS Org 官方博客** —— `https://lmsys.org/blog/2024-01-17-sglang/`（2024-01-17，
  作者含 Lianmin Zheng、Ying Sheng 等；抓取日 **2026-10-09**）。★ 早期猜测的 URL
  `.../2023-12-18-sglang/` 实测 **HTTP 404**。逐字依据：radix tree 保留 KV · 「省空间的 trie 替代」·
  LRU 淘汰 · `page = 1 token` · 「与 continuous batching / paged attention 兼容」。
- **`[B]` vLLM 源码**（`main` 分支）—— `vllm/config/cache.py` 的 `DEFAULT_BLOCK_SIZE`（`block = 16`）；
  ★ 文档页 `configuration/engine_args/` 的 `--enable-prefix-caching`（**未声明默认值**）。
- **`[E]` 本主题的实现** —— radix tree / 未压缩 trie / 列表扫描 LRU 都是**本文自己写的参照**；
  ★★ 缓存模型的容量单位 = **条序列**，是**本主题的算例**，不是任何引擎的配置。
- ★ **一条术语纪律**：`block` / `page` **同名不同指** —— 引用时必须**写明是哪一家**；
  本文一律写「**KV 缓存**」「**请求调度器**」全称。
- ★ **跨域回链**（**正文在彼处，本文只回链**）：
  · 前缀共享与淘汰机制 → 本域 `kv-cache/03`；分页与块管理 → 本域 `kv-cache/02`；
  · PagedAttention 的算式 → `130-model-optimization/quantization/18-kv-cache-basics:68-84`。

## 编译与运行

```sh
cd code && make run                       # 三个编号程序，任一断言失败即非零退出
python3 02_sglang_and_radix.py             # 只跑本文的配套程序
make 02_sglang_and_radix                  # ★★ 与上一行**不等价**：它先过 check-env（查解释器 / numpy）再跑
```

★ 本机的运行条件：`python3 3.13.5`、`numpy 2.4.4`、macOS arm64，**无 GPU、无任何引擎**。
★★★ **本文的程序【有真 A 类】**（`engines` 主题里唯一的一篇）：`A.1`（最长前缀：压缩树 vs 逐 token 朴素比对）·
`A.2`（空间：radix vs 未压缩 trie）· `A.3`（LRU：树判成员 vs 列表扫描）；
★★ `D.4`/`D.5` 是**两个守卫实验**（不切分 / MRU）；★ `D.0` 机械证明本文件**有** A 类；
★ `D.9` 是**术语纪律**（`裸写被禁裸词的计数 == 显式豁免清单覆盖数`）。
★ **本文对应 `code/02_sglang_and_radix.py`**（三类断言全在程序内部，失败即非零退出）。
