---
title: 编写 pass 与 pattern rewrite
slug: mlir-writing-a-pass
domain: 60-compilers
topic: mlir
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/mlir/01-ir-and-dialects
tags: [mlir, pass, pattern-rewrite, fixpoint, greedy, non-convergence, canonicalization]
related:
  - 60-compilers/mlir/01-ir-and-dialects
  - 60-compilers/mlir/02-tensor-vs-memref
code:
  - code/04_pattern_rewrite.py
status: stable
updated: 2026-09-16
---

# 编写 pass 与 pattern rewrite

> 一个 pass 通常不是"遍历 IR 改一遍"，而是"反复套用规则直到改不动"。这个"直到改不动"有名字叫**不动点**，而它的实现里有一个默认值 `maxIterations = 10`。**达到这个上限而停，返回值是 failure**——也就是说，一个写不好的 pass 在真实系统里的表现是"报未收敛"，而不是"改得少一点"。

## 0. 背景与动机

最朴素的重写做法是这样的：

```text
for op in 所有 op:
    if 匹配某条规则(op):
        改掉 op
```

这段代码有个已经讨论烂了的问题：**改完之后可能又出现新的可改写点**。经典例子是常量折叠——`(2 + 3) * x` 折掉 `2 + 3` 变 `5 * x`，但如果 `5 * x` 的 `5` 又和别的东西能折，就得再来一遍。于是做法变成：

```text
while 还有规则被套用:
    走一遍 IR
```

这里就是本文要讲的三个问题：

1. **"还有规则被套用"这句话怎么判定？** —— 两轮之间一次都没改写，就叫**不动点**（fixpoint）。
2. **如果永远到不了不动点呢？** —— 必须有上限。官方默认是 **10**。
3. **到达上限时会发生什么？** —— 这一条最反直觉：**它不是"改得少一点"，而是 failure**。

第 3 条为什么重要？因为它决定了这个机制**在行为上是不是可靠的**。如果到上限就算"差不多收敛了"，那么一条写反的规则（比如同时存在 `x + 0 → x` 和 `x → x + 0`）会**静默地把 IR 改坏**——pipeline 跑完、日志干净、结果不对。而事实是官方明写：达到上限而停**就是没收敛**，返回值是 failure。

本文配套程序把这条跑成了可执行断言（§5 的 C1）：一组互相抵消的规则下，driver 在 10 轮处报 `converged=False`；把上限抬到 500 仍报不收敛；而**把收敛判定改成"永远 True"，C1 的判据立刻判为失败**——最后一条证明的是"这个判据真的在检查东西"。

## 1. 原理

**第一步：官方对贪心 driver 的语义定义。** 源码 `GreedyPatternRewriteDriver.h` 里 `applyPatternsGreedily` 的注释原文：

> Rewrite ops in the given region … by **repeatedly applying the highest benefit patterns in a greedy worklist driven manner until a fixpoint is reached**.

这句话里有三个机制词，缺一不可：

- **highest benefit**：多条规则同时匹配时，按 benefit 从高到低挑。benefit 是规则自带的数值，不是运行时学习来的。
- **worklist driven**：维护一个待处理队列，而不是每轮从头扫。
- **until a fixpoint is reached**：终止条件是不动点。

**第二步：终止与失败——两条独立的退出路径。**

同一份注释紧接着写：

> The greedy rewrite **may prematurely stop after a maximum number of iterations**, which can be configured in the configuration parameter.

以及返回值：

> Returns "success" if the iterative process **converged** (i.e., fixpoint was reached) and no more patterns can be matched within the region.

把两句合起来读，就得到 driver 的**两条退出路径**：

| 退出原因 | 返回值 |
|---|---|
| 某一轮**一次改写都没发生** | success（这就是不动点） |
| 轮数用尽（默认 10）**而最后一轮仍在改写** | **failure** |

这里有一处容易读漏的地方：**「没有 pattern 被成功匹配」也算收敛**。官方对收敛的定义不是"IR 达到了某个标准形态"，而是"**再套一轮什么也不会改**"。所以一个"什么都没做"的 pass 返回 success 是**正常**的，不是 bug。

**第三步：`GreedyRewriteConfig` 的默认值。**

源码里这几个成员的初值（逐行，见 `code/specs.py`）：

| 成员 | 默认值 | 含义 |
|---|---|---|
| `maxIterations` | **10** | 迭代上限；用尽而停 = failure |
| `fold` | **true** | 是否在改写时顺带做 folding |
| `useTopDownTraversal` | **false** | **默认自底向上**遍历 |
| `regionSimplificationLevel` | **Aggressive** | region 化简的力度 |

第三行值得停一下：**默认是 bottom-up**。它有一个非常具体的后果，本文 §5 的边界实验把它测出来了：对一条左嵌套的 `addf` 链，bottom-up 下的剥离规则**每轮只能剥掉一层**，于是"需要多少轮"就等于"链有多深"。这正好让"上限 10 到底卡在哪里"变成一个可以放在天平上量的东西。

**第四步：为什么上限是 10、以及它会咬人。**

`maxIterations = 10` 是个**经验值**，不是从某个不变量推出来的。它的合理性在于：绝大多数规则的依赖链都很短，10 轮足够。但它会咬人的地方在于**"够不够"取决于规则集，而不是取决于 IR 有多大**。

本文 §5 的边界实验给出了一个具体的分界（本机实测，迷你 driver）：

| 左嵌套层数 | 收敛 | 用掉的轮数 | 改写次数 |
|---|---|---|---|
| 8 | 是 | 9 | 8 |
| 9 | **是** | **10** | 9 |
| 10 | **否** | 10（用尽） | 10 |
| 11 | 否 | 10（用尽） | 10 |

注意**收敛所需轮数 = 层数 + 1**（多出来的那一轮是"什么都没改"的那轮，用来确认不动点）。于是"9 层收敛、10 层不收敛"这条分界的成因完全清楚了：**10 轮的上限意味着最多 9 次有效改写**。这不是"10 层就会失败"这种模糊说法，而是一次可以复算的算术。

**第五步：DCE 是 driver 的一部分。**

官方注释里还有一句很容易被忽略：

> Also performs simple dead-code elimination **before attempting to match** any of the provided patterns.

"before attempting to match"是顺序——**先 DCE，再匹配**。这一条的实际价值很高：改写经常留下死掉的常量与中间结果，如果 driver 不顺手清，IR 会随着轮数增长而膨胀。本机实测里这一点很直观：一个"插入 `+0`"的规则单独存在时，IR 的 op 数在 10 轮里从 4 涨到 7（有 DCE）；**第一版没有 DCE 时，同一个地形在 10 轮里从 4 涨到 2050**（见 §5 的第一版被推翻项）。

**第六步：这个 driver 就是 canonicalization 用的那个。**

`mlir/lib/Transforms/Canonicalizer.cpp` 里调用的正是同一个函数（PR #104649 的 diff 里可以逐行看到）。这条的意义是：**canonicalize 不是另一套机制**，它就是"规则集里装满各种局部化简、然后跑贪心 driver"。

## 2. 执行映射

**组件一：pattern 的注册与冻结。** 规则不是散着传的，而是先收进一个集合再交给 driver（源码里的 `FrozenRewritePatternSet`）。"冻结"这个词有实际后果——**规则集在 driver 跑的过程中不变**，所以"每轮重新算一遍规则"这种玩法不在这条路上。

**组件二：worklist 与遍历方向。** `GreedyRewriteConfig` 有两种初始填充方式（官方文档页）：

- **Top-down**（pre-order）：官方说它 "generally more efficient in compile time"；
- **Bottom-up**：默认。

**组件三：benefit 的排序。** driver 每轮对每个 op 找出所有匹配的规则，按 benefit 排序取最高的。这带来一个值得记的推论：**如果两条规则的 benefit 相等且互相抵消，驱动方向就由排序的稳定性决定**。本文配套程序里给排序加了名字做次级键（`(-benefit, name)`），就是为了让"相等时选哪条"是确定的——**不确定的排序会让不收敛的复现变得困难**。

**组件四：诊断。** 真实系统里"不收敛"的症状是 pass 报 failure；官方还提示在 `testConvergence=true` 下，这类 pipeline 会被判为 non-convergent（即"这条 pipeline 本身没收敛"这个性质会被测试基础设施单独标出来）。

**量级**：一条小链（12 个 op）通常 2 轮收敛（本机实测：改写前 12 个 op、改写后 4 个、累计 4 次改写、2 轮）。**轮数是 O(依赖深度)，不是 O(op 数)**——这一点在 §5 的边界表里体现得很清楚：层数从 8 加到 11，op 数只涨了 3，但轮数一直顶在 10。

## 3. 实现与演进

**第一代：手写遍历。** 每个 pass 自己决定怎么走、改几遍。问题很直接：**"改到不动点"这件事没有统一语义**，每个 pass 的收敛行为都不一样。

**第二代：把 driver 抽出来。** `applyPatternsGreedily` 这一族 API 把"反复套用到不动点"做成公共设施，规则只写"匹配 + 改写 + benefit"。canonicalize 也换成了同一套（§1 第六步）。

**第三代：配置化。** `GreedyRewriteConfig` 把迭代上限、遍历方向、region 化简力度都变成可配置项。**默认值就是这一代的核心内容**——`maxIterations = 10` 这个数决定了所有 pass 的收敛预算。

**一处版本注记（这一处凭印象写几乎必错）**：

当前 API 名是 **`applyPatternsGreedily`**。旧名 `applyPatternsAndFoldGreedily` 在 2024-12 那次改动（llvm-project PR #104649，标题 "[mlir] Enable decoupling two kinds of greedy behavior."，merged 2024-12-20）里被改成了 `LLVM_DEPRECATED("Use applyPatternsGreedily() instead", "applyPatternsGreedily")` 形式的**兼容别名**。

**版本边界要说准**：那次改动**加的是别名，不是立刻删除**。这一点在分支上可以直接复核（2026-09-16 取用，同一个头文件里 grep `applyPatternsAndFoldGreedily`）：

```text
release/20.x : 2 命中
release/21.x : 2 命中
release/22.x : 2 命中
release/23.x : 0 命中
main         : 0 命中
```

所以"LLVM 21 上这个名字已经没了"这种说法是**错的**——**旧名在 20/21/22 上仍在，从 23 起消失**。引用这条时**必须写明版本**。

这个名字的历史还**翻过一次**：据本计划的记录，2020 年官方曾把 `applyPatternsGreedily` 改成 `applyPatternsAndFoldGreedily`，2024 年又改回来。**本文只把 2024 年那次列为一手核实过的**（PR #104649 的 diff 可逐行查证）；2020 年那次方向翻转本文未取到一手证据，**不作为事实陈述**。

**关于 "greedy driver 就是 canonicalization 用的那个"**：这条有一手依据——`mlir/lib/Transforms/Canonicalizer.cpp` 调用的就是同一个函数（PR #104649 的 diff 里有该文件的改动）。

## 4. 踩坑与失败模式

五条。前两条是本文的核心，都**能真的跑出来**。

**失败模式 1：规则集互相抵消，driver 静默地改坏 IR。**

- **现象**：一组本该"简化"的规则，跑完之后 IR 不但没变小，反而变大；如果不看返回值，整个 pipeline 会把这当成一次成功的改写。
- **成因**：两条规则的作用方向相反且都能一直匹配。本文配套程序用的地形是 `x + 0 → x`（benefit 10）与 `x → x + 0`（benefit 30）同时存在。注意 benefit **更高的是那条"变复杂"的规则**——于是每轮都有改写发生，**永远到不了"一轮什么都没改"这个终止条件**。
- **规避**：**把 driver 的返回值当成必须检查的东西**，而不是让它默默结束。官方对这条的立场很清楚：达到上限而停 = failure。本机实测：该地形下 driver 报 `converged=False iterations=10 applied=10`（10 轮全都改了东西），IR 的 op 数从 4 涨到 7；**去掉反向规则后，同一个地形在第 2 轮收敛**。

  > 一条**反证**（程序里跑过）：把收敛判定改成"永远返回 True"，C1 用的那个判据（"未收敛且用满 10 轮"）立刻判为失败。这条证明的是**判据本身在检查东西**，不是恒真的。

**失败模式 2：把 `maxIterations` 当成"应该够用"而不去核对。**

- **现象**：一条需要 11 轮才能到不动点的规则链被 10 轮上限截断，pass 报 failure；实现者以为是"IR 太怪"，于是去查 IR，方向就错了。
- **成因**：上限约束的是**改写轮数**，而改写轮数由**规则之间的依赖深度**决定，与 IR 规模关系不大。IR 里多几个无关的 op 不会增加轮数；规则之间多一层依赖就多一轮。
- **规避**：核对 `轮数 = 依赖深度 + 1` 这条算术（§1 第四步那张表）。本机实测的分界：**9 层收敛（用满 10 轮），10 层不收敛**。所以这不是"10 层这个数不吉利"，而是"默认预算允许 9 次有效改写"。
- **交叉验证**：把上限抬到 500，互逆规则下仍不收敛（本机实测 `iterations=500 applied=500`，IR 规模基本不变）——**这一条排除了"不收敛只是因为上限太小"这个解释**。这是本域对"自造地形"类断言的硬性要求的落实：地形必须能因为"把被检验的那一方换成正确的"而失效，做法是**换一个上限再跑一次**。

**失败模式 3：以为"没有规则匹配"是异常。**

- **现象**：写一个 pass，跑在一条已经化简过的 IR 上，发现"什么都没改"，于是加了一堆诊断想找出"为什么规则没生效"。
- **成因**：官方对收敛的定义就是"**再套一轮什么也不会改**"。一个已经是不动点的 IR 上不匹配任何规则，是**正确结果**，返回值是 success。
- **规避**：区分"没匹配"（正常）与"匹配了但改了又改回来"（异常）。本机实测里这两者都跑到了：前者是"幂等"那条 A 类断言——对第一遍的输出再跑一遍，改写次数为 0、`converged=True`（14 个地形全过）。

**失败模式 4：忘了 driver 自带 DCE，手工清理反而打乱顺序。**

- **现象**：为了让 IR 干净，在规则之间手工插删除死代码的动作，结果某些规则的匹配前提被这次删除破坏。
- **成因**：driver 的 DCE 发生在**每一轮匹配之前**（"before attempting to match"），是一个固定位置。手工再插一个，就变成两个位置，行为不再可预期。
- **规避**：把清理交给 driver。本机实测里 DCE 的效果很直接：同一份互逆规则，**有 DCE 时 10 轮后 op 数 4 → 7，没有 DCE 时 4 → 2050**（§5 的第一版被推翻项）。**2050 这个数不是笔误**——它是每轮插入新 op 而不清理的自然结果。

**失败模式 5：benefit 相等时依赖"稳定排序"。**

- **现象**：两条 benefit 相同的规则互相抵消，改写的方向在不同运行之间不一致，不收敛现象时有时无。
- **成因**：排序是 driver 的职责，而"相等时怎么办"没有唯一的官方语义。若实现里用了不稳定的排序，行为就不可复现。
- **规避**：给排序加确定的次级键。本文配套程序用的是 `sort(key=(-benefit, name))`，`name` 只是为了让行为可复现，**没有语义**。这一条同样适用于"自造地形"类实验：**地形与排序规则都要先于结果冻结**。

## 5. 实测数据

**测量条件**：`code/04_pattern_rewrite.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（20 条断言：A 6 / B 8 / C 6）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库几百行的迷你 greedy rewrite driver**跑出来的，**不是 MLIR 的 `applyPatternsGreedily`**。规则表、benefit 值、遍历顺序的实现都是本仓库自己的；只有**默认值**（`maxIterations=10` / `fold=true` / `useTopDownTraversal=false` / `regionSimplificationLevel=Aggressive`）与官方一致，出处逐条列在 `code/specs.py` 里并注明取用日期。这些数字支持的是"这类 driver 的收敛行为会怎样被规则集决定"这类**机制**结论，**不支持任何关于真实 MLIR 的结论**。

**规则表**（本仓库设定，benefit 越大越先被挑中）：

| 规则 | benefit |
|---|---|
| `fold_add`（常量折叠） | 20 |
| `fold_mul`（常量折叠） | 20 |
| `mul_zero`（`x * 0 → 0`） | 15 |
| `add_zero`（`x + 0 → x`） | 10 |
| `mul_one`（`x * 1 → x`） | 10 |

（C 类地形另加 `add_zero_reverse`（30）、`mul2_to_add`（5）、`add_to_mul2`（4）三条，只在对应的反例里启用。）

**A 类：14 个随机地形上的三方一致性。** 参照物是 `code/exprlang.py` 里的**独立参考简化器**——它走的是嵌套元组（表达式树），与 driver 走的 IR op 图**没有任何共享代码**。地形由固定种子 `20260916` 生成（12 个随机 + 2 个手工构造），**种子先于结果冻结**：

| 检查 | 本机实测值 |
|---|---|
| 语义一致（两边求值相同） | 14 / 14，最大误差 0.000e+00 |
| 结构一致（读回表达式树逐节点比对） | 14 / 14 |
| 幂等（对输出再跑一遍，改写次数为 0） | 14 / 14 |
| 改写前的 IR 与表达式树求值器一致 | 14 个地形 |

**B 类：一次完整改写的账**（地形：`((x * 1 + 0) + 0) * (2 * 3)`）：

| 项 | 本机实测值 |
|---|---|
| 改写前 op 数 | 12 |
| 改写后 op 数 | 4 |
| 用到的迭代数 | 2 |
| 累计改写次数 | 4 |
| `maxIterations` 默认值 | 10 |

改写结果（程序打印的 IR，格式是本仓库迷你实现自己的）：

```text
%v180: f32 = arith.constant : () -> f32 {value = 6}
%v179: f32 = arith.addf(%arg0, %v180) : (f32, f32) -> f32
func.return(%v179)
```

12 → 4 个 op，剩下的正好是"一个常量 + 一次加法 + 返回"。**注意最后那个常量是 `6`**——它是 `2 * 3` 折叠出来的，说明常量折叠规则真的被套用了。

**C 类：不收敛必须被检出**：

| 检查 | 本机实测值 |
|---|---|
| C1 `x+0→x` 与 `x→x+0` 同时存在 | `converged=False iterations=10 applied=10`；IR op 数 **4 → 7** |
| C1 对照：去掉 `x→x+0` | `converged=True iterations=2 applied=1` |
| C2 上限抬到 500 | `converged=False iterations=500 applied=500`；IR op 数 3 → 4（**规模基本不变**） |
| C3 反证：把收敛判定改成永远 True | `c1_predicate` 从 True 变成 False（**判据不是恒真的**） |
| C4 把"未收敛"升级成异常 | `NonConvergenceError: 贪心改写在上限 10 轮内没有到达不动点` |
| C4 对照：去掉反向规则 | 同一个 driver 正常返回 |

C1 与 C2 是**同一件事的两个侧面**：C1 说"默认上限下会报未收敛"，C2 说"这不是因为上限太小"——两者的对照项分别是"去掉反向规则后收敛"与"抬高上限后仍不收敛"。

**边界：不动点恰好落在迭代上限的两侧**（专用规则 `peel_one_layer`，构造规则写死在程序里）：

| 嵌套层数 | 收敛 | 用掉轮数 | 改写次数 |
|---|---|---|---|
| 8 | 是 | 9 | 8 |
| 9 | **是** | **10** | 9 |
| 10 | **否** | 10（用尽） | 10 |
| 11 | 否 | 10（用尽） | 10 |

收敛侧的最大层数是 **9**、不收敛侧的最小层数是 **10**，两者相邻且分界正好等于 `maxIterations`。**这就是"上限不是装饰"的可执行证据。**

**第一版被实跑推翻的三项**：

1. **"左嵌套链每轮剥一层"是错的（第一版）。** 第一版用的是 `useTopDownTraversal=True`，实测 8/9/10/11 层**全部在第 2 轮收敛**——因为迷你实现的 op 列表顺序是"内层 op 在前"，于是"按列表顺序遍历"从内层开始，一趟就把整条链剥光了。改成**默认的 bottom-up**（`use_top_down=False`）之后，遍历顺序变成外层优先，每轮才真的只剥一层。**这条修正是理解"遍历方向为什么是一个配置项"的最好例子**：同一个规则集，换个方向，收敛轮数从 2 变成"层数 + 1"。
2. **"不收敛的 pass 只是多改写几次"是错的。** 第一版没有实现 DCE，互逆规则下 10 轮把 IR 从 4 个 op 撑到 **2050** 个。补上 DCE（官方注释里那句 "simple dead-code elimination before attempting to match"）之后，同一个地形只涨到 7。**"不收敛"的危害不只是结果不确定，还包括 IR 规模失控。**
3. **"`applied` 是有界的"是错的（第一版）。** 第一版 B 类表把累计改写次数猜成 5，**本仓库迷你 driver** 实测是 4。这个数被 table 逐行比对当场抓出——它本来是个容易顺着"直觉"写下的数字。

## 6. 小结与延伸

- 贪心 driver 的终止条件是**不动点**：某一轮**一次改写都没发生**。所以"什么都没改"是 success，不是异常。
- **两条退出路径**：到不动点 → success；轮数用尽而仍在改写 → **failure**。
- 默认值：`maxIterations = 10`、`fold = true`、`useTopDownTraversal = false`（**bottom-up**）、`regionSimplificationLevel = Aggressive`。
- **上限约束的是改写轮数，而轮数由规则的依赖深度决定**：本机实测"9 层收敛、10 层不收敛"，即默认预算最多允许 9 次有效改写。
- driver **自带 DCE，且在匹配之前做**；不实现它，不收敛的 pass 会把 IR 撑大（本机实测 4 → 2050）。
- 当前 API 名是 **`applyPatternsGreedily`**；旧名是 2024-12 起的**兼容别名**（PR #104649），**在 LLVM 21/22 上仍可用，不能在 `main` 上照样断言**。

**延伸**：

- 在这套结构上被改写的那些 op 从哪里来 → [01-ir-and-dialects](01-ir-and-dialects.md)
- 被改写出来的 memref IR 怎么继续往下走 → [03-progressive-lowering](03-progressive-lowering.md)
- canonicalize 与"规则集装满局部化简再跑贪心 driver"的关系 → `60-compilers/inductor`（同一套机制在 Inductor 侧的对应物）

## 编译与运行

```bash
cd 60-compilers/mlir/code
make 04_pattern_rewrite      # 只跑本篇
make run                     # 跑全部四篇
```

程序：`code/04_pattern_rewrite.py`（共享迷你 IR `code/miniir.py`、独立参照物 `code/exprlang.py`、外部事实源 `code/specs.py`）。

## 参考

- llvm-project 源码 `mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h` — 贪心语义、premature stop、返回值定义、`GreedyRewriteConfig` 四个默认值（取用日期 2026-09-16）
- MLIR PatternRewriter 文档 — worklist 语义、两种遍历方向：`https://mlir.llvm.org/docs/PatternRewriter/`
- llvm-project PR #104649 — "[mlir] Enable decoupling two kinds of greedy behavior."（merged 2024-12-20）：`https://github.com/llvm/llvm-project/pull/104649`；diff 里可见 `LLVM_DEPRECATED("Use applyPatternsGreedily() instead", …)` 与 `mlir/lib/Transforms/Canonicalizer.cpp` 的改动
- 以上每条来源都在 `code/specs.py` 里逐条对应并标注取用日期
