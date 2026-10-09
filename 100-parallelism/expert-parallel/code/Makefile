# 100-parallelism/expert-parallel 示例构建
#
# ★ 本目录与同域另外五个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**，
#   也**没有** `meta_falsify.py`（那是 `50-performance` 这种**测量域**的形态，见域 README）。
#
# ★★ **Chunk 5 已落盘**（提交日 2026-10-08）：三个编号程序与 `mutate.py` 都已就位 ⇒
#    `make run` 走**非空分母**分支，`SCRIPTS` 一行就是下面那三个。
#    01_expert_sharding.py   ↔ 01-expert-sharding.md   （专家怎么分到设备 · ★★ 两次 all-to-all 的位置与语义）
#    02_load_imbalance.py    ↔ 02-load-imbalance.md    （★★★ EP 的负载账：发出量均衡、接收量不均）
#    03_ep_and_other_axes.py ↔ 03-ep-and-other-axes.md （EP × TP / DP 的组合 · EP 组的划分 · EP 的约束）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**。）
#
# ★★★ **本主题最容易写错的三条结论**（★★ **三篇文档的抬头各自回链它讲的那一条**）：
#    · ★★★ **本域最自然的 A 类判据「EP 切分 == 单卡」是【恒真】的** ——
#      token 分发是**纯索引搬运** ⇒ `max|diff|` **恰为 `0.000e+00`**
#      （`01` 的 `D.4` 逐档量它，与 `C.1`/`C.2` 的 `2.47`/`2.81` **并列**）。
#      ⇒ **A 类钉在【整数划分】**（`01` 的 `A.1`/`A.2`），**内容判据在 C 类**。
#    · ★★★ **发出量是【恒等式】、接收量是【统计量】**（`02`）——
#      「每源设备恒发 `T·top_k/EP`」对**每个种子、每档 EP** 都成立 ⇒ 报**一个数**即可；
#      而「接收量的 `max/min`」**逐种子不同** ⇒ **必须报多个种子的均值【与范围】**，
#      ★★ **并写明【初始分布】**（换分布会给出不同的比值）。
#    · ★★★ **`EP` 的约束只写【取到过】的那一条** —— 原文（**Megatron-LM MoE README，
#      [B] 级实现文档，不是规范**）逐字说的是 **`EP` 与 `DP`** 的关系；
#      ★★ 本库**没有**在任何载体上取到「**专家数与 EP 度之间的大小关系**」这类规则 ⇒
#      `03` 的 `B` 类（`EP > E`）是**本库设计的一个情景**，**不是**转述。
#    ★★ **另有两条硬边界**：**容量因子的定义与语义**（定义式 / 取整口径 /
#      dropping vs No-Token-Left-Behind / 丢弃率口径）→ `80-model-architectures/moe/02-capacity-factor.md`
#      （**全篇 563 行**）；**all-to-all 的通信量分析** → `110-communication/collectives`）
#      与 `20-gpu-architecture/interconnect/03`（★ `40-kernels/moe-kernels/01:243` 已逐字让渡）。
#
# ★★★ **本主题最要紧的三处形态风险**（每一条都对应一个程序里的真实现场）：
#    · **`01` 的 A 类不是「实现 vs 参照」** —— 替代方案在本组形状下**恒真**
#      ⇒ 本程序把 A 类钉在**整数划分**上，并让 `D.4` **只报告**那个 `0.0`（**不是判据**）。
#    · **`02` 的两类判据写法必须不同**（恒等式 vs 统计量）——
#      把 `B` 类的数当恒等式用，是本主题最容易犯的错。
#    · **`03` 的 A 类把两个条件写进【同一条】**（`tp = 1` 恰为 `0` **且** `tp > 1` 严格 `> 0`）——
#      ★★ 少了后半条，前半条就是**恒真**的伪断言。
#    · **每条 A 类都配一条「真的会改变被判量」的守卫**（`CONTRIBUTING:235`）：
#      `01` 的 `A.1g`（`ceil` 分配 / 越界）、`A.2g`（拿不整除的那组过条件）；
#      `02` 的 `A.1g`（把发出量换成接收量）；`03` 的 `A.1g`（TP 少加几片）。
#
# 用法：
#   make                    同 make run
#   make run                依次跑本目录全部编号程序（任一断言失败即非零退出）
#   make <程序名>           只跑一个（省略 .py 后缀）
#   make lint               AST 检查：本目录**全部** .py 里没有 import 目标框架
#   make mutate             ★ 叶子级变异测试（非编号工具 mutate.py；**不进 run**）
#   make lines              列出本目录各 .py 的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，CONTRIBUTING §2）───────────
#   specs.py        外部事实常量源：本主题**真的会断言**的那批常量。
#                   ★ 每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`；
#                   ★★ **来源必须分级并写明是哪一级** —— 本域的头号错误模式是
#                     **把「某实现的文档」当成「规范的文档」**（B 级当 A 级）
#                     ⇒ 本主题的两条来源分属 **[C] 论文 ×2 + [B] 实现文档 ×1**。
#                   ★★ 本域所有数值常量**一律 `< 2^53`**（`mutate.py` 的浮点变异是 `v + 1.0`），
#                     且**不做幂次字面量**（`_walk()` 不下钻 `BinOp`）。
#   mutate.py       ★ **叶子级变异测试**（**Chunk 5 已落**，与三个编号程序同时）。
#                   用 **AST** 从编号程序里读出 `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，
#                   看断言是否捕获。**只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无多卡、无 NCCL、无真实网络**；`python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域不 import 任何第三方框架**（`CONTRIBUTING` §2 的「永不 import 目标框架」，
#     原文的措辞比 torch 宽 —— 本域按宽的口径执行），`numpy` 是唯一例外（见 requirements.txt）。
#   ★★ **本主题的三个程序【都要】`numpy`** —— `01`/`02`/`03` 都要矩阵乘
#     （`x @ Wg` 的路由打分、`x @ We[e]` 的专家前向）⇒ **没有“某一个程序不需要它”这一档**：
#     `requirements.txt` 与下面的 `check-env` 按**三个程序的并集**要求它（★ 这里的并集
#     与「三个都要」**恰好重合** —— 本主题**不写**「单跑某个程序不需要 numpy」那种措辞）。
#     ★ 本主题的**路由**只是 `argsort` 取 `top_k`，**不含 gating 权重**（等权 `1/top_k`）
#       —— 那是 `80-model-architectures/moe/01,03` 的正文。
#   ★★ **通信被实现为一个显式的张量搬运**（all-to-all = **索引搬运的计数**）——
#     本域的知识对象是**切分策略**，不是通信原语的实现
#     （通信算法的正文在 `110-communication/collectives` —— ★ ）。
#     ⇒ 一切程序验证的是**切分的代数**（划分算术 / 搬运条目数 / 负载分布 / 组划分），
#       **不是**任何关于真实集群性能的结论。★ 强制限定语：**「不是 NCCL，也不是真机」**。
#   ★★ **本主题的定位**（与 `requirements.txt` **抬头那段**同一句，两处同步）：
#      · **专家怎么分到设备**（每设备 `E/EP` 个）+ ★★ **两次 all-to-all** 的位置与语义；
#      · ★★★ **EP 的负载账**：**发出量均衡（恒等式）、接收量不均（统计量）**；
#      · **EP 与 TP / DP 的组合**、**EP 组的划分**、**`EP` 的约束**（只写取到过的那一条）。
#   ★★ **本主题的禁区**：
#     **容量因子的定义与语义** → `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）；
#     **aux loss 的式子 / 路由 gating 的机制** → `80-model-architectures/moe/01,03`；
#     **all-to-all 的通信量分析** → `110-communication/collectives`（与
#     `20-gpu-architecture/interconnect/03`；**kernel 侧的 permute / unpermute / grouped GEMM** →
#     `40-kernels/moe-kernels/02,03` 与 `00-meta/glossary.md` 的 `grouped GEMM` 条；
#     **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
#     `10-foundations/performance-models/02:417`（★ **不是 `:223`**）。
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`\|\| FAIL: expert-parallel`（**每个主题一个，不跨主题照抄**；
#     标记**只允许出现在 `print()` 的字符串里**，**不得**出现在任何 shell 命令行里
#     ⇒ ★★ **本文件里它只出现在【注释】中**，且**转义写成 `\|\|`** —— recipe 行里**没有**它）；
#     每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量比较**）。
#   ★ 同域的 `../probes/` 是**域级**探测目录（**只打印、不断言**），**不在本 Makefile 的扫描范围内**。

PYTHON  ?= python3
SCRIPTS := 01_expert_sharding 02_load_imbalance 03_ep_and_other_axes

.PHONY: all run lint mutate lines clean check-env

all: run

# ★★ 整个目标**必须写成一条 shell**（用 `; \` 串起来），不能拆成多行 recipe。
#   原因：make 的每一行 recipe 各起一个 shell，`exit 0` 只结束**那一行** ——
#   拆开写时，空分母分支里的 `exit 0` 之后，尾巴上的
#   `=== 全部 0 个编号程序退出码 0 ===` **照样会打印**，
#   于是 `make run` 会输出一句自相矛盾的「全绿」。
#   （`80-model-architectures/transformer/code/Makefile` 的同名目标有这个隐患；
#    `90-training/autograd-and-backward/code/Makefile` 已按此串起来，本文件照搬。）
run: check-env
	@if [ -z "$(SCRIPTS)" ]; then \
		echo "=== 本主题当前有 0 个编号程序 ==="; \
		echo "    ★ 此刻 make run **不构成任何证据** —— 它一条断言也没跑，"; \
		echo "      不要把它当成「绿」写进任何文档。"; \
		exit 0; \
	fi; \
	rm -f .last_out; \
	for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; echo; \
			echo "!! $$s.py 非零退出 —— 断言失败（详情见上）"; \
			rm -f .last_out; exit 1; }; \
		cat .last_out; \
		echo; \
	done; \
	rm -f .last_out; \
	echo "=== 全部 $(words $(SCRIPTS)) 个编号程序退出码 0 ==="; \
	echo "    （失败标记的纪律在**每个程序内部**：它只打印本主题自己的 ASSERT_FAIL_MARK，"; \
	echo "      并有一条「标记自检」断言守着；跨主题照抄标记由 mutate.py 的 AST 读取挡住。）"

# ★ 这个目标的第一个动作是**空值检查**，不是靠 `PYTHON ?=` 的默认值兜底。
#   `?=` 的语义是「**未被赋值时**才赋值」，而命令行传 `PYTHON=`（**空值**）也算
#   「已赋值」—— 默认值不会被填上，探测被整个跳过（`CONTRIBUTING:196-199`）。
check-env:
	@if [ -z "$(PYTHON)" ]; then \
		echo "!! PYTHON 是空的。"; \
		echo "   make 的 ?= 在命令行传空值时不会填默认值，所以这里必须再查一次。"; \
		echo "   请显式给一个解释器： make run PYTHON=python3"; \
		exit 1; \
	fi
	@if ! command -v "$(PYTHON)" >/dev/null 2>&1 && [ ! -x "$(PYTHON)" ]; then \
		echo "!! 找不到解释器: $(PYTHON)"; \
		echo "   ★ make **不做** ~ 展开：若你写 PYTHON=~/env/bin/python，"; \
		echo "     引号里的 ~ 是字面量，[ -x ] 必然失败。请改用 $$HOME/env/bin/python。"; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
		|| { echo "!! 需要 python >= 3.9"; exit 1; }
	@$(PYTHON) -c 'import numpy' 2>/dev/null \
		|| { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }

# ★ `ast` 而非 grep：注释与文档串里出现框架名**不算违规**，真 import 才算。
#   ★ 扫的是本目录**全部** `*.py`（含 specs.py / mutate.py 与本主题的编号程序）。
#   ★★ 本域比 `50-performance` 的 lint 更严：那边禁的是 torch/triton/numba/cuda 四类，
#      这边禁的是**全部深度学习框架与运行时**（本域**任何第三方框架都不 import**）。
#   ★★ 本主题**特别需要这一条**：三个程序讲的是 **EP 的切分与负载账** ——
#      文档串里会出现框架的 MoE 并行 API 名字（`moe_token_dispatcher` 等），但**一行都不 import 它们**。
#   ⚠ 某个 .py 有**语法错误**时，`ast.parse` 会抛异常、lint 以非零码结束 —— 这是**故意的**。
lint: check-env
	@$(PYTHON) -c 'import ast,glob,sys; \
	BAN={"torch","tensorflow","jax","flax","triton","numba","cupy","vllm","transformers","flash_attn","tensorrt","deepspeed","bitsandbytes","apex"}; \
	F=sorted(glob.glob("*.py")); \
	hit=[(f,n.lineno,m) for f in F for n in ast.walk(ast.parse(open(f,encoding="utf-8").read(),f)) \
	     if isinstance(n,(ast.Import,ast.ImportFrom)) \
	     for m in ([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or ""]) \
	     if m.split(".")[0] in BAN]; \
	[print("!! 违规 import: %s:%d  import %s" % (f,ln,m)) for f,ln,m in hit]; \
	print("lint: 扫了 %d 个 .py 文件（%s），违规 %d 处" % (len(F), ", ".join(F) if F else "-", len(hit))); \
	print("lint: numpy 是唯一允许的第三方库（见 requirements.txt）"); \
	sys.exit(1 if hit else 0)'

# ★ **不进 `run`**：变异测试要跑很多副本，慢；而且它的结论是「元层」的。
# ★ 退出码的含义：
#   0 = 全部捕获 · 1 = **有漏网**（逐条列出）· 2 = 读数不可信（仓库文件被改过）·
#   3 = 空分母。
#   ⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
#     不是 `mutate.py` 的。**要看退出码就用 `python3 mutate.py`。**
mutate: check-env
	@$(PYTHON) mutate.py || { rc=$$?; echo; \
		echo "★ mutate.py 的退出码 = $$rc 。它的含义：1 = 有漏网（见上「漏网」清单）；"; \
		echo "  2 = 仓库文件在本次运行中被改动，读数不可信；3 = 空分母。"; \
		exit $$rc; }

lines:
	@$(PYTHON) -c 'import os; [print("%-32s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith((".py", ".txt", ".md"))]'

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
$(SCRIPTS): check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .last_out .run_summary
