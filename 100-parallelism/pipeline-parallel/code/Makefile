# 100-parallelism/pipeline-parallel 示例构建
#
# ★ 本目录与同域另外五个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**，
#   也**没有** `meta_falsify.py`（那是 `50-performance` 这种**测量域**的形态，见域 README）。
#
# ★★ **Chunk 3 已落盘**（2026-09-23）：三个编号程序与 `mutate.py` 都已就位 ⇒
#    `make run` 走**非空分母**分支，`SCRIPTS` 一行就是下面那三个。
#    01_bubbles.py      ↔ 01-microbatches-and-bubbles.md（微批 / 气泡率 / **两个口径**）
#    02_schedules.py    ↔ 02-schedules.md         （GPipe / 1F1B / interleaved）
#    03_balance.py      ↔ 03-memory-balance.md    （各 stage 的**负载均衡**）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**。）
#
# ★★★ **本主题最容易写错的一条结论（三个程序都在 `preflight()` 里回链它）**：
#    **1F1B 换的是【显存】，不是气泡率。**
#    原文逐字（`specs.MEGATRON_1F1B_SAME_BUBBLE_QUOTE`）：
#      `The time spent in the bubble is the same for this new schedule`
#    ⇒ ★★ **常见转述把 1F1B 说成「气泡从 `(p−1)/(m+p−1)` 降到 `(p−1)/m`」，那是错的** ——
#      那两个公式的差别是**分母口径**（GPipe 除以**总时间**、Megatron 除以**理想时间**），
#      **不是任何改善**（`01` 的 `B.2` 用**换算恒等式** `b_m == b_g/(1−b_g)` 钉住这一点）。
#    ★★★ **改善气泡的是 interleaved**（`§2.2.2`）—— 代价是**通信量增加**（`02` 的 `B.2`/`C.1`）。
#    ★ 另有两条硬边界：**重算换显存**的机制（`O(√n)` vs `≈2n`）→
#      `90-training/autograd-and-backward/02-gradient-checkpointing.md`（本域**只引**它压的是什么量、
#      **只写**「PP 下用重算做显存均衡的取舍」）；**transformer block 的结构** →
#      `80-model-architectures/transformer/`；**点对点通信的实现**与**通信-计算重叠** →
#      `110-communication/`（**未建成**）。
#
# ★★★ **本主题最要紧的一处形态风险**：三个程序**全是模拟器**
#    （`01` 是离散事件模拟、`02` 是两种调度的模拟、`03` 是纯算术）⇒ A 类判据极易写成**自证**。
#    落法（三条，逐条对应一个程序里的真实现场）：
#      · **模拟器与闭式必须独立** —— `01` 的 `D.3` 用 `inspect.getsource` 证明
#        **闭式 `(m+p−1)` 的任何形式都不出现**在 `simulate()` 的源码里；`02` 的 `D.4` 证明
#        **两个调度器各自生成 `order`、不共用时间函数**；
#      · **每条 A 类都配一个「真的会改变被判量」的守卫**（`CONTRIBUTING:235`）——
#        ★★★ **本主题的第一版把守卫写成了「对调 `t_f`/`t_b`」（`01`）与「层数倒序」（`03`），
#        两者都【恒真】**（前者因闭式对称、后者因首尾 embedding 对称）⇒ 已换成
#        **「漏 fwd 串行」（`01` 的 `C.2`）** 与 **「embedding 归属改变」（`03` 的 `C.2`）**；
#      · **参照实现不得调用被测实现的任何函数** —— `01` 的 `D.3` 与 `03` 的 `D.2` 是源码级证据。
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
#                     **把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
#                   ★★ 本域所有数值常量**一律 `< 2^53`**（`mutate.py` 的浮点变异是 `v + 1.0`），
#                     且**幂次写成十进制字面量**（`_walk()` 不下钻 `BinOp`）。
#   mutate.py       ★ **叶子级变异测试**（**Chunk 3 已落**，与三个编号程序同时）。
#                   用 **AST** 从编号程序里读出 `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，
#                   看断言是否捕获。**只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无多卡、无 NCCL、无真实网络**；`python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域不 import 任何第三方框架**（`CONTRIBUTING` §2 的「永不 import 目标框架」，
#     原文的措辞比 torch 宽 —— 本域按宽的口径执行），`numpy` 是唯一例外（见 requirements.txt）。
#   ★★ **通信被实现为一个显式的张量搬运**（`all_reduce = Σ局部`、`all_gather = concat`）——
#     本域的知识对象是**切分策略**，不是通信原语的实现（通信算法的正文在 `110-communication`）。
#     ⇒ 一切程序验证的是**切分的代数**（等价性 / 通信次数 / 字节账 / 气泡率 / 负载分布），
#       **不是**任何关于真实集群性能的结论。★ 强制限定语：**「不是 NCCL，也不是真机」**。
#   ★★ **本主题只写**：PP 的**调度与气泡**：`1F1B` 的时序、气泡率的两个口径、stage 间的显存均衡
#   ★★ **本主题的禁区**：
#     **通信-计算重叠的机制**（双缓冲 / 分块流水 / 异步落盘）与**重算换显存**的机制
#     —— 正文回链 `110-communication` 的 `overlap` 主题（未建成）与
#     `90-training/autograd-and-backward/02-gradient-checkpointing.md`
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`== FAIL: pipeline-parallel`（**每个主题一个，不跨主题照抄**；
#     标记**只允许出现在 `print()` 的字符串里**，**不得**出现在任何 shell 命令行里）；
#     每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量比较**）。
#   ★ 同域的 `../probes/` 是**域级**探测目录（**只打印、不断言**），**不在本 Makefile 的扫描范围内**。

PYTHON  ?= python3
SCRIPTS := 01_bubbles 02_schedules 03_balance

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
#   ★★ 本主题**特别需要这一条**：三个程序是**离散事件模拟**（`1F1B` 的时序与气泡）—— 文档串里会出现框架流水线 API 的名字，
#      但**一行都不 import 它们**。
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
# ★ Chunk 0 **尚未落 `mutate.py`** ⇒ 此刻本目标会以「没有 mutate.py」失败，**这是预期行为**。
# ★ 退出码的含义（Chunk 1 起）：
#   0 = 全部捕获 · 1 = **有漏网**（逐条列出）· 2 = 读数不可信（仓库文件被改过）·
#   3 = 空分母。
#   ⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
#     不是 `mutate.py` 的。**要看退出码就用 `python3 mutate.py`。**
mutate: check-env
	@$(PYTHON) mutate.py || { rc=$$?; echo; \
		echo "★ mutate.py 的退出码 = $$rc 。它的含义：1 = 有漏网（见上「漏网」清单）；"; \
		echo "  2 = 仓库文件在本次运行中被改动，读数不可信；3 = 空分母。"; \
		echo "  ★ 若本主题还没有 mutate.py（Chunk 0），也会给 2 —— 那是「文件不存在」，不是读数不可信。"; \
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
