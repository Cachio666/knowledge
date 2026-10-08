# 110-communication/collectives 示例构建
#
# ★ 本目录与同域另外三个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束。**没有 `verify.py`**，也**没有** `meta_falsify.py`。
#
# ★★ **Chunk 1 已落盘**（提交日 2026-10-08）：三个编号程序与 `mutate.py` 都已就位 ⇒
#    `make run` 走**非空分母**分支，`SCRIPTS` 一行就是下面那三个。
#    01_algorithm_family.py   ↔ 01-algorithm-family.md  （算法族：ring / 递归倍增 / Bruck /
#                                                         二项树 / halving-doubling，各自的步数与每卡搬运量）
#    02_stages_and_costs.py   ↔ 02-stages-and-costs.md  （两段式：reduce-scatter + all-gather 的
#                                                         通信阶段划分 · 四个原语 · 为什么是 2(N−1)/N × S）
#    03_all_to_all_and_p2p.py ↔ 03-all-to-all-and-p2p.md（all-to-all 的算法与总搬运量口径 · P2P 原语）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**。）
#
# ★★★ **本主题的头号判据纪律**（★ `§3.2`）：
#    **A 类的参照必须是【真消息序列模拟】** —— 逐步穷举「**哪一步、谁发给谁、哪个块、多少元素**」，
#    把每一步的 `(sender, receiver, block, n_elem)` 真的枚举出来，**再**与闭式对拍。
#    ★★★ **不许把闭式用循环写一遍** —— 规划 `§D` 的 `D2` 就是这么写的
#      （`for _ in range(N-1): sent += PER`），那样的「相对差 `0.0e+00`」是**恒真**的
#      （`P12` 的现场）。★ 本主题的三个程序都**驱动真实数据**（按块搬运、按块累加）。
#
# ★★★ **一条本主题自己发现的【结构性事实】**（写进 `01` 的抬头与 `D.4`）：
#    **每卡搬运量**对「块索引写错」**不敏感**（`±` 只是重排块）——
#    所以「搬运量 == 闭式」这一条**过不了** `CONTRIBUTING:235` 的守卫；
#    **过守卫的是【结果正确性】那条**（`A.1` 的「与单进程参照逐位相同」）。
#
# ★★★ **本主题的禁区（回链，一行都不重写）**：
#    **`V(N) = 2(N−1)/N × S` 的推导与饱和** → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`；
#    **bisection 下界** / **all-to-all 跨切口量 `N·V/4`** / **分层 all-reduce 三段式** →
#      `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
#    **带宽口径** → `interconnect/01-bandwidth-conventions.md`；
#    **`algbw` vs `busbw` 与修正系数表** → `interconnect/04-measuring-bandwidth.md`。
#
# ★★ **本主题的三处【同名不同指】**（★ 写正文时必须消歧）：
#    · **`stage` / 阶段**：库内已有**四个**含义 ⇒ 一律写「**通信阶段**」或
#      「**reduce-scatter / all-gather 两段**」，**不得裸写 `stage`**（★ 第二篇的篇名里就有它）；
#    · **`ring`**：另有 ring 拓扑 / ring attention ⇒ 一律写「**ring allreduce**」全称；
#    · **`rank`**：另有「张量秩」⇒ **尽量写「设备」**。
#    ★ 另有一条**同名不同指**（引文层）：论文里 **`Trec dbl` 出现在两处** ——
#      allgather 的（带宽项 `(p−1)/p · n`）与 allreduce 的（带宽项 `n lg p`），**差一个 `lg p`**；
#      且 **`Bruck` 是 allgather 与 all-to-all(index) 的算法，不是 allreduce 的**。
#
# ★★ **一条引文读法纪律**（本主题唯一的载体是 MPICH 那篇 [C] 级论文）：
#    `pdftotext -layout` 把 **`⌈lg p⌉`** 提取成 **`dlg pe`**、**`⌊lg p⌋`** 提取成 **`blg pc`**、
#    而**裸 `lg p`** 表示**没有取整** ⇒ ★★★ **误判不会有任何报错**，只会让公式悄悄错掉。
#
# 用法：
#   make                    同 make run
#   make run                依次跑本目录全部编号程序（任一断言失败即非零退出）
#   make <程序名>           只跑一个（省略 .py 后缀）
#   make lint               AST 检查：本目录**全部** .py 里没有 import 目标框架
#   make mutate             ★ 叶子级变异测试（非编号工具 mutate.py；**不进 run**）
#   make lines              列出本目录各文件的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，CONTRIBUTING §2）───────────
#   specs.py        外部事实常量源：**每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`**。
#                   ★★ 本主题**只有一条来源**（Thakur / Rabenseifner / Gropp 2005，[C] 级，
#                     同行评审：是）⇒ **没有「B 级当 A 级」的风险**，但**另有**一条：
#                     **取整符号的三种形态**（见上）。
#                   ★★ 所有数值常量**一律 `< 2^53`**、**不做幂次字面量**、**字符串一律单行**
#                     （跨行字面量会被 `mutate.py` 的 `_patch()` 整条跳过 ⇒ 记入分母缺口）。
#   mutate.py       ★ **叶子级变异测试**（与三个编号程序同时落）。用 **AST** 从编号程序里读出
#                   `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，看断言是否捕获。
#                   **只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无 NCCL、无 InfiniBand / RoCE 网卡、无 NVLink**；
#   `python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域最强的一条限定语**：**「不是 NCCL，也不是真机」** ——
#     本主题四个程序验证的都只是**算法的代数**（步数 / 每卡搬运量 / 块索引 / 载荷口径）。
#   ★ **多进程只进 `../probes/`**（★ 实测：ring allreduce 的原型**连跑通都不稳**，约 80% 概率挂）
#     ⇒ **编号程序一律单进程**，保证 `make run` 快且稳。
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`&& FAIL: collectives`（**每个主题一个，不跨主题照抄**；
#     标记**只允许出现在 `print()` 的字符串里**，**不得**出现在任何 shell 命令行里
#     ⇒ ★★ **本文件里它只出现在【注释】中**，recipe 行里**没有**它）；
#     每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量**比较）。

PYTHON  ?= python3
SCRIPTS := 01_algorithm_family 02_stages_and_costs 03_all_to_all_and_p2p

.PHONY: all run lint mutate lines clean check-env

all: run

# ★★ 整个目标**必须写成一条 shell**（用 `; \` 串起来），不能拆成多行 recipe。
#   原因：make 的每一行 recipe 各起一个 shell，`exit 0` 只结束**那一行** ——
#   拆开写时，空分母分支里的 `exit 0` 之后，尾巴上的
#   `=== 全部 0 个编号程序退出码 0 ===` **照样会打印**，于是 `make run` 会输出
#   一句自相矛盾的「全绿」。
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
#   ★★ 本主题**特别需要这一条**：三篇文档与程序的抬头里会出现 `NCCL` / `torch` 这类名字
#      （讲**它们与本主题的分工**），但**一行都不 import 它们**。
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
#   0 = 全部捕获 · 1 = **有漏网**（逐条列出）· 2 = 读数不可信（仓库文件被改过）· 3 = 空分母。
#   ⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
#     不是 `mutate.py` 的。**要看退出码就直接 `python3 mutate.py`。**
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
