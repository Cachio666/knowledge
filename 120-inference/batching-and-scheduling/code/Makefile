# 120-inference/batching-and-scheduling 示例构建
#
# ★ 本目录与同域另外四个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束。**没有 `verify.py`**，也**没有** `meta_falsify.py`。
#
# ★★ **本目录的编号程序已随 Chunk 2 落盘**（提交日 2026-10-09）——
#    `SCRIPTS` 已填 **3 个**（`01_iteration_level_scheduling` / `02_chunked_prefill` /
#    `03_preemption_and_fairness`），`make run` 走**非空分母**分支，跑满三个程序。
#    ★ 空分母分支仍然保留（`SCRIPTS` 为空时的自我保护），但**本主题走不到它**。
#
#    本主题的定位：**迭代级调度**的机制与收益来源（continuous batching）·
#      **chunked prefill 的分块数权衡**（`k* = ⌈t_p/t_d⌉`）· **抢占与公平**。
#
#   ★★ 本主题的边界（**只写增量，不重推既有正文**）：
#     · **continuous batching / chunked prefill 的【定义】** → `10-foundations/performance-models/01:251`；
#     · **prefill/decode 的 kernel 差异**（两者的 `AI` 差 120 倍）→ `40-kernels/attention/05:229-236`；
#     · ★★ **静态 batching 的填充浪费 `1 − 均值/最大值`** → `40-kernels/moe-kernels/02-grouped-gemm.md:89`
#       （★ 本域**只回链 + 写「同式在调度里的含义不同」**，**不重算** —— **错误模式 `P37`** / `§0.7 D5`）。
#   ★★ 判据纪律：★ **收益来自【长度方差】，不来自【到达过程】**（`§0.7 D3` 的**定性结论**；
#      ★★ 本主题一律用**程序自身**的数：等长 `1.0000×` / 均匀 `1.6460×` / 指数 `2.8135×`；
#      泊松与突发到达均 `1.00–1.03×`）——
#      ★★ 任何「X 带来 N 倍」的句子**必须写明【基线】**（`§0.7 D4`：同一件事的两个加速比
#      差 **16.9 倍** —— ★ 这是**程序自己**的数，**不是规划探针的采样**）。
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无 vLLM / SGLang / TensorRT-LLM、无 transformers**；
#   `python3` = 3.13.5（★ **默认解释器没有 `torch`**）、`numpy` = 2.4.4（已装）。
#   ★ **本域最强的一条限定语**：**「不是任何真实引擎，也不是真机」** ——
#     本主题验证的只是**调度的算术与不变量**（makespan / 下界 / 分块数），不是任何真实系统的性能。
#   ★ **本域【不设】`probes/`**（★ 没有「必须真跑多进程」的知识对象，与 `110-communication` 不同）
#     ⇒ `code/` **自包含**，编号程序一律单进程。
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`-- FAIL: batching-and-scheduling`（**每个主题一个，不跨主题照抄**；
#     标记**只允许出现在 `print()` 的字符串里**，**不得**出现在任何 shell 命令行里
#     ⇒ ★★ **本文件里它只出现在【注释】中**，recipe 行里**没有**它）；
#     每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量**比较）。
#
# 用法：
#   make                    同 make run
#   make run                依次跑本目录全部编号程序（任一断言失败即非零退出）
#   make <程序名>           只跑一个（省略 .py 后缀）
#   make lint               AST 检查：本目录**全部** .py 里没有 import 目标框架
#   make mutate             ★ 叶子级变异测试（非编号工具；**不进 run**）
#   make lines              列出本目录各文件的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，`CONTRIBUTING` §2）───────────
#   specs.py        外部事实常量源：**每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`**。
#                   ★★ 每条常量**都必须有消费者**（无消费者的常量是 `make mutate` 的漏网候选）。
#                   ★★ 所有数值常量**一律 `< 2^53`**、**不做幂次字面量**、**字符串一律单行**
#                     （跨行隐式拼接会被 `ast` 折成一个 `Constant` ⇒ `mutate.py` 静默计入「跳过」）。
#   mutate.py       ★ **叶子级变异测试**（与编号程序同时落；**本目录已落**）。用 **AST** 从编号程序里
#                   读出 `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，看断言是否捕获（只在副本上做）。
#                   盲区：①只变异常量 ②负字面量不进分母 ③前缀式检查漏网 ④`_walk` 不下钻 `BinOp`
#                         ⑤`None` 计为「跳过」 ⑥浮点变异是 `v + 1.0`（大浮点吞掉它 ⇒ **常量 < 2^53**）
#                         ⑧**「整体钉住」代替不了「元素钉住」**（每个元组/列表常量都要逐元素 `==`）
#   Makefile / requirements.txt   构建入口

PYTHON  ?= python3
SCRIPTS := 01_iteration_level_scheduling 02_chunked_prefill 03_preemption_and_fairness

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
	echo "      并有一条「标记自检」断言守着。）"

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

# ★ 默认 `python3` **只有 numpy 之外的第三方包为零** —— 本域程序**不 import torch**
#   （实测：默认解释器 `import torch` 会 `ModuleNotFoundError`），故这里**只探测 numpy**。
# ★ `ast` 而非 grep：注释与文档串里出现框架名**不算违规**，真 import 才算。
#   ★ 扫的是本目录**全部** `*.py`（含 specs.py / mutate.py 与本主题的编号程序）。
#   ★★ 本主题**特别需要这一条**：三篇文档与程序的抬头里会出现 `vLLM` / `ORCA` 这类名字
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
