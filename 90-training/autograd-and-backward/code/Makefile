# 90-training/autograd-and-backward 示例构建
#
# ★ 本目录与同域另外四个主题是**同一形态**（CONTRIBUTING §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**，
#   也**没有** `meta_falsify.py`（那是 `50-performance` 这种**测量域**的形态，见域 README）。
#
# ★★ Chunk 0 状态：**本目录此刻还没有编号程序**（`SCRIPTS` 是空的）。
#    Chunk 1 会落盘三个，与同目录的三篇文档一一对应（`CONTRIBUTING` §2 的「编号即外键」）：
#      01_backward.py        ↔ 01-backward-from-scratch.md   （反向传播本身 + 数值梯度对拍的方法学）
#      02_checkpointing.py   ↔ 02-gradient-checkpointing.md  （重算换显存：√n 的存储 / ≈2n 的前向）
#      03_grad_lifecycle.py  ↔ 03-gradient-lifecycle.md      （训练循环里的梯度：累积 / 清零 / 裁剪）
#    届时本目标自动开始真跑。
#    ⇒ **此刻 `make run` 不构成任何证据** —— 它一条断言也没跑（见下面 `run` 里的空值分支）。
#    ⇒ `make mutate` 同样只走到**空分母自检**，以**说明性非零码 3** 退出（见 `mutate.py` 顶部）。
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
#                     **把「某实现的文档」当成「规范的文档」**（计划 §0.1 的 **P19**）；
#                     凡引用 `O0/O1/O2/O3`、`GradScaler`、`clip_grad_norm_` 这类
#                     「某实现的规格」，**必须标 B 级并写明是哪个实现**。
#   mutate.py       ★ **叶子级变异测试**（计划 §1.4 的决定）。用 **AST** 从编号程序里
#                   读出 `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，看断言是否捕获。
#                   **只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、不训练任何真模型**；`python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域不 import 任何第三方框架**（计划 §0.2；CONTRIBUTING §2 的「永不 import 目标框架」），
#     `numpy` 是唯一例外（见 requirements.txt）。
#   ★ **本主题不需要网络**：`make run` 只读本目录的文件。
#   ★ 同域的 `../probes/` 是**域级**探测目录，**不在本 Makefile 的扫描范围内**
#     （它不参与「文档数 = 编号程序数」的核对，口径见域 README 与 `../probes/README.md`）。

PYTHON  ?= python3
# ★★ Chunk 1 已落盘（2026-09-23）：
#    01_backward.py        ↔ 01-backward-from-scratch.md   （反向传播本身 + 数值梯度对拍的方法学）
#    02_checkpointing.py   ↔ 02-gradient-checkpointing.md  （重算换显存：√n 的存储 / ≈2n 的前向）
#    03_grad_lifecycle.py  ↔ 03-gradient-lifecycle.md      （训练循环里的梯度：累积 / 清零 / 裁剪 / 二阶）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**）。
SCRIPTS := 01_backward 02_checkpointing 03_grad_lifecycle

.PHONY: all run lint mutate lines clean check-env

all: run

# ★★ 整个目标**必须写成一条 shell**（用 `; \` 串起来），不能拆成多行 recipe。
#   原因：make 的每一行 recipe 各起一个 shell，`exit 0` 只结束**那一行** ——
#   拆开写时，空分母分支里的 `exit 0` 之后，尾巴上的
#   `=== 全部 0 个编号程序退出码 0 ===` **照样会打印**，
#   于是 Chunk 0 的 `make run` 会输出一句自相矛盾的「全绿」。
#   （`80-model-architectures/transformer/code/Makefile` 的同名目标有这个隐患，
#     只是那边 `SCRIPTS` 非空、走不到；本域 Chunk 0 **正是空的那一侧**，所以必须串起来。）
run: check-env
	@if [ -z "$(SCRIPTS)" ]; then \
		echo "=== 本主题当前有 0 个编号程序（Chunk 0 骨架）==="; \
		echo "    Chunk 1 会落盘 01/02/03 三个编号程序，本目标届时自动开始真跑。"; \
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
#   「已赋值」—— 默认值不会被填上，探测被整个跳过（CONTRIBUTING:196-199）。
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
		echo "     引号里的 ~ 是字面量，[ -x ] 必然失败。请改用 \$$HOME/env/bin/python。"; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
		|| { echo "!! 需要 python >= 3.9"; exit 1; }
	@$(PYTHON) -c 'import numpy' 2>/dev/null \
		|| { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }

# ★ `ast` 而非 grep：注释与文档串里出现 "torch" 这个词**不算违规**，真 import 才算。
#   ★ 扫的是本目录**全部** `*.py`（含 specs.py / mutate.py 与将来的编号程序）——
#     只扫编号程序会在 Chunk 0 退化成「扫 0 个文件」，那种 lint 是假绿。
#     （此刻扫到的是 specs.py 与 mutate.py 两个文件 —— 所以现在跑 lint 不是空转。）
#   ★ 本域比 `50-performance` 的 lint 更严：那边禁的是 torch/triton/numba/cuda 四类，
#     这边禁的是**全部深度学习框架与运行时**（计划 §0.2：本域**任何第三方框架都不 import**，
#     requirements.txt 里只有 numpy）。
#   ⚠ 某个 .py 有**语法错误**时，`ast.parse` 会抛异常、lint 以非零码结束 —— 这是**故意的**：
#     一个读不了的源文件不该被静默跳过。
lint: check-env
	@$(PYTHON) -c 'import ast,glob,sys; \
	BAN={"torch","tensorflow","jax","flax","triton","numba","cupy","vllm","transformers","flash_attn","tensorrt","deepspeed","bitsandbytes"}; \
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
#   3 = 空分母（Chunk 0 的骨架行为）。
#   ⇒ Chunk 0 的 `make mutate` **预期是 3，不是 0**；把 3 读成「失败」或把 0 读成
#     「通过了」都是错的 —— 见 `mutate.py` 顶部与计划 §3 Chunk 0 的验收 5。
#   ⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
#     不是 `mutate.py` 的。**要看退出码就用 `python3 mutate.py`。**
mutate: check-env
	@$(PYTHON) mutate.py || { rc=$$?; echo; \
		echo "★ mutate.py 的退出码 = $$rc 。它的含义：1 = 有漏网（见上「漏网」清单）；"; \
		echo "  2 = 仓库文件在本次运行中被改动，读数不可信；3 = 空分母（Chunk 0 骨架）。"; \
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
