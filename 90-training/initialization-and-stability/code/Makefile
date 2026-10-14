# 90-training/initialization-and-stability 示例构建
#
# ★ 本目录与同域另外四个主题是**同一形态**（CONTRIBUTING §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**，
#   也**没有** `meta_falsify.py`（那是 `50-performance` 这种**测量域**的形态，见域 README）。
#
# ★ 本目录与 `../autograd-and-backward/code/` 是**同一套形态**（Chunk 1 已落盘）：
#   本 Makefile 与 `mutate.py` **从 `../optimizers/code/`（Chunk 3，最新）复制过来**并只改与
#   本主题有关的常量（`SCRIPTS` 的名字、`mutate.py` 的临时目录前缀、路径常量）。
#   **没有第二个模板。**
#
# ★★ Chunk 4 已落盘（2026-09-23）：
#    01_init.py           ↔ 01-initialization.md               （He / Xavier / μP；★ 含 Y1–Y3 三条订正）
#    02_clip_monitor.py   ↔ 02-clipping-and-monitoring.md      （梯度裁剪 · **范数监控的判据**）
#    03_spikes.py         ↔ 03-instability-and-spikes.md       （loss spike：现象 / 已证伪 / 仍是假设）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**）。
#    ★ 本主题**没有**上游指名的文件名（那是 `optimizers/01` 的待遇）。
#    ★★ **本主题的禁区最密**（计划 §2 的**四行**）：warmup 的机制解释、残差流的范数随深度增长、
#       误差沿深度的传播、MoE 的 aux loss 尺度 —— **三个程序里各有一段回链声明**。
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
#                     本主题的三条链**分属三级**：He / Glorot / μP 的公式与论断是
#                     **[C] 论文**（逐条注明是否确认同行评审），`torch.nn.init.*` 与
#                     `clip_grad_norm_` 的签名/上下界/返回值是 **[B] PyTorch 2.14 的文档**，
#                     而本机量出的方差比值/斜率/更新量比值是**推导值**（**不进本文件**）。
#                   ★★ **本主题的三条「与常见转述不一致」**（计划 §0.7-C 的 **Y1/Y2/Y3**）
#                     全靠这里的字面量钉住：那个错的 arXiv 编号、那个原文里不存在的名字、
#                     以及那个错的分布与分母。
#   mutate.py       ★ **叶子级变异测试**（计划 §1.4 的决定）。用 **AST** 从编号程序里
#                   读出 `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，看断言是否捕获。
#                   **只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、不训练任何真模型**；`python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域不 import 任何第三方框架**（计划 §0.2；CONTRIBUTING §2 的「永不 import 目标框架」），
#     `numpy` 是唯一例外（见 requirements.txt）。
#   ★★ **本主题的一个特有约束**：`01` 的 `kaiming_uniform_` / `xavier_uniform_` /
#     `calculate_gain`、`02` 的 `clip_grad_norm_` 的返回值与两个默认值，**都是某个实现**的规格，
#     但**三个程序一行 torch 也不 import** —— 它们把那些规格当作 **specs.py 里的 [B] 常量**用，
#     按**该实现文档里逐字写的规则**在 numpy 上重演一遍。
#     ⇒ 这些数**不是「PyTorch 在本机跑出来的」**，而是「按 PyTorch 文档的规则在本机推出来的」
#       —— **推导值**。这个区别写在每个程序的 `main()` 抬头与文档的 §5 里。
#   ★★ **本主题的禁区（四行，计划 §2）**：`warmup` 的机制解释（`80-…/normalization/02`）、
#     残差流的范数随深度增长（`80-…/transformer/01`）、误差沿深度的传播
#     （`130-…/07-outliers-and-propagation`）、MoE 的 aux loss 尺度（`80-…/moe/03`）—— 三个程序只写
#     **工程判据与设定**，并且**每个程序里都有一段回链声明**（不进正文，只留接口）。
#   ★ **本主题不需要网络**：`make run` 只读本目录的文件。
#   ★ 同域的 `../probes/` 是**域级**探测目录，**不在本 Makefile 的扫描范围内**。
#   ★★ **本主题与 `probes/` 的关系最紧**：`D9`（He vs Xavier 的方差）、`D14`（裁剪闭式）、
#     `D15`（spike 的反直觉形态）三条探测值就是本主题三个程序要复现/否证的靶子 ——
#     ★ 而 `probes/README.md` 的「缺口与不可用的数」列着它们**不能**支撑什么。

PYTHON  ?= python3
SCRIPTS := 01_init 02_clip_monitor 03_spikes

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

# ★ `ast` 而非 grep：注释与文档串里出现框架名**不算违规**，真 import 才算。
#   ★ 扫的是本目录**全部** `*.py`（含 specs.py / mutate.py 与三个编号程序）。
#   ★ 本域比 `50-performance` 的 lint 更严：那边禁的是 torch/triton/numba/cuda 四类，
#     这边禁的是**全部深度学习框架与运行时**（计划 §0.2：本域**任何第三方框架都不 import**）。
#   ★★ 本主题**特别需要这一条**：`01`/`02` 讲的是 `torch.nn.init.*` 与
#      `torch.nn.utils.clip_grad_norm_` 的规格 —— 注释与文档串里**满是
#      `kaiming_uniform_` / `xavier_uniform_` / `calculate_gain` / `clip_grad_norm_`
#      这些名字**，只有 AST 能分清「提到」与「import」。
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
