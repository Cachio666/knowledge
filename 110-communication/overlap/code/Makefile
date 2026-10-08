# 110-communication/overlap 示例构建
#
# ★ 本目录与同域另外三个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束。**没有 `verify.py`**，也**没有** `meta_falsify.py`。
#
# ★★ **Chunk 4 已落盘**：本目录现有 **3 个编号程序** ⇒ `SCRIPTS` 非空，
#    `make run` **真跑**三个程序（★ 走不到空分母分支）。
#
#    ★★★ **本主题【有真 A 类】**（★ 与同域的 `nccl` 相反）：它的知识对象是
#      **两条时间轴的调度**（计算引擎 + 通信/落盘引擎）⇒ **本机可以真做**，做法是
#      **时隙模拟 vs 独立闭式**（★ 两条独立路径）：
#        · `01`（机制）：`sim_ticks` **逐拍**排两条引擎 ⇒ 与 `(C+T)/K + (K−1)·max(C/K,T/K)` 对拍；
#        · `02`（能藏多少）：**三条免费守卫**（`K=1`⇒无重叠 / `K→∞`⇒下界 / 中间值不下穿）
#          + **可藏上限 = `min(C,T)`**；
#        · `03`（异步落盘）：**同一个模型、第三组参数**（`SAVE_C_S_GRID`）。
#      ★★ **三个程序的 A 类都钉在【模拟器】上**（照 `§1.3.2b`：参照的控制流不得依赖被测的量）。
#
#    ★★★ **本主题的判据模型是 `[E]` 级（本库推导）** —— 全库**没有**「能藏多少」的一手公式
#      （计划 `§C.4`）。★★ **本主题的 C 类是【主力】**：**错式 `C + T/K` 在 `T > C` 时低于物理下界**
#      必须失败（见计划 `§D` 的 `D4` / `P35`）—— 三个程序各判一次。
#
#    本主题的定位：通信-计算重叠的**机制**（双缓冲 / 通信分块 / 异步发起）、**「能藏多少」的判据**、**异步落盘**
#
#   ★ 本主题的禁区：**异步落盘的【存储侧】机制** → `90-training/checkpointing/01,02,03`（本主题只写「与计算重叠的那一半」）；**点对点通信的实现** → 本域 `collectives`
#   ★★ 判据纪律：本主题 **`[E]` 级**的模型 **不得写成「研究表明」**（计划 `§C.5` 的 `F8`）；★★ 凡引外部载体**必带 URL + 版本 + 抓取日**
#
#   ★★ **本主题的三个编号程序【只用标准库】**（`ast` / `collections` / `inspect` /
#     `sys` / `textwrap` / `fractions`）—— ★ 它们**不 import `numpy`**（本主题是精确有理数与
#     整数计数的比对，没有数组运算）⇒ ★★ `check-env` **不探测 `numpy`**（`requirements.txt` 也不列它）——
#       否则**没装 numpy 的读者**会在一个**根本不用 numpy** 的主题上**硬失败**（依赖与用途不符）。
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无 NCCL、无 InfiniBand / RoCE 网卡、无 NVLink**；
#   `python3` = 3.13.5（★ 本主题**不需要** `numpy`，`check-env` 也不探测它）。
#   ★ **本域最强的一条限定语**：**「不是 NCCL，也不是真机」**。
#   ★ **多进程只进 `../probes/`**（实测：ring allreduce 原型连跑通都不稳，约 80% 概率挂）
#     ⇒ **编号程序一律单进程**，保证 `make run` 快且稳。
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`+= FAIL: overlap`（**每个主题一个，不跨主题照抄**；
#     标记**只允许出现在 `print()` 的字符串里**，**不得**出现在任何 shell 命令行里）；
#     每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量**比较）。
#
# 用法：
#   make        同 make run      make run      依次跑本目录全部编号程序
#   make <程序名>                 make lint     AST 检查：没有 import 目标框架
#   make mutate                   make lines    行数      make clean
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，`CONTRIBUTING` §2）───
#   specs.py        外部事实常量源：**每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`**
#                   ★★ 每条常量**都必须有消费者**（否则是 `make mutate` 的漏网候选）
#                   ★★★ **所有字符串常量一律写成【单行】**：跨行隐式拼接会被 `ast` 折成一个
#                     `Constant`、行号跨行 ⇒ **静默计入「跳过」**（`mutate.py` 盲区⑦；
#                     同域 `network-topology` 第一版因此**29 个叶子全被跳过**）⇒ **本主题 跳过 = 0**。
#   mutate.py       ★ 叶子级变异测试（**与编号程序同时落**）。用 **AST** 读出 `ASSERT_FAIL_MARK`（不得写死）。
#                   盲区：①只变异常量 ②负字面量不进分母 ③前缀式检查漏网 ④`_walk` 不下钻 `BinOp`
#                         ⑤`None` 计为「跳过」 ⑥浮点变异是 `v + 1.0`（大浮点吞掉它 ⇒ **常量 < 2^53**）
#                         ⑦**跨行字符串被整条跳过** ⑧**「整体钉住」代替不了「元素钉住」**
#   Makefile / requirements.txt   构建入口

PYTHON  ?= python3
SCRIPTS := 01_mechanisms 02_how_much_can_hide 03_async_io

.PHONY: all run lint mutate lines clean check-env

all: run

# ★★ 整个目标**必须写成一条 shell**（用 `; \` 串起来）：make 的每一行 recipe 各起一个 shell，
#   `exit 0` 只结束**那一行** —— 拆开写时空分母分支之后照样打印「全绿」。
run: check-env
	@if [ -z "$(SCRIPTS)" ]; then \
		echo "=== 本主题当前有 0 个编号程序 ==="; \
		echo "    ★ 此刻 make run **不构成任何证据** —— 它一条断言也没跑。"; \
		exit 0; \
	fi; \
	for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; rm -f .last_out; exit 1; }; \
		cat .last_out; echo; \
	done; rm -f .last_out; \
	echo "=== 全部 $(words $(SCRIPTS)) 个编号程序退出码 0 ==="

# ★ 第一个动作是**空值检查**（`PYTHON ?=` 在命令行传空值时不会填默认值）。
check-env:
	@if [ -z "$(PYTHON)" ]; then echo "!! PYTHON 是空的。请显式给：make run PYTHON=python3"; exit 1; fi
	@if ! command -v "$(PYTHON)" >/dev/null 2>&1 && [ ! -x "$(PYTHON)" ]; then echo "!! 找不到解释器: $(PYTHON)"; exit 1; fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "!! 需要 python >= 3.9"; exit 1; }
# ★★ 本主题**不探测 `numpy`**（三个编号程序只用标准库 ⇒ 依赖与用途相符；见文件抬头）。

# ★ `ast` 而非 grep：注释里出现框架名不算违规，真 import 才算。
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
	sys.exit(1 if hit else 0)'

# ★ 退出码：0=全部捕获 · 1=有漏网 · 2=读数不可信 · 3=空分母。
#   ⚠ `make mutate > file; echo $?` 拿到的是 **make 自己的码（2）** ⇒ 要看退出码就 `python3 mutate.py`。
mutate: check-env
	@$(PYTHON) mutate.py || { rc=$$?; echo "★ mutate.py 退出码=$$rc（1=有漏网 / 2=读数不可信 / 3=空分母）"; exit $$rc; }

lines:
	@$(PYTHON) -c 'import os; [print("%-32s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith((".py", ".txt", ".md"))]'

$(SCRIPTS): check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .last_out .run_summary
