# 50-performance/tuning-practice 示例构建
#
# ★ 本目录与同域另外两个主题是**同一形态**（CONTRIBUTING §2 的第二种情形）：
#   三个编号程序是纯 `.py`，本机默认可跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                    同 make run
#   make run                依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_sigma_tau       只跑一个（省略 .py 后缀）
#   make env                打印本主题需要的**外部工具与版本**（python / numpy）
#   make docs               探测两份 NVIDIA 文档在不在（不在 ⇒ 逐字核对整体 SKIP）
#   make lint               AST 检查：确认没有 import torch / triton / numba / cuda
#   make falsify            ★ **元检验**（非编号工具 meta_falsify.py）：往**测量引擎**里
#                           注入 5 种缺陷，按「期望结果」判定（**全部期望失败**）
#   make mutate             ★ **叶子级变异测试**（非编号工具 mutate.py；**不进 `run`**）
#                           报「测 N / 捕获 M / 漏网 K」+ harness 异常数 + 对照组；
#                           断言捕获与崩溃捕获**分开数**
#   make mutate-all        同上，分母含 mini_search.py 与三个编号程序的模块级常量
#   make quick              用 PERF_QUICK=1 跑一遍（诊断用；断言相同）
#   make lines              列出本目录各 .py 的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）────────
#   specs.py         外部事实常量源：**F1–F15 的逐字引文**（A 级，带 URL + 章节 +
#                    抓取日 + 文档页脚 last-updated + 原始字节 sha256）、
#                    **`45` 三处「表外输入」的逐字引文**、`60` 的 `MEASURE_NOISE = 0.0`、
#                    本机硬约束、计划 §0.7 **D-4 / D-4b / D-8 / D-10 / D-12** 的记录值
#                    （含**已撤回的错值 250 ns**，用于机器化反证），
#                    以及 `45/02` 的 δ* 系数（**代入，不重推**）
#   mini_search.py   ★ **非编号共享模块**：人造代价地形 + 迷你搜索**独立实现**
#                    （**不 import `60` 域的任何代码**）。它最关键的一条是：
#                    **噪声不是造出来的，是量出来的** —— `measure()` 真的把那份工作量
#                    跑一遍并计时。`02` 用它。★ 02 与 03 都**不**依赖外部工具。
#   meta_falsify.py  **元检验**：往测量引擎里注入缺陷，断言编号程序按「期望结果」反应 ——
#                    这是本主题对 **P13（量法口径错）** 的正面回答
#   mutate.py        计划的 §3 验收把「完整叶级 mutation（含对照组）」列为**固定验收动作**，
#                    所以 harness 必须在仓库里、必须能由 `make` 复现。
#                    它**只在副本上做**（仓库文件只读，跑完再校验一次哈希），
#                    并且**区分「断言捕获」与「崩溃捕获」**。
#                    ★★ 它的 `ASSERT_FAIL_MARK` **不是**照抄任何别的主题的：
#                       本主题三个程序统一打印 `  >>> FAIL <<< <标签> …`。
#                       抄 `30-gpu-programming/cutlass` 的 `[FAIL]`、
#                       `45-kernel-lifecycle` 的 `[FAIL]|✗ N 条断言失败`、
#                       **同域** `profiling-tools` 的 `**FAIL**`、或
#                       `bottleneck-diagnosis` 的 `*** FAIL ***` 都会匹配到其中一部分，
#                       那正是「跨主题照抄」要禁的那件事。
#                    ★ **并行度固定为 1**：`03` 的 A 段**自己拉起忙进程**，
#                       并行会把它测的「本机噪声形状」搅乱并**灌水捕获数**。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束（域 README，已核实）────────────────────────────────────────
#   无 GPU、无 `ncu`/`nsys`/`nvprof`/`nvidia-smi`/`CUPTI`/`perf`。
#   ★ **本主题不写 `.cu`、不 import 装不上的东西、不需要网络、也不需要 clang/sample**：
#     三个编号程序是**纯 `python3` + 标准库**。
#     `03_ab_orchestration.py` 的 A 段会**临时拉起 `cpu_count()-1` 个
#     `python3 -c 'while True: pass'` 子进程**并在结束时全部终止
#     （它们不写文件、不联网）。**这是它测量口径的一部分**：
#     在闲机上 `A.1`（`max/中位 > 100`）必然失败。

PYTHON  ?= python3
SCRIPTS := 01_sigma_tau 02_noise_and_search 03_ab_orchestration
# ★ 本主题**不设**（也不需要）外部工件路径变量：真值来自
#   ① 本机 CPU 上的真实测量（`perf_counter` / `get_clock_info` / 真实计时）；
#   ② NVIDIA 文档的规格事实（F1–F15，逐字引文已内联进 specs.py）。
#   两份 HTML **不是**运行前提 —— 不在场时逐字核对整体 SKIP，程序照常跑完其余断言。
#   想补上逐字核对：`PERF_DOC_DIR=/path/to/ncu_src make run`。

.PHONY: all run env docs lint falsify mutate mutate-all quick lines clean check-env

all: run

run: check-env
	@rm -f .run_summary .last_out
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; echo; \
			echo "!! $$s.py 非零退出 —— 断言失败（详情见上）"; \
			rm -f .last_out .run_summary; exit 1; }; \
		cat .last_out; \
		cat .last_out >> .run_summary; \
		echo; \
	done
	@rm -f .last_out
	@$(PYTHON) -c 'import re,sys; \
	txt=open(".run_summary",encoding="utf-8").read(); \
	ms=re.findall(r"断言 (\d+) 条，通过 (\d+) 条，失败 (\d+) 条", txt); \
	t=[int(a) for a,b,c in ms]; p=[int(b) for a,b,c in ms]; f=[int(c) for a,b,c in ms]; \
	sk=len(re.findall(r"\[SKIP\]", txt)); \
	print("=== 三个编号程序全部退出码 0 ==="); \
	print("    断言 %d 条 · 通过 %d 条 · 失败 %d 条 · 跳过 %d 项（跳过不计入通过）" \
	      % (sum(t), sum(p), sum(f), sk)); \
	sys.exit(1 if sum(f) or sum(t)!=sum(p) else 0)'
	@rm -f .run_summary

# ★ 这个目标的第一个动作是**空值检查**，不是靠 `PYTHON ?=` 的默认值兜底。
#   `?=` 的语义是「**未被赋值时**才赋值」，而命令行传 `PYTHON=`（**空值**）也算
#   「已赋值」—— 默认值不会被填上，探测被整个跳过（CONTRIBUTING 第 178–181 行）。
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
	@$(PYTHON) -c 'import numpy; sys_exit=0' 2>/dev/null \
		|| { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }

# ★ 外部工具的**可见性检查**。本主题**不需要 clang / sample**：
#   三个程序是纯 python3 + 标准库。这里只报解释器与 `cpu_count()`。
env: check-env
	@echo "  解释器 = $$($(PYTHON) -c 'import sys; print(sys.executable)')"
	@echo "           $$($(PYTHON) -c 'import sys, numpy; print("python %s / numpy %s" % (sys.version.split()[0], numpy.__version__))')"
	@$(PYTHON) -c 'import os,sys,time; \
	i=time.get_clock_info("perf_counter"); \
	print("  cpu_count = %d（03 的 A 段会拉起 cpu_count()-1 个忙进程）" % (os.cpu_count() or 0)); \
	print("  perf_counter: implementation=%s resolution=%.10g s (= %.4f ns)" \
	      % (i.implementation, i.resolution, i.resolution*1e9)); \
	print("  ★ 不需要 clang / sample / 网络")'

# ★ 探测两份 NVIDIA 文档在不在。**不在不是错误** —— 逐字核对整体 SKIP，
#   其余断言照跑；但 `must_contain ⊂ quote` 那一半自洽检查**始终在守**。
docs: check-env
	@$(PYTHON) -c 'import specs,sys; \
	d=specs.find_doc_dir(verbose=True); \
	[print("     %s ⇒ %s %s" % (n, st, det or "")) for n,(st,det) in sorted(specs.doc_status(d).items())]; \
	print("  ★ 候选列表（相对项以 specs.py 所在目录为基准，不是 cwd）："); \
	[print("     %s" % c) for c in specs.resolve_candidates()]; \
	print("  ★ 本库**不复刻** NVIDIA 的文档，也不内置它的副本。"); \
	print("     取法：curl -sS -o profiling_guide.html https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"); \
	print("           curl -sS -o ncu_cli.html       https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html"); \
	print("     然后：PERF_DOC_DIR=<目录> make run")'

# ★ `ast` 而非 grep：注释与文档串里出现 "torch" 这个词**不算违规**，真 import 才算。
#   检查器写在三个编号程序各自的 `--lint` 里。
lint: check-env
	@$(PYTHON) 01_sigma_tau.py --lint
	@$(PYTHON) 02_noise_and_search.py --lint
	@$(PYTHON) 03_ab_orchestration.py --lint
	@echo "✓ 没有 import torch / triton / numba / cuda（含 specs.py / mini_search.py / meta_falsify.py / mutate.py）"

# ★ **不进 `run`**：元检验要跑 5 个副本，慢；而且它的结论是「元层」的。
#   基线检查（不打补丁时必须全清）在它自己内部做。
falsify: check-env
	@$(PYTHON) meta_falsify.py

# ★ **不进 `run`**：诊断工具，跑一次要几十分钟（本主题的并行度**固定为 1**）。
mutate: check-env
	@$(PYTHON) mutate.py specs

mutate-all: check-env
	@$(PYTHON) mutate.py all

# ★ 快档：断言相同、只缩短重复次数 —— 给上面两个诊断工具与人工排查用。
quick: check-env
	@for s in $(SCRIPTS); do \
		echo "=== $$s（PERF_QUICK=1） ==="; \
		PERF_QUICK=1 $(PYTHON) $$s.py || exit 1; \
		echo; \
	done

lines:
	@$(PYTHON) -c 'import os; [print("%-32s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith((".py", ".txt"))]'

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
01_sigma_tau 02_noise_and_search 03_ab_orchestration: check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .run_summary .last_out
