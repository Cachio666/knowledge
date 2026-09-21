# 50-performance/profiling-tools 示例构建
#
# ★ 本目录与同域另外两个主题是**同一形态**（CONTRIBUTING §2 的第二种情形）：
#   三个编号程序是纯 `.py`，本机默认可跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                    同 make run
#   make run                依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_collection_mechanisms   只跑一个（省略 .py 后缀）
#   make env                打印本主题需要的**外部工具与版本**（clang / sample / python）
#   make docs               探测两份 NVIDIA 文档在不在（不在 ⇒ 逐字核对整体 SKIP）
#   make lint               AST 检查：确认没有 import torch / triton / numba / cuda
#   make falsify            ★ **元检验**（非编号工具 meta_falsify.py）：往**测量引擎**里
#                           注入 6 种缺陷，按「期望结果」判定（多数必须失败，
#                           **D3 是方向相反的一条：期望通过**）
#   make mutate             ★ **叶子级变异测试**（非编号工具 mutate.py；**不进 `run`**）
#                           报「测 N / 捕获 M / 漏网 K」+ harness 异常数 + 对照组；
#                           断言捕获与崩溃捕获**分开数**
#   make mutate-all        同上，分母含三个编号程序的模块级常量
#   make lines              列出本目录各 .py 的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）────────
#   specs.py         外部事实常量源：**F1–F15 的逐字引文**（A 级，带 URL + 章节 +
#                    抓取日 + 文档页脚 last-updated + 原始字节 sha256）、F14 的两套
#                    同名指标定义、F15 的两个默认值、上游契约（`45` 三处「表外输入」）、
#                    本机硬约束、以及三个程序共用的**文档探测**与**自洽不变量**
#   meta_falsify.py  **元检验**：往测量引擎里注入缺陷，断言程序按「期望结果」反应 ——
#                    这是本主题对 **P13（量法口径错）** 的正面回答
#   mutate.py        计划的 §3 验收把「完整叶级 mutation（含对照组）」列为**固定验收动作**，
#                    所以 harness 必须在仓库里、必须能由 `make` 复现。
#                    它**只在副本上做**（仓库文件只读，跑完再校验一次哈希），
#                    并且**区分「断言捕获」与「崩溃捕获」**。
#                    ★★ 它的 `ASSERT_FAIL_MARK` **不是**照抄任何别的主题的：
#                       本主题三个程序统一打印 `  **FAIL** <标签> …`。
#                       抄 `30-gpu-programming/cutlass` 的 `[FAIL]` 或
#                       `45-kernel-lifecycle` 的 `[FAIL]|✗ N 条断言失败` 都会匹配到，
#                       那正是「跨主题照抄」要禁的那件事。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束（域 README，已核实）────────────────────────────────────────
#   无 GPU、无 `ncu`/`nsys`/`nvprof`/`nvidia-smi`/`CUPTI`/`perf`；
#   有 `sample`（免 root）、`clang`、`dtrace`、`xctrace`。
#   ★ **本主题不写 `.cu`、不 import 装不上的东西、也不需要网络**：
#     `make run` 只读本机的 `/usr/bin/clang`、`/usr/bin/sample` 与本目录的文件。
#   ★ `clang` **必需**：找不到时编号程序**以可操作的错误 rc=2 退出**，
#     **不静默跳过、不回落到 Python 循环**（回落会得到 D-7 的假数）。
#   ★ `sample` 不在 ⇒ 相关断言 **SKIP**（单独计数，**不计入通过**），`make run` 仍可过；
#     此时 `make falsify` 的 D6 靶子不会被触及，元检验会**明说**这一点。

PYTHON  ?= python3
SCRIPTS := 01_collection_mechanisms 02_replay_multipass 03_defaults_attribution
# ★ 本主题**不设**（也不需要）外部工件路径变量：真值来自
#   ① 本机 CPU 上的真实测量；② NVIDIA 文档的规格事实（F1–F15，逐字引文已内联进 specs.py）。
#   两份 HTML **不是**运行前提 —— 不在场时逐字核对整体 SKIP，程序照常跑完其余断言。
#   想补上逐字核对：`PERF_DOC_DIR=/path/to/ncu_src make run`。

.PHONY: all run env docs lint falsify mutate mutate-all lines clean check-env

all: run

run: check-env
	@rm -f .run_summary .last_out
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; echo; \
			echo "!! $$s.py 非零退出 —— 断言失败或外部工具不可用（详情见上）"; \
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

# ★ 外部工具的**可见性检查**。找不到 `clang` 时**在这里就报**，
#   不让三个程序各炸一遍再汇总成一句含糊的话。
#   `sample` 不在**不算失败**（相关断言会 SKIP），但要**明说**。
env: check-env
	@echo "  解释器 = $$($(PYTHON) -c 'import sys; print(sys.executable)')"
	@echo "           $$($(PYTHON) -c 'import sys, numpy; print("python %s / numpy %s" % (sys.version.split()[0], numpy.__version__))')"
	@$(PYTHON) -c 'import shutil,sys,os; \
	c=os.environ.get("CLANG") or shutil.which("clang"); s=shutil.which("sample"); \
	print("  clang  = %s" % (c or "**不在**（编号程序会以 rc=2 退出，不回落）")); \
	print("  sample = %s" % (s or "**不在**（相关断言将 SKIP，不计入通过）")); \
	sys.exit(0 if c else 2)'

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
#   检查器写在 `01_collection_mechanisms.py --lint` / `02… --lint` / `03… --lint` 里。
lint: check-env
	@$(PYTHON) 01_collection_mechanisms.py --lint
	@$(PYTHON) 02_replay_multipass.py --lint
	@$(PYTHON) 03_defaults_attribution.py --lint
	@echo "✓ 没有 import torch / triton / numba / cuda（含 specs.py / meta_falsify.py / mutate.py）"

# ★ **不进 `run`**：元检验要跑 6 个副本，慢；而且它的结论是「元层」的。
#   基线检查（不打补丁时必须全清）在它自己内部做。
falsify: check-env
	@$(PYTHON) meta_falsify.py

# ★ **不进 `run`**：诊断工具，跑一次要几分钟（并行跑上百个副本）。
mutate: check-env
	@$(PYTHON) mutate.py specs

mutate-all: check-env
	@$(PYTHON) mutate.py all

lines:
	@$(PYTHON) -c 'import os; [print("%-32s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith((".py", ".txt"))]'

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
01_collection_mechanisms 02_replay_multipass 03_defaults_attribution: check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .run_summary .last_out .d6_child.py
