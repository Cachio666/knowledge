# 30-gpu-programming/ptx-sass 示例构建
#
# ★ 本目录与同域的另外三个主题是**同一形态**（CONTRIBUTING「本域的代码形态」）：
#   三个编号程序是纯 `.py`，本机默认可跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                    同 make run
#   make run                依次跑三个编号程序（任一断言失败即非零退出）
#   make 02_warp_execution  只跑一个（省略 .py 后缀）
#   make lint               AST 检查：确认没有 import torch / triton / numba / cuda
#   make anchors            ★ 联网：重新下载 PTX ISA 原文，在**原始字节**上重新计数、
#                           重新抽取 9 条引文、重新定位它们的章节号、逐条核对 14 条表外
#                           mnemonics（本主题**唯一**的联网目标；`make run` 全程离线）
#   make falsify            ★ **元检验**（非编号工具 meta_falsify.py）：往 `ptxsub.py`
#                           注入 6 种缺陷，断言编号程序必须退出非 0；并确认不打补丁时全过
#   make mutate             ★ **叶子级变异测试**（非编号工具 mutate.py；**不进 `run`**）
#                           对每个叶子级常量逐个「改错」，报「测 N / 捕获 M / 漏网 K」
#                           + harness 异常数 + 对照组；断言捕获与崩溃捕获**分开数**
#   make lines              列出本目录各 .py 的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）────────
#   ptxsub.py        ★ **本主题的引擎**：PTX 子集解释器（分派表 / 闭合的覆盖范围 /
#                      warp 级指令 / bank 冲突计数）。三个编号程序都 `import ptxsub`。
#   specs.py         外部事实常量源（9 条 A 级 ISA 锚点 + 原始字节计数 + PTX 子集表
#                     + bank 冲突表 + `.up` 的未定案项），**每条带 `# 来源：` 与来源等级**
#   meta_falsify.py  **元检验**：往 `ptxsub.py` 注入 6 种缺陷，断言程序必须失败 ——
#                    这是本主题对 **P12（自造真值：模拟器同时是被测对象）** 的正面回答
#   mutate.py        计划的 §4 把「全量叶子级变异测试（含对照组）」列为**固定验收动作**，
#                    所以 harness 必须在仓库里、必须能由 `make` 复现 ——
#                    只在 /tmp 里跑过一次的脚本不构成可复现的验收。
#                    它**只在副本上做**（仓库文件只读，跑完再校验一次哈希），
#                    并且**区分「断言捕获」与「崩溃捕获」**（后者是 C6 模式一）。
#                    ★★ 它的 `ASSERT_FAIL_MARK` **不是**照抄 cuda 主题的那一条：
#                        本主题三个程序统一打印 `  [FAIL] <标签> …`，
#                        照抄 `\[[ABC]/FAIL|FAILED: ` 会导致**一条都匹配不到**。
#   Makefile / requirements.txt   构建入口
#
# ── ★★ 本目录**不依赖外部源码树**，也**不联网**（`make run` 全程离线）────────────
#   与 `cuda` / `cutlass` 主题不同：本主题的真值来源是**PTX ISA 的逐字引文**（`specs.ISA`），
#   而引文是**离线常量**；只有 `make anchors` 会联网**重新抽取**并逐字比对。
#   ★ 三个源仓库里的 `.ptx` / `.sass` / `.cubin` / `.fatbin` **共 0 个**（计划 §0.7 A），
#     所以本主题**没有** B 级来源可引 —— 全部是 A 级（ISA 原文）与 E 级（本库推导）。
#
# ── 本机的硬约束（域 README，已核实）────────────────────────────────────────
#   无 GPU、无 nvcc / cuobjdump / nvdisasm / ptxas / ncu。
#   ⇒ **本目录不写 `.cu`**（连语法都无法验证），也绝不 import 装不上的东西；
#     `make lint` 钉住后者，`01_virtual_isa.py` 的 `C.7` 钉住「本目录没有 spawn 能力」。

PYTHON  ?= python3
SCRIPTS := 01_virtual_isa 02_warp_execution 03_smem_and_sass

.PHONY: all run lint anchors falsify mutate lines clean check-env

all: run

run: check-env
	@rm -f .run_summary
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; echo; \
			echo "!! $$s.py 非零退出 —— 断言失败（详情见上）"; \
			rm -f .last_out; exit 1; }; \
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
	print("    断言 %d 条 · 通过 %d 条 · 失败 %d 条 · 跳过 %d 项（跳过不计入通过数）" \
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
		echo "   请显式给一个解释器："; \
		echo "       make run PYTHON=python3"; \
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
	@echo "  解释器 = $$($(PYTHON) -c 'import sys; print(sys.executable)')"
	@echo "           $$($(PYTHON) -c 'import sys, numpy; print("python %s / numpy %s" % (sys.version.split()[0], numpy.__version__))')"

# ★ `ast` 而非 grep：注释与 docstring 里出现 "triton" 这个词**不算违规**，真 import 才算。
#   ★ **必须含 specs.py 与 ptxsub.py**（计划 §2 P6）。★ `import_module` 这类**动态导入**
#     也算违规。检查器放在 `01_virtual_isa.py` 的 `--lint` 模式里（**同一份扫描代码**，
#     所以它同时是 `make run` 的 `A.8` 断言）。
lint: check-env
	@$(PYTHON) 01_virtual_isa.py --lint
	@echo "✓ 没有 import torch / triton / numba / cuda（含 specs.py / ptxsub.py，含 import_module）"

# ★★ 联网入口（本主题**唯一**需要网络的目标）。引文与计数**由脚本从下载的原始 HTML 重算**，
#   不手打；对不上就以非零码退出（不得静默放宽 specs.py）。
#   ★★ 它的第一个动作是**在原始字节上数计数** —— 这是计划 §0.7 E9 那条教训的机械执行：
#      本主题的规划阶段就是「抽取脚本打印对了、转写时抄错了」栽的跟头。
anchors: check-env
	@$(PYTHON) 01_virtual_isa.py --anchors-net

# ★ **不进 `run`**：元检验要跑 6 × 4 次子进程，慢；而且它的结论是「元层」的。
#   基线检查（不打补丁时必须全清）在它自己内部做。
falsify: check-env
	@$(PYTHON) meta_falsify.py

# ★ **不进 `run`**：诊断工具，跑一次要几分钟（并行跑几百个副本）。
mutate: check-env
	@$(PYTHON) mutate.py

lines:
	@$(PYTHON) -c 'import os; [print("%-24s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith(".py")]'

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
01_virtual_isa 02_warp_execution 03_smem_and_sass: check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .run_summary .last_out
