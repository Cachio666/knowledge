# 30-gpu-programming/cutlass 示例构建
#
# ★ 本目录与同域的另外三个主题是**同一形态**（CONTRIBUTING「本域的代码形态」）：
#   三个编号程序是纯 `.py`，本机默认可跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                    同 make run
#   make run                依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_layout_algebra  只跑一个（省略 .py 后缀）
#   make cutlass-path       只做外部工件路径探测，打印选中的 CUTLASS 源码树
#   make lint               AST 检查：确认没有 import torch / triton / numba / cuda
#   make falsify            ★ **元检验**（非编号工具 meta_falsify.py）：把**本仓库那份
#                           参照实现**改坏（9 种缺陷，含 2 条方向相反的），断言编号程序
#                           必须退出非 0；并确认不打补丁时全过
#   make mutate             ★ **叶子级变异测试**（非编号工具 mutate.py；**不进 `run`**）
#                           对每个叶子级常量逐个「改错」，报「测 N / 捕获 M / 漏网 K」
#                           + harness 异常数 + 对照组；断言捕获与崩溃捕获**分开数**
#   make lines              列出本目录各 .py 的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）────────
#   specs.py         外部事实常量源（CUTLASS 版本的**四处对照** / 31 条带 file:line 的
#                    源码锚点 / pycute 的 22 行实测输出表 / cutlass_library 的枚举与形状表
#                    / 两条陷阱），**每条带 `# 来源：` 与来源等级**；
#                    另含**外部工件路径探测**（三个程序共用同一份候选列表）
#   meta_falsify.py  **元检验**：把本仓库的参照实现改成坏的，断言程序必须失败 ——
#                    这是本主题对 **P11（把外部工件当作不会错的正典）** 的正面回答的
#                    **镜像**：**「对拍」这一侧自己也要被反驳**
#   mutate.py        计划的 §4 把「全量叶子级变异测试（含对照组）」列为**固定验收动作**，
#                    所以 harness 必须在仓库里、必须能由 `make` 复现 ——
#                    只在 /tmp 里跑过一次的脚本不构成可复现的验收。
#                    它**只在副本上做**（仓库文件只读，跑完再校验一次哈希），
#                    并且**区分「断言捕获」与「崩溃捕获」**（后者是 C6 模式一）。
#                    ★★ 它的 `ASSERT_FAIL_MARK` **不是**照抄 cuda / triton / ptx-sass 的：
#                        本主题三个程序统一打印 `  [FAIL] <标签> …`，
#                        照抄 `\[[ABC]/FAIL|FAILED: ` 会导致**一条都匹配不到**。
#   Makefile / requirements.txt   构建入口
#
# ── ★★ 外部工件：本主题的 `code/` 不依赖仓库中其他目录，但**依赖一棵外部源码树** ──
#   本域代码形态的「真值来自外部真实工件」，在 `cutlass` 主题里就是：
#     · CUTLASS 4.8.0 源码树（**B 级来源：NVIDIA 自己的实现，不是规范**）。
#       本目录读它、也**真的 import 并跑**它自带的两个纯 Python 参考实现：
#         python/pycute/          —— CuTe layout 代数的纯 Python 实现（零第三方依赖）
#         python/cutlass_library/ —— CUTLASS 自己的生成器
#       行号、逐字引文、实测输出都记在 `specs.py` 里，程序逐条核对。
#   ★ 路径探测：候选列表见 `specs.CUTLASS_CANDIDATES`，**不得写死构造者本机的目录布局**
#     （CONTRIBUTING 第 140 行）。找到后**打印它选中的路径**；找不到时
#     **以可操作的错误信息退出**，**不静默回落** ——
#     「静默跳过对拍」会把 A 类断言变成**空操作**，那是本域最不能接受的一种通过。
#
#   ★ 与 CONTRIBUTING 第 36 行「`code/` 必须自包含」的关系：本域代码形态对 `Makefile`
#     的规定是「需要探测外部工件路径时不得写死布局，找到后打印路径，找不到以可操作的
#     错误退出」——本条与第 36 行**不冲突**：不依赖的是**仓库里其他目录**，
#     依赖的是**外部真实工件**，而后者正是本形态的真值来源。
#     （域 README 已把「`cutlass` 例外」写明；`cuda` 是同一个例外的第二种情形。）
#
# ── 本机的硬约束（域 README，已核实）────────────────────────────────────────
#   无 GPU、无 nvcc / cuobjdump / nvdisasm / ptxas / ncu；triton 全机器不存在；
#   numba / cuda-python 未安装。⇒ **本目录不写 `.cu`，也不 import 装不上的东西**。
#   ★ 本主题**不需要网络**：`make run` 全程离线，只读本机的 CUTLASS 源码树。

PYTHON  ?= python3
SCRIPTS := 01_layout_algebra 02_tiling_decompose 03_swizzle_probe

# ★ 外部工件的路径。默认从环境变量取；也可用 `make run CUTLASS_PATH=...` 覆盖。
#   留空时由 `specs.find_cutlass()` 按候选列表探测（相对本目录的兄弟目录等）。
#
#   ★★ 留空**也是开箱即用的**（本节原先把「留空」说成「原样透传」，那**不成立**：
#      `CUTLASS_PATH ?=` 留空 + `export` ⇒ 环境里是**空字符串**，
#      「**空值原样透传等于没传**」）。实际的透传链是：
#        · `make run`：三个编号程序各自用 `specs.require_cutlass()` 探测；
#          `cutlass-path` 先探一次并把**选中路径**打印出来；
#        · `make falsify` / `make mutate`：两个工具**自己先探测**
#          （`specs.find_cutlass()`），再把**探测到的绝对路径**写进
#          「跑编号程序的那个临时副本」的 env（`child_env()`）——
#          副本里那些**相对**候选（`../cutlass` 等）**必然**解析不到，
#          靠的只有这一条透传。（控制者实测：这一版之前的 `make falsify` rc=2。）
#        · 显式传 `CUTLASS_PATH=...` 时它是候选列表里的**第一顺位**。
#   ★ 探测失败时三个入口**都以可操作的错误退出**，不静默跳过对拍。
CUTLASS_PATH ?=
export CUTLASS_PATH

.PHONY: all run cutlass-path lint falsify mutate lines clean check-env

all: run

run: check-env cutlass-path
	@rm -f .run_summary
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; echo; \
			echo "!! $$s.py 非零退出 —— 断言失败或外部工件缺失（详情见上）"; \
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

# ★★ 这个目标是**本主题独有的**（`CONTRIBUTING` 第 140 行那条规定的机械执行）：
#   探测外部工件路径 → **打印选中路径** → 找不到就**以可操作的错误退出**（rc 非 0）。
#   ★ 它是 `run` 的前置依赖，所以「找不到 CUTLASS」**不会**变成「静默跳过对拍」。
cutlass-path: check-env
	@$(PYTHON) -c 'import specs,sys; \
	r=specs.find_cutlass(verbose=True); \
	print("  版本        : include/cutlass/version.h ⇒ **%s**（基准）" % specs.CUTLASS_VERSION_BASELINE); \
	print("  四处来源    : %s" % ", ".join("%s=%s%s" % (s["label"], s["value"], "" if s["agree"] else " ← **不一致**") for s in specs.VERSION_SOURCES)); \
	sys.exit(0 if r else 2)' || { \
		echo ""; \
		echo "!! 找不到 CUTLASS 源码树 —— **本目录不会静默跳过对拍**。"; \
		echo "   本主题的真值来源是 CUTLASS 4.8.0 自带的两个 Python 参考实现："; \
		echo "     python/pycute/          (CuTe layout 代数)"; \
		echo "     python/cutlass_library/ (CUTLASS 自己的生成器)"; \
		echo ""; \
		echo "   请显式指定（三个入口同理）："; \
		echo "     make run CUTLASS_PATH=/path/to/cutlass"; \
		echo "     make falsify CUTLASS_PATH=/path/to/cutlass"; \
		echo "     make mutate  CUTLASS_PATH=/path/to/cutlass"; \
		echo "     export CUTLASS_PATH=/path/to/cutlass"; \
		echo "   或把源码树放到下列任一位置（相对本目录）："; \
		$(PYTHON) -c 'import specs; [print("     %s" % c) for c in specs.CUTLASS_CANDIDATES]'; \
		echo ""; \
		echo "   ★ 本库**不复刻** CUTLASS 的源码树，也不内置它的副本。"; \
		exit 2; }

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

# ★ `ast` 而非 grep：注释与 docstring 里出现 "torch" 这个词**不算违规**，真 import 才算。
#   ★ **必须含 specs.py / meta_falsify.py / mutate.py**（计划 §2 P6）。
#   ★ 检查器放在 `01_layout_algebra.py` 的 `--lint` 模式里（**同一份扫描代码**，
#     所以它同时是 `make run` 的 A.14 断言）。`make lint` 是**不读外部工件**的快速入口。
lint:
	@$(PYTHON) 01_layout_algebra.py --lint
	@echo "✓ 没有 import torch / triton / numba / cuda（含 specs.py / meta_falsify.py / mutate.py，含 import_module）"

# ★ **不进 `run`**：元检验要跑 9 × 3 次子进程，慢；而且它的结论是「元层」的。
#   基线检查（不打补丁时必须全清）在它自己内部做。
#   ★ 它**自己先探测**外部工件路径并写进子进程 env（**不依赖 `CUTLASS_PATH` 非空**，
#     见 Makefile 头部的透传链）。`cutlass-path` 这个前置依赖的用处是
#     「先探一次、把选中路径打印出来」：脱离原位时它就在这里给出可操作的错误，
#     而不是让 9 × 3 个副本各炸一遍。
falsify: check-env cutlass-path
	@$(PYTHON) meta_falsify.py

# ★ **不进 `run`**：诊断工具，跑一次要几分钟（并行跑几百个副本）。
#   ★ 与 `falsify` 同一形态：自己探测 + 把绝对路径写进每个任务的子进程 env。
mutate: check-env cutlass-path
	@$(PYTHON) mutate.py

lines:
	@$(PYTHON) -c 'import os; [print("%-24s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith(".py")]'

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
01_layout_algebra 02_tiling_decompose 03_swizzle_probe: check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .run_summary .last_out
