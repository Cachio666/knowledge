# 30-gpu-programming/cuda 示例构建
#
# ★ 本目录与同域的另外三个主题是**同一形态**（CONTRIBUTING「本域的代码形态」）：
#   三个编号程序是纯 `.py`，本机默认可跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                    同 make run
#   make run                依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_sync_contract   只跑一个（省略 .py 后缀）
#   make lint               AST 检查：确认没有 import torch / triton / numba / cuda
#   make anchors            ★ 联网：从官方文档**再抽取**全部 22 条语义锚点并与 specs.py 逐字比对
#   make show               打印两个 DAG 的可读形式（★ 不是时间线）
#   make mutate             ★ **叶子级变异测试**（非编号工具 mutate.py；**不进 `run`**）
#                           对每个叶子级常量逐个「改错」，报「测 N / 捕获 M / 漏网 K」
#                           + harness 异常数 + 对照组；断言捕获与崩溃捕获**分开数**
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）──────
#   specs.py            本主题的外部事实常量源，**每条带 `# 来源：` 与来源等级（A–E）**
#   mutate.py           计划的 §4 把「全量叶子级变异测试（含对照组）」列为**固定验收动作**，
#                       所以 harness 必须在仓库里、必须能由 `make` 复现 ——
#                       只在 /tmp 里跑过一次的脚本不构成可复现的验收（订正轮补）。
#                       它**只在副本上做**（仓库文件只读，跑完再校验一次哈希），
#                       并且**区分「断言捕获」与「崩溃捕获」**（后者是 C6 模式一）。
#   Makefile / requirements.txt   构建入口
#
# ── ★★ 外部工件：本主题的 `code/` 不依赖仓库中其他目录，但**依赖两棵外部源码树** ──
#   本域代码形态的「真值来自外部真实工件」，在 `cuda` 主题里就是：
#     · CUDA_TRAINING_SERIES —— NVIDIA 官方训练系列（B 级来源：厂商材料，不是规范）
#       本目录读它的 exercises/hw2、hw5、hw10、hw13 下的**真实 .cu 文件**：
#       行号、逐字行内容、sha256 前 16 位全部记在 specs.py 里，程序逐条核对。
#     · CUDA_LEARNING —— T4 / sm_75 目标的实现集（用于 C2/C3 的**误报回归**）。
#   两者都用候选列表探测；**不得写死构造者本机的目录布局**（CONTRIBUTING 第 167–189 行）。
#   找到后程序会**打印它选中的路径**；找不到时**以可操作的错误信息退出**，不静默回落。
#
#   ★ 与 CONTRIBUTING 第 36 行「`code/` 必须自包含」的关系：本域代码形态对
#     `Makefile` 的规定是「需要探测外部工件路径时不得写死布局，找到后打印路径，
#      找不到以可操作的错误退出」——本条与第 36 行**不冲突**：
#     不依赖的是**仓库里其他目录**，依赖的是**外部真实工件**，而后者正是本形态的真值来源。
#     （域 README 已把「`cutlass` 例外」写明；`cuda` 是同一个例外的第二种情形，
#      实施时发现计划只点名了 `cutlass`，**这里一并写明**。）
#
# ── 本机的硬约束（域 README，已核实）────────────────────────────────────────
#   无 GPU、无 nvcc / cuobjdump / nvdisasm / ptxas / ncu；triton 全机器不存在；
#   numba / cuda-python 未安装。⇒ **本目录不写 `.cu`，也不 import 装不上的东西**。

PYTHON  ?= python3
SCRIPTS := 01_sync_contract 02_stream_dag 03_graph_capture

# ★ 外部工件的路径。默认从环境变量取；也可用 `make run CUDA_TRAINING_SERIES=...` 覆盖。
#   留空时由程序自己按候选列表探测（相对本目录的兄弟目录）。
CUDA_TRAINING_SERIES ?=
CUDA_LEARNING        ?=
export CUDA_TRAINING_SERIES
export CUDA_LEARNING

.PHONY: all run lint anchors show mutate clean check-env

all: run

run: check-env
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
		echo "     引号里的 ~ 是字面量，[ -x ] 必然失败（未加引号的命令位置才会被"; \
		echo "     /bin/sh 展开）。请改用 \$$HOME/env/bin/python。"; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
		|| { echo "!! 需要 python >= 3.9（三个程序都用了 f-string 以外的现代语法）"; exit 1; }
	@$(PYTHON) -c 'import numpy; sys_exit=0' 2>/dev/null \
		|| { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }
	@echo "  解释器 = $$($(PYTHON) -c 'import sys; print(sys.executable)')"
	@echo "           $$($(PYTHON) -c 'import sys, numpy; print("python %s / numpy %s" % (sys.version.split()[0], numpy.__version__))')"

# ★ `ast` 而非 grep：注释与 docstring 里出现 "torch" 这个词不算违规，真 import 才算。
#   ★ **必须含 specs.py**（计划 §2 P6：声称「13 个程序都不 import torch」却只查了编号程序）。
#   ★ 判据按 CONTRIBUTING：`import_module("torch")` 这类**动态导入**也算违规。
#   ★ 检查器放在 `01_sync_contract.py` 的 `--lint` 模式里（**同一个程序**，
#     所以它同时是 `make run` 的 A.10 断言）。`make lint` 只是不读外部工件的快速入口。
lint:
	@$(PYTHON) 01_sync_contract.py --lint
	@echo "✓ 没有 import torch / triton / numba / cuda（含 specs.py，含 import_module 动态导入）"

# ★★ 联网入口。保留探针的核心做法：**引文由脚本从下载的 HTML 里抓取，不手打**。
#   抓不到、或与 specs.py 记录的 quote 不一致，都以非零码退出。
#   注：`curl` 必须带 `-L`，否则拿到 HTTP 301 且 size=0；本目标用 urllib，同样带重定向。
anchors: check-env
	@$(PYTHON) 02_stream_dag.py --anchors-net

show: check-env
	@$(PYTHON) 02_stream_dag.py --show

# ★ **不进 `run`**：这是诊断工具，跑一次要几分钟（并行跑几百个副本）。
#   它自己会做基线检查（三个程序必须全清），并在结尾校验原文件哈希未变。
mutate: check-env
	@$(PYTHON) mutate.py

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
01_sync_contract 02_stream_dag 03_graph_capture: check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .run_summary .last_out
