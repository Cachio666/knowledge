# 70-frameworks/pytorch 示例构建
#
# ★ 本目录是 CONTRIBUTING §2 的**第三种形态**：编号程序需要一个**非默认环境**
#   （要 `import torch`，而默认 python3 里没有它）。三类断言的分工是：
#     · B 代价模型  → **编号程序内部**（代价模型的参照是真系统，只有在能 import
#                     到它的环境里测才有意义）
#     · A / C       → **verify.py**（默认 python3 下也能复核的那一半）
#   `verify.py` **永不 import torch**，产出**一律是推导**，不得标成本机实测值。
#
# 用法：
#   make                    同 make run
#   make run                自动探测一个装了 torch 的解释器，依次跑全部编号程序
#                           （断言失败即非零退出）
#   make verify             用**默认 python3** 跑 verify.py（纯 python3 + 标准库）
#   make 01_dispatch_layers 只跑一个（省略 .py 后缀）
#   make clean              清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   verify.py      纯 python3 的独立复核（A/C），**不 import torch**
#   specs.py       PyTorch 的外部事实源与版本锚点，每条带 `# 来源：`
#   Makefile / requirements.txt  构建入口
#
# ── 环境探测（CONTRIBUTING §2 的环境探测约定）──────────────────────────────
# ⚠ 这里**不写条数**：该小节的约定会增补（写「6 条」就会过期）。
# ⚠ 本主题的编号程序不依赖任何「2.8+ 才有的特性」，所以**不加版本下限**；
#    需要下限的域见 CONTRIBUTING 该小节的最后一条（`130-model-optimization` 是首个例子）。
# 1. **不得硬编码某台机器的目录布局**；候选以通用位置为主，本机布局只作兜底并注明。
# 2. 探测判据用 `importlib.util.find_spec("torch")`，**不要用 `import torch`**——
#    前者只回答「模块是否可见」，不把 torch 载进内存。
#    本仓库作者机器实测：find_spec 约 0.01 s，import torch 约 1.0 s（冷启动首次 2.7 s），
#    **差约两个数量级**，且随机器与版本变——**可移植的是量级差，不是具体数**。
# 3. 找到后**打印选中的是哪一个（含 torch 版本）**，供文档 §5 直接引用为测量条件。
# 4. 找不到时**以可操作的错误信息退出**，不要静默回落到 python3 去跑一个假失败。
# 5. ⚠ `PYTHON ?=` 挡不住空值：命令行传 `PYTHON=`（空值）也算「已赋值」，
#    默认值不会被填上、探测被整个跳过。所以下面 `run` 里**另有一次空值检查**。
# 6. ⚠ `PYTHON=~/...` 的坑：make 本身不做 `~` 展开，值就是字面量 `~/...`。
#    它在**未加引号的命令位置**能跑（recipe 交给 /bin/sh，shell 会展开词首的 `~`），
#    但**加引号的位置**（`[ -x "$(PYTHON)" ]`）永远是字面量、必然失败。
#    所以报错信息里要教用户写 `PYTHON=$HOME/...`。

SCRIPTS := 01_dispatch_layers 02_autograd_graph 03_strides_and_views 04_custom_ops

# 候选顺序：通用位置在前，本机布局兜底。每个候选都先要求「文件可执行」，再问
# 「torch 是否可见」——两问都不进 python 的 import 路径。
PYTHON ?= $(shell for p in \
            "$$(command -v python3 2>/dev/null)" \
            "$$(command -v python 2>/dev/null)" \
            "$$CONDA_PREFIX/bin/python" \
            "$$VIRTUAL_ENV/bin/python" \
            "$$HOME/.venv/bin/python" \
            $$HOME/*/.venv/bin/python \
            $$HOME/miniconda3/envs/*/bin/python \
            $$HOME/anaconda3/envs/*/bin/python \
            $$HOME/miniforge3/envs/*/bin/python ; do \
            [ -n "$$p" ] && [ -x "$$p" ] || continue; \
            $$p -c 'import importlib.util as u,sys; sys.exit(0 if u.find_spec("torch") else 1)' \
                2>/dev/null && { echo "$$p"; break; }; \
          done)

.PHONY: all run verify clean
all: run

# ★ 空值检查不能用 `[ -z "$(PYTHON)" ]` 之外的写法；也不能只靠 `?=` 的默认值兜底。
run:
	@if [ -z "$(PYTHON)" ]; then \
		echo "错误：找不到装了 torch 的 Python 解释器。"; \
		echo ""; \
		echo "本目录的编号程序需要真实 torch（默认 python3 里没有）。可用的做法："; \
		echo "  1) 激活一个装了 torch 的环境后再跑，或"; \
		echo "  2) 显式指定解释器：make run PYTHON=/path/to/python"; \
		echo "     ⚠ 路径里若要用家目录，请写 \$$HOME/... 而**不要**写 ~/...："; \
		echo "        make 自身不做 ~ 展开，而探测/校验位置一定会给变量加引号，"; \
		echo "        加了引号 ~ 就永远是字面量、必然被判定为「不存在」。"; \
		echo "  3) 只想复核判据（不需要 torch）：make verify"; \
		exit 1; \
	fi
	@echo "解释器：$(PYTHON)"
	@$(PYTHON) -c 'import sys, torch; print("torch %s / python %s / CUDA 可用: %s" % (torch.__version__, sys.version.split()[0], torch.cuda.is_available()))'
	@echo
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py || exit 1; \
		echo; \
	done

# 纯 python3 + 标准库；**不 import torch**；产出是推导，不是实测。
verify:
	@echo "解释器：python3（默认）"
	@python3 verify.py

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
