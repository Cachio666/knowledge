# 130-quantization/numerical-formats 示例构建
#
# ★ 本目录是 CONTRIBUTING §2 的**第三种形态**：编号程序需要一个**非默认环境**
#   （要 `import torch`，而默认 python3 里没有它）。三类断言的分工是：
#     · B 代价模型  → **编号程序内部**（代价模型的参照是真系统，只有在能 import
#                     到它的环境里测才有意义）
#     · A / C       → **verify.py**（默认 python3 下也能复核的那一半）
#   `verify.py` **永不 import torch**，产出**一律是推导**，不得标成本机实测值。
#   本主题的 A/C 验的是**可独立推导的数学与位域**（从位域反解 max/次正规/eps、
#   e8m0 的取整规则、MX 的 Floor 取整致溢边界、饱和/溢出解析边界），
#   编号程序测的是**真实实现的行为**。**两者不得互相背书。**
#
# 用法：
#   make                    同 make run
#   make run                自动探测一个装了 torch ≥ 2.8 的解释器，依次跑全部编号程序
#                           （断言失败即非零退出）
#   make verify             用**默认 python3** 跑 verify.py（纯 python3 + 标准库）
#   make 01_fp8_formats     只跑一个（省略 .py 后缀）
#   make clean              清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   verify.py      纯 python3 的独立复核（A/C），**不 import torch**
#   specs.py       本主题的外部规格常量源（ONNX / OCP MX / 位域），每条带 `# 来源：`
#   Makefile / requirements.txt  构建入口
#
# ── 环境探测（CONTRIBUTING §2 的环境探测约定）──────────────────────────────
# ⚠ 这里**不写条数**：该小节的约定会增补（写「6 条」就会过期）。
# ⚠ **★ 本主题是全仓库第一个给探测加「版本下限」的目录。**
#    下面比 `70-frameworks/pytorch` 多一次版本检查，理由是：
#      · 本主题的 02_e8m0_and_mx.py 依赖 `torch.float8_e8m0fnu`（**≥ 2.8**）；
#      · `find_spec("torch")` 只回答「模块**可见**」，**不回答「版本够不够」**；
#      · 本机同时装着 torch 2.6.0 / 2.7.1 两个旧版本，**且在候选顺序上排在前面**，
#        只查 find_spec 会选中它们，程序会崩在「找不到 float8_e8m0fnu」上
#        ——**错误信息指向错误的方向**（看着像 torch 没装，其实是版本太旧）。
#    **⚠ 版本比较必须转成整数元组，不能用字符串**：
#      字符串比较 `"2.10" >= "2.8"` 是 **False**（逐字符比，`'1' < '8'`），
#      **恰好在 2.10 这个版本上判断错误**，而且**不报错，只会静默跳过或选错**。
# 1. **不得硬编码某台机器的目录布局**；候选以通用位置为主，本机布局只作兜底并注明。
# 2. 探测判据用 `importlib.util.find_spec("torch")`，**不要用 `import torch`**——
#    前者只回答「模块是否可见」，不把 torch 载进内存。
# 3. 找到后**打印选中的是哪一个（含 torch 版本）**，供文档 §5 直接引用为测量条件。
# 4. 找不到时**以可操作的错误信息退出**，不要静默回落到 python3 去跑一个假失败。
# 5. ⚠ `PYTHON ?=` 挡不住空值：命令行传 `PYTHON=`（空值）也算「已赋值」，
#    默认值不会被填上、探测被整个跳过。所以下面 `run` 里**另有一次空值检查**。
# 6. ⚠ `PYTHON=~/...` 的坑：make 本身不做 `~` 展开，值就是字面量 `~/...`。
#    它在**未加引号的命令位置**能跑（recipe 交给 /bin/sh，shell 会展开词首的 `~`），
#    但**加引号的位置**（`[ -x "$(PYTHON)" ]`）永远是字面量、必然失败。
#    所以报错信息里要教用户写 `PYTHON=$HOME/...`。
#
# ── 跨版本对拍（本主题特有）────────────────────────────────────────────────
# 文档里「fp8 溢出行为跨版本变了」这条结论需要 **< 2.8** 那一侧的反例，而下限会跳过它们。
# 做法是**显式指定**（本机路径，仅供对拍；正常跑用 make run 即可）：
#   make 01_fp8_formats PYTHON=$HOME/miniconda3/envs/deep-prac/bin/python   # torch 2.6.0
#   make 01_fp8_formats PYTHON=$HOME/miniconda3/envs/dp-test/bin/python    # torch 2.7.1
#   make 01_fp8_formats PYTHON=$HOME/PyCharmMiscProject/.venv/bin/python    # torch 2.13.0
# ⚠ 2.6.0 / 2.7.1 上**只有 01 能跑**：02 依赖 `float8_e8m0fnu`（≥ 2.8），
#   03 / 04 不依赖 ≥ 2.8 的特性，但本目录的 `make run` 是一整串，会在 02 处停。
#   → 跨版本对拍请用 `make 01_fp8_formats ...` 这种**单程序**形式。

SCRIPTS := 01_fp8_formats 02_e8m0_and_mx 03_bf16_fp16_tf32 04_integer_formats

# ★ 版本下限。整数序列比较（见上文），**不用字符串**。
TORCH_MIN_MAJOR := 2
TORCH_MIN_MINOR := 8
#
# ⚠ 下面探测里**不能**把这两个变量放进一对圆括号里**再**跟一个多余的 `)`，
#   写法必须是 `v>=[$(TORCH_MIN_MAJOR),$(TORCH_MIN_MINOR)]`（**方括号**），
#   不能写成 `v>=($(TORCH_MIN_MAJOR),$(TORCH_MIN_MINOR))`。原因是 make 在展开
#   `$(shell ...)` 之前要先**按括号配对找出它的结尾**，而配对是**纯字符计数**：
#   `sys.exit(` 已经开了一层，随后 `)` 若是源码里字面的那一只，就把 `$(shell)`
#   提前闭合了——剩下的文本会被当成变量值的一部分，且**不报「语法错」，
#   只报一个指向别处的 `/bin/sh: unexpected EOF`**（本目录实测踩过）。
#   方括号不参与配对，所以 `[...]` 写法没有这个问题。

# 候选顺序：通用位置在前，本机布局兜底。每个候选都先要求「文件可执行」，再问
# 「torch 是否可见」，最后问「版本够不够」——三步都不进 python 的 import 路径
# （第三步要 import torch，是唯一的例外：`find_spec` 回答不了版本）。
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
                2>/dev/null || continue; \
            $$p -c 'import torch,sys; v=[int(x) for x in torch.__version__.split(".")[:2]]; sys.exit(0 if v>=[$(TORCH_MIN_MAJOR),$(TORCH_MIN_MINOR)] else 1)' \
                2>/dev/null && { echo "$$p"; break; }; \
          done)

.PHONY: all run verify clean
all: run

# ★ 空值检查不能用 `[ -z "$(PYTHON)" ]` 之外的写法；也不能只靠 `?=` 的默认值兜底。
run:
	@if [ -z "$(PYTHON)" ]; then \
		echo "错误：找不到装了 torch (>= $(TORCH_MIN_MAJOR).$(TORCH_MIN_MINOR)) 的 Python 解释器。"; \
		echo ""; \
		echo "本目录的编号程序需要真实 torch，且 02 依赖 torch >= $(TORCH_MIN_MAJOR).$(TORCH_MIN_MINOR) 才有的"; \
		echo "  torch.float8_e8m0fnu（旧版本里这个 dtype 根本不存在）。可用的做法："; \
		echo "  1) 激活一个装了 torch >= $(TORCH_MIN_MAJOR).$(TORCH_MIN_MINOR) 的环境后再跑，或"; \
		echo "  2) 显式指定解释器：make run PYTHON=/path/to/python"; \
		echo "     ⚠ 路径里若要用家目录，请写 \$$HOME/... 而**不要**写 ~/...："; \
		echo "        make 自身不做 ~ 展开，而探测/校验位置一定会给变量加引号，"; \
		echo "        加了引号 ~ 就永远是字面量、必然被判定为「不存在」。"; \
		echo "        （对照：make 01_fp8_formats PYTHON=~/x/bin/python 能跑，"; \
		echo "          因为那个位置没有引号、由 /bin/sh 展开。）"; \
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
