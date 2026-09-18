# 130-quantization/sparsity-and-distillation 示例构建
#
# ★ 本目录是 CONTRIBUTING §2 的**第三种形态**：编号程序需要一个**非默认环境**
#   （要 `import torch`，而默认 python3 里没有它）。三类断言的分工是：
#     · B 代价模型  → **编号程序内部**（代价模型的参照是真系统，只有在能 import
#                     到它的环境里测才有意义）
#     · A / C       → **verify.py**（默认 python3 下也能复核的那一半）
#   `verify.py` **永不 import torch**（也**不 import numpy**），产出**一律是推导**，
#   不得标成本机实测值。本主题的 A/C 验的是**可独立推导的算术与判据**：
#   2:4 的压缩率账（含两个分母）、4:8 paired 的两种 bit/元素口径、
#   三种剪枝判据的公式与归约关系、蒸馏的 T² 标度。
#   编号程序测的是**真实实现的行为**：torch 内部转换函数算出的 metadata 位宽与
#   往返、float32 路径用的是 1:2 还是 2:4、真 torch 上三种判据选出的集合与输出误差、
#   真 autograd 算出的温度梯度。**两者不得互相背书。**
#
# 用法：
#   make                    同 make run
#   make run                自动探测一个装了 torch ≥ 2.8 的解释器，依次跑全部编号程序
#                           （断言失败即非零退出）
#   make verify             用**默认 python3** 跑 verify.py（纯 python3 + 标准库）
#   make 01_sparsity        只跑一个（省略 .py 后缀）
#   make clean              清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   verify.py      纯 python3 的独立复核（A/C），**不 import torch / numpy**
#   specs.py       本主题的外部常量源（硬件粒度、压缩率算术、本机实测基线），
#                  每条带 `# 来源：`
#   Makefile / requirements.txt  构建入口
#
# ── 环境探测（CONTRIBUTING §2 的环境探测约定）──────────────────────────────
# ⚠ 这里**不写条数**：该小节的约定会增补（写「6 条」就会过期）。
#
# ⚠ **★ 本主题的版本下限 (2, 8) 与同域其余主题字面相同，但理由要看清楚，
#    照抄时不要把那几边的理由搬过来。**
#      · `numerical-formats` 的 ≥ 2.8 是**真实依赖**：它的 02 需要 `torch.float8_e8m0fnu`。
#      · **本主题没有这个依赖**：三个编号程序只用到
#        `torch.sparse._semi_structured_conversions`（两个内部函数）、
#        `torch.linalg.inv`、`torch.topk` / `argsort`、`torch.nn.functional.softmax`、
#        `torch.autograd`、`nn.Linear` —— **这些在 2.6.0 上全部存在**。
#        本仓库已在五个解释器（2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）上
#        跑过全部三个程序，**全部通过，且 01 的 metadata 位宽逐格一致**。
#        （`torch.sparse.to_sparse_semi_structured` 这个**公开** API 在 CPU 上会
#          `RuntimeError: Only CUDA tensors are currently supported`，
#          所以本主题用的是内部函数——这一点与版本无关，五个版本都一样。）
#      · 那为什么还要设下限？—— **对齐**：使 `make run` 在装多个 torch 的机器上
#        选中与本域其余主题同一个解释器，从而 §5 的测量条件全域可比。
#    **这是一条「选择」，不是一条「依赖」。** 写成依赖就是 P4（编机制从句）。
#
# ⚠ 下面探测里**不能**把两个变量放进一对圆括号里**再**跟一个多余的 `)`，
#   写法必须是 `v>=[$(TORCH_MIN_MAJOR),$(TORCH_MIN_MINOR)]`（**方括号**），
#   不能写成 `v>=($(TORCH_MIN_MAJOR),$(TORCH_MIN_MINOR))`。原因是 make 在展开
#   `$(shell ...)` 之前要先**按括号配对找出它的结尾**，而配对是**纯字符计数**：
#   `sys.exit(` 已经开了一层，随后 `)` 若是源码里字面的那一只，就把 `$(shell)`
#   提前闭合了——剩下的文本会被当成变量值的一部分，且**不报「语法错」，
#   只报一个指向别处的 `/bin/sh: unexpected EOF`**（同域 `numerical-formats` 实测踩过）。
#   方括号不参与配对，所以 `[...]` 写法没有这个问题。
#
# ⚠ **版本比较必须转成整数元组，不能用字符串。**
#   字符串比较 `"2.10" >= "2.8"` 是 **False**（逐字符比，`'1' < '8'`）——
#   **恰好在 2.10 这个版本上判断错误**，而且**不报错，只会静默跳过或选错**。
#   ⚠ 元组比较还有第二种形态：`(2, 8) >= (2, 8, 0)` 是 **False**——
#   前缀相同时短的更小。所以比较的两边都取 `[:2]`。
#
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
# ── 跨版本对拍 ────────────────────────────────────────────────────────────
# 本主题**没有**依赖版本行为的断言：三个程序在五个解释器上逐格一致（见上）。
# 下限会跳过 2.6.0 / 2.7.1，做法是**显式指定**：
#   make 01_sparsity PYTHON=$HOME/miniconda3/envs/deep-prac/bin/python  # torch 2.6.0
#   make 02_pruning  PYTHON=$HOME/miniconda3/envs/dp-test/bin/python    # torch 2.7.1
#   make 03_distillation PYTHON=/usr/bin/python3                        # torch 2.8.0 (py 3.9)
#   make 01_sparsity PYTHON=$HOME/PyCharmMiscProject/.venv/bin/python   # torch 2.13.0
# ⚠ 用**单程序**形式：`make run PYTHON=...` 是一整串，跑到一半若某个程序非零退出会停住。
# ⚠ **`/usr/bin/python3` 是 Python 3.9.6**，所以三个编号程序**不得使用 3.10+ 的语法**
#   （`match`、`X | Y` 类型标注、`zip(strict=)`）。本主题三个程序都遵守这一条。
# ⚠ 本主题**不撞** §0.5 里那条 scipy 陷阱：三个程序与 verify.py 都不用 scipy。

SCRIPTS := 01_sparsity 02_pruning 03_distillation

# ★ 版本下限。整数序列比较（见上文），**不用字符串**。
#   ⚠ (2, 8) 是**对齐选择**，不是依赖 —— 理由见上面的长注释。
TORCH_MIN_MAJOR := 2
TORCH_MIN_MINOR := 8

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
		echo "本目录的编号程序需要真实 torch（本主题**不依赖**某个新 dtype，"; \
		echo "  $(TORCH_MIN_MAJOR).$(TORCH_MIN_MINOR) 这个下限是为了与本域其余主题选中同一个解释器）。可用的做法："; \
		echo "  1) 激活一个装了 torch 的环境后再跑，或"; \
		echo "  2) 显式指定解释器：make run PYTHON=/path/to/python"; \
		echo "     ⚠ 路径里若要用家目录，请写 \$$HOME/... 而**不要**写 ~/...："; \
		echo "        make 自身不做 ~ 展开，而探测/校验位置一定会给变量加引号，"; \
		echo "        加了引号 ~ 就永远是字面量、必然被判定为「不存在」。"; \
		echo "        （对照：make 01_sparsity PYTHON=~/x/bin/python 能跑，"; \
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

# 纯 python3 + 标准库；**不 import torch、不 import numpy**；产出是推导，不是实测。
verify:
	@echo "解释器：python3（默认）"
	@python3 verify.py

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
