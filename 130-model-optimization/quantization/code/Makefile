# 130-model-optimization/quantization 示例构建
#
# ★ 本主题是 2026-10-13 由 6 个子主题（numerical-formats / theory / ptq / qat /
#   quantized-kernels / kv-cache-quantization）**合并**而成的单一 `quantization` 主题。
#   20 篇文档编号 01–20，与 `code/` 下 20 个编号程序一一对应（CONTRIBUTING §2 的外键）。
#
# ★ 代码形态（CONTRIBUTING §2 判定表**第三种**）：编号程序需要**非默认环境**（要 `import torch`，
#   而默认 python3 里没有它）。三类断言的分工是：
#     · **B 代价模型** → **编号程序内部**（代价模型的参照是真系统，只有在能 import
#                        到它的环境里测才有意义）
#     · **A / C**       → **各个 verify 模块**（默认 python3 下也能复核的那一半）
#   `make verify` 用默认 python3；`make run` 自动探测一个装了 torch ≥ 2.8 的解释器。
#   **两个入口不得混用。**
#
# ★ **非编号辅助模块**（CONTRIBUTING §2 允许，且**不计入**「文档数 = 编号程序数」的核对）。
#   合并 6 个子主题时，它们各自的 `specs.py` / `verify.py` **没有合并成一份**，而是按来源
#   改名后并列 —— 这是一条**刻意的选择**，理由有两条：
#     ① 6 份 specs 有 **8 个跨主题重名**常量（`MEASURED_VERSIONS` ×6、`TORCH_MIN_REASON` ×5、
#        `FP16_MAX` ×2、`TERMS` ×2 等），其中若干条**同名不同值**（各主题的版本下限理由不同）
#        —— 合成一份就得给它们改名，而**改名会同时改动编号程序里的断言**，风险大而收益为零；
#     ② 每个子主题的 `verify` 有**自己的 A/C 编号**，20 篇文档按 `verify_<组>.py 的 A.n`
#        逐条引用它；合并编号会让**全部文档的行内引用同时失真**。
#   ⇒ 保留 6 + 6 个辅助模块，`verify.py` 只做**编排**（见下）。
#
# 用法：
#   make                    同 make run
#   make run                自动探测装了 torch (>= 2.8) 的解释器，依次跑 20 个编号程序
#   make verify             用**默认 python3** 跑 verify.py（它依次调用 6 个 verify 模块）
#   make 09_gptq            只跑一个（省略 .py 后缀）
#   make clean              清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」）：
#   verify.py            纯 python3 的**编排入口**：依次调用下面 6 个模块的 `main()`
#   verify_formats.py    对应文档 01–04（数值格式）
#   verify_theory.py     对应文档 05–07（量化误差理论）
#   verify_ptq.py        对应文档 08–11（训练后量化）
#   verify_qat.py        对应文档 12–14（量化感知训练）
#   verify_kernels.py    对应文档 15–17（量化算子）
#   verify_kv.py         对应文档 18–20（KV cache 量化）
#   specs_*.py           上面 6 组各自的事实源（外部常量、论文印值、本机实测基线），
#                        每条带 `# 来源：`；**编号程序与 verify 模块各自 import 自己那一份**
#   Makefile / requirements.txt   构建入口
#     ⚠ requirements 里的 **scipy 是可选的**：`08_calibration.py` 里 TensorRT entropy 校准器
#       那份移植代码**首选** `scipy.stats.entropy`（与 TensorRT 源码逐字对应）；
#       **没装 scipy 时退到 numpy 向量化的 `Σ p·ln(p/q)`**。装不装都不影响其余 19 个程序。
#
# ── 文档与代码的外键 ──────────────────────────────────────────────────────
#   01-fp8-family.md              ↔ 01_fp8_formats.py      05-error-bounds.md             ↔ 05_error_bounds.py
#   02-e8m0-and-mx.md             ↔ 02_e8m0_and_mx.py      06-granularity.md              ↔ 06_granularity.py
#   03-bf16-fp16-tf32.md          ↔ 03_bf16_fp16_tf32.py   07-outliers-and-propagation.md ↔ 07_outliers.py
#   04-integer-formats.md         ↔ 04_integer_formats.py  08-calibration.md              ↔ 08_calibration.py
#   09-gptq.md                    ↔ 09_gptq.py             10-awq-and-smoothquant.md      ↔ 10_awq_smoothquant.py
#   11-rounding.md                ↔ 11_rounding.py         12-ste.md                      ↔ 12_ste.py
#   13-fake-quant-and-lsq.md      ↔ 13_fake_quant_lsq.py   14-pact-and-comparison.md      ↔ 14_pact.py
#   15-w8a8.md                    ↔ 15_w8a8.py             16-w4a16-packing.md            ↔ 16_w4a16_packing.py
#   17-kernel-design.md           ↔ 17_kernel_design.py    18-kv-cache-basics.md          ↔ 18_kv_cache_size.py
#   19-kivi-asymmetry.md          ↔ 19_kivi.py             20-low-bit-and-engines.md      ↔ 20_lowbit_engines.py
#
# ── 环境探测（CONTRIBUTING §2 的环境探测约定）──────────────────────────────
# ⚠ 这里**不写条数**：该小节的约定会增补（写「6 条」就会过期）。
#
# ⚠ **版本下限 (2, 8) 的理由按组不同，照抄时不要互相搬。**
#    · `numerical-formats` 的 ≥ 2.8 是**真实依赖**：它的 02 需要 `torch.float8_e8m0fnu`，
#      旧版本里那个 dtype 根本不存在。
#    · 其余各组的 ≥ 2.8 是**对齐选择**：使 `make run` 在多 torch 的机器上选中同一解释器，
#      从而各文档的测量条件可比。**写成依赖就是编机制从句**（那会编出一句假话）。
#
# ⚠ 探测里**不能**把版本变量放进圆括号：写法必须是
#   `v>=[$(TORCH_MIN_MAJOR),$(TORCH_MIN_MINOR)]`（**方括号**）。原因是 make 在展开
#   `$(shell ...)` 前要按**纯字符计数**配对括号，`sys.exit(` 已经开了一层，
#   源码里字面的 `)` 会提前闭合它，且**不报语法错、只报一个指向别处的
#   `/bin/sh: unexpected EOF`**（同域历史实测踩过）。
#
# ⚠ **版本比较必须转成整数元组，不能用字符串。**
#   字符串比较 `"2.10" >= "2.8"` 是 **False**（逐字符比，`'1' < '8'`）——
#   **恰好在 2.10 这个版本上判断错误**，而且**不报错，只会静默跳过或选错**。
#
# 1. **不得硬编码某台机器的目录布局**；候选以通用位置为主，本机布局只作兜底并注明。
# 2. 探测判据用 `importlib.util.find_spec("torch")`，**不要用 `import torch`**——
#    前者只回答「模块是否可见」，不把 torch 载进内存。
# 3. 找到后**打印选中的是哪一个（含 torch 版本）**，供各文档 §5 直接引用为测量条件。
# 4. 找不到时**以可操作的错误信息退出**，不要静默回落到 python3 去跑一个假失败。
# 5. ⚠ `PYTHON ?=` 挡不住空值：命令行传 `PYTHON=`（空值）也算「已赋值」，
#    默认值不会被填上、探测被整个跳过。所以下面 `run` 里**另有一次空值检查**。
# 6. ⚠ `PYTHON=~/...` 的坑：make 本身不做 `~` 展开，值就是字面量 `~/...`。
#    它在**未加引号的命令位置**能跑（recipe 交给 /bin/sh，shell 会展开词首的 `~`），
#    但**加引号的位置**（`[ -x "$(PYTHON)" ]`）永远是字面量、必然失败。
#    所以报错信息里要教用户写 `PYTHON=$HOME/...`。

SCRIPTS := 01_fp8_formats 02_e8m0_and_mx 03_bf16_fp16_tf32 04_integer_formats \
           05_error_bounds 06_granularity 07_outliers \
           08_calibration 09_gptq 10_awq_smoothquant 11_rounding \
           12_ste 13_fake_quant_lsq 14_pact \
           15_w8a8 16_w4a16_packing 17_kernel_design \
           18_kv_cache_size 19_kivi 20_lowbit_engines

# ★ 版本下限。整数序列比较（见上文），**不用字符串**。
#   ⚠ (2, 8) 对 `numerical-formats` 那一组是**依赖**，对其余各组是**对齐选择** —— 见上文。
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

# ★ 空值检查不能只靠 `?=` 的默认值兜底。
run:
	@if [ -z "$(PYTHON)" ]; then \
		echo "错误：找不到装了 torch (>= $(TORCH_MIN_MAJOR).$(TORCH_MIN_MINOR)) 的 Python 解释器。"; \
		echo ""; \
		echo "本主题的 20 个编号程序需要真实 torch。可用的做法："; \
		echo "  1) 激活一个装了 torch 的环境后再跑，或"; \
		echo "  2) 显式指定解释器：make run PYTHON=/path/to/python"; \
		echo "     ⚠ 路径里若要用家目录，请写 \$$HOME/... 而**不要**写 ~/...："; \
		echo "        make 自身不做 ~ 展开，而探测/校验位置一定会给变量加引号，"; \
		echo "        加了引号 ~ 就永远是字面量、必然被判定为「不存在」。"; \
		echo "        （对照：make 09_gptq PYTHON=~/x/bin/python 能跑，"; \
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
