# 110-communication/nccl 示例构建
#
# ★ 本目录与同域另外三个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束。**没有 `verify.py`**，也**没有** `meta_falsify.py`。
#
# ★★ **Chunk 0 只落骨架**：本目录**尚无任何编号程序** ⇒ `SCRIPTS` 为空，
#    `make run` 走**空分母**分支（★ **它不构成证据**）。
#
#    本主题的定位：NCCL 的算法取值（`NCCL_ALGO`）与协议（`NCCL_PROTO`）、拓扑感知、报错归因
#
#   ★ 本主题的禁区：**框架 API 的用法**（全库无正文）· **任何 NCCL 的性能数字**（本机无 NCCL / 无 GPU）
#   ★★ 判据纪律：★★ **本主题【没有 A 类】**（本机无 NCCL）⇒ 按 `CONTRIBUTING` §2 的第三种形态【写明不可测】；★ **不得把「常量钉住」冒充 A 类**
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无 NCCL、无 InfiniBand / RoCE 网卡、无 NVLink**；
#   `python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域最强的一条限定语**：**「不是 NCCL，也不是真机」**。
#   ★ **多进程只进 `../probes/`**（实测：ring allreduce 原型连跑通都不稳，约 80% 概率挂）
#     ⇒ **编号程序一律单进程**，保证 `make run` 快且稳。
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`>< FAIL: nccl`（**每个主题一个，不跨主题照抄**；
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
#   mutate.py       ★ 叶子级变异测试（**与编号程序同时落**）。用 **AST** 读出 `ASSERT_FAIL_MARK`（不得写死）。
#                   盲区：①只变异常量 ②负字面量不进分母 ③前缀式检查漏网 ④`_walk` 不下钻 `BinOp`
#                         ⑤`None` 计为「跳过」 ⑥浮点变异是 `v + 1.0`（大浮点吞掉它 ⇒ **常量 < 2^53**）
#                         ⑧**「整体钉住」代替不了「元素钉住」**（每个元组/列表常量都要逐元素 `==`）
#   Makefile / requirements.txt   构建入口

PYTHON  ?= python3
SCRIPTS :=

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
	@$(PYTHON) -c 'import numpy' 2>/dev/null || { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }

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
