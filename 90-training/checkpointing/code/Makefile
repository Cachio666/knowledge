# 90-training/checkpointing 示例构建
#
# ★ 本目录与同域另外四个主题是**同一形态**（CONTRIBUTING §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**，
#   也**没有** `meta_falsify.py`（那是 `50-performance` 这种**测量域**的形态，见域 README）。
#
# ★ 本 Makefile 与 `mutate.py` **从 `../initialization-and-stability/code/`（Chunk 4，最新）
#   复制过来**并只改与本主题有关的常量（`SCRIPTS` 的名字、`mutate.py` 的临时目录前缀、
#   路径常量）。**没有第二个模板。**
#
# ★★ Chunk 5 已落盘（2026-09-23）：
#    01_state_bytes.py   ↔ 01-state-and-bytes.md                （状态的组成与字节账；★ 判据 ①）
#    02_sharded.py       ↔ 02-sharded-save-and-load.md          （分片保存与恢复；★ 判据 ② / ③）
#    03_consistency.py   ↔ 03-consistency-and-fault-tolerance.md（一致性恢复；★ 判据 ④）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**）。
#    ★ 本主题**没有**上游指名的文件名（那是 `optimizers/01` 的待遇）。
#    ★★ **本主题的禁区最少（一行）**：导出/推理格式的选择（`.keras` vs SavedModel）
#       → `70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；
#       三个程序里各有一段回链声明（另有两条**接口级**回链：异步落盘 → `110-communication`、
#       状态分片与通信量 → `100-parallelism`）。
#    ★★★ **本主题与其余四个主题最大的一处不同**：它的上游来源是 **[B] 某实现的文档**
#       （`torch.distributed.checkpoint`，PyTorch 2.14）而不是论文 ⇒
#       三个程序**一行 torch 都不 import**，把那份文档的**契约**当作 `specs.py` 里的
#       [B] 常量、按它逐字写的规则在本机用 numpy 重演一遍。
#       ⇒ 由此得到的数**不是「PyTorch 在本机跑出来的」**，而是「按 PyTorch 的**文档**
#         在本机推出来的」—— **推导值**。这个区别写在每个程序的抬头与文档的 §5 里。
#
# 用法：
#   make                    同 make run
#   make run                依次跑本目录全部编号程序（任一断言失败即非零退出）
#   make <程序名>           只跑一个（省略 .py 后缀）
#   make lint               AST 检查：本目录**全部** .py 里没有 import 目标框架
#   make mutate             ★ 叶子级变异测试（非编号工具 mutate.py；**不进 run**）
#   make lines              列出本目录各 .py 的行数
#   make clean              清理 __pycache__ 与临时文件
#
# ── 非编号文件（不计入「文档数 = 编号程序数」的核对，CONTRIBUTING §2）───────────
#   specs.py        外部事实常量源：本主题**真的会断言**的那批常量。
#                   ★ 每条带 `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`；
#                   ★★ **来源必须分级并写明是哪一级** —— 本域的头号错误模式是
#                     **把「某实现的文档」当成「规范的文档」**（计划 §0.1 的 **P19**）；
#                     本主题有三节**分属三级**：
#                       · **ZeRO 的字节账与 PaLM 的逐位确定性** → **[C] 论文**
#                         （逐条注明是否确认同行评审）；
#                       · **DCP 的分片语义** → **[B] PyTorch 2.14 的文档**
#                         （★ 写明 URL + 字节数 + `sha256[:16]`，与计划 §C.2b 的 T3 互钉）；
#                       · **分片格式的头字节数 / 版本号** → **设计常量（E 级）**，
#                         单独一节存放 —— ★ 因为 DCP 的文档**不给**这个数
#                         （已登记 `UNSOURCED`：`dcp_per_shard_header_bytes`）。
#                   ★★ **本主题的一处量法口径现场**：同一页上 `torch.distributed.checkpoint`
#                      在**原始字节**上计 **547** 次、**剥标签 + 折叠空白后**计 **71** 次
#                      ⇒ 两个数**都存**，引用时必须写明载体（Y15 / T34）。
#   mutate.py       ★ **叶子级变异测试**（计划 §1.4 的决定）。用 **AST** 从编号程序里
#                   读出 `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，看断言是否捕获。
#                   **只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、不训练任何真模型**；`python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域不 import 任何第三方框架**（计划 §0.2），`numpy` 是唯一例外。
#   ★★ **本主题的引用链是「实现文档」而不是「规范」**（P19 的反面做法）：
#      `specs.py` 的第二节把 [B] 的出处写成 **URL + 页面字节数 + sha256 前缀**，
#      因为没有这些就无法确认「引的是哪一版」。
#   ★ **本主题需要写临时文件**（分片要落盘才能量字节）：`02` 与 `03` 各自在
#      `tempfile.mkdtemp()` 里做事、跑完清理 ⇒ **不碰仓库目录**（`CONTRIBUTING:36` 的
#      「`code/` 必须自包含」不受影响：它们只写**系统临时目录**）。
#   ★ 同域的 `../probes/` 是**域级**探测目录，**不在本 Makefile 的扫描范围内**。
#   ★★ **本主题是五个主题里唯一「全库零让渡」的**（计划 §0.7-B 的边界表）⇒
#      它的知识点由本域**自己立**，因此上表的禁区与接口声明更要逐条守住。

PYTHON  ?= python3
SCRIPTS := 01_state_bytes 02_sharded 03_consistency

.PHONY: all run lint mutate lines clean check-env

all: run

# ★★ 整个目标**必须写成一条 shell**（用 `; \` 串起来），不能拆成多行 recipe。
#   原因：make 的每一行 recipe 各起一个 shell，`exit 0` 只结束**那一行** ——
#   拆开写时，空分母分支里的 `exit 0` 之后，尾巴上的
#   `=== 全部 0 个编号程序退出码 0 ===` **照样会打印**，
#   于是 `make run` 会输出一句自相矛盾的「全绿」。
#   （`80-model-architectures/transformer/code/Makefile` 的同名目标有这个隐患；
#    `90-training/initialization-and-stability/code/Makefile` 已按此串起来，本文件照搬。）
run: check-env
	@if [ -z "$(SCRIPTS)" ]; then \
		echo "=== 本主题当前有 0 个编号程序 ==="; \
		echo "    ★ 此刻 make run **不构成任何证据** —— 它一条断言也没跑，"; \
		echo "      不要把它当成「绿」写进任何文档。"; \
		exit 0; \
	fi; \
	rm -f .last_out; \
	for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py > .last_out 2>&1 || { cat .last_out; echo; \
			echo "!! $$s.py 非零退出 —— 断言失败（详情见上）"; \
			rm -f .last_out; exit 1; }; \
		cat .last_out; \
		echo; \
	done; \
	rm -f .last_out; \
	echo "=== 全部 $(words $(SCRIPTS)) 个编号程序退出码 0 ==="; \
	echo "    （失败标记的纪律在**每个程序内部**：它只打印本主题自己的 ASSERT_FAIL_MARK，"; \
	echo "      并有一条「标记自检」断言守着；跨主题照抄标记由 mutate.py 的 AST 读取挡住。）"

# ★ 这个目标的第一个动作是**空值检查**，不是靠 `PYTHON ?=` 的默认值兜底。
#   `?=` 的语义是「**未被赋值时**才赋值」，而命令行传 `PYTHON=`（**空值**）也算
#   「已赋值」—— 默认值不会被填上，探测被整个跳过（CONTRIBUTING:196-199）。
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
	@$(PYTHON) -c 'import numpy' 2>/dev/null \
		|| { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }

# ★ `ast` 而非 grep：注释与文档串里出现框架名**不算违规**，真 import 才算。
#   ★ 扫的是本目录**全部** `*.py`（含 specs.py / mutate.py 与三个编号程序）。
#   ★★ 本主题**特别需要这一条**：三个程序讲的**全是** `torch.distributed.checkpoint` 的
#      契约 —— 注释与文档串里**满是** `torch.distributed.checkpoint` / `FileSystemWriter` /
#      `DefaultLoadPlanner` / `state_dict` 这些名字，只有 AST 能分清「提到」与「import」。
#   ⚠ 某个 .py 有**语法错误**时，`ast.parse` 会抛异常、lint 以非零码结束 —— 这是**故意的**。
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
	print("lint: numpy 是唯一允许的第三方库（见 requirements.txt）"); \
	sys.exit(1 if hit else 0)'

# ★ **不进 `run`**：变异测试要跑很多副本，慢；而且它的结论是「元层」的。
# ★ 退出码的含义：
#   0 = 全部捕获 · 1 = **有漏网**（逐条列出）· 2 = 读数不可信（仓库文件被改过）·
#   3 = 空分母。
#   ⚠ **读数陷阱**：`make mutate > file; echo $?` 拿到的是 **make 自己的码（2）**，
#     不是 `mutate.py` 的。**要看退出码就用 `python3 mutate.py`。**
mutate: check-env
	@$(PYTHON) mutate.py || { rc=$$?; echo; \
		echo "★ mutate.py 的退出码 = $$rc 。它的含义：1 = 有漏网（见上「漏网」清单）；"; \
		echo "  2 = 仓库文件在本次运行中被改动，读数不可信；3 = 空分母。"; \
		exit $$rc; }

lines:
	@$(PYTHON) -c 'import os; [print("%-32s %5d" % (f, sum(1 for _ in open(f, encoding="utf-8")))) for f in sorted(os.listdir(".")) if f.endswith((".py", ".txt", ".md"))]'

# 单独跑一个编号程序（显式目标优先于下面的模式规则）
$(SCRIPTS): check-env
	@$(PYTHON) $@.py

%:
	@$(PYTHON) $@.py

clean:
	rm -rf __pycache__ .last_out .run_summary
