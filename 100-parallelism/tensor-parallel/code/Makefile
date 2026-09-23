# 100-parallelism/tensor-parallel 示例构建
#
# ★ 本目录与同域另外五个主题是**同一形态**（`CONTRIBUTING` §2 的第二种情形）：
#   编号程序是纯 `.py`，**默认 `python3` 下可跑** ⇒ **三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**，
#   也**没有** `meta_falsify.py`（那是 `50-performance` 这种**测量域**的形态，见域 README）。
#
# ★★ **Chunk 2 已落盘**（2026-09-23）：三个编号程序与 `mutate.py` 都已就位 ⇒
#    `make run` 走**非空分母**分支，`SCRIPTS` 一行就是下面那三个。
#    01_column_row.py   ↔ 01-column-and-row.md    （列切输出维 / 行切输入维 / 配对约束）
#    02_heads_vocab.py  ↔ 02-heads-and-vocab.md   （头切分 / vocab 并行 / 不整除的三种处置）
#    03_saturation.py   ↔ 03-saturation-and-limits.md（TP 的通信量【也饱和】）
#    （**编号即外键**：与同目录的三篇文档一一对应；`CONTRIBUTING` §2：`<名称>` 与
#     `<程序名>` 不要求相同 —— 文档名讲**讲什么**、程序名讲**这个程序做什么**。）
#
# ★★★ **本主题最重要的一条边界（三个程序都在 `preflight()` 里回链它）**：
#    TP 的三条 ——「通信发生在**每一层内部**」「通信量与**激活**成正比而不是与参数量成正比」
#    「**只在高带宽互联上可用 / 通常限于单节点 8 卡**（规模上限由节点内互联决定）」
#    —— 已经由 `10-foundations/performance-models/02-amdahl-and-scaling.md:223` **逐字写完**。
#    ⇒ ★★ **本主题不重写它们的【结论】**（★ 「复述它哪一句、补它哪一句」写在 `01` 的
#       §1.6/§5 与 `03` 的 §1.7/§5 —— 复述与补写都在 `:223` 的结论范围内，不构成第二处正文；
#       否则是 `P2`「同一个载体重复」）。
#    **本主题只写它没写的**：
#      ① **切法本身**（列/行的**配对**）与「列并行为什么非得在前面」的**机制**（`01`）；
#      ② **`2 + 2` 的逐字出处**（`:223` 只说「各一次」＝前向 2 次，**漏了 backward 那 2 次**）；
#      ③ **通信量也饱和**这一层（`(t−1)/t → 1` 与 DP 的 `2(N−1)/N → 2` 是同一个结构，
#         但**饱和值不同**：DP `2S` ∝ 参数量、TP `8bsh` ∝ 激活量）—— `03`。
#    ★ 另有两条硬边界：**GEMM 的 tiling/分块/层级** → `40-kernels/gemm/` 与
#      `30-gpu-programming/cutlass/02-hierarchical-tiling.md`（本域只写「矩阵怎么切到设备上」）；
#      **注意力变体的结构**（MHA/MQA/GQA、KV head 配比）→
#      `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`（本域只写「头怎么分给设备」）。
#
# ★★★ **本主题最要紧的一处形态风险**：三个程序**全是模拟器**（`all_reduce = Σ局部`、
#    `all_gather = concat`、`03` 更是**纯公式**）⇒ A 类判据极易写成**自证**。
#    落法（三条，逐条对应一个程序里的真实现场）：
#      · **纯搬运处的等价性判据不当 A 类** —— `01` 的 `D.10` 与 `02` 的 `D.10` 把
#        「列切 == 单卡」的**恒真性**变成一条**机械检查**；★ `02` 的 `D.10` 还量出
#        「vocab 维列切的差**随形状变**」（`0.0` 与非零都有）⇒ **它连恒真都不是**；
#      · **每条 A 类都配一个「把参照故意改错」的守卫**（`CONTRIBUTING:235`）——
#        `A.1g` / `A.4g` / `A.3` 的跨格比较 / `C.1` 的 2 倍差，而那些「改错的目标」全部是
#        **模块级常量** ⇒ 它们会进 `make mutate` 的分母（本主题因此**特别需要**下面那个目标）；
#      · **参照实现不得调用被测实现的任何函数** —— 三个程序都带调用计数器（`D.7c`），
#        且**先证明计数器非空**（被测路径跑一次必须 > 0；否则那是恒真的「没调用」）。
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
#                     **把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
#                   ★★ 本域所有数值常量**一律 `< 2^53`**（`mutate.py` 的浮点变异是 `v + 1.0`），
#                     且**幂次写成十进制字面量**（`_walk()` 不下钻 `BinOp`）。
#                   ★★★ 本主题的两篇论文**都不含公式文本**（公式在原始字节上是 `<math>`）
#                     ⇒ 引文常量**一律取不含公式的散文**，每一条都在原始字节上核过次数；
#                     公式的**系数**（`8` / `4` / `2`）单独寄存为**整数**。
#   mutate.py       ★ **叶子级变异测试**（**Chunk 2 已落盘**；从 `100-parallelism/data-parallel/code/`
#                   逐字复制，只改**主题名、临时目录前缀与抬头散文**）。用 **AST** 从编号程序里读出
#                   `ASSERT_FAIL_MARK`（**不得写死**），逐个叶子改值，看断言是否捕获。
#                   **只在副本上做**（仓库文件只读）。分母为 0 时以**说明性非零码**退出。
#   Makefile / requirements.txt   构建入口
#
# ── 本机的硬约束 ──────────────────────────────────────────────────────────────
#   **无 GPU、无多卡、无 NCCL、无真实网络**；`python3` = 3.13.5、`numpy` = 2.4.4（已装）。
#   ★ **本域不 import 任何第三方框架**（`CONTRIBUTING` §2 的「永不 import 目标框架」，
#     原文的措辞比 torch 宽 —— 本域按宽的口径执行），`numpy` 是唯一例外（见 requirements.txt）。
#   ★★ **通信被实现为一个显式的张量搬运**（`all_reduce = Σ局部`、`all_gather = concat`）——
#     本域的知识对象是**切分策略**，不是通信原语的实现（通信算法的正文在 `110-communication`）。
#     ⇒ 一切程序验证的是**切分的代数**（等价性 / 通信次数 / 字节账 / 负载分布），
#       **不是**任何关于真实集群性能的结论。★ 强制限定语：**「不是 NCCL，也不是真机」**。
#     ★★★ 而 `03_saturation.py` 更进一步：它**没有张量**，是**纯公式对拍**
#       （所以它的数字绝大多数要标为**推导值**）。
#   ★★ **本主题只写**：TP 的**列并行 / 行并行与配对约束**、**头切分与 vocab 并行**、
#     **不整除的三种处置**、**通信量的饱和**（切在哪个轴 / 怎么分 / 切不匀怎么办 / 代价多大）
#   ★ **本主题的 `ASSERT_FAIL_MARK`**：`^^ FAIL: tensor-parallel`（**每个主题一个，不跨主题照抄**；
#     标记**只允许出现在 `print()` 的字符串里**，**不得**出现在任何 shell 命令行里）；
#     每个编号程序另加一条**标记自检**断言（去重后只有一个、等于本主题的值、含 `FAIL`，**字面量比较**）。
#   ★ 同域的 `../probes/` 是**域级**探测目录（**只打印、不断言**），**不在本 Makefile 的扫描范围内**。

PYTHON  ?= python3
SCRIPTS := 01_column_row 02_heads_vocab 03_saturation

.PHONY: all run lint mutate lines clean check-env

all: run

# ★★ 整个目标**必须写成一条 shell**（用 `; \` 串起来），不能拆成多行 recipe。
#   原因：make 的每一行 recipe 各起一个 shell，`exit 0` 只结束**那一行** ——
#   拆开写时，空分母分支里的 `exit 0` 之后，尾巴上的
#   `=== 全部 0 个编号程序退出码 0 ===` **照样会打印**，
#   于是 `make run` 会输出一句自相矛盾的「全绿」。
#   （`80-model-architectures/transformer/code/Makefile` 的同名目标有这个隐患；
#    `90-training/autograd-and-backward/code/Makefile` 已按此串起来，本文件照搬。）
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
#   「已赋值」—— 默认值不会被填上，探测被整个跳过（`CONTRIBUTING:196-199`）。
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
		echo "     引号里的 ~ 是字面量，[ -x ] 必然失败。请改用 $$HOME/env/bin/python。"; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
		|| { echo "!! 需要 python >= 3.9"; exit 1; }
	@$(PYTHON) -c 'import numpy' 2>/dev/null \
		|| { echo "!! 缺 numpy。请先: $(PYTHON) -m pip install -r requirements.txt"; exit 1; }

# ★ `ast` 而非 grep：注释与文档串里出现框架名**不算违规**，真 import 才算。
#   ★ 扫的是本目录**全部** `*.py`（含 specs.py / mutate.py 与本主题的编号程序）。
#   ★★ 本域比 `50-performance` 的 lint 更严：那边禁的是 torch/triton/numba/cuda 四类，
#      这边禁的是**全部深度学习框架与运行时**（本域**任何第三方框架都不 import**）。
#   ★★ 本主题**特别需要这一条**：三个程序讲的是**切哪个轴 / 头怎么分 / 通信量多大**，
#      文档串里满是 `all_reduce` / `all_gather` / `Megatron` 这些名字，
#      只有 AST 能分清「提到」与「import」。
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
		echo "  ★ 若本主题还没有 mutate.py（Chunk 0），也会给 2 —— 那是「文件不存在」，不是读数不可信。"; \
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
