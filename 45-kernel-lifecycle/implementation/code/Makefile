# 45-kernel-lifecycle/implementation 示例构建
#
# ★ 本目录与同域的 `requirement-discovery` / `operator-design` 是**同一形态**
#   （CONTRIBUTING §2 的第二种形态）：三个编号程序是纯 `.py`，本机可直接跑，
#   因此**三类断言放在编号程序内部**，失败时以**非零退出码**结束
#   —— 即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                         同 make run
#   make run                     依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_reference_independence   只跑一个（省略 .py 后缀）
#   make lint                    AST 检查：确认没有 import torch（不用 grep）
#   make clean                   清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   specs.py          本主题的外部事实常量源，**每条带 `# 来源：` 与来源等级（A–E）**
#   Makefile / requirements.txt   构建入口
#
# ── 与同域前两个主题的形态差异（写出来，避免读者以为这里不一致）──────────────
#   ① 三个程序**都用 numpy**（01 的数组构造、02 的 fp16/bf16 存储舍入、
#      03 的 fp32 累加与随机置换）。域 README 的口径「纯 python3 + numpy」成立。
#   ② **本主题有一套第 ② 类额外机制**（与 `operator-design` 同源）：
#      `01_reference_independence.py` 用 `ast` 读同目录另外两个程序的 `COVERED_*`
#      元组，断言**三者的并集 == specs.py 的全部模块级常量**。
#      ⇒ **在 specs.py 里加一个没人断言的常量，`make run` 立刻失败。**
#      这是「常量清单规模 = 变异测试规模」这条纪律的机器化防线。
#      ★★ **粒度是「字面量叶子」，不是「常量名」。** 元组 / 列表 / 字典 / 嵌套结构里
#         **每一个字面量**都算一个待钉住的叶子，`N` 应当是**叶子数**。
#         本主题的叶子级变异测试：`N = 151`（`specs.py` 的 66 个常量展开成 151 个叶子；
#         `VLLM_DEFAULT_TOLERANCES` 一个常量就有 21 个叶子）。
#         **只改变量名那一层会漏**：实测漏过 `ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE
#         = (100, 200)` 的两条叶子 —— 它原先只出现在两处 f-string 的 `print` 和一个名字
#         清单里（清单不是断言），改成 `(100, 201)` 时 `make run` 仍退出 0。
#      ★ 三个程序各有一段 **§L 结构守卫**（在**任何使用之前**先钉住容器常量的逐格字面量）。
#        它有两个作用：① 把「只 print」的叶子逼出来；② 字典的**键**被改错时，
#        `SP.DICT[k]` 会抛 `KeyError`（**harness 异常 = 假捕获**），
#        先断言结构就变成干净的 `[FAIL]` + `exit 1`。
#      ★ 做变异测试时**要清 `__pycache__`**：长度不变的字面量变异（`100`→`101`）在
#        陈旧 .pyc 被判有效时会让程序读不到新值 ⇒ **假漏网**（`make clean` 即可）。
#   ③ ★ 本主题**没有 `verify.py` 也没有外部数据文件**：
#      02 里 bf16 的舍入由程序**自己写两条机制不同的路径**并交叉核对
#      （numpy 没有 bfloat16 这个 dtype）——这是本主题「参考实现要独立」那条
#      论断在自己身上的应用。
#
# ── 每个程序「可以被反驳」的判据（域 README）────────────────────────────────
# 把文档里对应的论断改错，程序应当报错或给出相反结论。三个程序各自的反驳点在其
# 文件头的 `★ 本程序的可证伪断言` 段里逐条列出。★ 三篇都有一条**被本程序推翻的
# 规格条目**在输出里明写出来（02 的 §A.3/A.4 最典型）。
#
# ★ `make lint` 是给「不 import torch」这一条硬约束用的**机械检查**：
#   用 `ast.parse` 读真实语法树，**不用 grep** —— grep 会被注释和字符串骗过
#   （本目录的注释里就出现 "torch" 这个词，grep 必然假阳性）。

PYTHON  ?= python3
SCRIPTS := 01_reference_independence 02_verification_matrix 03_debug_order

.PHONY: all run lint clean
all: run

run:
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py || exit 1; \
		echo; \
	done
	@echo "=== 三个编号程序全部退出码 0 ==="

# ★ `ast.parse` 而非 grep：注释里出现 "torch" 不算违规，真 import 才算。
lint:
	@$(PYTHON) -c 'import ast, sys, pathlib; \
scripts = sorted(pathlib.Path(".").glob("[0-9][0-9]_*.py")); \
bad = [(p.name, a.name) for p in scripts \
       for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) \
       if isinstance(n, ast.Import) for a in n.names if a.name.split(".")[0] == "torch"]; \
bad += [(p.name, n.module) for p in scripts \
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) \
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == "torch"]; \
print("AST 检查：编号程序共 %d 个，import torch 的 %d 个" % (len(scripts), len(bad))); \
print("           （含 ImportFrom，比 v1 的 Import-only 检查多覆盖一种写法）"); \
sys.exit(1 if bad else 0)'
	@echo "✓ 没有编号程序 import torch"

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
