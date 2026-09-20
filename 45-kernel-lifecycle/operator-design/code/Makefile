# 45-kernel-lifecycle/operator-design 示例构建
#
# ★ 本目录与 `requirement-discovery` 是**同一形态**（CONTRIBUTING §2 的第二种形态）：
#   三个编号程序是纯 `.py`，本机可直接跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。
#   **没有 `verify.py`**：本主题不像 `40-kernels` 那样有「本机跑不了的内核本体」。
#
# 用法：
#   make                         同 make run
#   make run                     依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_contract_check       只跑一个（省略 .py 后缀）
#   make lint                    AST 检查：确认没有 import torch（不用 grep）
#   make clean                   清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   specs.py          本主题的外部事实常量源，**每条带 `# 来源：` 与来源等级（A–E）**
#   Makefile / requirements.txt   构建入口
#
# ── 与 `requirement-discovery` 的两处形态差异（写出来，避免读者以为这里不一致）──
#   ① 本主题**用了 numpy**（02/03 需要 fp16/fp32 标量累加与整数矩阵乘），
#      上一主题三个程序连 numpy 都没用。域 README 的口径「纯 python3 + numpy」在本主题
#      **成立**，在上一主题是**子集**。
#   ② 本主题多了一条**跨程序的机械检查**：`01_contract_check.py` 用 `ast` 读同目录的
#      `02_*.py` / `03_*.py`，断言三个 `COVERED_*` 元组的**并集 == specs.py 的全部
#      模块级常量**。⇒ **在 specs.py 里加一个没人断言的常量，`make run` 立刻失败。**
#      这是上一轮 C6（19 条常量 0 引用）的机器化防线。
#
# ── 每个程序「可以被反驳」的判据（域 README）────────────────────────────────
# 把文档里对应的论断改错，程序应当报错或给出相反结论。三个程序各自的反驳点在其
# 文件头的 `★ 本程序的可证伪断言` 段里逐条列出。
# ★ `make lint` 是给「不 import torch」这一条硬约束用的**机械检查**：
#   用 `ast.parse` 读真实语法树，**不用 grep** —— grep 会被注释和字符串骗过
#   （本目录的注释里就出现 "torch" 这个词，grep 必然假阳性）。

PYTHON  ?= python3
SCRIPTS := 01_contract_check 02_error_budget 03_tiling_check

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
bad = []; \
[bad.append((p.name, n.names[0].name.split(".")[0])) \
 for p in sorted(pathlib.Path(".").glob("[0-9][0-9]_*.py")) \
 for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) \
 if isinstance(n, ast.Import) and any(a.name.split(".")[0] == "torch" for a in n.names)]; \
print("AST 检查：编号程序共 %d 个，import torch 的 %d 个" % (len(list(pathlib.Path(".").glob("[0-9][0-9]_*.py"))), len(bad))); \
sys.exit(1 if bad else 0)'
	@echo "✓ 没有编号程序 import torch"

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
