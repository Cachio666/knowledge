# 45-kernel-lifecycle/requirement-discovery 示例构建
#
# ★ 本目录是 CONTRIBUTING §2 的**第二种形态**（编号程序跑得了）：
#   三个编号程序是纯 `.py`、默认 `python3` 即可跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束 —— 即「断言失败 ⇒ `make run` 非零」。
#   **没有 `verify.py`**：本主题不像 `40-kernels` 那样有「本机跑不了的内核本体」，
#   所以不存在「另置一份本机复核」的必要（域 README「代码约定」一节）。
#
# 用法：
#   make                         同 make run
#   make run                     依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_operator_inventory   只跑一个（省略 .py 后缀）
#   make lint                    AST 检查：确认没有 import torch（不用 grep）
#   make clean                   清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   specs.py          本主题的外部事实常量源，**每条带 `# 来源：` 与来源等级（A–E）**
#   Makefile / requirements.txt   构建入口
#
# ── 为什么不用环境探测（对照 `130-model-optimization/*` 的第三种形态）───────────────
# 本主题的三个程序**不 import torch**（域 README：本域无 GPU，不碰框架），
# 只 import `specs` 与标准库。所以 CONTRIBUTING 那套「探测一个装了框架的解释器」
# 的机器在这里**没有对象**，写了就是给读者增加一条永远不会触发的分支。
# ⚠ 但也**不得**反过来把下限写死成某个本机版本：本主题只用标准库里
#   自 Python 3.8 起就稳定的语法与 API，所以 `python3` 就是正确的目标。
#
# ── 每个程序「可以被反驳」的判据（域 README）────────────────────────────────
# 把文档里对应的论断改错，程序应当报错或给出相反结论。三个程序各自的反驳点在
# 其文件头的 `★ 本程序的可证伪断言` 段里逐条列出。
# ★ `make lint` 是给「不 import torch」这一条硬约束用的**机械检查**：
#   用 `ast.parse` 读真实语法树，**不用 grep** —— grep 会被注释和字符串骗过
#   （本目录的注释里就大量出现 "torch" 这个词，grep 必然假阳性）。

PYTHON  ?= python3
SCRIPTS := 01_operator_inventory 02_roofline_verdict 03_optimization_priority

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
