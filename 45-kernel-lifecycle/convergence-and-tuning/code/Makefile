# 45-kernel-lifecycle/convergence-and-tuning 示例构建
#
# ★ 本目录与同域的另外三个主题是**同一形态**（CONTRIBUTING §2 的第二种形态）：
#   三个编号程序是纯 `.py`，本机可直接跑，因此**三类断言放在编号程序内部**，
#   失败时以**非零退出码**结束——即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**。
#
# 用法：
#   make                    同 make run
#   make run                依次跑三个编号程序（任一断言失败即非零退出）
#   make 01_benchmark_hygiene   只跑一个（省略 .py 后缀）
#   make lint               AST 检查：确认没有 import torch（不用 grep）
#   make clean              清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   specs.py          本主题的外部事实常量源，**每条带 `# 来源：` 与来源等级（A–E）**
#   Makefile / requirements.txt   构建入口
#
# ── 与同域前三个主题的形态差异（写出来，避免读者以为这里不一致）──────────────
#   ① 三个程序都用 numpy（01 的池化样本、02/03 的蒙特卡洛与向量化 t 统计量）。
#      域 README 的口径「纯 python3 + numpy」成立；**都没有 import torch**。
#   ② 02 **自带一个 t 分布实现**（正则化不完全 beta 的连分式 + 二分求分位数），
#      并用已知分位数自检。原因是本主题的核心论断之一是「临界值用 z 还是 t 会改变结论」，
#      而 `statistics` 里只有正态分布。**这个工具的自检本身就是 A 类断言（§A.4）**。
#   ③ 01 有一套**跨程序常量覆盖**的机械检查（与 `implementation` 同源）：
#      用 `ast` 读另外两个程序的 `COVERED_*` 元组，断言**并集 == specs.py 的全部
#      模块级常量**、且**两两不交**。⇒ 在 specs.py 里加一个没人断言的常量，
#      `make run` 立刻失败。**本主题的规模：136 条常量。**
#   ④ ★★ **本主题的来源强度是全库最弱的**：核心数字大量来自 **[D] 级**
#      （个人/小团队仓库）。因此 `specs.py` 的常量名**把来源编进去**
#      （`HIPFIRE_*` / `TRITON_*` / `VLLM_*` / `ARXIV_*` / `MYTKOWICZ_*` / `CUDA_*`），
#      使**误用可见**；并且这些值在程序里是**函数实参**，不是硬编码进逻辑的常数。
#
# ── 每个程序「可以被反驳」的判据（域 README）────────────────────────────────
#   把文档里对应的论断改错，程序应当报错或给出相反结论。
#   三个程序各自的反驳点在其文件头的 `★ 本程序的可证伪断言` 段里逐条列出。
#   ★★ 每个程序都有一条**根据实跑结果推翻规格**的断言（见 `specs.py` §8 的
#      `SPEC_CLAIM_*` 常量），这些不是「照抄规格」，而是**本库与规格不一致的地方**。
#
# ★ `make lint` 是给「不 import torch」这一条硬约束用的**机械检查**：
#   用 `ast.parse` 读真实语法树，**不用 grep** —— grep 会被注释和字符串骗过
#   （本目录的注释里就出现 "torch" 这个词，grep 必然假阳性）。

PYTHON  ?= python3
SCRIPTS := 01_benchmark_hygiene 02_significance 03_convergence

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
sys.exit(1 if bad else 0)'
	@echo "✓ 没有编号程序 import torch"

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
