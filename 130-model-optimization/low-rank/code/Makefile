# 130-model-optimization/low-rank 示例构建
#
# ★ 本目录是 CONTRIBUTING §2 判定表里的**第二种情形**：编号程序**本机跑得了**
#   （`.py`，默认 `python3` + `numpy` 即可），所以三类断言**全部在编号程序内部**，
#   失败时以非零码退出。因此本目录**不需要 `verify.py`**。
#   编号程序产出的是**推导值**——它只跑算术与 numpy，不碰硬件、不加载权重。
#
# 用法：
#   make                        同 make run
#   make run                    用默认 python3 跑全部编号程序（断言失败即非零退出）
#   make 01_optimal_low_rank    只跑一个（省略 .py 后缀）
#   make clean                  清理 __pycache__
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   specs.py            事实源（示例谱形状 / 与量化对照的档位 / 外部规格常量），每条带 `# 来源：`
#   Makefile / requirements.txt   构建入口
#
# ── 文档与代码的外键 ──────────────────────────────────────────────────────
#   01-optimal-low-rank-approximation.md  ↔ 01_optimal_low_rank.py
#
# ── 依赖 ─────────────────────────────────────────────────────────────────
# 只需要 numpy。`python3` 走 PATH 上的默认解释器——本主题刻意**不探测任何框架环境**：
# 本主题的账全部是可解析推导的（谱、秩、范数），不需要真系统。
#
# ── 可证伪性（CONTRIBUTING §2）─────────────────────────────────────────────
# 建成时做过变异测试，三处**全部被捕获**（退出码非零）：
#   · 截断**少留一个**分量（`u[:, :r]` → `u[:, :r-1]`）⇒ A.2 的「最优」被随机矩阵打败
#   · 把 `‖A − A_r‖_F² = Σ_{i>r} σ_i²` 的求和起点右移一位（`s1[r:]` → `s1[r+1:]`）⇒ A.3 失败
#   · 把 C.2 的 `B` 从 `0.5·11ᵀ` 改成 `1.0·11ᵀ` ⇒ C.2 的「逐元素更优」消失
# ⚠ 变异测试是**一次性核对**，不是 `make` 目标。

SCRIPTS := 01_optimal_low_rank
PYTHON ?= python3

.PHONY: all run clean
all: run

# ⚠ `PYTHON ?=` 挡不住空值：命令行传 `PYTHON=`（空值）也算「已赋值」，
#   默认值不会被填上。所以下面 run 里**另有一次空值检查**（CONTRIBUTING §2）。
run:
	@if [ -z "$(PYTHON)" ]; then \
		echo "错误：PYTHON 为空。本目录需要默认 python3（含 numpy）。"; \
		echo "  make run PYTHON=python3        # 显式指定"; \
		echo "  ⚠ 路径里要用家目录请写 \$$HOME/... 而不要写 ~/...："; \
		echo "     make 自身不做 ~ 展开，而校验位置会给变量加引号，"; \
		echo "     加了引号 ~ 就永远是字面量、必然被判定为不存在。"; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; print("解释器：%s / python %s" % (sys.executable, sys.version.split()[0]))'
	@$(PYTHON) -c 'import numpy; print("numpy %s" % numpy.__version__)' \
		|| { echo ""; echo "错误：当前解释器里没有 numpy。"; \
		     echo "  $(PYTHON) -m pip install -r requirements.txt"; exit 1; }
	@echo
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py || exit 1; \
		echo; \
	done

%:
	@$(PYTHON) -c 'import numpy' 2>/dev/null || { \
		echo "错误：当前解释器里没有 numpy。"; \
		echo "  $(PYTHON) -m pip install -r requirements.txt"; exit 1; }
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
