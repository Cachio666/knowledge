# 70-frameworks/tensorflow 示例构建
#
# 目标环境：python3 >= 3.10，numpy >= 2.0
#
# 用法：
#   make                               同 make run
#   make run                           依次运行全部编号程序（断言失败即非零退出）
#   make 01_tracing_and_retracing      只跑一个（省略 .py 后缀）
#   make clean                         清理 __pycache__
#
# ★★ 本目录**不需要 TensorFlow** ★★
#   本机一个 TF 环境都没有，编号程序是**本仓库自己写的迷你追踪器**（minitf.py），
#   只依赖 python3 + numpy。它借用 TF 的术语与机制（TraceType / FunctionType /
#   tf.cond / SavedModel …）以便对照，**结构是本仓库简化的**。
#   它跑出的数字只支撑关于**机制**的结论，**不支撑任何关于真实 TF 的结论**；
#   关于真实 TF 的每条行为断言都以 specs.py 里带 `# 来源：` 的一手 URL 为准。
#
# ★ 关于本目录属于 CONTRIBUTING §2 的哪种形态：
#   编号程序**跑得了**（默认 python3 即可），所以三类断言（A 算法正确性 /
#   B 代价模型 / C 边界与反例）**都写在编号程序内部**，失败时以非零码退出。
#   因此本目录**没有 verify.py**——它的作用（「在没有框架的机器上复核」）
#   在这里由编号程序本身承担，因为编号程序本来就不需要任何框架。
#   这与 70-frameworks/pytorch 的第三种形态不同（那里编号程序要 import torch，
#   故 A/C 另置 verify.py）。

PYTHON  ?= python3
SCRIPTS := 01_tracing_and_retracing 02_control_flow 03_custom_ops_and_export

.PHONY: all run clean
all: run

run:
	@$(PYTHON) -c 'import sys, numpy; print("测量条件：python3 %s / numpy %s（无 TensorFlow）" % (sys.version.split()[0], numpy.__version__))'
	@echo
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py || exit 1; \
		echo; \
	done

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
