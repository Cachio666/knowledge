# 60-compilers/mlir 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                   同 make run
#   make run               依次运行全部程序（断言失败即非零退出）
#   make 01_ir_structure   只跑一个（省略 .py 后缀）
#   make clean             清理 __pycache__
#
# 这些程序只用 python3 + numpy，不依赖 torch / tvm / mlir / jax / tensorflow，
# 也不需要 GPU，可在本机直接跑。
#
# ★ 本目录里没有 verify 目标：三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）
#   写在**编号程序内部**，失败时以非零码退出，因此没有独立的 verify.py 可跑。
#   这与 40-kernels 的 .cu 双轨形态不同（那里编号程序跑不了，断言才另置）。
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   miniir.py   迷你 IR / 解释器 / 三类断言框架
#   specs.py    MLIR 的外部事实源，每条带 # 来源：URL
#   exprlang.py 表达式树上的独立求值器与参考简化器（04 号程序的 A 类参照物）

PYTHON  ?= python3
SCRIPTS := 01_ir_structure 02_bufferization 03_lowering_pipeline 04_pattern_rewrite

.PHONY: all run clean
all: run

run:
	@for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py || exit 1; \
		echo; \
	done

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__
