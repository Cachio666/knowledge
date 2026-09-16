# 60-compilers/tvm 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                   同 make run
#   make run               依次运行全部程序（断言失败即非零退出）
#   make 01_compute_schedule   只跑一个（省略 .py 后缀）
#   make clean             清理 __pycache__
#
# 这些程序只用 python3 + numpy，不依赖 torch / tvm / mlir / jax / tensorflow，
# 也不需要 GPU，可在本机直接跑。
#
# ★ 本目录里没有 verify 目标：三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）
#   写在**编号程序内部**，失败时以非零码退出，因此没有独立的 verify.py 可跑。
#   这与 40-kernels 的 .cu 双轨形态不同（那里编号程序跑不了，断言才另置）。
#
# ★ 本目录里**没有 tvm 包**，也不 import 它：所有程序跑的都是本仓库的迷你实现
#   （minitir.py + 各编号程序自己造的地形）。关于真实 TVM 的一切结论都只出现在文档里，
#   逐条带来源，且**不作为程序断言**（那会是自证）。
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   minitir.py  迷你 TIR / 调度 / 代价模型 / 解释器 / native 轨 / 三类断言框架
#   specs.py    TVM 的外部事实源，每条带 # 来源：，全部钉在 tag v0.26.0

PYTHON  ?= python3
SCRIPTS := 01_compute_schedule 02_schedule_primitives 03_two_tracks 04_scheduling_paths

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
