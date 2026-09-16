# 40-kernels/reduce-scan 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0；GPU 算力 >= sm_80
#
# 用法：
#   make                 构建默认 TARGETS（全部 sm_80 可移植）
#   make run             构建并依次运行默认 TARGETS
#   make 02_warp_reduce  构建单个目标
#   make verify          **只需 python3 + numpy，本机（macOS）可跑**
#   make regs            查看每线程寄存器用量与 spill（默认 01_reduce_basics）
#   make regs T=02_warp_reduce
#   make clean           清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80
#
# ⚠ 本仓库的写作环境（macOS）**没有 CUDA 工具链也没有 GPU**，`make` / `make run`
#   在这里跑不了；**只有 `make verify` 是本机可跑的**。
#   `.cu` 内核的运行结果本仓库没有——文档里的数字分「规格值 / 推导值 / 引用值」三类。
#
# 本主题**四个内核都是 sm_80 可移植的**：全部只用 `__shfl_*_sync`、`__syncthreads()`、
# `__syncwarp()`、共享内存与 `atomicAdd(float)`（sm_20 起可用），
# **不依赖 `sm_90a` / `sm_100a` 的加速变体**，所以四个全进默认 TARGETS，
# 不需要 gemm 主题那样的 `hopper` / `blackwell` 独立目标。

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LDFLAGS   ?=

PYTHON ?= python3

# --- 全部 sm_80 可移植 ---
TARGETS := 01_reduce_basics 02_warp_reduce 03_scan 04_multi_block_scan

# make regs 的目标，可覆盖：make regs T=03_scan
T ?= 01_reduce_basics

.PHONY: all clean run verify regs
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $<

# 查看每线程寄存器用量与 spill（只打印 ptxas 报告，不产出目标文件）
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(T).cu -o /dev/null

run: all
	@for t in $(TARGETS); do \
		echo "=== $$t ==="; \
		./$$t || exit 1; \
		echo; \
	done

# 本机可跑的验证模型：只跑算术与 numpy，不碰硬件。产出是**推导**，不是实测。
verify:
	$(PYTHON) verify.py

clean:
	rm -f $(TARGETS)
	rm -rf __pycache__
