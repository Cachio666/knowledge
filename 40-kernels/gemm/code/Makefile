# 40-kernels/gemm 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0；GPU 算力 >= sm_80（Hopper 档需 H100，Blackwell 档需 B200）
#
# 用法：
#   make               构建默认 TARGETS（sm_80 可移植）
#   make run           构建并依次运行默认 TARGETS
#   make 02_tiled_gemm 构建单个目标
#   make verify        **只需 python3 + numpy，本机（macOS）可跑**
#   make regs          查看每线程寄存器用量与 spill（默认 01_naive_gemm）
#   make regs T=03_register_tiled_gemm
#   make hopper        构建需要 -arch=sm_90a 的目标（04，需 H100）
#   make blackwell     构建需要 -arch=sm_100a 的目标（05，需 B200）
#   make clean         清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80
#
# ⚠ 本仓库的写作环境（macOS）**没有 CUDA 工具链也没有 GPU**，`make` / `make run` /
#   `make hopper` / `make blackwell` 在这里都跑不了；**只有 `make verify` 是本机可跑的**。
#   `.cu` 内核的运行结果本仓库没有——文档里的数字分「规格值 / 推导值 / 引用值」三类。

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LDFLAGS   ?=

PYTHON ?= python3

# --- 默认档：sm_80 可移植，全部内核不依赖任何 "accelerated" 变体 ---
TARGETS           := 01_naive_gemm 02_tiled_gemm 03_register_tiled_gemm 06_split_k_gemm

# --- 前沿档：**不混进 TARGETS**，否则在 A100 上 `make` 直接失败 ---
# `sm_90a` / `sm_100a` 是 "accelerated" 变体：为它编出的 cubin 跑不了 sm_90 / sm_100。
# 反方向是可以的——为 `sm_90` 编出的 cubin 能在 H100 上正常加载运行。
HOPPER_TARGETS    := 04_wgmma_gemm
BLACKWELL_TARGETS := 05_tcgen05_persistent_gemm

# make regs 的目标，可覆盖：make regs T=03_register_tiled_gemm
T ?= 01_naive_gemm

.PHONY: all clean run verify regs hopper blackwell
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $<

# 查看每线程寄存器用量与 spill（只打印 ptxas 报告，不产出目标文件）
.PHONY: regs
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(T).cu -o /dev/null

# 需要 sm_90a：wgmma。在 sm_80 上编不过属预期，不是 bug。
.PHONY: hopper
hopper:
	@for t in $(HOPPER_TARGETS); do \
		echo "=== $$t (-arch=sm_90a) ==="; \
		$(NVCC) $(NVCCFLAGS) -arch=sm_90a $(LDFLAGS) -o $$t $$t.cu || exit 1; \
	done

# 需要 sm_100a：tcgen05 / TMEM。在 H100 上编不过属预期。
.PHONY: blackwell
blackwell:
	@for t in $(BLACKWELL_TARGETS); do \
		echo "=== $$t (-arch=sm_100a) ==="; \
		$(NVCC) $(NVCCFLAGS) -arch=sm_100a $(LDFLAGS) -o $$t $$t.cu || exit 1; \
	done

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
	rm -f $(TARGETS) $(HOPPER_TARGETS) $(BLACKWELL_TARGETS)
	rm -rf __pycache__
