# 40-kernels/softmax-normalization 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0；GPU 算力 >= sm_80
#
# 用法：
#   make                   构建默认 TARGETS（sm_80 可移植）
#   make run               构建并依次运行默认 TARGETS
#   make 02_online_softmax 构建单个目标
#   make verify            **只需 python3 + numpy + scipy，本机（macOS）可跑**
#   make regs              查看每线程寄存器用量与 spill（默认 01_softmax_stable）
#   make regs T=03_layernorm
#   make clean             清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80
#
# ⚠ 本仓库的写作环境（macOS）**没有 CUDA 工具链也没有 GPU**，`make` / `make run` /
#   `make regs` 在这里都跑不了；**只有 `make verify` 是本机可跑的**。
#   `.cu` 内核的运行结果本仓库没有——文档里的数字分「规格值 / 推导值 / 引用值」三类。

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LDFLAGS   ?=

PYTHON ?= python3

# --- 默认档：sm_80 可移植 ---
# 本主题的三个内核**都只用 warp shuffle + shared memory + MUFU 超越函数**，
# 没有任何一条指令需要 "accelerated" 变体（sm_90a / sm_100a），
# 所以三个全部进默认 TARGETS，在 A100 上也能直接 `make`。
TARGETS := 01_softmax_stable 02_online_softmax 03_layernorm

# --- 前沿档：**本主题用不到** ---
# 这里保留 HOPPER_TARGETS / BLACKWELL_TARGETS 两个骨架只是为了让
# `make hopper` / `make blackwell` 在本主题下也有确定的含义（跑一遍、
# 报一句「本主题没有前沿档目标」），而不是命令不存在。
# 若将来本主题加入需要 wgmma / tcgen05 的内核，按 40-kernels/gemm 的写法
# 把目标名填进这两个变量即可——**不要把 sm_90a / sm_100a 的目标混进 TARGETS**，
# 否则在 A100 上 `make` 直接失败。
HOPPER_TARGETS    :=
BLACKWELL_TARGETS :=

# make regs 的目标，可覆盖：make regs T=03_layernorm
T ?= 01_softmax_stable

.PHONY: all clean run verify regs hopper blackwell
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $<

# 查看每线程寄存器用量与 spill（只打印 ptxas 报告，不产出目标文件）
.PHONY: regs
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(T).cu -o /dev/null

# 本主题没有需要 sm_90a 的内核。
.PHONY: hopper
hopper:
	@if [ -z "$(HOPPER_TARGETS)" ]; then \
		echo "本主题没有 Hopper 档目标（三个内核都是 sm_80 可移植）。"; \
	else \
		for t in $(HOPPER_TARGETS); do \
			echo "=== $$t (-arch=sm_90a) ==="; \
			$(NVCC) $(NVCCFLAGS) -arch=sm_90a $(LDFLAGS) -o $$t $$t.cu || exit 1; \
		done; \
	fi

# 本主题没有需要 sm_100a 的内核。
.PHONY: blackwell
blackwell:
	@if [ -z "$(BLACKWELL_TARGETS)" ]; then \
		echo "本主题没有 Blackwell 档目标（三个内核都是 sm_80 可移植）。"; \
	else \
		for t in $(BLACKWELL_TARGETS); do \
			echo "=== $$t (-arch=sm_100a) ==="; \
			$(NVCC) $(NVCCFLAGS) -arch=sm_100a $(LDFLAGS) -o $$t $$t.cu || exit 1; \
		done; \
	fi

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
