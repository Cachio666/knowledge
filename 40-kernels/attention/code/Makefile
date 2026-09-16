# 40-kernels/attention 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0；GPU 算力 >= sm_80（Hopper 档需 H100）
#
# 用法：
#   make                        构建默认 TARGETS（sm_80 可移植）
#   make run                    构建并依次运行默认 TARGETS
#   make 02_flash_attention     构建单个目标
#   make verify                 **只需 python3 + numpy + scipy，本机（macOS）可跑**
#   make regs                   查看每线程寄存器用量与 spill（默认 01_naive_attention）
#   make regs T=02_flash_attention
#   make hopper                 构建需要 -arch=sm_90a 的目标（04，需 H100）
#   make clean                  清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80
#
# ⚠ 本仓库的写作环境（macOS）**没有 CUDA 工具链也没有 GPU**，`make` / `make run` /
#   `make regs` / `make hopper` 在这里都跑不了；**只有 `make verify` 是本机可跑的**。
#   `.cu` 内核的运行结果本仓库没有——文档里的数字分「规格值 / 推导值 / 引用值」三类。

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LDFLAGS   ?=
PYTHON    ?= python3

# --- 默认档：sm_80 可移植，全部内核不依赖任何 "accelerated" 变体 ---
# 01 朴素（物化 S）/ 02 flash（在线 softmax + 不物化 S）/ 03 反向（重算 S）
# 05 decode（GQA·MQA / split-KV / paged KV）：四个都只用 mma.sync + cp.async +
# warp shuffle，在 A100 上也能直接 `make`。
TARGETS           := 01_naive_attention 02_flash_attention \
                     03_flash_attention_backward 05_decode_attention

# --- 前沿档：**不混进 TARGETS**，否则在 A100 上 `make` 直接失败 ---
# 04 用 wgmma（`sm_90a` 的 "accelerated" 变体：为它编出的 cubin 跑不了 sm_90，
# 反之可以）。见 specs.py 的 FEATURE_FIRST_TARGET 与 ACCELERATED_TARGETS。
HOPPER_TARGETS    := 04_flash_attention_ws
BLACKWELL_TARGETS :=

# make regs 的目标，可覆盖：make regs T=02_flash_attention
T ?= 01_naive_attention

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

# 本主题没有需要 sm_100a 的内核（tcgen05 在 attention 上尚无本仓库认可的形态）。
.PHONY: blackwell
blackwell:
	@if [ -z "$(BLACKWELL_TARGETS)" ]; then \
		echo "本主题没有 Blackwell 档目标（04 是唯一的非默认档，且只需 sm_90a）。"; \
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
