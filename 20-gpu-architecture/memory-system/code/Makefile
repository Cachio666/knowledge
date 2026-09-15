# memory-system 示例构建
#
# 目标环境：CUDA Toolkit >= 12.0，GPU 算力 >= sm_80
# （03 的 64 位一档需要每 block 动态共享内存的 opt-in 上限 >= 67584 B
#  = 66 KB，sm_80 起满足）
#
# 用法：
#   make              构建全部
#   make 01_bandwidth_probe
#   make run          构建并依次运行
#   make regs         查看寄存器用量与 spill（默认 01_bandwidth_probe）
#   make regs T=03_bank_conflict_demo
#   make clean        清理
#
# 若 -arch=native 不可用，改为显式指定，如：make ARCH=-arch=sm_80

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
LDFLAGS   ?=

TARGETS := 01_bandwidth_probe 02_coalescing_demo 03_bank_conflict_demo

# make regs 的目标，可覆盖：make regs T=03_bank_conflict_demo
T ?= 01_bandwidth_probe

.PHONY: all clean run
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $<

# 查看每线程寄存器用量、共享内存用量与 spill 情况（只打印 ptxas 报告，不产出目标文件）
.PHONY: regs
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(T).cu -o /dev/null

run: all
	@for t in $(TARGETS); do \
		echo "=== $$t ==="; \
		./$$t || exit 1; \
		echo; \
	done

clean:
	rm -f $(TARGETS)
