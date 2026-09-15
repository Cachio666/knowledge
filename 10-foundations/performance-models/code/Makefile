# performance-models 示例构建
#
# 目标环境：
#   09 / 10 —— Python >= 3.10 + numpy >= 2.0（纯解析模型，本机可跑）
#   11      —— CUDA Toolkit >= 12.0，GPU 算力 >= sm_80（需真实硬件）
#
# 用法：
#   make                       同 make py
#   make py                    只跑 09 / 10（本机即可）
#   make 11_roofline_probe     构建 CUDA 程序
#   make regs                  查看 11 的寄存器用量与 spill
#   ./11_roofline_probe        运行（需 GPU）
#   make clean

PYTHON    ?= python3
NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo

PYSCRIPTS := 09_roofline_model 10_amdahl_scaling
CUTARGET  := 11_roofline_probe
T         ?= 11_roofline_probe

.PHONY: all py regs clean
all: py

py:
	@for s in $(PYSCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s.py || exit 1; \
		echo; \
	done

$(CUTARGET): $(CUTARGET).cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) -o $@ $<

# 11 用 16 路独立累加器打破 FMA 依赖链。寄存器不够时 ptxas 会把累加器
# spill 到本地内存，实测「峰值算力」会大幅偏低。判据：spill 应为 0。
regs:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas -v -c $(T).cu -o /dev/null

%:
	$(PYTHON) $@.py

clean:
	rm -rf __pycache__ $(CUTARGET)
