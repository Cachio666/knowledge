# tensor-core 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                   同 make run
#   make run               依次运行全部程序
#   make 01_mma_primitive  只跑一个（省略 .py 后缀）
#   make clean             清理 __pycache__
#
# 这些程序是纯数值与组合推导，只用 numpy，不需要 GPU，可在本机直接跑。
# 它们不产出任何实测数据——跑的是算术，不是硬件。
#
# 注意 specs.py 是本主题的**共享事实源**（从官方来源抄来的硬件规格常量），
# 没有 NN 编号，不计入「文档数 = 编号源文件数」的核对（见 CONTRIBUTING §2）。

PYTHON  ?= python3
SCRIPTS := 01_mma_primitive 02_fragment_layout 03_wgmma_pipeline \
           04_tcgen05_tmem 05_structured_sparsity

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
