# floating-point 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                   同 make run
#   make run               依次运行全部程序
#   make 01_format_probe   只跑一个（省略 .py 后缀）
#   make clean             清理 __pycache__
#
# 这些程序是纯数值演示，只用 numpy，不需要 GPU，可在本机直接跑。
#
# 注意 fpbits.py 是 bf16 / fp8 / tf32 的**位运算模拟**模块，没有 NN 编号，
# 不计入「文档数 = 编号源文件数」的核对（见 CONTRIBUTING §2）。

PYTHON  ?= python3
SCRIPTS := 01_format_probe 02_rounding_and_error \
           03_summation_and_fma 04_low_precision_formats

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
