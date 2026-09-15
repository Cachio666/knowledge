# numerical-linear-algebra 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                   同 make run
#   make run               依次运行全部程序
#   make 05_norms_and_condition   只跑一个（省略 .py 后缀）
#   make clean             清理 __pycache__
#
# 这些程序是纯数值演示，只用 numpy，不需要 GPU，可在本机直接跑。
#

PYTHON  ?= python3
SCRIPTS := 05_norms_and_condition 06_gemm_error \
           07_factorization 08_iterative_methods

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
