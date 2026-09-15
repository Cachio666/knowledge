# interconnect 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                   同 make run
#   make run               依次运行全部程序
#   make 02_topology_bisection  只跑一个（省略 .py 后缀）
#   make clean             清理 __pycache__
#
# 这四个程序是纯数值推导，只用 numpy，不需要 GPU，可在本机直接跑。
# 它们做的是**单位换算、下界推导与拓扑组合**，不是硬件测量——规格值来自
# specs.py（每条带来源注释），其余数字都是规格值代入公式的结果。
#
# 注意 specs.py 是本主题的**共享事实源**，没有 NN 编号，
# 不计入「文档数 = 编号源文件数」的核对（见 CONTRIBUTING §2）。

PYTHON  ?= python3
SCRIPTS := 01_bandwidth_conventions 02_topology_bisection \
           03_collectives_topology 04_measuring_bandwidth

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
