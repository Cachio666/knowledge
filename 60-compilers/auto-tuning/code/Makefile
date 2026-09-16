# 60-compilers/auto-tuning 示例构建
#
# 目标环境：Python >= 3.10，numpy >= 2.0
#
# 用法：
#   make                       同 make run
#   make run                   依次运行全部程序（断言失败即非零退出）
#   make 01_search_space       只跑一个（省略 .py 后缀）
#   make clean                 清理 __pycache__
#
# 这些程序只用 python3 + numpy，不依赖 torch / tvm / triton / mlir / jax，
# 也不需要 GPU，可在本机直接跑。
#
# ★ 本目录里没有 verify 目标：三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）
#   写在**编号程序内部**，失败时以非零码退出，因此没有独立的 verify.py 可跑。
#   这与 40-kernels 的 .cu 双轨形态不同（那里编号程序跑不了，断言才另置）。
#
# ★ 本目录里**没有 Ansor、没有 TVM、没有 Triton**，也不 import 它们：
#   所有程序跑的都是本仓库的迷你实现（mini_autotune.py + 各编号程序自己造的地形）。
#   关于真实 Ansor / Triton 的一切结论只出现在文档与 specs.py 里，逐条带来源，
#   且**不作为程序断言**（那会是自证）。
#
# ★ 本主题是全域唯一有大量「引用值」的一篇，且它们**分两处落正文**（CONTRIBUTING §7.1）：
#   · Ansor 论文 §7.5 的**模型质量**四数（RMSE / R² / pairwise / recall@30）
#     → 正文在 02-cost-models（那是代价模型自身的性质）
#   · Ansor 论文 §7.4 Table 3 的搜索成本、§7.3 的加速比、§7.4 的「1000 × n」实验设置
#     → 正文在 03-search-algorithms §5
#   两处全部标「引用值，本仓库不复核其测量口径」，且**不互相复制**。
#
# ★ 运行时间：三个程序合计约 55 秒（02 与 03 里有代价模型的反复重训与搜索，
#   比本域其它主题的秒级程序慢——这是主题本身的性质，不是卡住了）。
#
# 非编号文件（不计入「文档数 = 编号程序数」的核对，见 CONTRIBUTING §2）：
#   mini_autotune.py  迷你自动调优基础设施：搜索空间 / 代价地形 / 迷你 GBDT /
#                     搜索算法 / 迷你 autotuner（照 Triton v3.6.0 默认路径）/
#                     三类断言框架
#   specs.py          Ansor 与 Triton 的外部事实源，每条带 # 来源：

PYTHON  ?= python3
SCRIPTS := 01_search_space 02_cost_model 03_search_trace

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
