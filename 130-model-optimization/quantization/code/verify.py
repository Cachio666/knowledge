#!/usr/bin/env python3
"""130-model-optimization/quantization —— 纯 python3 复核入口（**编排**）。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`**（CONTRIBUTING §2 第三种形态）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**把本主题的**判据**重算一遍，再用手写期望值判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

★ 合并说明（2026-10-13）：本主题由 6 个子主题合并而成（20 篇文档编号 01–20）。
  各组的 A/C 断言**没有合并成一份**，而是留在 6 个按来源命名的模块里，本文件只做编排：

      模块                 覆盖文档   内容
      verify_formats.py    01–04      数值格式（fp8 族 / e8m0 与 MX / bf16-fp16-tf32 / 整数）
      verify_theory.py     05–07      量化误差理论（误差界 / 粒度 / outlier 与传播）
      verify_ptq.py        08–11      训练后量化（校准 / GPTQ / AWQ+SmoothQuant / 舍入）
      verify_qat.py        12–14      量化感知训练（STE / 伪量化与 LSQ / PACT 与对比）
      verify_kernels.py    15–17      量化算子（W8A8 / W4A16 打包 / kernel 设计）
      verify_kv.py         18–20      KV cache 量化（账 / KIVI 非对称 / 低位宽与引擎）

  **为什么不合并成一份**：各组有自己的 `specs_<组>.py`，而 6 份 specs 之间有
  **8 个跨主题重名**常量，其中若干条**同名不同值**；合并要给它们改名，而改名会连带
  改动编号程序里的断言。另一条：20 篇文档按 `verify_<组>.py 的 A.n` **逐条引用**各自的
  断言编号，合并编号会让全部行内引用同时失真。详见同目录 `Makefile` 的头注释。

★ 三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 6 个 verify 模块；参照物是它们各自的 `ref_*` 独立实现与手写期望值。
    · **C 边界与反例** —— 同上；断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件与 6 个模块**都不做** B。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import sys

import verify_formats
import verify_theory
import verify_ptq
import verify_qat
import verify_kernels
import verify_kv

MODULES = [
    verify_formats,
    verify_theory,
    verify_ptq,
    verify_qat,
    verify_kernels,
    verify_kv,
]


def main():
    print("=" * 78)
    print("130-model-optimization/quantization —— 纯 python3 复核（**不 import torch**）")
    print("★ 编排 6 个 verify 模块；产出**一律是推导**，不是实测值。")
    print("=" * 78)
    failed = []
    for mod in MODULES:
        name = mod.__name__
        print()
        try:
            rc = mod.main()
        except AssertionError as exc:          # 模块内若直接用 assert
            print(f"  **{name} 抛断言异常**：{exc}")
            rc = 1
        if rc != 0:
            failed.append(name)
    print()
    print("=" * 78)
    if failed:
        print(f"汇总：{len(failed)} / {len(MODULES)} 个模块有失败 → {'、'.join(failed)}")
        return 1
    print(f"汇总：{len(MODULES)} 个模块全部通过（A / C 两类断言）。")
    print("★ 提醒：本入口**不做 B 类代价模型** —— 那在 `make run`（需要真实 torch）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
