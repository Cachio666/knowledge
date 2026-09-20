# -*- coding: utf-8 -*-
r"""三类断言的记账框架（A 算法正确性 / B 代价模型 / C 边界与反例）。

本文件是**非编号辅助模块**，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

★ 与 `30-gpu-programming/cuda/code/` 同口径：
  - 三类断言写在**编号程序内部**，失败时以**非零退出码**结束 ——
    即「断言失败 ⇒ `make run` 非零」。**没有 `verify.py`**（`CONTRIBUTING` 第 121 行第二种情形）。
  - **每个编号程序至少有一个可证伪的断言** —— 判据是问「我故意把这段改错，程序会报错吗？」
  - C 类的名字是「C 边界与反例」；它断言的是「**失败行为成立**」，不是「成功」。

★ 汇总行的格式**照抄 `cuda/code/checks.py`**（`断言 N 条，通过 P 条，失败 F 条`）：
  `Makefile` 的聚合器用 `断言 (\d+) 条，通过 (\d+) 条，失败 (\d+) 条` 从**全部程序**的输出里
  汇总总数，所以这一行的形状是**与 Makefile 的约定**，不是随便写的。
"""

import sys


class Checks:
    """一个编号程序的断记账本。

    ★ 它**同时记「通过」与「失败」**：`summary()` 打印的 `A x/y` 里的分母是**本程序发出的
      A 类断言条数**，分子是通过数 —— 两者不等时程序以非零码退出（`finish()`）。
    """

    def __init__(self, title):
        self.title = title
        self.n = {"A": 0, "B": 0, "C": 0}          # 发出的条数
        self.ok = {"A": 0, "B": 0, "C": 0}         # 通过的条数
        self.failed = []

    # -- 内部 ---------------------------------------------------------------
    def _rec(self, kind, ok, msg, detail=""):
        self.n[kind] += 1
        if ok:
            self.ok[kind] += 1
        tag = "PASS" if ok else "FAIL"
        line = "  [%s] %s%d  %s" % (tag, kind, self.n[kind], msg)
        if detail:
            line += "\n         " + str(detail).replace("\n", "\n         ")
        print(line)
        if not ok:
            self.failed.append((kind, self.n[kind], msg))

    # -- 三类 ---------------------------------------------------------------
    def A(self, cond, msg, detail=""):
        self._rec("A", bool(cond), msg, detail)

    def B(self, cond, msg, detail=""):
        self._rec("B", bool(cond), msg, detail)

    def C(self, cond, msg, detail=""):
        self._rec("C", bool(cond), msg, detail)

    # -- 便捷 ---------------------------------------------------------------
    def expect_raises(self, fn, exc, msg):
        """断言 `fn()` 抛 `exc`（**C 类**：断言的是「拒绝成立」）。"""
        try:
            fn()
        except exc as e:
            self._rec("C", True, msg + " → 抛出 %s" % type(e).__name__,
                      "错误信息：" + str(e)[:200])
            return
        except Exception as e:                        # noqa: BLE001
            self._rec("C", False, msg + " → 抛出的**不是** %s" % exc.__name__,
                      "实得 %s: %s" % (type(e).__name__, str(e)[:160]))
            return
        self._rec("C", False, msg + " → **没有抛错**（这条断言本该失败）")

    def expect_ok(self, fn, msg, detail=""):
        """断言 `fn()` **不**抛错（C 类的对照组）。"""
        try:
            r = fn()
        except Exception as e:                        # noqa: BLE001
            self._rec("C", False, msg + " → 意外抛错",
                      "%s: %s" % (type(e).__name__, str(e)[:160]))
            return None
        self._rec("C", True, msg, detail)
        return r

    # -- 汇总 ---------------------------------------------------------------
    def summary(self):
        total = sum(self.n.values())
        passed = sum(self.ok.values())
        print()
        print("  汇总: A %d/%d · B %d/%d · C %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
              % (self.ok["A"], self.n["A"], self.ok["B"], self.n["B"],
                 self.ok["C"], self.n["C"], total, passed, len(self.failed)))
        if self.failed:
            for k, i, m in self.failed:
                print("    FAILED %s%d: %s" % (k, i, m))
        return 1 if self.failed else 0


def rule(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def finish(code):
    sys.exit(code)
