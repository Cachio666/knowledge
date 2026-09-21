#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""microbench.py —— **非编号共享模块**：内嵌 C 源码 + `clang` 编译执行。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 为什么必须是一个**共享**的非编号模块
===============================================================================

`01_latency_bandwidth.py`（复现依赖链曲线）与 `03_diagnosis_walkthrough.py`
（对热点判 bound）**用的是同一台仪器**。`CONTRIBUTING:36` 要求 **`code/` 自包含**，
所以这份 C 源码**必须落在同一个 `code/` 目录内**，而不是让 `03` 去 import `01`
（`01` 是**编号程序**，把它当模块 import 会同时引发两个问题：文档↔程序的外键被绕开，
以及 `import 01_latency_bandwidth` 这种名字在 Python 里根本不是合法标识符）。
同主题内共享**非编号**模块是允许的（`CONTRIBUTING:32`）。

===============================================================================
★ 两段 C 源码 —— **不是同一段代码，不得并列比较**（P13 的教科书实例）
===============================================================================

| 标签 | C 源码 | 循环结构 | 它回答什么 |
|---|---|---|---|
| `DEP` | `DEP_SRC` | **runtime-K 内层循环**：`for(i; i+=K) for(k<K) s[k] += a[i+k]` | 「同时有几个**未完成的独立读**」对带宽的影响（依赖链曲线） |
| `UNROLL8` | `UNROLL8_SRC` | **8 路写死的展开**（`s0 += a[i+0]; …`，**无内层循环**） | 同一台机器上**顺序读**的有效带宽随**规模**怎么变（cache 层次） |

**两张表各自的数都是真测量**，但**它们不是同一配置的两次测量**：
`DEP` 的 K=8 与 `UNROLL8` 的 8 路展开**看起来像同一个配置**，其实是两段不同的代码
（一个走 runtime 内层循环、一个把 8 条链写死），**量出来的带宽也不同**。
⇒ **凡引用这两个数，必须写明是哪一段代码。** 这是本域 `P13`（量法口径错）的正面教材：
`18.18 GB/s` 与 `51.28 GB/s` 都是真的，把它们并列比较才是错的。

===============================================================================
★ 本模块的**可证伪**职责（`make falsify` 的注入项就打在下面两个锚点上）
===============================================================================

  · **累加器个数写死为 1**（改 `DEP_SRC` 的内层循环）⇒ 依赖链曲线**塌平**，
    `01` 的 `A.2`（K=1 的带宽 < K=32 的 1/8）必须失败。
  · **去掉 `volatile`**（改 `sink` 的声明）⇒ `-O2` 会把整个循环**消掉**
    （`sink` 只被写、从不被读）⇒ 测到的带宽变成**荒谬值**，
    `01` 的 `A.1`（「这个数荒谬不荒谬」的上限）必须失败。

★★ 这就是本模块对 **P13** 的处置：**仪器自己坏掉时，读数必须叫。**
   若把累加器写死为 1 之后程序仍然报「通过」，那说明它**没有在测带宽**。

===============================================================================
★ 本机实测值的**测量条件**（逐处都要带）
===============================================================================

  本机 CPU（`Apple M1 Pro` / `macOS`）、`clang`（`/usr/bin/clang`）、
  python3 **3.13.5**、numpy **2.4.4**。**不是 GPU、不是 Nsight Compute。**
  `clang` 找不到时**以可操作的错误 rc=2 退出**，**不静默跳过、不回落到 Python 循环**
  —— 回落会得到 `10-foundations/performance-models/03` §4.1 记录的假数
  （用 Python `for` 循环测访存带宽得到 `0.10 GB/s`，**且不随规模变化**）。
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 编译与执行
# ===========================================================================
CLANG_ENV = "CLANG"
DEFAULT_OPT = "O2"          # ★ 注入项 2 的锚点之一（另一个是 DEP_SRC 里的 volatile）
BUILD_TIMEOUT = 120
RUN_TIMEOUT = 600

# ★ 断言的**荒谬性上限**（P13 的判据第②句：「这个数荒谬不荒谬」）。
#   本机内存带宽是**几十** GB/s（`§0.7` 的实测：顺序读 51–65 GB/s）。
#   取 1000 GB/s 作上限：它比任何真实存储层次都高一个量级以上，
#   却能可靠地抓住「循环被优化器消掉」（那时算出来的带宽是 10^17 量级）。
#   **不写死本机的具体带宽**，是为了让这条判据在别的机器上仍然成立。
ABSURD_BANDWIDTH_GBPS = 1000.0


def find_clang():
    """返回 `clang` 的路径；找不到返回 `None`。`CLANG` 环境变量优先。"""
    return os.environ.get(CLANG_ENV) or shutil.which("clang")


def require_clang():
    """`clang` **必需**：找不到时以可操作的错误退出（rc=2）。

    ★ 计划 §1.3：**不得静默跳过、不得回落到 Python 循环**——
      回落会得到 `10-foundations/performance-models/03` §4.1 的假数
      （用 Python `for` 循环测访存带宽得到 `0.10 GB/s`，**且不随规模变化**，
      因为瓶颈是解释器每次迭代的开销，**根本没碰到内存子系统**）。
    """
    p = find_clang()
    if not p:
        print()
        print("!! 找不到 `clang` —— 本模块**不会静默跳过、也不会回落到 Python 循环**。")
        print("   为什么不能回落：`10-foundations/performance-models/03` §4.1 记录的实测——")
        print("   用 Python `for` 循环测访存带宽会得到 0.10 GB/s，**且不随规模变化**，")
        print("   因为瓶颈是解释器每次迭代的开销，**根本没碰到内存子系统**。")
        print("   请安装命令行工具（macOS: `xcode-select --install`），")
        print("   或显式指定：`CLANG=/path/to/clang make run`。")
        sys.exit(2)
    return p


def clang_version(clang=None):
    clang = clang or require_clang()
    r = subprocess.run([clang, "--version"], capture_output=True, text=True)
    txt = (r.stdout or r.stderr) or ""
    return txt.splitlines()[0] if txt.strip() else "?"


def build(src_text, tag, tmpdir, opt=DEFAULT_OPT, clang=None):
    """把内嵌的 C 源码写进 `tmpdir`、用 `clang -<opt>` 编出来。

    返回 `(cpath, bpath)`。编译失败时**以 rc=2 退出**（不静默）。
    ★ `tmpdir` 由调用方用 `tempfile.mkdtemp()` 提供，**不往仓库里写任何东西**。
    """
    clang = clang or require_clang()
    cpath = os.path.join(tmpdir, "%s_%s.c" % (tag, opt))
    bpath = os.path.join(tmpdir, "%s_%s" % (tag, opt))
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(src_text)
    r = subprocess.run([clang, "-%s" % opt, "-o", bpath, cpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("!! clang -%s 编译 `%s` 失败：\n%s" % (opt, tag, r.stderr))
        sys.exit(2)
    return cpath, bpath


def build_all(tmpdir, clang=None, opt=DEFAULT_OPT):
    """一次编出两段 C 源码，返回 `{"DEP": bpath, "UNROLL8": bpath}`。"""
    out = {}
    for tag, src in (("DEP", DEP_SRC), ("UNROLL8", UNROLL8_SRC)):
        _, out[tag] = build(src, tag, tmpdir, opt=opt, clang=clang)
    return out


def run_binary(bpath, argv, timeout=RUN_TIMEOUT):
    """跑一个微基准二进制，返回它打印的**秒数**（`float`）。

    ★ 二进制只打印一个数（秒）。解析失败时**抛异常**而不是返回 0 ——
      「没读懂输出」与「耗时是 0」是两件不同的事。
    """
    r = subprocess.run([bpath] + [str(a) for a in argv],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("微基准 %s 退出 %d：%s" % (bpath, r.returncode, r.stderr[:200]))
    line = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
    return float(line)


# ===========================================================================
# 1. DEP：依赖链曲线（runtime-K 内层循环）
# ===========================================================================
# ★ 来源：`docs/superpowers/plans/2026-09-21-performance-domain.md` §0.7 **D-2**
#   （`clang -O2`、规模固定 64 MB、只改独立累加器个数 K）。
#   ★★ 这段代码**就是 D-2 的 `dep.c`**，实施者已逐点复现那条曲线。
DEP_SRC = r'''\
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 单调时钟。与 Python 的 time.perf_counter 同一个底层（mach_absolute_time）。 */
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ★★ volatile：写回必须**真的发生**。
   否则 -O2 会把整个循环消掉（`sink` 只被写、从不被读），
   于是这个程序测的是**一次不存在的访存** —— 那正是 §0.7 D-7 第 2 条的现场：
   把 memcpy 的耗时测成 `inf`，因为死代码消除把整个 memcpy 删掉了。
   ⇒ **`make falsify` 的注入项 2 就打在下面这一行上。** */
static volatile double dep_sink = 0.0;

int main(int argc, char **argv) {
    size_t bytes = (size_t)strtoull(argv[1], 0, 10);
    int K = atoi(argv[2]);
    int reps = atoi(argv[3]);
    if (K < 1 || K > 64) { fprintf(stderr, "K out of range\n"); return 4; }
    size_t n = bytes / sizeof(double);
    double *a = (double *)aligned_alloc(4096, n * sizeof(double));
    if (!a) { fprintf(stderr, "aligned_alloc failed\n"); return 3; }
    for (size_t i = 0; i < n; i++) a[i] = 1.0 + (double)(i & 7);

    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double s[64];
        for (int k = 0; k < K; k++) s[k] = 0.0;
        double t0 = now_sec();
        /* ★★ 依赖链：K 条**互不依赖**的累加链，即「同时有多少个未完成的独立读」。
           K=1 时每一次加法都等上一次的结果 ⇒ 测到的是**延迟**；
           K=32 时 32 条链并行 ⇒ 才可能测到**带宽**。 */
        for (size_t i = 0; i + (size_t)K <= n; i += (size_t)K) {
            for (int k = 0; k < K; k++) s[k] += a[i + (size_t)k];
        }
        double dt = now_sec() - t0;
        double sum = 0.0;
        for (int k = 0; k < K; k++) sum += s[k];
        dep_sink = sum;
        if (dt < best) best = dt;
    }
    printf("%.9f\n", best);
    free(a);
    return 0;
}
'''

# ===========================================================================
# 2. UNROLL8：8 路**写死**展开的顺序读（**不是**上面那段代码）
# ===========================================================================
# ★ 来源：计划 §0.7 **D-3**（`mb2.c`，8 路写死展开，顺序读）。
#   ⚠ **它与 DEP 不是同一段代码**：这里没有 runtime 内层循环，
#     8 条链是**写死**的。两张表**不得并列比较**（模块头已写明）。
UNROLL8_SRC = r'''\
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static volatile double u8_sink = 0.0;

/* ★★ `inner`：**每次计时里把整个数组扫多少遍**，报出来的耗时是「一遍」的平均值。
   为什么需要它（实施者实测）：64 KB 的数组一遍只有 **1 µs**，
   而这个尺度上**一次抢占就能让读数翻倍**（实测 1 µs / 2 µs 交替出现，
   报出来的带宽就在 65.5 与 32.8 GB/s 之间跳）。
   把小规模的 `inner` 调大，让每次计时都够长 —— 这与 `03_diagnosis_walkthrough.py`
   里「短操作要在一次测量内重复 R 次」是**同一条口径纪律**。
   ★ 窗口取多长由 `unroll8_inner()` 按 `UNROLL8_MIN_MEASURE_S` 算（模块头有实测标定）。 */
int main(int argc, char **argv) {
    size_t bytes = (size_t)strtoull(argv[1], 0, 10);
    int reps = atoi(argv[2]);
    int inner = atoi(argv[3]);
    if (inner < 1) inner = 1;
    size_t n = bytes / sizeof(double);
    double *a = (double *)aligned_alloc(4096, n * sizeof(double));
    if (!a) { fprintf(stderr, "aligned_alloc failed\n"); return 3; }
    for (size_t i = 0; i < n; i++) a[i] = 1.0 + (double)(i & 7);

    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        double t0 = now_sec();
        for (int it = 0; it < inner; it++) {
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                s0 += a[i + 0]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
                s4 += a[i + 4]; s5 += a[i + 5]; s6 += a[i + 6]; s7 += a[i + 7];
            }
            for (; i < n; i++) s0 += a[i];
        }
        double dt = (now_sec() - t0) / (double)inner;
        u8_sink = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        if (dt < best) best = dt;
    }
    printf("%.9f\n", best);
    free(a);
    return 0;
}
'''

# ===========================================================================
# 3. 参数与换算
# ===========================================================================
# ★ `reps` 与「预热」都是**测量口径的一部分**（`DEP_REPS` 写在这里，不是魔法数）：
#   · `reps=7` 取**最小值**：最小值是「这台机器最不受打扰时跑出的那个数」，
#     比中位数更接近曲线本身；实测 `reps<7` 时曲线右端（K=32）会偶发掉到 25 GB/s 一带
#     （编译刚结束、CPU 还没安静下来），把 `K=32/K=1` 从 16.7 压到 11.5。
#   · **每个点先跑一遍丢掉**（`warm=True`）：第一遍要付页错误与频率爬升的钱。
#     ⚠ 这**不是** `45/01` 讲的「丢掉前 k 次」（那是**重复测量**内部的 warmup，正文在 45）；
#       这里丢掉的是**每个点各自的第一遍**，理由是它测的不是稳态的访存。
DEP_REPS = 7               # 每个 K 点跑几遍取**最小值**
UNROLL8_REPS = 5
# ★ 前 6 个点是计划 §0.7 D-2 的**原表**（K=1…32）；**后两个是本程序加上去的**：
#   计划那张表在 K=32 处**还没有到平台**（K=16→32 仍涨 1.5 倍），
#   而 01 的 §5 判据表里有「增加并发度带宽**不变** ⇒ 已到带宽平台」这一行 ——
#   不把曲线跑到底，那一行就没有本机证据。实测：K=48→64（并发度 +33%）
#   带宽的增益已趋零（8 次实测 0.94–1.13 倍），**本机的平台起点在 K≈48**。
DEP_K_POINTS = (1, 2, 4, 8, 16, 32, 48, 64)
MB = 1024 * 1024
DEP_BYTES = 64 * MB        # ★ D-2 的口径：规模固定 64 MB（远超本机 L2 = 4 MB）
DEP_K_FAST = 48            # ★ 曲线右端：本机实测的**带宽平台起点**（不是计划原表的 32）
UNROLL8_SIZES = (64 * 1024, 256 * 1024, 1024 * 1024, 4 * MB, 8 * MB, 64 * MB, 256 * MB)


def dep_time(bpaths, nbytes, k, reps=DEP_REPS):
    """跑一次 DEP 微基准，返回**最小耗时**（秒）。"""
    return run_binary(bpaths["DEP"], [nbytes, k, reps])


def dep_curve(bpaths, nbytes=DEP_BYTES, ks=DEP_K_POINTS, reps=DEP_REPS, warm=True):
    """复现依赖链曲线，返回 `{K: (秒, 有效带宽 GB/s)}`。

    ★ **顺序采样**：每个 K 点在自己的进程里跑 `reps` 遍、取最小值。
      （不是「每轮把每个 K 各跑一次」，见下。）

    ★★ **一次被实测推翻的「改进」**（实施者踩到，记在这里 —— 这是一条 P13）：
      实施者第一版按本域 §0.7 **D-12** 的教训把采样改成**交错**的
      （`reps` 轮，每轮把 8 个 K 各跑一遍，再对每个 K 取 min）——
      理由听起来完全正确：「顺序采样时，不同 K 落在各自的时间段里，机器负载一变
      就被记上不同的负载」。
      **实测结果：整条曲线在带宽那一端下沉约 45%**（K=32 从 ~37 掉到 ~20 GB/s，
      `K32/K1` 从 16.7 掉到 10.2），**平台判据跟着失效**。
      **真因**：交错之后，**每一个样本都是一个新进程，都要先填一遍 64 MB 的数组**
      （`a[i] = 1.0 + …`）；于是每一次计时循环都紧跟在 **64 MB 脏行的写回**之后，
      测到的是**被写回污染的带宽**。顺序采样时，一次填充换来 7 遍计时，
      只有第 1 遍挨着写回，而**最小值把那 1 遍丢掉了**。
      ⇒ **「交错」在这里是量法改动，不是编排改动**：D-12 讲的是**两组比较**时的编排，
        而这里是**一台仪器的采样方式**。两者不可套用。
    """
    out = {}
    for k in ks:
        if warm:
            dep_time(bpaths, nbytes, k, 1)          # ★ 丢掉的那一遍
        sec = dep_time(bpaths, nbytes, k, reps)
        out[k] = (sec, bandwidth_gbps(nbytes, sec))
    return out


# ★★ 一次计时**至少要有这么长**（秒）。短于此的测量会被「一次抢占」翻倍：
#   实测 64 KB 的一遍只有 1 µs，读出来在 65.5 与 32.8 GB/s 之间跳（正好差 2 倍）——
#   那**不是 cache 效应**，是**计时尺度太短**（`01` 的 `C.1` 因此随机失败过）。
#
# ★★ 为什么是 **20 ms**（**实测标定，不是随手取大**）：
#   窗口 `W = inner × 一遍耗时` 里混进一次时长 `P` 的抢占时，报出来的「一遍耗时」被抬高
#   `P / W`，于是 `C.1` 判的那个极差 ≈ `(W + P_a) / (W + P_b)`。要让它 < 1.5，
#   就得让**两点抢占时长之差**远小于窗口：`P_a − P_b < 0.5 · W`（`P ≪ W` 时）。
#   ★★ 关键的经验事实是：**`P` 不随 `W` 成比例增长**（否则放大窗口毫无用处）。
#      实测（自造 9 个忙进程打满 10 核，看 `UNROLL8_SIZES` 的 cache 侧三点）：
#        · `W = 50 µs`（`inner` = 46/11/3）⇒ 三点被抬高 **23 / 87 / 95 µs**，
#          带宽 **41.9 / 21.7 / 21.6 GB/s**（真值 ≈ 62）⇒ 极差 **1.94 倍**，`C.1` 红；
#        · `W = 1 ms`（`inner` = 916/229/57）⇒ 仍撞到过 **1.8 ms** 的一次抢占
#          （256 KB 掉到 21.4 GB/s），极差 **2.90 倍**；
#        · `W = 20 ms` ⇒ 三点的极差 **1.004–1.173**（本会话满载下 23 次实测）。
#      ⇒ 窗口放大 400 倍，`P` 的**典型值**没有跟着放大 400 倍，于是 `P/W` 被摊薄。
#   ★ 另一条**必须保住**的性质（它在旧值下也成立，**不是**这次改动带来的）：
#     `inner ∝ 1/一遍耗时` 使**三点的窗口等长**（都 ≈ MIN）—— 这让「一次抢占抬高多少比例」
#     对三点是同一个量级，极差才可能小。**把它改掉（例如给 `inner` 设上限）就会破坏 `C.1`。**
#   ★ 它**不改变量法**：还是同一个「扫一遍的耗时」，只是让每次计时更长；
#     而代价可以忽略（`UNROLL8_SIZES` 七档 × 6 次计时 × 20 ms ≈ 0.9 s）。
UNROLL8_MIN_MEASURE_S = 2.0e-2
UNROLL8_EST_BW = 60e9          # 粗估：本机 8 路展开顺序读 50–65 GB/s


def unroll8_inner(nbytes):
    """每个计时里扫几遍，使一次计时 ≥ `UNROLL8_MIN_MEASURE_S`。

    ★★ **它同时把各规模的窗口拉齐**（`inner ∝ 1/一遍耗时`）：只要一遍的耗时 < MIN，
      每个规模的窗口就都是 `≈ MIN`。**这是一条必须保住的性质，但它不是这次改动带来的**
      （旧的 50 µs 下窗口同样是等长的）：它让「一次抢占抬高多少**比例**」对三点同量级，
      极差才可能小。给 `inner` 设上限（或按固定遍数扫）会破坏它。
    ★ 真正随 MIN 变的是**窗口长度**本身 —— 抢占时长 `P` 不随窗口成比例增长，
      于是 `P/W` 被摊薄（模块头有实测标定）。
    ★ 代价：规模越小扫的遍数越多，总时长 = `档数 × (reps+1) × MIN`（与规模无关）。
    """
    est = nbytes / UNROLL8_EST_BW
    return max(1, int(round(UNROLL8_MIN_MEASURE_S / est)))


def unroll8_time(bpaths, nbytes, reps=UNROLL8_REPS, inner=None):
    """跑一次 UNROLL8，返回**一遍扫描的耗时**（秒）。

    ★ **不要在这里再除一次 `inner`**：C 端已经把计时除以了 `inner`
      （它报的是「一遍」的时间）。实施者第一版在两边各除一次，
      于是 64 KB 报出 **3014 GB/s** —— 一个荒谬到一眼能看出的数
      （这正是 P13 判据第②句要抓的东西，而 `01` 的 `C 类` 现在也有荒谬性闸门）。
    """
    inner = unroll8_inner(nbytes) if inner is None else max(1, inner)
    return run_binary(bpaths["UNROLL8"], [nbytes, reps, inner])


def unroll8_curve(bpaths, sizes=UNROLL8_SIZES, reps=UNROLL8_REPS, warm=True):
    """跑 8 路写死展开的顺序读，返回 `{字节数: (秒, 有效带宽 GB/s)}`。"""
    out = {}
    for b in sizes:
        if warm:
            unroll8_time(bpaths, b, 1)              # ★ 丢掉的那一遍（同 DEP 的理由）
        sec = unroll8_time(bpaths, b, reps)
        out[b] = (sec, bandwidth_gbps(b, sec))
    return out


def bandwidth_gbps(nbytes, seconds):
    """有效带宽 = 搬运的字节数 ÷ 耗时。

    ★ **口径就写在这里**：`nbytes` 是**数组的字节数**（顺序读一遍）。
      `seconds <= 0`（计时器读数相同、或循环被优化器消掉）时返回 `inf`
      —— **不是**抛异常，也不是编一个数：`inf` 会被 `01` 的荒谬性上限抓住。
    """
    if seconds <= 0.0:
        return float("inf")
    return nbytes / seconds / 1e9


def describe(nbytes):
    """把字节数写成人类可读的规模（`64 KB` / `8 MB`）。"""
    if nbytes >= MB:
        return "%g MB" % (nbytes / MB)
    return "%g KB" % (nbytes / 1024)


def make_tmpdir(prefix="perfmb_"):
    return tempfile.mkdtemp(prefix=prefix)
