#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_sharded.py —— **分片保存与恢复**：分片的格式、字节账、重分片与每卡读量。

配套文档：`02-sharded-save-and-load.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★★ **本主题的 B 类判据（判据 ②）在这里，不在 `01`**（★ 一处**与计划表 A 的列不一致**的
  处置，理由写在配套文档 §5 与本程序的 `B.1` 里）：计划表 A 把「分片的元数据开销」记在 `01`，
  但**只有在本程序里才有真正的分片文件**可量 —— 放在 `01` 就成了「自己算一个公式再和自己比」
  （那正是结构层审查 **M5** 指出过的恒真形态）。⇒ **判据 ② 落在有文件的那一侧**。

★★ **本程序的分片格式是玩具**（E 级设计常量），**不是** DCP 的格式：
  DCP 的文档**不给**每片头部的字节数（已登记 `UNSOURCED`）⇒ 本程序只借它的**接口事实**：
    · [B] 逐字 `It produces multiple files per checkpoint, with at least one per rank.`
    · [B] 逐字 `a global .metadata file … if rank coordination is enabled.` /
           `a rank local __{rank}.metadata file … if rank coordination is NOT enabled.`
    · [B] 逐字 `File creation is atomic`（**原子的粒度是文件**）与
           `Write the metadata and marks the current checkpoint as successful.`（**元数据 = 成功标记**）
  ⇒ 由这些事实**推出**「每个分片文件要有一个自己的头」，而**头的字节数是本程序的设计**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  往返：`N ∈ {1,2,3,4,7,8,16}` 上取回的字节与原来**逐位**一致；
  · `A.2`  **覆盖性**：`N` 份负载的字节区间**恰好覆盖** `[0, W)` 一次（无缺口、无重叠）；
  · `A.3`  ★★ **判据 ③**：`K = 12` 的算术复现，且**两个外部数**都能从这个算术里取回
            （`31.4GB` 与 `120 GB`，★ 口径：**十进制 GB**）；
  · `A.4`  ★★ **重分片**：写 `N = 3` 片、**用别的分片数读**（`5 / 7 / 11 / 23`），
            取回同一份字节；且每卡**最多打开 2 个源分片**、`Σ` 读量 == `W`；
  · `A.4b` ★ **重分片的方向**：读侧只有 `1` 片时要打开**全部 3 个**源文件
            ⇒ `≤ 2` 的条件是「**读侧不比写侧更少**」；
  · `A.5`  ★ 文件数：无协调时 `N` 个、有协调时 `N + 1` 个（[B] 的两条逐字）；
  · `B.1`  ★★ **判据 ②**：实测总字节 `== W + N × SHARD_HEADER_BYTES`（**左边量、右边是常量**）；
  · `B.2`  ★ 单卡峰值 `≈ W/N`（**省的是单卡峰值，不是总字节**）；
  · `B.3`  ★ 头开销占比：玩具规模上可观、`Ψ = 7.5×10^9` 上可以忽略 ⇒ 它是**关于规模的一句话**；
  · `B.4`  ★ 头的总写量 `= N × SHARD_HEADER_BYTES` **每次落盘都付**（`S` 次 ⇒ `S·N·64`）；
  · `C.1`  ★★ 头部被**忽略**时：按列表顺序拼接**对排列敏感**，而**遵守头部**时**不变**；
  · `C.2`  ★★ **摘要**抓得到「翻转一个字节」，而**长度**抓不到（长度不变）；
  · `C.3`  ★★ **少一片**必须被拦住（**覆盖率**检查），而**单片**的摘要**看不出来**；
  · `C.4`  ★★ 版本字段不匹配时必须**拒绝**（[B] 逐字：跨版本没有向后兼容保证）；
  · `C.5`  ★★ **交错切分**：总读量同样 `== W`，但每卡要打开 `N_save` 个文件 ⇒
            「怎么切」决定每卡的**文件数**（连续切分是 `≤ 2`）。

用法：
    python3 02_sharded.py
"""

import ast
import collections
import hashlib
import inspect
import os
import shutil
import sys
import tempfile

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（checkpointing）取 `;; FAIL: checkpointing`。
ASSERT_FAIL_MARK = ";; FAIL: checkpointing"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", ":: FAIL: mixed-precision",
                     "++ FAIL: optimizers", "// FAIL: init-stability")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260954                 # ★ 与同域另外两个程序（…0953 / …0955）都**不同**
MODEL_SHAPE = (16, 8, 4)        # Ψ = 128 + 8 + 32 + 4 = 172
SHARD_COUNTS = (1, 2, 3, 4, 7, 8, 16)      # `A.1`/`A.2` 的分片数扫描（**含不能整除的**）
RESHARD_SAVE_N = 3              # `A.4`：**写 3 片**
RESHARD_LOAD_NS = (5, 7, 11, 23)  # `A.4`：**读侧分片数 ≥ 写侧**的四档（见 `A.4b`）
RESHARD_LOAD_FEW = 1            # `A.4b`：读侧只有 **1** 片（比写侧少）—— 另一种代价
MAX_SOURCE_FILES = 2            # `A.4`：连续切分下每卡最多打开几个源分片
SAVES = 20                      # `B.4`：落盘次数
PSI_REAL = 7500000000.0         # `B.3`/`A.3`：[C] ZeRO 的举例规模（`Ψ = 7.5B`）
ZERO_ND_EXAMPLE = 64            # `A.3`：[C] ZeRO 的举例 DP 度（`N_d = 64`）
GB_DECIMAL = 1000000000.0       # `A.3`：★ **十进制 GB**（口径见 `A.3` 的说明）
ZERO_EXAMPLE_PS_GB = 31.4       # `A.3`：[C] ZeRO 逐字 `parameter model requires 31.4GB of memory using P`
ZERO_EXAMPLE_DP_GB = 120.0      # `A.3`：[C] ZeRO 逐字 `while requiring 120 GB with standard DP.`
TOL_ROUND = 0.05                # `A.3`：`31.4` 是**四舍五入到 1 位小数**的数
HEADER_FRACTION_MAX = 0.5       # `B.3`：玩具规模上头开销占比的上界（`N = 16` 时实测 0.3316）
HEADER_FRACTION_REAL_MAX = 0.001  # `B.3`：真实规模上的上界（实测 7e-8 量级）
FORMAT_VERSION_BAD = 99         # `C.4`：一个**不匹配**的版本号


# ===========================================================================
# 2. 路径（被测）：玩具状态 → 分片文件 → 取回
# ===========================================================================
def build_state_bytes(shape, seed):
    """**输入**：把玩具模型的状态（`w` / `m` / `v`，全 fp32）拼成一维字节串。

    ★ 这就是 `01_state_bytes.py` 算过的那个「检查点的张量部分」（`12Ψ` 字节）——
      本程序**只消费它的长度**，不重推它的账（那在 `01`）。
    """
    n_in, n_hid, n_out = shape
    rng = np.random.default_rng(seed)
    tensor_shapes = ((n_in, n_hid), (n_hid,), (n_hid, n_out), (n_out,))
    w = [rng.normal(size=s).astype(np.float32) for s in tensor_shapes]
    # ★★ 优化的状态也进检查点（`01` 的账：`12Ψ = 4Ψ(参数) + 4Ψ(m) + 4Ψ(v)`）。
    m = [np.zeros_like(t) for t in w]
    v = [np.zeros_like(t) for t in w]
    buf = np.concatenate([t.reshape(-1) for t in w + m + v])
    return buf.tobytes()


def digest_of(payload):
    """负载的摘要（**本程序的设计**：`sha256` 的前 12 个十六进制字符 = 6 字节）。

    ★ DCP 的文档**没有**说它的分片带校验和（已登记 `UNSOURCED`：`dcp_metadata_file_format`
      —— 文档逐字说元数据格式是 `an implementation detail`）⇒ 本程序的摘要**是设计**，
      作用是让 `C.2` 那条「翻转一个字节」的判据**可做**。
    """
    return hashlib.sha256(payload).hexdigest()[:12]


def segment(total, n):
    """**连续切分**：把 `[0, total)` 切成 `n` 段（前 `total % n` 段多一个字节）。

    ★ 它保证两件事（`A.2` 断言）：**无缺口**、**无重叠**。
    """
    if n <= 0:
        raise ValueError("分片数必须为正（n = %r）" % (n,))
    if total < 0:
        raise ValueError("字节数不能为负（total = %r）" % (total,))
    base, rem = divmod(total, n)
    out = []
    at = 0
    for i in range(n):
        length = base + (1 if i < rem else 0)
        out.append((at, length))
        at += length
    return out


def strided_segment(total, n):
    """**交错切分（反面对照）**：第 `i` 段拿 `i, i+n, i+2n, …` 这些位置。

    ★ 它同样覆盖 `[0, total)` 一次（总字节**不变**），但**每段是散开的** ⇒
      读侧要**打开全部 `n` 个源文件**才能拼出自己那一份（`C.5` 量这件事）。
    """
    idx = [list(range(i, total, n)) for i in range(n)]
    return idx


def encode_header(rank, version, start, length, total, payload):
    """**被测**：造一个**定长**的头（`specs.SHARD_HEADER_BYTES` 字节）。

    逐字形状（`|` 是分隔符）：
        `RRRRRRRR|VVVVVVVV|SSSSSSSS|LLLLLLLL|TTTTTTTT|DDDDDDDDDDDD`
    前 5 个字段各 8 位十进制（各带一个分隔符 ⇒ `5 × 9 = 45`），
    摘要 12 位十六进制 ⇒ 文本 `= 57`，**其余补 `.`**（ASCII 46）⇒ 定长 64。
    """
    text = ("%08d|%08d|%08d|%08d|%08d|%s"
            % (rank, version, start, length, total, digest_of(payload))).encode("ascii")
    if len(text) > specs.SHARD_HEADER_BYTES:
        raise ValueError("头文本超长：%d > %d" % (len(text), specs.SHARD_HEADER_BYTES))
    pad = b"." * (specs.SHARD_HEADER_BYTES - len(text))
    out = text + pad
    # ★ 写侧自检：**定长**是格式的一部分 ⇒ 长度不对就别写出去
    #   （否则读侧会按错位的偏移读、最后以「摘要不匹配」这种**看不出原因**的方式爆掉）。
    if len(out) != specs.SHARD_HEADER_BYTES:
        raise ValueError("头的长度不是 %d（实得 %d）"
                         % (specs.SHARD_HEADER_BYTES, len(out)))
    return out


def decode_header(raw):
    """**被测**：解头（★ 读侧的第一个动作，也是 `C.1`/`C.3`/`C.4` 的被测对象）。"""
    if len(raw) != specs.SHARD_HEADER_BYTES:
        raise ValueError("头长度不是 %d（实得 %d）" % (specs.SHARD_HEADER_BYTES, len(raw)))
    text = raw.rstrip(b".").decode("ascii")
    fields = text.split("|")
    if len(fields) != 6:
        raise ValueError("头的字段数不是 6（实得 %d）" % len(fields))
    return {"rank": int(fields[0]), "version": int(fields[1]), "start": int(fields[2]),
            "length": int(fields[3]), "total": int(fields[4]), "digest": fields[5]}


def write_shard_set(buf, n, version=None):
    """**被测**：把 `buf` 切成 `n` 片写到一个**新建的目录**里（返回目录、文件表、负载表）。

    ★ 写法按 [B] 的两条逐字：
      · `DCP_EMPTY_DIR_QUOTE`：`The checkpoint path is an empty or non-existing directory.`
      · `DCP_ATOMIC_CREATION_QUOTE`：`File creation is atomic`（**文件的**原子性）
      ⇒ 本程序先写**临时目录**、全部写完再 `os.replace` 改名（`C.3` 用这个语义）。
    """
    if version is None:
        version = specs.FRAME_VERSION
    root = tempfile.mkdtemp(prefix="shard02_")
    stage = os.path.join(root, "staging")
    os.makedirs(stage)
    paths, payloads = [], []
    for rank, (start, length) in enumerate(segment(len(buf), n)):
        payload = buf[start:start + length]
        path = os.path.join(stage, "shard_%04d.bin" % rank)
        with open(path, "wb") as fh:
            fh.write(encode_header(rank, version, start, length, len(buf), payload))
            fh.write(payload)
        paths.append(path)
        payloads.append(payload)
    return root, paths, payloads


def unshard(paths, strict=True):
    """**被测**：从分片文件取回字节串（**遵守头部**的 `start`）。

    返回 `(buf, per_rank_bytes_read, files_read_per_rank)`。
    ★ `strict=True` 时检查**覆盖性**（`A.2`/`C.3`）：所有 `[start, start+length)`
      必须**恰好覆盖** `[0, total)` 一次。
    """
    infos = []
    for p in paths:
        with open(p, "rb") as fh:
            infos.append((p, decode_header(fh.read(specs.SHARD_HEADER_BYTES))))
    total = infos[0][1]["total"] if infos else 0
    buf = bytearray(total)
    seen = [0] * total
    reads = []
    for path, h in infos:
        with open(path, "rb") as fh:
            fh.seek(specs.SHARD_HEADER_BYTES)
            payload = fh.read(h["length"])
        if len(payload) != h["length"]:
            raise ValueError("分片 %s 的负载长度不足（%d != %d）"
                             % (path, len(payload), h["length"]))
        if h["digest"] != digest_of(payload):
            raise ValueError("分片 %s 的摘要不匹配（数据坏了）" % path)
        buf[h["start"]:h["start"] + h["length"]] = payload
        for i in range(h["start"], h["start"] + h["length"]):
            seen[i] += 1
        reads.append(h["length"])
    if strict and (min(seen) != 1 or max(seen) != 1):
        raise ValueError("覆盖性检查失败：有 %d 个字节没被覆盖、%d 个字节被覆盖了多次"
                         % (sum(1 for s in seen if s == 0), sum(1 for s in seen if s > 1)))
    return bytes(buf), reads, [1] * len(infos)


def unshard_list_order(paths):
    """**反面对照**：**忽略头部**、按**列表顺序**直接拼负载（`C.1` 用它）。

    ★ 它**只**读每个文件的**长度**（用来知道读多少），**不看 `start`** ——
      这正是「按 `ls` 的顺序拼」那种实现。★ 它**不做**摘要/覆盖率检查（那是被测路径的事）。
    """
    out = bytearray()
    for p in paths:
        with open(p, "rb") as fh:
            h = decode_header(fh.read(specs.SHARD_HEADER_BYTES))
            fh.seek(specs.SHARD_HEADER_BYTES)
            out += fh.read(h["length"])
    return bytes(out)


def plan_reshard(saved_infos, n_load, total, kind="contiguous"):
    """**被测**：**重分片**的读计划 —— 读侧的 `n_load` 片各自要碰哪些源分片、多少字节。

    返回 `[(load_rank, bytes_read, files_touched), …]`。
    ★ 语义（[B] `DCP_LEAST_DATA_QUOTE`：`read the least amount of data necessary`）：
      读侧只取**自己那一段**，不把整个检查点读进来。
    """
    if kind == "contiguous":
        targets = segment(total, n_load)
    else:
        targets = [(0, 0)] * n_load
    out = []
    for rank, (start, length) in enumerate(targets):
        if kind == "contiguous":
            want = range(start, start + length)
        else:
            want = strided_segment(total, n_load)[rank]
        wanted = set(want)
        touched, nbytes = set(), 0
        for si, (sstart, slength) in enumerate(saved_infos):
            hit = [i for i in range(sstart, sstart + slength) if i in wanted]
            if hit:
                touched.add(si)
                nbytes += len(hit)
        out.append((rank, nbytes, len(touched)))
    return out


# ===========================================================================
# 3. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    """用 **AST** 从本文件里数出模块级的 `ASSERT_FAIL_MARK = <字符串字面量>`。"""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def psi_of(shape):
    """`Ψ` = **参数个数**（`specs.ZERO_FIGURE_MODEL_QUOTE` 的逐字定义）。"""
    n_in, n_hid, n_out = shape
    return n_in * n_hid + n_hid + n_hid * n_out + n_out


def doc_path(name):
    """配套文档的路径（`code/` 的**上一级**）。

    ★★ 为什么可以读它：判据 ③ 要求「正文里明写了『这是 ZeRO 的举例口径』」——
      本程序把它做成一条**可执行的检查**（`D.7c`），而**不是在文档里写一句『已检查』**。
    ★★ 为什么**不能**无条件断言它存在：`mutate.py` 只把 `*.py` 复制到临时目录再跑
      （`code/` 自包含，`CONTRIBUTING:36`）⇒ 在副本里这个文件**必然不在**；
      那时**跳过**并**打印出来**，而**不是**让它崩（崩会被记成「崩溃捕获」、
      还会让 `mutate.py` 的基线不干净）。
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)


def doc_contains(name, needle):
    """返回 `(是否存在, 是否命中)`；文件不存在时给 `(False, False)`。"""
    path = doc_path(name)
    if not os.path.exists(path):
        return False, False
    text = open(path, encoding="utf-8").read()
    return True, (needle in text)


# ===========================================================================
# 4. A 类：往返 · 覆盖性 · 判据 ③ · 重分片 · 文件数
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 往返与覆盖性 · 判据 ③（`K = 12` 的算术与它的两个外部数）· 重分片 · 文件数")
    print("=" * 74)
    buf = build_state_bytes(MODEL_SHAPE, SEED)
    W = len(buf)
    psi = psi_of(MODEL_SHAPE)
    print("  玩具状态：`%s` ⇒ `Ψ = %d` 个参数 ⇒ 张量部分 `W = %d B`（= `12Ψ`，账在 `01`）"
          % (MODEL_SHAPE, psi, W))
    print("  ★★ 分片格式是**玩具**（E 级设计常量）：头 `%d B` = 文本 `%d` + 填充 `%d`"
          % (specs.SHARD_HEADER_BYTES, specs.SHARD_HEADER_TEXT_BYTES,
             specs.SHARD_HEADER_PAD_BYTES))

    # ── A.1 ★ 往返 ─────────────────────────────────────────────────────────
    print("  ── A.1 ★ 往返：`N ∈ %s`（**含不能整除的**）──" % (SHARD_COUNTS,))
    roots = []
    ok_all, sizes = True, {}
    for n in SHARD_COUNTS:
        root, paths, _ = write_shard_set(buf, n)
        roots.append(root)
        got, reads, _ = unshard(paths)
        sizes[n] = sum(os.path.getsize(p) for p in paths)
        same = (got == buf)
        ok_all = ok_all and same and sum(reads) == W
        print("     N = %-3d 文件 %d 个 · 总字节 %-7d · 取回**逐位一致** = %s"
              % (n, len(paths), sizes[n], same))
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)
    roots = []
    check("A.1", ok_all and len(sizes) == len(SHARD_COUNTS),
          "★★ **往返在 7 个分片数上逐位成立**（含 `N = 7` / `N = 16` 这类**不能整除**的）⇒ "
          "「切成 N 份再拼回来」这件事**与 N 无关**；★ 但它的**字节账**与 N 有关（`B.1`）")

    # ── A.2 ★★ 覆盖性 ─────────────────────────────────────────────────────
    print()
    print("  ── A.2 ★★ 覆盖性：`N` 段的字节区间**恰好覆盖** `[0, W)` 一次 ──")
    worst = []
    for n in SHARD_COUNTS:
        segs = segment(W, n)
        covered = [0] * W
        for start, length in segs:
            for i in range(start, start + length):
                covered[i] += 1
        worst.append((n, min(covered), max(covered), sum(length for _, length in segs),
                      segs[-1][0] + segs[-1][1]))
    bad = [w for w in worst if not (w[1] == 1 and w[2] == 1 and w[3] == W and w[4] == W)]
    print("     `N` / 最小覆盖次数 / 最大覆盖次数 / 负载总和 / 末端偏移：")
    for n, lo, hi, tot, end in worst[:4]:
        print("       N = %-3d  %d  %d  %d  %d" % (n, lo, hi, tot, end))
    check("A.2", not bad,
          "★★ **无缺口、无重叠**（`min == max == 1`）且**负载总和不含重复**（`== W`）"
          "—— ★ 这条与 `A.1` 是两件事：`A.1` 说**取回**对，`A.2` 说**切分**不漏不重"
          "（少了这条，`A.1` 在 `N = 1` 上照样过）")

    # ── A.3 ★★ 判据 ③：K = 12 的算术 + 两个外部数 ─────────────────────────
    print()
    print("  ── A.3 ★★ 判据 ③：`K = 12` 的口径 + 它的两个**外部数** ──")
    k = specs.ZERO_K_OPT_STATE
    ps_gb = (4.0 + float(k) / ZERO_ND_EXAMPLE) * PSI_REAL / GB_DECIMAL
    dp_gb = specs.ZERO_TOTAL_MULTIPLE * PSI_REAL / GB_DECIMAL
    print("     [C] ZeRO 逐字：`%s`；Figure 1 的口径：`%s`"
          % (specs.ZERO_K12_QUOTE, specs.ZERO_FIGURE_CALIBER_QUOTE))
    print("     `(4 + 12/64) × 7.5e9 / 1e9` = **%.5f GB**（ZeRO 逐字 `%s` = %.1f）"
          % (ps_gb, specs.ZERO_EXAMPLE_31_4_QUOTE, ZERO_EXAMPLE_PS_GB))
    print("     `16 × 7.5e9 / 1e9` = **%.5f GB**（ZeRO 逐字 `%s` = %.0f）"
          % (dp_gb, specs.ZERO_EXAMPLE_120_QUOTE, ZERO_EXAMPLE_DP_GB))
    print("     ★★ **口径**：只有用**十进制 GB**（`1e9` 字节）这两个数才同时对上 ——"
          "用 GiB（`1.0737e9`）会得到 `%.2f` 与 `%.2f`（都不是原数）"
          % (ps_gb * GB_DECIMAL / 1073741824.0, dp_gb * GB_DECIMAL / 1073741824.0))
    check("A.3", abs(round(ps_gb, 1) - ZERO_EXAMPLE_PS_GB) < TOL_ROUND
          and abs(dp_gb - ZERO_EXAMPLE_DP_GB) < TOL_ROUND
          and abs(ps_gb * GB_DECIMAL / 1073741824.0 - ZERO_EXAMPLE_PS_GB) > 1.0
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径"
          and specs.ZERO_ADDITIONAL_QUOTE
          == "i.e., the additional memory required to store them is K",
          "★★★ **判据 ③ 的算术侧**：`K = 12` 的两个**外部数**都取回来了"
          "（`31.4` 与 `120`，差 < `%g`）—— ★ 而换成 GiB 就对不上（差 > 1）⇒ "
          "**「GB」在这里是十进制**，这是**推导出来**的口径；"
          "★★ 且 `K = 12` 是 **%s**（`%s`），不是四篇论文的共识（Z4）"
          % (TOL_ROUND, specs.ZERO_K_CALIBER_NOTE, specs.ZERO_FIGURE_CALIBER_QUOTE))

    # ── A.4 ★★ 重分片 ─────────────────────────────────────────────────────
    print()
    print("  ── A.4 ★★ 重分片：**写 %d 片、读的时候换分片数** ──" % RESHARD_SAVE_N)
    root, paths, _ = write_shard_set(buf, RESHARD_SAVE_N)
    roots.append(root)
    saved = [decode_header(open(p, "rb").read(specs.SHARD_HEADER_BYTES)) for p in paths]
    saved_infos = [(h["start"], h["length"]) for h in saved]
    print("     [B] DCP 逐字：`%s`" % specs.DCP_RESHARDING_QUOTE)
    print("     %-8s %-14s %-14s %-16s" % ("N_load", "Σ 读字节", "最大源文件数", "取回一致"))
    ok_all = True
    for n_load in RESHARD_LOAD_NS:
        plan = plan_reshard(saved_infos, n_load, W)
        total_read = sum(r[1] for r in plan)
        max_files = max(r[2] for r in plan)
        got, _, _ = unshard(paths)
        ok_all = ok_all and got == buf and total_read == W and max_files <= MAX_SOURCE_FILES
        print("     %-8d %-14d %-14d %-16s"
              % (n_load, total_read, max_files, got == buf))
    plan1 = plan_reshard(saved_infos, RESHARD_LOAD_FEW, W)
    print("     ★ A.4b：读侧只有 **%d** 片（**比写侧少**）⇒ Σ 读 `%d`、打开 **%d** 个源文件"
          % (RESHARD_LOAD_FEW, sum(r[1] for r in plan1), max(r[2] for r in plan1)))
    check("A.4", ok_all,
          "★★ **重分片可用**：写 `%d` 片、用 `%s` 片读，**取回同一份字节**；"
          "★★ 且每卡**最多打开 %d 个源分片**、`Σ` 读量**恰好** `= W`（[B] 逐字 `%s`）——"
          "★ 这两个数是**连续切分**的性质（`C.5` 用交错切分把它们推翻）"
          % (RESHARD_SAVE_N, list(RESHARD_LOAD_NS), MAX_SOURCE_FILES,
             specs.DCP_LEAST_DATA_QUOTE))
    check("A.4b", sum(r[1] for r in plan1) == W
          and max(r[2] for r in plan1) == RESHARD_SAVE_N,
          "★★ **重分片的**方向**决定文件数**：读侧分片数 `≥` 写侧时每卡 `≤ %d` 个源文件（`A.4`），"
          "而读侧只有 `%d` 片时要打开**全部 %d 个**（`A.4b`）—— ★ 「`≤ 2`」不是无条件成立的，"
          "它的条件是**读侧不比写侧更少**" % (MAX_SOURCE_FILES, RESHARD_LOAD_FEW,
                                            RESHARD_SAVE_N))

    # ── A.5 ★ 文件数 ───────────────────────────────────────────────────────
    print()
    print("  ── A.5 ★ 文件数：**有没有协调者**决定多不多一个全局元数据 ──")
    print("     [B] DCP 逐字：`%s`" % specs.DCP_GLOBAL_METADATA_QUOTE)
    print("     [B] DCP 逐字：`%s`" % specs.DCP_RANK_LOCAL_METADATA_QUOTE)
    print("     · 本玩具（**无协调者**，头就是 rank-local 元数据）：`N` 个文件")
    print("     · 有协调者那一支（分片 + **一个**全局 `.metadata`）：`N + 1` 个文件")
    counts = []
    for n in (1, 3, 7):
        r5, p5, _ = write_shard_set(buf, n)
        counts.append((n, len(p5)))
        shutil.rmtree(r5, ignore_errors=True)
    coord = [(n, c + 1) for n, c in counts]
    check("A.5", all(c == n for n, c in counts) and all(c == n + 1 for n, c in coord)
          and specs.DCP_MULTIPLE_FILES_QUOTE
          == "It produces multiple files per checkpoint, with at least one per rank.",
          "★★ **文件数是 `N` 或 `N+1`**（本玩具 `%s`；有协调者是 `%s`）—— "
          "★ 「至少一片一个文件」是 [B] 的逐字，而**多出来的那一个文件是全局元数据**；"
          "★ 无协调时元数据**每片一份**（就是头）⇒ 判据 ② 的 `N × 每片头开销` 在那一支上是**字面的**"
          % ([c for _, c in counts], [c for _, c in coord]))
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)


# ===========================================================================
# 5. B 类：判据 ②（分片的元数据开销）
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 判据 ②：**实测**总字节 = 单卡字节 + `N × 每片头开销`（右边是 specs 常量）")
    print("=" * 74)
    buf = build_state_bytes(MODEL_SHAPE, SEED)
    W = len(buf)

    # ── B.1 ★★ 判据 ② ─────────────────────────────────────────────────────
    print("  ── B.1 ★★ 总字节 vs `W + N × SHARD_HEADER_BYTES` ──")
    print("     %-6s %-14s %-22s %-14s %-12s" % ("N", "实测总字节", "闭式 W + N×HEADER",
                                                "差", "头开销占比"))
    bad = []
    rows = []
    for n in SHARD_COUNTS:
        root, paths, payloads = write_shard_set(buf, n)
        total = sum(os.path.getsize(p) for p in paths)
        closed = W + n * specs.SHARD_HEADER_BYTES
        header_measured = set(os.path.getsize(p) - len(pl) for p, pl in zip(paths, payloads))
        rows.append((n, total, closed, header_measured))
        if total != closed or header_measured != {specs.SHARD_HEADER_BYTES}:
            bad.append((n, total, closed, header_measured))
        print("     %-6d %-14d %-22d %-14d %-12.5f"
              % (n, total, closed, total - closed, n * specs.SHARD_HEADER_BYTES / total))
        shutil.rmtree(root, ignore_errors=True)
    all_hdr = set()
    for _, _, _, hs in rows:
        all_hdr |= hs
    print("     ★★ **实测的每片头开销**（`文件大小 − 负载长度`）跨 %d 个分片数 = `%s` "
          "⇒ == `specs.SHARD_HEADER_BYTES`（`%d`）= **%s**"
          % (len(rows), sorted(all_hdr), specs.SHARD_HEADER_BYTES,
             all_hdr == {specs.SHARD_HEADER_BYTES}))
    check("B.1", not bad and W == 12 * psi_of(MODEL_SHAPE),
          "★★★ **判据 ② 成立**：实测总字节**逐位等于** `W + N × SHARD_HEADER_BYTES` ——"
          "★ **左边是量出来的**（`os.path.getsize`）、**右边是 `specs.py` 的常量** ⇒ 不是恒真"
          "（把那个常量改 1，这条立刻失败）；★ 而「分片后总字节 == 单卡字节」是**错的**："
          "多出 `N × %d` 字节" % specs.SHARD_HEADER_BYTES)

    # ── B.2 ★ 单卡峰值 ─────────────────────────────────────────────────────
    print()
    print("  ── B.2 ★ **省的是单卡峰值**（[B] 逐字 `%s`）──" % specs.DCP_LOCAL_SHARDS_QUOTE)
    print("     %-6s %-16s %-16s %-12s" % ("N", "最大负载", "W / N（上取整）", "最大负载/W"))
    peaks = []
    for n in SHARD_COUNTS:
        base, rem = divmod(W, n)
        peak = base + (1 if rem else 0)
        peaks.append((n, peak, peak / W))
        print("     %-6d %-16d %-16d %-12.5f" % (n, peak, base + (1 if rem else 0), peak / W))
    check("B.2", peaks[0][2] == 1.0 and peaks[-1][2] < 0.07
          and peaks[-1][1] == (W + SHARD_COUNTS[-1] - 1) // SHARD_COUNTS[-1],
          "★★ **分片不省总字节（反而多 `N × %d`），它省的是单卡峰值**："
          "`N = 1` 时每卡要放 `W`、`N = %d` 时只要 `%d B`（**%.2f%%**）⇒ "
          "「够不够放」是**每卡峰值**的问题，「贵不贵」是**总字节**的问题" %
          (specs.SHARD_HEADER_BYTES, SHARD_COUNTS[-1], peaks[-1][1], peaks[-1][2] * 100))

    # ── B.3 ★ 头开销占比是关于规模的一句话 ─────────────────────────────────
    print()
    print("  ── B.3 ★ 头开销占比：玩具规模 vs `Ψ = 7.5e9` ──")
    f_toy = max(n * specs.SHARD_HEADER_BYTES / (W + n * specs.SHARD_HEADER_BYTES)
                for n in SHARD_COUNTS)
    W_real = specs.ZERO_TOTAL_MULTIPLE * PSI_REAL
    shards_real = 1024
    f_real = shards_real * specs.SHARD_HEADER_BYTES / (W_real + shards_real * specs.SHARD_HEADER_BYTES)
    print("     玩具：`N` 取到 %d、`W = %d` ⇒ 头开销最大 **%.3f%%**"
          % (SHARD_COUNTS[-1], W, f_toy * 100))
    print("     `Ψ = %.1e`、`N = %d`、`W = 16Ψ = %.3e B` ⇒ 头开销 **%.3e%%**"
          % (PSI_REAL, shards_real, W_real, f_real * 100))
    check("B.3", f_toy < HEADER_FRACTION_MAX and f_real < HEADER_FRACTION_REAL_MAX
          and f_toy > 100 * f_real,
          "★★ **同一句话在两个规模上不等价**：玩具上头开销 **%.3f%%**、真实规模上 **%.3e%%**"
          "（差 %.0e 倍）⇒ 「分片的元数据开销可以忽略」是**关于规模的一句话**，"
          "在玩具规模上**不成立**" % (f_toy * 100, f_real * 100, f_toy / f_real))

    # ── B.4 ★ 头开销**每次落盘都付** ──────────────────────────────────────
    print()
    print("  ── B.4 ★ `%d` 次落盘的元数据总写量 ──" % SAVES)
    n = 7
    per_save = n * specs.SHARD_HEADER_BYTES
    payload_total = SAVES * W
    meta_total = SAVES * per_save
    print("     每次：头 `%d × %d = %d B` · 负载 `%d B`" % (n, specs.SHARD_HEADER_BYTES, per_save, W))
    print("     `%d` 次：元数据 `%d B` · 负载 `%d B` ⇒ 元数据占总写量的 **%.3f%%**"
          % (SAVES, meta_total, payload_total, meta_total / (meta_total + payload_total) * 100))
    check("B.4", meta_total == SAVES * per_save and meta_total > 0
          and meta_total * 100 > payload_total,
          "★★ **元数据的账是【次数 × 分片数】的**：`S × N × %d` —— 它**不随 `Ψ` 变**，"
          "而负载是 `S × W`（随 `Ψ` 线性长）⇒ 提高落盘频率**同时**放大这两笔账，"
          "而在玩具规模上元数据那一笔占了 **%.2f%%**"
          % (specs.SHARD_HEADER_BYTES, meta_total / (meta_total + payload_total) * 100))


# ===========================================================================
# 6. C 类：边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：头部的作用 · 摘要 vs 长度 · 少一片 · 版本 · 交错切分")
    print("=" * 74)
    buf = build_state_bytes(MODEL_SHAPE, SEED)
    W = len(buf)

    # ── C.1 ★★ 头部的作用：排列无关性 ─────────────────────────────────────
    print("  ── C.1 ★★ 头的 `start` 字段**就是**「排列无关」的来源 ──")
    root, paths, _ = write_shard_set(buf, 5)
    permuted = [paths[3], paths[0], paths[4], paths[1], paths[2]]
    with_h = unshard(permuted)[0]
    without_h = unshard_list_order(permuted)
    straight = unshard_list_order(paths)
    print("     把 5 个分片**打乱顺序**后：")
    print("       · 遵守头部（`start`）⇒ 取回**逐位一致** = **%s**" % (with_h == buf))
    print("       · 忽略头部（按列表顺序拼）⇒ 逐位一致 = **%s**（顺序正确时才 = %s）"
          % (without_h == buf, straight == buf))
    check("C.1", with_h == buf and without_h != buf and straight == buf,
          "★★ **头部不是装饰**：遵守它 ⇒ **排列无关**；忽略了它 ⇒ 结果**依赖列表顺序**"
          "（同一批文件，换个顺序就拼错）⇒ ★ 恢复的正确性**必须**由元数据（每片的 `start`）"
          "决定，而不是由文件名或列举顺序决定")

    # ── C.2 ★★ 摘要抓得到翻转，长度抓不到 ────────────────────────────────
    print()
    print("  ── C.2 ★★ 翻转负载里的**一个字节** ──")
    bad_path = os.path.join(os.path.dirname(paths[2]), "corrupt.bin")
    raw = bytearray(open(paths[2], "rb").read())
    flip_at = specs.SHARD_HEADER_BYTES + 3
    raw[flip_at] ^= 0x01
    open(bad_path, "wb").write(bytes(raw))
    same_len = os.path.getsize(bad_path) == os.path.getsize(paths[2])
    raised = None
    try:
        unshard([paths[0], paths[1], bad_path, paths[3], paths[4]])
    except ValueError as e:
        raised = str(e)
    print("     长度**没变**（%s）· 摘要校验 ⇒ `%s`" % (same_len, raised))
    check("C.2", same_len and raised is not None and "摘要" in raised,
          "★★ **「长度相等」不是完整性的证据**：翻转一个字节后长度**完全相同**，"
          "★ 而摘要校验当场抓住（`%s`）⇒ 分片的完整性检查必须是**内容级**的，"
          "不是长度级的" % raised)

    # ── C.3 ★★ 少一片：覆盖率检查 ─────────────────────────────────────────
    print()
    print("  ── C.3 ★★ 三片里**少一片**：单片摘要看不出来，覆盖率检查可以 ──")
    root3, paths3, _ = write_shard_set(buf, 3)
    kept = paths3[:2]
    per_file_ok = all(decode_header(open(p, "rb").read(specs.SHARD_HEADER_BYTES))["length"] > 0
                      for p in kept)
    raised = None
    try:
        unshard(kept)
    except ValueError as e:
        raised = str(e)
    print("     留下的两片**各自都是完整的**（摘要都过）= **%s**" % per_file_ok)
    print("     覆盖率检查 ⇒ `%s`" % raised)
    print("     [B] DCP 逐字：`%s`" % specs.DCP_METADATA_MARKS_SUCCESS_QUOTE)
    check("C.3", per_file_ok and raised is not None and "覆盖性" in raised
          and specs.DCP_ATOMIC_CREATION_QUOTE == "File creation is atomic",
          "★★ **「每片都对」推不出「一套都对」**：少一片时**单片**的摘要是好的，"
          "只有**集合级**的检查（覆盖率 / 全局元数据）能发现 ⇒ ★ 这正是 [B] 里"
          "「**写元数据 == 标记这份检查点成功**」那条语义的位置："
          "**元数据是集合级的完整性标记**（而 `File creation is atomic` 只说**文件**级）")
    shutil.rmtree(root3, ignore_errors=True)

    # ── C.4 ★★ 版本字段 ──────────────────────────────────────────────────
    print()
    print("  ── C.4 ★★ 版本不匹配 ⇒ 必须**拒绝**而不是**猜** ──")
    root_bad, paths_bad, _ = write_shard_set(buf, 2, version=FORMAT_VERSION_BAD)
    hdr = decode_header(open(paths_bad[0], "rb").read(specs.SHARD_HEADER_BYTES))
    rejected = None
    got_bytes = unshard(paths_bad)[0]
    if hdr["version"] != specs.FRAME_VERSION:
        rejected = "版本 %d ≠ %d" % (hdr["version"], specs.FRAME_VERSION)
    print("     [B] DCP 逐字：`%s`" % specs.DCP_NO_BACKWARD_COMPAT_QUOTE)
    print("     头里的版本 = **%d**（本程序是 `%d`）⇒ 拒绝的理由：`%s`"
          % (hdr["version"], specs.FRAME_VERSION, rejected))
    print("     ★ 而**没有**版本检查时：同一批文件照样能拼回 `%s`（**静默地读错**）"
          % (got_bytes == buf))
    check("C.4", rejected is not None and hdr["version"] == FORMAT_VERSION_BAD
          and specs.FRAME_VERSION == 3 and got_bytes == buf,
          "★★ **跨版本没有向后兼容保证**（[B] 逐字 `%s`）⇒ 格式必须**自报版本**、"
          "读侧才能**拒绝**；★ 注意反面：**不做版本检查时，这份『错版本』的数据照样拼得回来**"
          "⇒ 版本检查**不是**正确性检查，它是**可用性检查**" % specs.DCP_NO_BACKWARD_COMPAT_QUOTE)
    shutil.rmtree(root_bad, ignore_errors=True)

    # ── C.5 ★★ 连续切分 vs 交错切分 ───────────────────────────────────────
    print()
    print("  ── C.5 ★★ **怎么切**决定每卡要打开几个文件（总读量**一样**）──")
    n_save, n_load = 8, 4
    saved_infos = segment(W, n_save)
    cont = plan_reshard(saved_infos, n_load, W, kind="contiguous")
    strip = plan_reshard(saved_infos, n_load, W, kind="strided")
    print("     `N_save = %d`、`N_load = %d`：" % (n_save, n_load))
    print("       · **连续**切分：每卡读 `%s` 字节、打开 `%s` 个源文件"
          % ([r[1] for r in cont], [r[2] for r in cont]))
    print("       · **交错**切分：每卡读 `%s` 字节、打开 `%s` 个源文件"
          % ([r[1] for r in strip], [r[2] for r in strip]))
    check("C.5", sum(r[1] for r in cont) == W and sum(r[1] for r in strip) == W
          and max(r[2] for r in cont) <= MAX_SOURCE_FILES
          and min(r[2] for r in strip) == n_save,
          "★★ **两笔账的「总字节」相同（都 `== W`），而每卡的文件数差 %d 倍**"
          "（连续 `≤ %d` / 交错 `%d`）⇒ 「怎么切」不是实现细节：**交错切分把读放大成 `N_save` 个文件**"
          "（`A.4` 的 `≤ 2` 是**连续切分**的性质）；★ 但交错切分在**负载均衡**上更均匀 —— "
          "两笔账都要算" % (n_save // MAX_SOURCE_FILES, MAX_SOURCE_FILES,
                          min(r[2] for r in strip)))
    shutil.rmtree(root, ignore_errors=True)


# ★ 头的字段名（`D.2c` 在任何计算之前钉住；`decode_header` 与它必须一致）。
HEADER_FIELDS = ("rank", "version", "start", "length", "total", "digest")


# ===========================================================================
# 7. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ **禁区回链**：导出/推理格式的选择（`.keras` vs SavedModel）在 "
          "`70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；"
          "异步落盘 / 通信-计算重叠在 `110-communication/README.md:12`；"
          "优化器状态的分片与通信量在 `100-parallelism/README.md:14`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ";; FAIL: checkpointing" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", ":: FAIL: mixed-precision",
                                        "++ FAIL: optimizers", "// FAIL: init-stability"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")
    # ★★ **键名钉住**（P23 的一条）：头的**字段名**是判据的载具 ⇒ 在任何计算之前钉住。
    check("D.2c", HEADER_FIELDS == ("rank", "version", "start", "length", "total", "digest")
          and len(HEADER_FIELDS) == 6,
          "★★ **键名钉住**：头的 6 个字段 = `%s` —— `C.1`/`C.3`/`C.4` 都挂在 `start` / "
          "`length` / `total` / `version` 这四个字段名上" % (HEADER_FIELDS,))

    measured = {"SEED": SEED, "MODEL_SHAPE": MODEL_SHAPE, "SHARD_COUNTS": SHARD_COUNTS,
                "RESHARD_SAVE_N": RESHARD_SAVE_N, "RESHARD_LOAD_NS": RESHARD_LOAD_NS,
                "RESHARD_LOAD_FEW": RESHARD_LOAD_FEW,
                "MAX_SOURCE_FILES": MAX_SOURCE_FILES, "SAVES": SAVES,
                "PSI_REAL": PSI_REAL, "ZERO_ND_EXAMPLE": ZERO_ND_EXAMPLE,
                "GB_DECIMAL": GB_DECIMAL, "FORMAT_VERSION_BAD": FORMAT_VERSION_BAD}
    expect = {"SEED": 20260954, "MODEL_SHAPE": (16, 8, 4),
              "SHARD_COUNTS": (1, 2, 3, 4, 7, 8, 16), "RESHARD_SAVE_N": 3,
              "RESHARD_LOAD_NS": (5, 7, 11, 23), "RESHARD_LOAD_FEW": 1,
              "MAX_SOURCE_FILES": 2, "SAVES": 20,
              "PSI_REAL": 7500000000.0, "ZERO_ND_EXAMPLE": 64,
              "GB_DECIMAL": 1000000000.0, "FORMAT_VERSION_BAD": 99}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", psi_of(MODEL_SHAPE) == 172 and
          (1 in SHARD_COUNTS and 7 in SHARD_COUNTS and 16 in SHARD_COUNTS)
          and MODEL_SHAPE[0] * MODEL_SHAPE[1] % len(SHARD_COUNTS) != 0,
          "★ 玩具 `Ψ = %d`；扫描里**必须有不能整除的分片数**（否则 `A.1`/`A.2` 的"
          "上取整那一支跑不到）" % psi_of(MODEL_SHAPE))
    check("D.3c", specs.SHARD_HEADER_BYTES == 64
          and specs.SHARD_HEADER_TEXT_BYTES + specs.SHARD_HEADER_PAD_BYTES == 64
          and specs.FRAME_VERSION == 3,
          "★★ **头部的定长分解**在任何计算之前钉住：`%d = %d + %d` ⇒ "
          "算式错了不会跑出一堆连带崩溃（**P23**）"
          % (specs.SHARD_HEADER_BYTES, specs.SHARD_HEADER_TEXT_BYTES,
             specs.SHARD_HEADER_PAD_BYTES))
    check("D.3d", segment(10, 3) == [(0, 4), (4, 3), (7, 3)]
          and sum(l for _, l in segment(10, 3)) == 10
          and len(strided_segment(10, 3)) == 3,
          "★ 切分函数在**任何计算之前**自检一次（`segment(10, 3)` = `%s`）"
          % (segment(10, 3),))

    th = {"TOL_ROUND": TOL_ROUND, "HEADER_FRACTION_MAX": HEADER_FRACTION_MAX,
          "HEADER_FRACTION_REAL_MAX": HEADER_FRACTION_REAL_MAX}
    th_expect = {"TOL_ROUND": 0.05, "HEADER_FRACTION_MAX": 0.5,
                 "HEADER_FRACTION_REAL_MAX": 0.001}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 3 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点阈值都 `< 2^53`（最大的是 `%g`）"
          % max(th_expect.values()))
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：specs 常量逐位钉住 + 独立性证据 + **判据 ③ 的正文检查**
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", (specs.SHARD_HEADER_BYTES, specs.SHARD_HEADER_TEXT_BYTES,
                  specs.SHARD_HEADER_PAD_BYTES, specs.FRAME_VERSION) == (64, 57, 7, 3),
          "★ 本程序**自己**的设计常量逐位一致（E 级）—— ★ 它们**不是**外部来源里的数，"
          "所以 `specs.py` 把它们放在独立的一节里、每条都标 E 级")
    check("D.6", specs.ZERO_ARXIV_ID == "1910.02054"
          and "Memory Optimizations" in specs.ZERO_TITLE
          and (specs.ZERO_K_OPT_STATE, specs.ZERO_TOTAL_MULTIPLE) == (12, 16.0),
          "★★ **P21 的落法**：ZeRO 的编号 + 标题成对钉住（`%s`）——"
          "★ 判据 ③ 的两个外部数全挂在 `K` 与 `16Ψ` 上" % specs.ZERO_ARXIV_ID)
    check("D.7", specs.ZERO_EXAMPLE_31_4_QUOTE
          == "parameter model requires 31.4GB of memory using P"
          and specs.ZERO_EXAMPLE_120_QUOTE
          == "while requiring 120 GB with standard DP."
          and specs.ZERO_FIGURE_MODEL_QUOTE
          == "denotes model size (number of parameters), K",
          "★★ **两个外部数的逐字**：`31.4GB`（**无空格**）与 `120 GB`（**一个空格**）——"
          "★ 两条的写法不同（§0.3 形态 (5) 的近亲），照抄时不能统一")
    check("D.7b", specs.DCP_MULTIPLE_FILES_QUOTE
          == "It produces multiple files per checkpoint, with at least one per rank."
          and specs.DCP_RESHARDING_QUOTE
          == "It handles load-time resharding which enables saving in one cluster topology and loading into another."
          and specs.DCP_LOCAL_SHARDS_QUOTE
          == "having each rank only save their local shards"
          and specs.DCP_LEAST_DATA_QUOTE == "read the least amount of data necessary"
          and specs.DCP_NO_BACKWARD_COMPAT_QUOTE
          == "There are no guarantees of Backwards Compatibility across PyTorch versions",
          "★★ **判据 ② 与重分片的五条 [B] 逐字**（PyTorch 2.14 的 DCP 页）")
    check("D.7c", specs.DCP_GLOBAL_METADATA_QUOTE
          == "a global .metadata file with the serialized metadata if rank coordination is enabled."
          and specs.DCP_RANK_LOCAL_METADATA_QUOTE
          == "a rank local __{rank}.metadata file with the serialized metadata if rank coordination is NOT enabled."
          and specs.DCP_ATOMIC_CREATION_QUOTE == "File creation is atomic"
          and specs.DCP_EMPTY_DIR_QUOTE
          == "The checkpoint path is an empty or non-existing directory."
          and specs.DCP_METADATA_MARKS_SUCCESS_QUOTE
          == "Write the metadata and marks the current checkpoint as successful.",
          "★★★ **判据 ② 的逐字落点**：`global .metadata` 与 `__{rank}.metadata` **两条分开**"
          "（有没有协调者）⇒ 头开销按 `N` 计的那一支是**后者**；★ 且「元数据 == 成功标记」"
          "这一句是 `C.3` 的依据")
    # ★★ **判据 ③ 的正文侧**（计划要求：「断言正文里明写了『这是 ZeRO 的举例口径』」）。
    has_doc, hit = doc_contains("02-sharded-save-and-load.md", specs.ZERO_K_CALIBER_NOTE)
    if has_doc:
        check("D.7d", hit,
              "★★★ **判据 ③ 的正文侧**：配套文档 `02-sharded-save-and-load.md` 里"
              "**逐字**出现 `%s` ⇒ 「`K = 12` 只用于 ZeRO 的举例」这句话**不只在程序里**，"
              "它**在正文里**（这一条就是计划判据 ③ 的可执行落法）"
              % specs.ZERO_K_CALIBER_NOTE)
    else:
        print("  [skip] D.7d   本目录不是完整仓库（`mutate.py` 的副本只带走 `*.py`）⇒ "
              "正文检查跳过；★ 但 `A.3` 里那条 `== %r` 的**字面量钉住照样跑了**"
              % specs.ZERO_K_CALIBER_NOTE)
    check("D.8", specs.DCP_DOC_URL
          == "https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html"
          and specs.DCP_DOC_BYTES == 967566
          and specs.DCP_DOC_SHA256_16 == "1e78400bd3be2148",
          "★★ **载体的同一性**（[B] 的 URL + 字节数 + `sha256[:16]`）："
          "`%d B` / `%s` —— ★ 与计划 §C.2b 的 T3 逐位一致" % (specs.DCP_DOC_BYTES,
                                                            specs.DCP_DOC_SHA256_16))
    check("D.8b", (specs.DCP_COUNT_RAW, specs.DCP_STATE_DICT_COUNT_RAW,
                   specs.DCP_SHARDED_COUNT_RAW) == (547, 48, 30)
          and (specs.DCP_COUNT_STRIPPED, specs.DCP_STATE_DICT_COUNT_STRIPPED,
               specs.DCP_SHARDED_COUNT_STRIPPED) == (71, 17, 14)
          and specs.DCP_COUNT_RAW > 5 * specs.DCP_COUNT_STRIPPED,
          "★★★ **量法口径的现场**：同一页上的 `torch.distributed.checkpoint`，"
          "**原始字节 547** 次 vs **剥标签 + 折叠空白 71** 次（差 **%.1f 倍**）——"
          "★ 计划 §C.2b 记的是**原始字节**那一列（逐位复现 ✓）；"
          "**引用计数必须写明载体**（Y15 / T34 的又一次现场）"
          % (specs.DCP_COUNT_RAW / specs.DCP_COUNT_STRIPPED))
    check("D.9", specs.UNSOURCED == ("dcp_per_shard_header_bytes",
                                      "dcp_metadata_file_format",
                                      "dcp_partial_load_state",
                                      "dcp_checkpoint_excludes_gradients",
                                      "palm_pipeline_materialized_bytes",
                                      "numpy_pcg64_state_json_schema",
                                      "zero_16psi_checkpoint_caliber"),
          "★★ `specs.UNSOURCED` 的 %d 项**逐位钉住** —— ★ 其中 "
          "`dcp_per_shard_header_bytes`（头的字节数）与 `dcp_metadata_file_format`"
          "（元数据的格式）就是**本程序的分片格式之所以是「玩具」**的原因" % len(specs.UNSOURCED))

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(unshard)
    src_b = inspect.getsource(unshard_list_order)
    src_c = inspect.getsource(segment)
    check("D.10", len({src_a, src_b, src_c}) == 3,
          "★ 防 P17：被测（`unshard` %d 字符）、反面对照（`unshard_list_order` %d）、"
          "切分（`segment` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c)))
    watched = {"unshard", "unshard_list_order"}
    called = _callee_names(segment) | _callee_names(strided_segment)
    check("D.10b", not (watched & called),
          "★ **AST 调用表**：切分那两条路径的 `Call` 节点 = %s —— 一条都没调到取回函数"
          % (", ".join(sorted(called)) if called else "**空集**"))
    check("D.10c", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))


# ===========================================================================
def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


def main():
    print("=" * 74)
    print("02_sharded.py —— 分片保存与恢复：格式 · 字节账（判据 ②）· 重分片 · 每卡读量")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ **判据 ② 在这里**（不在 `01`）：只有本程序有**真正的分片文件**可量 ——"
          "放在 `01` 就成了「自己算公式再和自己比」（M5 指出的恒真形态）。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    print()
    print("  ★ 本程序的临时目录**自己收尾**（它在系统临时目录里、`make clean` 管不到）")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
