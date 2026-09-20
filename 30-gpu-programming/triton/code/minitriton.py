# -*- coding: utf-8 -*-
"""迷你 Triton：纯 Python + numpy 的块级 DSL、解释器、布局分析与归约代价模型。

★ 这是本主题的**迷你引擎**（非编号辅助模块），不是 Triton 的实现，也不是 Triton 的一部分。
  它借用 Triton 的术语（program / program_id / tl.arange / mask / other /
  blocked layout / layout conversion / 跨 warp 归约）以便与真实系统对照，
  但**没有任何一行是 Triton 的代码**，也**跑不出 Triton 的输出**，本机也没有 GPU。
  由它产生的每一个数都是**模型算出来的**，不是实测值。

★ 证据强度声明（必须随结论一起引用）：
  - A 类（算法正确性）：本模块的解释器跑 numpy，与 refs.py 的**独立参照**逐元素比对。
    这是最强的一档，但参照与被测跑在同一台机器、同一个 numpy 上。
  - B 类（代价模型）：**全部是模型值，没有任何一条是本机实测**。
    单位是模型自定的「标量操作数 / 共享内存字节 / barrier 次数」，
    不是指令周期、不是带宽。它们只能支撑**关于机制与相对量的结论**。
  - C 类（边界与反例）：故意注入缺陷后断言必须失败。

★ 边界掩码的建模口径（本模型自定，不是 Triton 的语义）：
  越界索引**不**返回垃圾，而是：
    strict=True  直接抛 OutOfBoundsError（用来抓「忘了加掩码」）
    strict=False 返回毒值 POISON（用来抓「掩码写错但没崩」——毒值会流进输出）
  两者合起来才能把「掩码真的挡住了越界贡献」变成可证伪的断言。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

NP_OF = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
         "i32": np.int32, "i64": np.int64, "i8": np.int8, "bool": np.bool_}
ESIZE = {"f32": 4, "f64": 8, "f16": 2, "i32": 4, "i64": 8, "i8": 1, "bool": 1}

# 毒值：任何正常计算都产生不出它，一旦出现即可判定「越界贡献漏进了输出」
POISON = np.float32(1.0e30)
POISON_GUARD = np.float64(1.0e29)   # 「输出里不该出现 >= 这个数的值」


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class MiniTritonError(Exception):
    """本模型的错误基类。"""


class ShapeError(MiniTritonError):
    """张量形状不匹配。"""


class LayoutError(MiniTritonError):
    """布局本身非法（切分不整除、threadsPerWarp 不是 32）。"""


class LayoutConflict(MiniTritonError):
    """消费者要求的布局与生产者给出的不同，且调用方**拒绝插入转换**。"""


class OutOfBoundsError(MiniTritonError):
    """走了一条越界的访存路径——掩码没写或写错。"""


class PlanInfeasible(MiniTritonError):
    """代价模型的前置条件不满足。"""


# ---------------------------------------------------------------------------
# 带越界检测的缓冲
# ---------------------------------------------------------------------------

class Buff:
    """一块全局内存。后面挂一段毒区，用来接住越界读。

    `flat[size:]` 是毒区；越界索引会被映射到毒区首元素，从而把「读越界」
    显式化成「输出里出现 POISON」。
    """

    def __init__(self, name, arr, poison=POISON, pad=8192):
        self.name = name
        src = np.asarray(arr)
        self.shape = src.shape
        self.size = int(src.size)
        self.dtype = src.dtype
        self.pad = pad
        self._base = np.concatenate([src.reshape(-1), np.full(pad, poison, src.dtype)])
        self.poison = poison
        self.oob_allowed = 0        # 掩码为真却越界（真 bug）
        self.oob_masked = 0         # 掩码为假而越界（合法）
        self.other_undefined = 0    # other=None 且掩码不全为真（Triton 的「未定义」）

    @property
    def flat(self):
        """底层存储（含毒区）。写入落在 [0, size)，毒区在 [size, size+pad)。"""
        return self._base

    @property
    def arr(self):
        """当前内容（写入可见）。"""
        return self._base[:self.size].reshape(self.shape)

    @property
    def poison_tail(self):
        return self._base[self.size:]

    def __len__(self):
        return self.size

    def flat_index(self, idx_arrays):
        if len(idx_arrays) == 1:
            return idx_arrays[0].astype(np.int64)
        if len(idx_arrays) == 2:
            return idx_arrays[0].astype(np.int64) * self.shape[1] + idx_arrays[1].astype(np.int64)
        raise ShapeError("Buff 只支持 1D/2D 索引，实得 %d 维" % len(idx_arrays))

    def record(self, idx_arrays, mask):
        fi = self.flat_index(idx_arrays)
        oob = (fi < 0) | (fi >= self.size)
        if not oob.any():
            return 0
        m = np.ones(fi.shape, bool) if mask is None else np.broadcast_to(mask, fi.shape)
        self.oob_allowed += int((oob & m).sum())
        self.oob_masked += int((oob & ~m).sum())
        return int((oob & m).sum())

    def reset_counters(self):
        self.oob_allowed = 0
        self.oob_masked = 0
        self.other_undefined = 0

    def __repr__(self):
        return "Buff(%s, %s)" % (self.name, self.shape)


# ---------------------------------------------------------------------------
# 块内张量
# ---------------------------------------------------------------------------

class Tile:
    """块内张量：数据用 numpy 表示，另带 dtype 与（可选的）布局。"""

    __slots__ = ("data", "dtype", "layout", "name")

    def __init__(self, data, dtype="f32", layout=None, name=""):
        self.data = np.asarray(data, dtype=NP_OF[dtype])
        self.dtype = dtype
        self.layout = layout
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    @property
    def numel(self):
        return int(self.data.size)

    def __repr__(self):
        return "Tile(%s, %s)" % (self.shape, self.dtype)


def const(v, dtype="f32", name=""):
    return Tile(np.asarray(v, dtype=NP_OF[dtype]), dtype=dtype, name=name)


def arange(lo, hi, dtype="i32", name=""):
    """`tl.arange` 的等价物：一维、半开区间 [lo, hi)。"""
    if hi <= lo:
        raise ShapeError("arange(%d, %d)：区间为空" % (lo, hi))
    return Tile(np.arange(lo, hi, dtype=NP_OF[dtype]), dtype=dtype, name=name)


def full(shape, v, dtype="f32", name=""):
    return Tile(np.full(shape, v, dtype=NP_OF[dtype]), dtype=dtype, name=name)


def zeros(shape, dtype="f32", name=""):
    return full(shape, 0, dtype, name)


def offs_add(offs, base):
    """块内下标整体平移（**保持整型**，不要走 f32 常量那条路）。"""
    return Tile(offs.data + base, dtype=offs.dtype, name=offs.name)


def less_than(offs, n):
    """`offs < n` 的掩码（块级 bool 张量）。"""
    return Tile(offs.data < n, dtype="bool")


def as_bool(x):
    return Tile(np.asarray(x.data if isinstance(x, Tile) else x, dtype=bool), dtype="bool")


# ---------------------------------------------------------------------------
# 访存
# ---------------------------------------------------------------------------

def _as_mask(mask, shape):
    if mask is None:
        return None
    m = np.asarray(mask.data if isinstance(mask, Tile) else mask, dtype=bool)
    return np.broadcast_to(m, shape)


def load(buff, index, mask=None, other=None, dtype="f32", strict=True):
    """带掩码的块级读。`index` 是 Tile 或 Tile 的元组。

    ★ `other` 的默认值是 `None`，这一点**照抄 Triton 官方语义**：
      官方 `triton.language.load` 的参数说明原文是
        「other (Block, optional) – if mask[idx] is false, return other[idx].
          If other is None, the masked-out value is undefined.」
      （来源：https://triton-lang.org/main/python-api/generated/triton.language.load.html）
      本模型把「undefined」具体化成 POISON——**「未定义」必须有个具体的坏值，
      否则它没法被断言**。所以 `other=None` + 掩码不全为真时，被掩掉的位置
      拿到 POISON，并且计数进 `buff.other_undefined`。

    ★ 本模型与 Triton 的已知分歧（据实记录）：本模型**不允许** unmasked 的越界读
      （strict 下直接抛错），而真实 GPU 上那是一次未定义行为。这是一处**加强**，
      目的是让「忘了加掩码」可证伪。
    """
    idx = index if isinstance(index, tuple) else (index,)
    arrays = np.broadcast_arrays(*[i.data for i in idx])
    shape = arrays[0].shape
    m = _as_mask(mask, shape)
    bad = buff.record(arrays, m)
    if bad and strict:
        raise OutOfBoundsError(
            "%s: %d 个位置在掩码为真时越界（掩码没写或写错）" % (buff.name, bad))
    fi = buff.flat_index(arrays)
    oob = (fi < 0) | (fi >= buff.size)
    safe = np.where(oob, buff.size, fi)      # 越界 → 指向毒区首元素
    vals = buff.flat[safe]
    if m is not None:
        if other is None:
            buff.other_undefined += int((~m).sum())
            fill = buff.poison
        else:
            fill = np.asarray(other, buff.arr.dtype)
        vals = np.where(m, vals, fill)
    return Tile(vals, dtype=dtype)


def store(buff, index, value, mask=None, strict=True):
    """带掩码的块级写。掩码为假的位置**不写**。"""
    idx = index if isinstance(index, tuple) else (index,)
    arrays = np.broadcast_arrays(*[i.data for i in idx])
    shape = arrays[0].shape
    m = _as_mask(mask, shape)
    bad = buff.record(arrays, m)
    if bad and strict:
        raise OutOfBoundsError(
            "%s: %d 个位置在掩码为真时越界（写越界）" % (buff.name, bad))
    fi = buff.flat_index(arrays)
    vals = np.broadcast_to(np.asarray(value.data if isinstance(value, Tile) else value,
                                      dtype=buff.arr.dtype), shape)
    if m is None:
        m = np.ones(shape, bool)
    write = m & (fi >= 0) & (fi < buff.size)
    if write.any():
        dst = buff.flat[:buff.size]
        dst[fi[write]] = vals[write]
    return None


# ---------------------------------------------------------------------------
# 逐元素与形状算子
# ---------------------------------------------------------------------------

def _b(a, b, fn):
    a = a if isinstance(a, Tile) else const(a)
    b = b if isinstance(b, Tile) else const(b)
    return Tile(fn(a.data, b.data), dtype=a.dtype)


def add(a, b):
    return _b(a, b, np.add)


def sub(a, b):
    return _b(a, b, np.subtract)


def mul(a, b):
    return _b(a, b, np.multiply)


def div(a, b):
    return _b(a, b, np.divide)


def maximum(a, b):
    return _b(a, b, np.maximum)


def exp(x):
    return Tile(np.exp(x.data.astype(np.float32)), dtype="f32")


def where(c, a, b):
    return _b(a, b, lambda x, y: np.where(np.asarray(c.data, bool), x, y))


def broadcast_to(x, shape):
    return Tile(np.broadcast_to(x.data, shape), dtype=x.dtype)


def expand_dims(x, axis):
    return Tile(np.expand_dims(x.data, axis), dtype=x.dtype)


def trans(x, perm=None):
    return Tile(np.transpose(x.data, perm), dtype=x.dtype)


def reshape(x, shape):
    return Tile(x.data.reshape(shape), dtype=x.dtype)


def reduce_op(x, axis, fn, keep_dims=False, neutral_init=None):
    """块内归约。axis=None 表示对**整个张量**归约（Triton 的默认口径）。"""
    d = x.data
    if axis is None:
        out = fn(d)
        if keep_dims:
            out = np.reshape(out, (1,) * d.ndim)
        return Tile(out, dtype=x.dtype)
    out = fn(d, axis=axis)
    if keep_dims:
        out = np.expand_dims(out, axis)
    return Tile(out, dtype=x.dtype)


def tsum(x, axis=None, keep_dims=False):
    return reduce_op(x, axis, np.sum, keep_dims)


def tmax(x, axis=None, keep_dims=False):
    return reduce_op(x, axis, np.max, keep_dims)


# ---------------------------------------------------------------------------
# program / grid 模型
# ---------------------------------------------------------------------------

class Grid:
    """块 id 网格。program_id(0) 是**最快变化**的那一维（Triton 口径）。"""

    def __init__(self, dims):
        self.dims = tuple(int(d) for d in dims)
        if any(d <= 0 for d in self.dims):
            raise ShapeError("grid 维度必须为正：%r" % (self.dims,))

    @property
    def size(self):
        return int(np.prod(self.dims))

    def ids(self):
        """按 axis 0 最快变化的顺序产出 pid 元组。"""
        out = []
        for flat in range(self.size):
            pid = []
            r = flat
            for d in self.dims:
                pid.append(r % d)
                r //= d
            out.append(tuple(pid))
        return out

    def __repr__(self):
        return "Grid(%r, size=%d)" % (self.dims, self.size)


class LaunchResult:
    def __init__(self, grid, trace, out):
        self.grid = grid
        self.trace = trace
        self.out = out

    def summary(self):
        return "grid=%r programs=%d 记账事件=%d" % (self.grid.dims, self.grid.size, len(self.trace))


def launch(kernel, grid, *args, **kwargs):
    """在 grid 上跑一个块级程序。

    kernel(pid, *args) —— pid 是元组，pid[i] 即 program_id(i)。
    """
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    trace = []
    rets = {}
    for pid in grid.ids():
        rets[pid] = kernel(pid, *args, trace=trace, **kwargs)
    return LaunchResult(grid, trace, rets)


# ---------------------------------------------------------------------------
# blocked layout：把 shape 精确切给 (warp, lane, register) 三层网格
# ---------------------------------------------------------------------------

class Layout:
    """blocked layout。

    - `spt`  sizePerThread：每个线程在每一维上连续持有的元素数
    - `tpw`  threadsPerWarp：一个 warp 在每一维上的线程数（乘积必须 == 32）
    - `wpc`  warpsPerCTA：CTA 在每一维上的 warp 数
    - `order` 维度主次序，`order[0]` 是**最快变化**的维（minor-to-major）

    本模型要求**精确切分**（无复制、无 padding）：
        spt[d] * tpw[d] * wpc[d] == shape[d]
    这条约束让「谁是主人」可以逐元素算出来，从而让「这次转换要不要走共享内存」
    变成可判定、可证伪的命题。
    """

    def __init__(self, shape, spt, tpw, wpc, order=None, dtype="f32", name=""):
        self.shape = tuple(int(s) for s in shape)
        self.rank = len(self.shape)
        self.spt = tuple(int(v) for v in spt)
        self.tpw = tuple(int(v) for v in tpw)
        self.wpc = tuple(int(v) for v in wpc)
        self.order = tuple(order) if order is not None else tuple(range(self.rank)[::-1])
        self.dtype = dtype
        self.name = name
        for attr in ("spt", "tpw", "wpc"):
            if len(getattr(self, attr)) != self.rank:
                raise LayoutError("%s 的 %s 长度必须等于 rank" % (self, attr))
        if int(np.prod(self.tpw)) != 32:
            raise LayoutError("threadsPerWarp 乘积必须是 32，实得 %d" % int(np.prod(self.tpw)))
        if sorted(self.order) != list(range(self.rank)):
            raise LayoutError("order 必须是 0..rank-1 的排列，实得 %r" % (self.order,))
        for d in range(self.rank):
            got = self.spt[d] * self.tpw[d] * self.wpc[d]
            if got != self.shape[d]:
                raise LayoutError(
                    "dim %d 切分不精确：spt*tpw*wpc = %d*%d*%d = %d != shape %d"
                    % (d, self.spt[d], self.tpw[d], self.wpc[d], got, self.shape[d]))

    @property
    def num_warps(self):
        return int(np.prod(self.wpc))

    @property
    def n_threads(self):
        return 32 * self.num_warps

    @property
    def numel(self):
        return int(np.prod(self.shape))

    @property
    def elemsize(self):
        return ESIZE[self.dtype]

    @property
    def minor_dim(self):
        return self.order[0]

    @property
    def vec_width(self):
        return self.spt[self.order[0]]

    def _coords(self):
        """每维的 (lane, warp) 坐标，形状广播到完整 shape。"""
        ts, ws = [], []
        for d in range(self.rank):
            e = np.arange(self.shape[d]) // self.spt[d]
            rsh = [-1 if k == d else 1 for k in range(self.rank)]
            ts.append((e % self.tpw[d]).reshape(rsh))
            ws.append((e // self.tpw[d]).reshape(rsh))
        return ts, ws

    def owners(self):
        """逐元素的属主线程号（0 .. 32*num_warps-1）。"""
        ts, ws = self._coords()
        lane = np.zeros(self.shape, np.int64)
        warp = np.zeros(self.shape, np.int64)
        for d in reversed(self.order):        # 从主序到次序：次序维最不重要
            lane = lane * self.tpw[d] + ts[d]
            warp = warp * self.wpc[d] + ws[d]
        return warp * 32 + lane

    def owner_of(self, idx):
        """单个下标的属主线程号。"""
        o = self.owners()
        return int(o[idx])

    def register_index(self, idx):
        """单个下标在该线程**寄存器数组**里的位置。"""
        r = 0
        for d in reversed(self.order):
            e = idx[d] % self.spt[d] if isinstance(idx, tuple) else idx % self.spt[d]
            r = r * self.spt[d] + e
        return r

    def contiguous_along(self, d):
        """内存维 d 上是否**由相邻线程读相邻地址**（合并访存）。"""
        return self.minor_dim == d

    def permute(self, perm):
        """布局的维度置换。**不搬数据**，只重新贴标签。

        真实系统里 tl.trans 也是这个性质：它是布局层面的重命名，
        代价由**消费者**在需要规范布局时付出。
        """
        if sorted(perm) != list(range(self.rank)):
            raise LayoutError("perm 必须是 0..rank-1 的排列")
        new_order = sorted(range(self.rank), key=lambda d: self.order.index(perm[d]))
        return Layout([self.shape[p] for p in perm],
                      [self.spt[p] for p in perm],
                      [self.tpw[p] for p in perm],
                      [self.wpc[p] for p in perm],
                      new_order, self.dtype, self.name + ".perm%r" % (perm,))

    def squeeze(self, axis):
        """去掉被**完全归约**掉的一维——**仅当该维没有跨线程摊开时合法**。

        若 `axis_threads(axis) > 1`，归约后的值只落在「沿 axis 坐标为零」的
        那些线程手里，结果**不再是本模型里的合法 blocked layout**——
        必须用 SliceLayout 表示。这里直接拒绝，避免静默产出一个漂亮但错的布局。
        """
        if self.axis_threads(axis) > 1:
            raise LayoutError(
                "维 %d 上摊开了 %d 个线程，归约结果不是紧凑的 blocked layout，"
                "必须用 SliceLayout（见 slice_along）" % (axis, self.axis_threads(axis)))
        keep = [d for d in range(self.rank) if d != axis]
        new_order = sorted(range(len(keep)), key=lambda i: self.order.index(keep[i]))
        return Layout([self.shape[d] for d in keep],
                      [self.spt[d] for d in keep],
                      [self.tpw[d] for d in keep],
                      [self.wpc[d] for d in keep],
                      new_order, self.dtype, self.name + ".sq%d" % axis)

    def slice_along(self, axis):
        """归约掉 axis 之后的**切片布局**。"""
        return SliceLayout(self, axis)

    def axis_threads(self, d):
        """沿维 d 摊开的线程数。"""
        return self.tpw[d] * self.wpc[d]

    def __repr__(self):
        return "Layout(shape=%r spt=%r tpw=%r wpc=%r order=%r)" % (
            self.shape, self.spt, self.tpw, self.wpc, self.order)


# ---------------------------------------------------------------------------
# 布局转换：免费与不免费，可判定
# ---------------------------------------------------------------------------

class Conversion:
    """从 src 布局到 dst 布局的一次转换。"""

    def __init__(self, shape, src, dst, dtype="f32", forced=False):
        self.shape = shape
        self.src = src
        self.dst = dst
        self.dtype = dtype
        self.forced = forced
        self.numel = int(np.prod(shape))
        self.elemsize = ESIZE[dtype]
        self.free = self._is_free()

    def _is_free(self):
        """**每个元素换布局后仍归同一个线程** → 只是寄存器内部重排，零代价。

        这一条是本模型里唯一「免费」的判据；它让代价模型不是退化的
        「转换一律昂贵」。
        """
        if self.src.shape != self.dst.shape:
            return False
        if self.src.owners().shape != self.dst.owners().shape:
            return False
        return bool(np.array_equal(self.src.owners(), self.dst.owners()))

    @property
    def smem_bytes(self):
        """不走共享内存的转换：0 字节。走了：写一遍 + 读一遍。"""
        return 0 if self.free else 2 * self.numel * self.elemsize

    @property
    def barriers(self):
        return 0 if self.free else 2

    @property
    def kind(self):
        return "register-permute" if self.free else "smem-round-trip"

    def __repr__(self):
        return "Conversion(%s, %sB, %d barrier, %s)" % (
            self.shape, self.smem_bytes, self.barriers, self.kind)


def insert_conversion(shape, src, dst, dtype="f32", allow=True):
    """要求把 src 布局的值转成 dst 布局。

    `allow=False` 表示调用方**拒绝**插入转换——转换需求的判定因此可证伪：
    判错时会抛 LayoutConflict，而不是静默产出一个错计划。

    返回 None **仅当两个布局的参数逐项相同**（那时连节点都不必建）。
    属主相同但参数不同的两个布局会返回一个 `free=True` 的转换——
    这一支必须可达，否则代价模型退化成「转换一律昂贵」。
    """
    if src.shape != dst.shape:
        raise LayoutConflict("布局形状不同：%r vs %r" % (src.shape, dst.shape))
    if (src.spt == dst.spt and src.tpw == dst.tpw
            and src.wpc == dst.wpc and src.order == dst.order):
        return None
    if not allow:
        raise LayoutConflict(
            "消费者要求 %r，生产者给出 %r，且调用方拒绝插入转换" % (dst, src))
    return Conversion(shape, src, dst, dtype)


def coalesced_layout(shape, num_warps, order=None, dtype="f32",
                     size_per_thread_floor=1):
    """构造一个**沿 order[0] 维合并访存**的精确切分布局。

    规则（本模型自定，不是 Triton 的）：
      先把 32 个 lane 尽量摊到 order[0] 维，摊完再摊到 order[1] 维……warp 同理。
      剩下来的每维元素数全归 sizePerThread。

    ★ 这个规则在两个具体形状上**复现了官方文档逐字给出的布局**（见 p3 的输出）：
        shape=[128,128], num_warps=4, order=[0,1] -> BlockedLayout([1,128],[32,1],[4,1],[0,1])
      与 Triton 官方 Gluon 「Tensor Layouts」教程里那一条一模一样。
      这不是抄的，是同一套「谁摊在哪一维」的规则算出来的。
    """
    rank = len(shape)
    if order is None:
        order = list(range(rank))[::-1]
    tpw = [1] * rank
    wpc = [1] * rank
    rem_lane, rem_warp = 32, num_warps
    for d in order:
        cover = int(shape[d])
        take = 1
        for cand in range(min(rem_lane, cover), 0, -1):
            if cover % cand == 0 and rem_lane % cand == 0:
                take = cand
                break
        tpw[d] = take
        rem_lane //= take
    for d in order:
        cover = int(shape[d]) // tpw[d]
        take = 1
        for cand in range(min(rem_warp, cover), 0, -1):
            if cover % cand == 0 and rem_warp % cand == 0:
                take = cand
                break
        wpc[d] = take
        rem_warp //= take
    spt = [int(shape[d]) // (tpw[d] * wpc[d]) for d in range(rank)]
    if rem_lane != 1 or rem_warp != 1:
        raise PlanInfeasible(
            "形状 %r / num_warps=%d 无法用 32 lane × %d warp 精确切分"
            % (tuple(shape), num_warps, num_warps))
    if any(s < size_per_thread_floor for s in spt):
        raise PlanInfeasible("sizePerThread 下溢：%r" % (spt,))
    return Layout(shape, spt, tpw, wpc, order, dtype,
                  name="coalesced(order=%r)" % (order,))


class SliceLayout:
    """归约结果的布局：值只落在 `parent` 中沿 `axis` 坐标为零的那些线程上。

    ★ 这是「归约后广播不免费」的根源。归约把值**收敛到少数线程**，
      而后续的逐元素运算需要每个线程都拿到它——中间那一步必须有人把值发出去。
      真实系统里这一步是 warp 内 shuffle 或共享内存往返；本模型把它算成
      `BroadcastConversion`。

    `squeeze` 在 axis 有跨线程摊开时会**拒绝**，就是不让这一层被静默抹掉。
    """

    def __init__(self, parent, axis):
        self.parent = parent
        self.axis = axis
        self.shape = tuple(s for d, s in enumerate(parent.shape) if d != axis)
        self.dtype = parent.dtype

    @property
    def threads_along(self):
        return self.parent.axis_threads(self.axis)

    def live_threads(self):
        """持有归约结果的那些线程号。"""
        o = self.parent.owners()
        sel = [slice(None)] * self.parent.rank
        sel[self.axis] = 0
        return np.unique(o[tuple(sel)])

    def dead_threads(self):
        """沿 axis 摊开、但**不**持有归约结果的线程号。"""
        return np.setdiff1d(np.arange(self.parent.n_threads), self.live_threads())

    def to_blocked(self, target=None, dtype=None):
        """广播回 blocked 布局的代价。"""
        return BroadcastConversion(self.parent, self.axis, dtype or self.dtype,
                                   target=target)

    def __repr__(self):
        return "SliceLayout(shape=%r, parent=%r, axis=%d, live=%d/%d)" % (
            self.shape, self.parent.shape, self.axis,
            len(self.live_threads()), self.parent.n_threads)


class BroadcastConversion:
    """把 slice 布局的值广播回 blocked 布局的代价。

    与普通 `Conversion` 的口径**不同**：搬的不是 numel 个元素，
    而是「值班线程写一次 + 全体各读一次」。
    """

    def __init__(self, parent, axis, dtype="f32", target=None):
        self.parent = parent
        self.axis = axis
        self.dtype = dtype
        self.esize = ESIZE[dtype]
        self.target = target or parent
        self.n_threads = parent.n_threads
        self.threads_along = parent.axis_threads(axis)
        self.positions_per_thread = int(
            parent.numel // (self.n_threads * parent.spt[axis]))
        if self.threads_along == 1:
            self.kind = "intra-thread(已复制)"
            self.ops = 0
            self.smem_bytes = 0
            self.barriers = 0
        elif parent.wpc[axis] == 1:
            self.kind = "warp-shuffle-broadcast"
            self.ops = self.positions_per_thread
            self.smem_bytes = 0
            self.barriers = 0
        else:
            self.kind = "smem-broadcast"
            self.ops = self.n_threads * self.positions_per_thread
            self.smem_bytes = self.positions_per_thread * self.esize * (1 + self.n_threads)
            self.barriers = 2

    @property
    def free(self):
        return self.threads_along == 1

    def __repr__(self):
        return "BroadcastConversion(%s, ops=%d, %dB, %d barrier)" % (
            self.kind, self.ops, self.smem_bytes, self.barriers)


# ---------------------------------------------------------------------------
# 归约的三相分解（intra-thread → intra-warp → cross-warp）
# ---------------------------------------------------------------------------

class ReducePlan:
    """一次块内归约的代价分解。

    ★ 这是**模型**，不是实测。三个相位的存在性来自算法结构：
      1. intra-thread   每个线程先把自己寄存器里的 spt[axis] 个元素加起来
      2. intra-warp     `__shfl_xor_sync` 蝶形，log2(tpw[axis]) 步
      3. cross-warp     各 warp 的局部和经共享内存汇总 —— **只要 wpc[axis] > 1 就必须有**
    """

    def __init__(self, layout, axis, dtype=None):
        self.layout = layout
        self.axis = axis
        self.dtype = dtype or layout.dtype
        self.esize = ESIZE[self.dtype]
        spt = layout.spt[axis]
        tpw = layout.tpw[axis]
        wpc = layout.wpc[axis]
        n_threads = layout.n_threads
        if layout.shape[axis] != spt * tpw * wpc:
            raise PlanInfeasible("维 %d 切分不精确" % axis)

        # 每个线程在「除 axis 外的每个位置」上持有一个局部和
        self.positions_per_thread = int(layout.numel // (n_threads * spt))

        # 相位 1：线程内
        self.phase1_ops = n_threads * (spt - 1) * self.positions_per_thread
        # 相位 2：warp 内蝶形
        self.warp_steps = int(np.log2(tpw)) if tpw > 1 else 0
        self.phase2_ops = self.warp_steps * n_threads * self.positions_per_thread
        # 相位 3：跨 warp 经共享内存
        self.cross_warp = wpc > 1
        if self.cross_warp:
            self.phase3_smem_bytes = 2 * wpc * self.positions_per_thread * self.esize
            self.phase3_barriers = 2
            self.phase3_ops = wpc * self.positions_per_thread
        else:
            self.phase3_smem_bytes = 0
            self.phase3_barriers = 0
            self.phase3_ops = 0

    @property
    def total_ops(self):
        return self.phase1_ops + self.phase2_ops + self.phase3_ops

    @property
    def smem_bytes(self):
        return self.phase3_smem_bytes

    @property
    def barriers(self):
        return self.phase3_barriers

    def phases(self):
        out = [("intra-thread", self.phase1_ops, 0, 0)]
        out.append(("intra-warp(butterfly x%d)" % self.warp_steps, self.phase2_ops, 0, 0))
        out.append(("cross-warp", self.phase3_ops, self.phase3_smem_bytes, self.phase3_barriers))
        return out

    def broadcast_back(self):
        """归约结果要广播回原形状时**额外**要付的代价。"""
        return BroadcastConversion(self.layout, self.axis, self.dtype)

    def __repr__(self):
        return ("ReducePlan(axis=%d, threads_along=%d, ops=%d, smem=%dB, barriers=%d)"
                % (self.axis, self.layout.axis_threads(self.axis),
                   self.total_ops, self.smem_bytes, self.barriers))


# ---------------------------------------------------------------------------
# 三相归约的**真跑**模拟：证明分解是算法而不是记账
# ---------------------------------------------------------------------------

class BarrierCounter:
    def __init__(self):
        self.n = 0

    def sync(self):
        self.n += 1


class ShuffleCounter:
    """真的按蝶形步数记账的 shuffle 计数器。"""

    def __init__(self):
        self.n = 0

    def xor_round(self, n_threads, positions):
        self.n += n_threads * positions


def simulate_three_phase_reduce(values, layout, axis, barrier=None, shuffle=None,
                                skip=()):
    """按三相**真的算一遍**，返回与布局无关的最终结果。

    `skip` 可以关掉若干相位（1/2/3），用来做**缺陷注入**：
    关掉跨 warp 相位后结果必须变错——那才证明这个相位不是记账，是算法。
    返回值用 reshape 回「归约后」的逻辑形状，可直接与 numpy 参照逐元素比对。
    """
    spt = layout.spt[axis]
    tpw = layout.tpw[axis]
    wpc = layout.wpc[axis]
    a = np.asarray(values if not isinstance(values, Tile) else values.data)
    if a.shape != layout.shape:
        raise ShapeError("数据形状 %r 与布局 %r 不符" % (a.shape, layout.shape))
    if barrier is None:
        barrier = BarrierCounter()
    if shuffle is None:
        shuffle = ShuffleCounter()
    positions = int(a.size // (layout.n_threads * spt))

    # 把 axis 维拆成 (warp, lane, reg)，次序维排在最后
    new_shape = a.shape[:axis] + (wpc, tpw, spt) + a.shape[axis + 1:]
    t = a.reshape(new_shape)

    # 相位 1：线程内
    if 1 in skip:
        t1 = np.take(t, [0], axis=axis + 2)                # 只取第一个寄存器，不合并
    else:
        t1 = t.sum(axis=axis + 2, keepdims=True)      # (..., wpc, tpw, 1, ...)

    # 相位 2：warp 内蝶形（真的按 xor 伙伴合并，不是直接 sum）
    t2 = np.broadcast_to(t1, new_shape[:axis] + (wpc, tpw, 1) + a.shape[axis + 1:]).copy()
    lane_axis = axis + 1
    steps = int(np.log2(tpw)) if tpw > 1 else 0
    if 2 not in skip:
        for k in range(steps):
            partner = np.arange(tpw) ^ (1 << k)
            t2 = t2 + np.take(t2, partner, axis=lane_axis)
            shuffle.xor_round(layout.n_threads, positions)
    else:
        shuffle.xor_round(0, 0)

    # 相位 3：跨 warp 经共享内存
    t3 = t2.take(0, axis=lane_axis)                    # 每个 warp 的局部和 (..., wpc, 1, ...)
    if wpc > 1:
        if 3 in skip:
            t3 = t3.take(0, axis=axis)                 # 只留 warp 0 的局部和（错的）
        else:
            smem = np.zeros(t3.shape, dtype=t3.dtype)  # 「共享内存」缓冲
            smem[...] = t3                             # 各 warp 写入
            barrier.sync()
            combined = smem.sum(axis=axis, keepdims=True)   # 某个 warp 汇总
            smem[...] = combined
            barrier.sync()
            t3 = smem.take(0, axis=axis)

    out = t3.reshape(a.shape[:axis] + a.shape[axis + 1:])
    return out, barrier.n


# ---------------------------------------------------------------------------
# 朴素参照（**故意与被测走不同代码路径**）
# ---------------------------------------------------------------------------

def naive_reduce(values, axis=None):
    """纯 Python 循环的归约。不调 numpy 的 sum/amax。

    这是 A 类断言的**第二条参照**：它和 numpy 的向量化实现、和
    simulate_three_phase_reduce 都不共用代码路径。
    """
    a = np.asarray(values if not isinstance(values, Tile) else values.data)
    if axis is None:
        it = a.reshape(-1).tolist()
        s = 0.0
        for v in it:
            s += float(v)
        return s
    a = np.moveaxis(a, axis, -1)
    outer = a.shape[:-1]
    n = a.shape[-1]
    res = np.empty(outer, dtype=np.float64)
    flat = a.reshape(-1, n)
    for i in range(flat.shape[0]):
        s = 0.0
        row = flat[i].tolist()
        for j in range(n):
            s += float(row[j])
        res.reshape(-1)[i] = s
    return res.reshape(outer)

# ---------------------------------------------------------------------------
# 迷你 IR + 前向布局推断 pass（「泄漏」检测器）
# ---------------------------------------------------------------------------

class Op:
    """一个块级算子。`inputs` 是图里前驱算子的下标。

    支持的 kind（本模型自定，只有这些）：
      load        产出「沿 coalesced_dim 合并访存」的规范布局
      trans       布局重标签，**不搬数据**
      reduce      沿 axis 归约，产出 SliceLayout
      broadcast   从 SliceLayout 广播回目标形状，**代价单独建模**
      elementwise 两个操作数必须同属主，不同则转换其中一个
      convert     显式的布局转换（用户写的）
      store       要求布局在 coalesced_dim 上连续；否则插入转换
    """

    __slots__ = ("kind", "shape", "inputs", "name", "kw", "layout")

    def __init__(self, kind, shape, inputs=(), name="", **kw):
        self.kind = kind
        self.shape = tuple(shape)
        self.inputs = tuple(inputs)
        self.name = name or ("%s%d" % (kind, id(self) % 1000))
        self.kw = kw
        self.layout = None

    def __repr__(self):
        return "Op(%s, shape=%r, in=%r)" % (self.kind, self.shape, self.inputs)


class PassResult:
    def __init__(self):
        self.events = []        # [(op_name, 对象)] —— 归约计划 / 转换 / 广播
        self.layouts = {}       # 算子下标 -> 布局
        self.conversions = []

    @property
    def smem_bytes(self):
        return sum(getattr(e, "smem_bytes", 0) for _, e in self.events)

    @property
    def barriers(self):
        return sum(getattr(e, "barriers", 0) for _, e in self.events)

    @property
    def ops(self):
        return sum(getattr(e, "ops", 0) + getattr(e, "total_ops", 0) for _, e in self.events)

    def report(self, indent="    "):
        out = []
        for name, e in self.events:
            out.append("%s%-34s %s" % (indent, name, e))
        return "\n".join(out)


def layout_pass(ops, num_warps, allow_convert=True, dtype="f32", verbose=False):
    """前向布局推断。每个算子的布局由生产者决定；消费者要求不同时插入转换。

    ★ 这是本模型里「layout conversion 泄漏」的检测器。它的判定**可证伪**：
      `allow_convert=False` 时，只要它认为需要转换就抛 LayoutConflict。
      故意把一个不需要转换的位置标成需要（或反之），断言必须失败。
    """
    res = PassResult()
    for i, o in enumerate(ops):
        if o.kind == "load":
            L = coalesced_layout(o.shape, num_warps, order=o.kw.get("order"), dtype=dtype)
            res.layouts[i] = L
        elif o.kind == "trans":
            res.layouts[i] = res.layouts[o.inputs[0]].permute(o.kw["perm"])
        elif o.kind == "reduce":
            src = res.layouts[o.inputs[0]]
            if isinstance(src, SliceLayout):
                raise LayoutError("对 slice 布局再归约：本模型不支持链式归约")
            res.layouts[i] = src.slice_along(o.kw["axis"])
            res.events.append(("reduce(axis=%d)" % o.kw["axis"], ReducePlan(src, o.kw["axis"])))
        elif o.kind == "broadcast":
            src = res.layouts[o.inputs[0]]
            if not isinstance(src, SliceLayout):
                raise LayoutError("broadcast 的源必须是 SliceLayout")
            dst = coalesced_layout(o.shape, num_warps, order=o.kw.get("order"), dtype=dtype)
            c = src.to_blocked(dst)
            res.events.append(("broadcast(axis=%d)" % src.axis, c))
            res.conversions.append(c)
            res.layouts[i] = dst
        elif o.kind == "elementwise":
            a = res.layouts[o.inputs[0]]
            b = res.layouts[o.inputs[1]]
            if isinstance(a, SliceLayout) or isinstance(b, SliceLayout):
                raise LayoutError("elementwise 的操作数不能是 slice 布局（先 broadcast）")
            if a.owners().shape != o.shape:
                raise LayoutError("操作数 0 的形状与结果不符")
            if not np.array_equal(a.owners(), b.owners()):
                if not allow_convert:
                    raise LayoutConflict(
                        "elementwise: 操作数 1 的布局与操作数 0 不同，且调用方拒绝插入转换")
                c = Conversion(o.shape, b, a, dtype)
                res.events.append(("convert(operand1->operand0)", c))
                res.conversions.append(c)
                b = a
            res.layouts[i] = a
        elif o.kind == "convert":
            src = res.layouts[o.inputs[0]]
            dst = o.kw["target"]
            c = insert_conversion(o.shape, src, dst, dtype, allow=allow_convert)
            if c is not None:
                res.events.append(("convert(%s)" % dst.name, c))
                res.conversions.append(c)
            res.layouts[i] = dst
        elif o.kind == "store":
            src = res.layouts[o.inputs[0]]
            if isinstance(src, SliceLayout):
                raise LayoutError("store 的源不能是 slice 布局")
            cd = o.kw["coalesced_dim"]
            if not src.contiguous_along(cd):
                # 目标布局必须**真的沿 coalesced_dim 合并**，所以 order 要重排，
                # 不能沿用源布局的 order（那样会算出一个与源完全相同的布局，
                # 转换变成恒等，非合并访存就被静默放过了）。
                tgt_order = [cd] + [d for d in range(src.rank) if d != cd][::-1]
                dst = coalesced_layout(src.shape, num_warps, order=tgt_order, dtype=dtype)
                c = insert_conversion(o.shape, src, dst, dtype, allow=allow_convert)
                if c is not None:
                    res.events.append(("convert(for-store)", c))
                    res.conversions.append(c)
                src = dst
            res.layouts[i] = src
        else:
            raise MiniTritonError("未知算子 %r" % o.kind)
    return res


def _find_owner(owners, idx):
    """给定属主映射，返回该下标所属的线程号。"""
    return int(owners[idx])


def graph_execute(ops, env):
    """在 numpy 上执行这张图（布局被完全忽略——转换是恒等映射）。

    用途：证明「同一份数学、不同转换代价」的两张图**输出逐位相同**。
    """
    vals = {}
    for i, o in enumerate(ops):
        if o.kind == "load":
            vals[i] = np.asarray(env[o.kw["buff"]], dtype=NP_OF[o.kw.get("dtype", "f32")])
        elif o.kind == "trans":
            vals[i] = np.transpose(vals[o.inputs[0]], o.kw["perm"])
        elif o.kind == "reduce":
            fn = np.sum if o.kw.get("fn", "sum") == "sum" else np.max
            vals[i] = fn(vals[o.inputs[0]], axis=o.kw["axis"], keepdims=o.kw.get("keep_dims", False))
        elif o.kind == "broadcast":
            vals[i] = np.broadcast_to(vals[o.inputs[0]], o.shape).copy()
        elif o.kind == "elementwise":
            vals[i] = o.kw["fn"](vals[o.inputs[0]], vals[o.inputs[1]])
        elif o.kind == "convert":
            vals[i] = vals[o.inputs[0]]
        elif o.kind == "store":
            env[o.kw["buff"]] = np.array(vals[o.inputs[0]])
            vals[i] = vals[o.inputs[0]]
        else:
            raise MiniTritonError("未知算子 %r" % o.kind)
    return vals


def free_conversion_demo(dtype="f32"):
    """一对**真的免费**的布局：属主逐位相同，但寄存器编号不同（寄存器重排）。

    免费判据的锚点（官方 Gluon「Tensor Layouts」原文）：
      「When converting between layouts you know are equivalent, or at most only
        require reordering registers within a thread (which is free), you can use
        gl.convert_layout(x, layout, assert_trivial=True) to ensure this.」
      （来源：https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html）

    这一对是**穷举搜出来**的（形状 (64,2)、1 个 warp、spt=[2,2]、tpw=[32,1]，
    两个 order）——不是编的，见 p3 里对它的逐元素复核。
    """
    src = Layout([64, 2], [2, 2], [32, 1], [1, 1], order=[0, 1], dtype=dtype, name="freeA")
    dst = Layout([64, 2], [2, 2], [32, 1], [1, 1], order=[1, 0], dtype=dtype, name="freeB")
    return src, dst

