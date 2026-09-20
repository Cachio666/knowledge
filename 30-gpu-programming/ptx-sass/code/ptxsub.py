# -*- coding: utf-8 -*-
"""ptxsub.py —— **PTX 子集解释器**（非编号引擎；三个编号程序 `import ptxsub`）。

★ 这是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★★ 它是模型，不是硬件（本域三条强制限定语的第 1 条）
===============================================================================

  1. **不是硬件**——本解释器**逐 warp 推进到阻塞点**（`_run_warp` 一个 warp 一路跑到
     `ret` / `bar.sync` 阻塞 / 结束，才轮到下一个 warp），**不是真实的 warp 交错调度**。
     真实 SM 上有多个 warp scheduler，指令级交错、双发射、延迟隐藏全都不同。
     ⇒ 凡本模块产出的数字（发射数、活跃 lane-指令数、发散步数），
       **都只是这个模型里的数**，不能读作硬件上的数。
  2. **语义按 PTX ISA 的章节转述实现**（引文与章节号见 `specs.ISA`，
     出处 `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html`）；
     **手写的 PTX 片段的语法从未被任何工具验证过**——本机**无 `ptxas`**（也
     无 `nvcc` / `cuobjdump` / `nvdisasm`）。本模块只解释**本文档手写的片段**。
  3. ★★ `shfl.sync.up` 的 `pval` 判据：**ISA 文本写的是 `maxLane`，本模块实现的是
     `minLane`——这是对 ISA 文本的**有意偏离**，且**这一点未定案**（三种可能见 `specs.py`
     的 `UP_MODE_UNDECIDED`，正文 `02-warp-level-execution.md` §4.1 逐字引原文）。
     `minimum` 与 `maximum` 在 ISA 自己的例子里取 `cval = 0`，此时两式**恰好重合**。

===============================================================================
★ 覆盖范围：一张**闭合**的表（`specs.PTX_SUBSET`）
===============================================================================

**表内的指令本模块实现；表外的指令一律 `PTXError`，绝不静默跳过**
（判据落点：`Sim.validate()`，以及 `01_virtual_isa.py` 的 `C.3`/`C.4`）。

`SUPPORTED_OPS` **不是手写的第二份清单**：它就是下面两张分派表的键的并集
（`_SCALAR` 与 `_WARP` 由装饰器 `@scalar(...)` / `@warp(...)` 登记）。
⇒ 「加了实现却没进表」与「进了表却没实现」在结构上都**不可能发生**，
   而 `01_virtual_isa.py` 的 `A.3` 会把 `SUPPORTED_OPS` 与 `specs.PTX_SUBSET['supported']`
   做**双向集合相等**断言 —— 改任何一边，`make run` 立刻失败。

===============================================================================
★ 与探针原型（`/tmp/ptx_probe/ptx_core.py`）的关系
===============================================================================

骨架照抄探针，三处**结构性**改动（都是为了让「表要闭合」可机械判定）：

  ① 标量指令从 `if/elif` 链改成**分派表**（`_SCALAR`），warp 级指令同理（`_WARP`）
     ⇒ `SUPPORTED_OPS` 可由表**导出**，而不是另抄一份；
  ② 新增 `Sim.validate()`：**表外指令在跑之前就被拒绝**（探针只在执行到时才报）；
  ③ 新增 `Program.declared` / `Program.used_regs()`：把「.reg 声明了多少」
     与「实际碰到了多少」都做成**可算的量**（`01` 篇的反直觉量化结论靠它）。

★ 探针原样保留的部分：`bar.sync` 的**到达计数按 warp 粒度 += 32**
  （ISA 原文见 `specs.ISA['B1']`/`B2`/`B3`，章节 `9.7.15.1`），
  以及 `max_bank_conflict` 的「**同一 word 是广播、不算冲突**」口径。

★★ 一处**不能**照抄的地方（`specs.py` 的 `MODEL_LIMITS` 逐条列了）：
  探针自报的 6 条可信度缺口**没有搬进 `specs.py`**——它们多数是散文，
  `specs.py` 只放**可被程序计算**的东西（计划 §0.6 第 3 条）。散文版本在正文里。
"""

import re
import struct

# ===========================================================================
# 0. 位运算与类型工具
# ===========================================================================

M32 = 0xFFFFFFFF


def u32(x):
    return x & M32


def s32(x):
    x &= M32
    return x - 0x100000000 if x > 0x7FFFFFFF else x


def f32b(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]


def b32f(b):
    return struct.unpack('<f', struct.pack('<I', b & M32))[0]


class PTXError(Exception):
    """解释器拒绝执行时抛出。**表外的指令、未声明的寄存器、死锁都走这里。**

    ★ 它不是「PTX 编译错误」的同义词：真实 `ptxas` 在**汇编期**就报未声明的寄存器，
      本解释器只能在**运行到那条指令时**才发现（`01` 篇 §4.1 与 `C.1`/`C.5`）。
    """


# ===========================================================================
# 1. 指令与程序
# ===========================================================================

class Instr:
    __slots__ = ('pred', 'neg', 'op', 'ops', 'pc', 'src')

    def __init__(self, pred, neg, op, ops, pc, src):
        self.pred = pred
        self.neg = neg
        self.op = op
        self.ops = ops
        self.pc = pc
        self.src = src

    def __repr__(self):
        p = '' if self.pred is None else '@%s%%p%d ' % ('!' if self.neg else '', self.pred)
        return '%3d: %s%s %s' % (self.pc, p, self.op, ', '.join(self.ops))


class Program:
    """一段 PTX 文本解析后的形态。

    ★ `declared` 是 `.reg` 行给出的**声明量**，与指令里**实际触及**的寄存器数是两件事
      （`01` 篇的反直觉量化结论）。两者都由本类算出，不由调用者手填。
    """

    def __init__(self):
        self.instrs = []
        self.labels = {}
        self.declared = {}          # kind -> 声明个数（'.reg .b32 %r<8>' => {'b32': 8}）
        self.src_lines = []

    @property
    def nreg(self):
        return max([v for k, v in self.declared.items() if k != 'pred'], default=0)

    @property
    def npred(self):
        return self.declared.get('pred', 0)

    def disasm(self):
        return '\n'.join(repr(i) for i in self.instrs)

    def used_regs(self):
        """**静态**扫出被指令引用过的 `%rN` 下标集合（含读与写，含谓词指令的守卫）。

        ★ 这是词法级的量：它数的是「文本里出现过」，不是「执行时被用到」。
          两者在**不可达分支**上不同——那正是 `01` 篇 `C.5` 的题材。
        """
        used = set()
        for it in self.instrs:
            toks = list(it.ops)
            if it.pred is not None:
                toks.append('%p%d' % it.pred)
            for t in toks:
                for m in re.finditer(r'%r(\d+)', t):
                    used.add(int(m.group(1)))
        return used

    def used_preds(self):
        used = set()
        for it in self.instrs:
            if it.pred is not None:
                used.add(it.pred)
            for t in it.ops:
                for m in re.finditer(r'%p(\d+)', t):
                    used.add(int(m.group(1)))
        return used

    def opcode_histogram(self):
        h = {}
        for it in self.instrs:
            h[it.op] = h.get(it.op, 0) + 1
        return h


# ===========================================================================
# 2. 解析
# ===========================================================================

_COMMENT = re.compile(r'//.*$')
_DECL = re.compile(r'\.reg\s+\.(\w+)\s+%(\w+?)<(\d+)>')


def _split_ops(s):
    """按逗号切分操作数，忽略 `[...]` 内部的逗号（地址表达式里有 `+`，没有逗号，但留余地）。"""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse(text):
    """把 PTX 片段解析成 `Program`。

    ★ 只认本子集需要的东西：`.reg` 声明、`标签:`、`@%p` 守卫、`op 操作数, …`。
      `.version` / `.target` / `.shared` 这类**声明行**以 `.` 开头，一律**跳过**
      （它们在真实的 PTX 里由汇编器消费，本解释器不模拟存储分配）。
    """
    p = Program()
    p.src_lines = text.splitlines()
    for raw in p.src_lines:
        line = _COMMENT.sub('', raw).strip()
        if not line:
            continue
        line = line.rstrip(';').strip()
        if not line:
            continue
        if line.startswith('.'):
            m = _DECL.match(line)
            if m:
                kind, n = m.group(1), int(m.group(3))
                p.declared[kind] = max(p.declared.get(kind, 0), n)
            continue
        if line.endswith(':'):
            p.labels[line[:-1].strip()] = len(p.instrs)
            continue
        pred, neg = None, False
        if line.startswith('@'):
            tok, _, rest = line.partition(' ')
            body = tok[1:]
            if body.startswith('!'):
                neg, body = True, body[1:]
            pred = int(body[2:])
            line = rest.strip()
        head, _, remainder = line.partition(' ')
        ops = _split_ops(remainder) if remainder.strip() else []
        p.instrs.append(Instr(pred, neg, head, ops, len(p.instrs), raw))
    return p


# ===========================================================================
# 3. 分派表（= 覆盖范围本身）
# ===========================================================================
#
# ★★ `_SCALAR` / `_WARP` 的键**就是** `SUPPORTED_OPS`。
#    加一个实现 = 表里多一条 = `specs.PTX_SUBSET['supported']` 必须同步（否则 A.3 失败）。

_SCALAR = {}
_WARP = {}


def scalar(*opcodes):
    """登记一条**逐 lane 执行**的指令（谓词 `@%p` 只过滤活跃 lane，不改变控制流）。"""
    def deco(fn):
        for op in opcodes:
            assert op not in _SCALAR, ('重复登记', op)
            _SCALAR[op] = fn
        return fn
    return deco


def warp(*opcodes):
    """登记一条**warp 级**指令（自己管 PC / 阻塞 / lane 间数据搬运）。"""
    def deco(fn):
        for op in opcodes:
            assert op not in _WARP, ('重复登记', op)
            _WARP[op] = fn
        return fn
    return deco


# ★★ `SUPPORTED_OPS` 的定义**在 `class Sim` 之后**（文件末尾附近）——
#    分派表由类体里的 `@scalar` / `@warp` 装饰器登记，而装饰器要等类体执行到才跑。
#    在类之前求值只会得到一个**空集**（第一版就是这么错的，`validate()` 于是拒绝一切）。


# ===========================================================================
# 4. SIMT 解释器
# ===========================================================================

class Sim:
    """一个 CTA 的 SIMT 解释器：warp 内每个 lane 各自持有 PC，**PC 不同即发散**。"""

    def __init__(self, prog, nthreads=64, smem_words=256, fill=-1, max_steps=None):
        if nthreads % 32:
            raise PTXError('nthreads 必须是 32 的倍数')
        self.max_steps = max_steps
        self.prog = prog
        self.nthreads = nthreads
        self.nwarp = nthreads // 32
        nr, npd = max(prog.nreg, 1), max(prog.npred, 1)
        self.regs = [[[0] * nr for _ in range(32)] for _ in range(self.nwarp)]
        self.preds = [[[False] * npd for _ in range(32)] for _ in range(self.nwarp)]
        self.pc = [[0] * 32 for _ in range(self.nwarp)]
        self.alive = [[True] * 32 for _ in range(self.nwarp)]
        self.smem = [u32(fill) for _ in range(smem_words)]
        # ---- 代价计数器（模型值，**不是硬件上的数**）----
        self.issue = [0] * self.nwarp            # 发射次数（一个 warp 发一条指令算 1）
        self.lane_issue = [0] * self.nwarp       # 活跃 lane-指令数（该次发射命中的 lane 数）
        self.divergent = [0] * self.nwarp        # 处于发散态（PC 不唯一）的调度步数
        # ---- barrier ----
        self.bar_count = {}
        self.bar_wait = {}
        self.bar_arrivals = []                   # (warp, barrier id, 活跃 lane 数)
        self.blocked = set()
        # ---- 可观测轨迹 ----
        self.ld_shared_trace = []                # (warp, lane, byte_addr)
        self.shfl_trace = []                     # (warp, mode, src, cval, membermask)
        self.trace_hook = None                   # fn(w, pc, op, n_active) —— 逐发射回调
        self.retired = [False] * self.nwarp

    # ------------------------------------------------------------------
    # 4.1 闭合性检查
    # ------------------------------------------------------------------
    def validate(self):
        """★ **表要闭合**：表外指令一律报错，**绝不静默跳过**。

        这是 `specs.PTX_SUBSET` 那张表唯一的执行落点 —— 表要是不能拒绝东西，
        它就只是一段说明文字（`CONTRIBUTING` 第 220–222 行对 `specs.py` 的判据：
        「把这条常量改错，程序会报错吗？」）。
        """
        bad = sorted({i.op for i in self.prog.instrs if i.op not in SUPPORTED_OPS})
        if bad:
            raise PTXError('指令不在 specs.PTX_SUBSET 覆盖范围内（表外一律不支持）：%s'
                           % ', '.join(bad))
        return self

    # ------------------------------------------------------------------
    # 4.2 操作数求值
    # ------------------------------------------------------------------
    def _val(self, w, l, tok):
        tok = tok.strip()
        if tok.startswith('%'):
            if tok == '%tid.x':
                return l + 32 * w
            if tok == '%laneid':
                return l
            if tok == '%warpid':
                return w
            if tok == '%ntid.x':
                return self.nthreads
            if tok == '%nwarpid':
                return self.nwarp
            if re.fullmatch(r'%r\d+', tok):
                return self.regs[w][l][self._ridx(tok)]
            raise PTXError('未知寄存器 %s' % tok)
        return int(tok, 0)

    def _addr(self, w, l, tok):
        t = tok.strip()
        if not (t.startswith('[') and t.endswith(']')):
            raise PTXError('非法地址操作数 %s' % tok)
        parts = re.split(r'\s*\+\s*', t[1:-1].strip())
        a = self._val(w, l, parts[0])
        for x in parts[1:]:
            a += self._val(w, l, x)
        return a

    def _ridx(self, tok):
        """寄存器下标 → 槽位。**未在 `.reg` 声明里分配的寄存器在这里报错。**

        ★ 真实 `ptxas` 在**汇编期**就拒绝这种引用；本解释器只能在**执行到那条指令时**
          才发现 —— 于是「不可达分支里的非法引用」会一路潜伏（`01` 篇 `C.5`）。
        """
        i = int(tok.strip()[2:])
        if i >= len(self.regs[0][0]):
            raise PTXError('寄存器 %s 未在 .reg 声明中分配（%d 个已分配）'
                           % (tok.strip(), len(self.regs[0][0])))
        return i

    def _prename(self, tok):
        """谓词下标 → 槽位。与 `_ridx` 同口径：未声明就报错。"""
        i = int(tok.strip()[2:])
        if i >= len(self.preds[0][0]):
            raise PTXError('谓词 %s 未在 .reg .pred 声明中分配（%d 个已分配）'
                           % (tok.strip(), len(self.preds[0][0])))
        return i

    # ------------------------------------------------------------------
    # 4.3 标量指令（逐 lane）
    # ------------------------------------------------------------------
    @scalar('mov.u32', 'mov.b32', 'mov.s32', 'mov.f32')
    def _h_mov(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]))

    @scalar('add.u32', 'add.b32')
    def _h_add_u(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) + g(ins.ops[2]))

    @scalar('add.s32')
    def _h_add_s(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(s32(g(ins.ops[1])) + s32(g(ins.ops[2])))

    @scalar('sub.u32')
    def _h_sub_u(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) - g(ins.ops[2]))

    @scalar('mul.lo.u32')
    def _h_mul_lo(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) * g(ins.ops[2]))

    @scalar('mad.lo.s32')
    def _h_mad_lo(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(s32(g(ins.ops[1])) * s32(g(ins.ops[2]))
                                        + s32(g(ins.ops[3])))

    @scalar('and.b32', 'and.u32')
    def _h_and(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) & g(ins.ops[2]))

    @scalar('or.b32', 'or.u32')
    def _h_or(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) | g(ins.ops[2]))

    @scalar('xor.b32', 'xor.u32')
    def _h_xor(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) ^ g(ins.ops[2]))

    @scalar('shl.b32')
    def _h_shl(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) << (g(ins.ops[2]) & 31))

    @scalar('shr.u32', 'shr.b32')
    def _h_shr(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = u32(g(ins.ops[1]) >> (g(ins.ops[2]) & 31))

    @scalar('add.f32')
    def _h_add_f(self, w, l, ins, R, g):
        R[self._ridx(ins.ops[0])] = f32b(b32f(g(ins.ops[1])) + b32f(g(ins.ops[2])))

    @scalar('setp.lt.s32', 'setp.le.s32', 'setp.gt.s32', 'setp.ge.s32',
            'setp.eq.s32', 'setp.ne.s32',
            'setp.lt.u32', 'setp.le.u32', 'setp.gt.u32', 'setp.ge.u32',
            'setp.eq.u32', 'setp.ne.u32')
    def _h_setp(self, w, l, ins, R, g):
        kind = ins.op.split('.')[1]
        a, b = s32(g(ins.ops[1])), s32(g(ins.ops[2]))
        ua, ub = u32(g(ins.ops[1])), u32(g(ins.ops[2]))
        r = {'lt': a < b, 'le': a <= b, 'gt': a > b, 'ge': a >= b,
             'eq': ua == ub, 'ne': ua != ub}[kind]
        self.preds[w][l][self._prename(ins.ops[0])] = r

    @scalar('ld.shared.u32', 'ld.shared.b32', 'ld.shared.s32', 'ld.shared.f32')
    def _h_ld_shared(self, w, l, ins, R, g):
        ad = self._addr(w, l, ins.ops[1])
        self.ld_shared_trace.append((w, l, ad))
        R[self._ridx(ins.ops[0])] = self.smem[(ad >> 2) % len(self.smem)]

    @scalar('st.shared.u32', 'st.shared.b32', 'st.shared.f32')
    def _h_st_shared(self, w, l, ins, R, g):
        ad = self._addr(w, l, ins.ops[0])
        self.smem[(ad >> 2) % len(self.smem)] = u32(g(ins.ops[1]))

    # ------------------------------------------------------------------
    # 4.4 warp 级指令
    # ------------------------------------------------------------------
    @warp('ret')
    def _h_ret(self, w, grp, ins, act):
        for l in grp:
            self.alive[w][l] = False
        self.retired[w] = True
        return 'ok'

    @warp('bra')
    def _h_bra(self, w, grp, ins, act):
        """`bra` + `@%p bra`。★ **重聚是 PC 相等自然发生的**：两条路径汇到同一个标签，
        不需要显式的 re-convergence 栈（`02` 篇的反直觉量化结论）。"""
        tgt = self.prog.labels[ins.ops[0]]
        for l in grp:
            self.pc[w][l] = tgt if l in act else ins.pc + 1
        return 'ok'

    @warp('bar.sync')
    def _h_bar_sync(self, w, grp, ins, act):
        """★ **到达计数按 warp 粒度递增**（ISA `9.7.15.1` 逐字：见 `specs.ISA['B1'/'B2']`）。

        不是按「活跃 lane 数」——即使这个 warp 只剩 1 个活跃 lane 执行到 `bar.sync`，
        整个 warp 也算到达（原文：`marks warps' arrival at barrier`）。
        """
        bid = self._val(w, 0, ins.ops[0])
        count = self._val(w, 0, ins.ops[1]) if len(ins.ops) > 1 else self.nthreads
        for l in grp:
            self.pc[w][l] = ins.pc + 1
        self.bar_arrivals.append((w, bid, len(act)))
        self.bar_count[bid] = self.bar_count.get(bid, 0) + 32
        self.bar_wait.setdefault(bid, set()).add(w)
        if self.bar_count[bid] >= count:
            for ww in self.bar_wait[bid]:
                self.blocked.discard(ww)
            self.bar_count[bid] = 0
            self.bar_wait[bid] = set()
            return 'ok'
        self.blocked.add(w)
        return 'blocked'

    @warp('shfl.sync.bfly.b32', 'shfl.sync.idx.b32',
          'shfl.sync.up.b32', 'shfl.sync.down.b32')
    def _h_shfl_sync(self, w, grp, ins, act):
        """`shfl.sync.<mode>.b32 d, a, b, c, membermask`（ISA `9.7.10.6`）。

        伪代码（原文逐字见 `specs.ISA['P1']`）：

            maxLane = (lane[4:0] & segmask[4:0]) | (cval[4:0] & ~segmask[4:0]);
            minLane = (lane[4:0] & segmask[4:0]);
            case .up:   j = lane - bval; pval = (j >= maxLane); break;   ← 原文
            case .down: j = lane + bval; pval = (j <= maxLane); break;
            case .bfly: j = lane ^ bval; pval = (j <= maxLane); break;
            case .idx:  j = minLane | (bval[4:0] & ~segmask[4:0]); pval = (j <= maxLane);
            if (!pval) j = lane;

        ★★ **本实现与原文有一处不同，且已明写在案**：`.up` 的 `pval` 这里用
          **`minLane`**（即 `j >= minLane`），而**原文写的是 `maxLane`**。
          理由：`cval = 0`（ISA 自己的 `shfl.sync.up.b32 Ry|p, Rx, 0x1, 0x0, 0xffffffff`
          就是这个 `cval`）时 `maxLane == minLane`，两式**恰好重合**；
          而按原文在 `cval = 31` 时 `.up` 会退化成空操作（`02` 篇 §4.1 逐字推了这一步）。
          **这一点未定案**（三种可能见 `specs.UP_MODE_UNDECIDED`）——
          本库**不声称规范规定 `minLane`**，只声称「本实现有意偏离 ISA 文本」。
        """
        mode = ins.op.split('.')[2]
        d, a, b, c, mm = ins.ops
        src = self._val(w, 0, b)
        cval = self._val(w, 0, c)
        mmask = self._val(w, 0, mm)
        clamp = cval & 0x1F
        segmask = (cval >> 8) & 0x1F
        self.shfl_trace.append((w, mode, src, cval, mmask))
        for l in act:
            maxLane = (l & segmask) | (clamp & ~segmask & 0x1F)
            minLane = (l & segmask)
            if mode == 'bfly':
                j = l ^ (src & 0x1F)
                pval = (j <= maxLane)
            elif mode == 'idx':
                j = minLane | (src & 0x1F & ~segmask & 0x1F)
                pval = (j <= maxLane)
            elif mode == 'up':
                j = l - (src & 0x1F)
                pval = (j >= minLane)          # ★ 有意偏离 ISA 文本（原文是 maxLane）
            elif mode == 'down':
                j = l + (src & 0x1F)
                pval = (j <= maxLane)
            else:
                raise PTXError('未实现的 shfl 模式 %s' % mode)
            if not pval:
                j = l                          # copy from own lane
            if not ((mmask >> j) & 1):
                raise PTXError('shfl 源 lane %d 不在 membermask 内（PTX：行为未定义）' % j)
            if not self.alive[w][j]:
                raise PTXError('shfl 源 lane %d 已退出（PTX：行为未定义）' % j)
            self.regs[w][l][self._ridx(d)] = self.regs[w][j][self._ridx(a)]
        for l in grp:
            self.pc[w][l] = ins.pc + 1
        return 'ok'

    # ------------------------------------------------------------------
    # 4.5 一步
    # ------------------------------------------------------------------
    def _step(self, w, grp, ins):
        if ins.pred is None:
            act = list(grp)
        else:
            # ★ 谓词下标也在**执行到这一条时**才查（与 `_ridx` 同一时机）：
            #   未声明的谓词 ⇒ `PTXError`，而不是 Python 的 `IndexError`。
            #   于是「不可达分支里的非法谓词引用」同样会潜伏（见 `01` 的 `C.2`）。
            self._prename("%%p%d" % ins.pred)
            act = [l for l in grp if self.preds[w][l][ins.pred] != ins.neg]
        if self.trace_hook is not None:
            self.trace_hook(w, ins.pc, ins.op, len(act))
        if ins.op in _WARP:
            return _WARP[ins.op](self, w, grp, ins, act)
        fn = _SCALAR[ins.op]                 # validate() 已保证不会 KeyError
        for l in act:
            fn(self, w, l, ins, self.regs[w][l], lambda t, w=w, l=l: self._val(w, l, t))
        for l in grp:
            self.pc[w][l] = ins.pc + 1
        return 'ok'

    # ------------------------------------------------------------------
    # 4.6 调度：逐 warp 推进到阻塞点
    # ------------------------------------------------------------------
    def _run_warp(self, w):
        """★ **不是硬件**：一个 warp 一路推进到 `ret` / `bar.sync` 阻塞 / 程序结束。

        真实 SM 上多个 warp 是指令级交错的；本模型里「warp0 先跑完」是**必然**的，
        这正是「缺 `bar.sync` 就会读到哨兵」能被构造出来的原因，
        也**正是它不能读作硬件行为的原因**（`03` 篇 §4.1 的限定语）。
        """
        while True:
            lanes = [l for l in range(32) if self.alive[w][l]]
            if not lanes:
                return 'done'
            pcs = [self.pc[w][l] for l in lanes]
            tgt = min(pcs)
            if tgt >= len(self.prog.instrs):
                for l in lanes:
                    self.alive[w][l] = False
                return 'done'
            grp = [l for l in lanes if self.pc[w][l] == tgt]
            if len(set(pcs)) > 1:
                self.divergent[w] += 1
            self.issue[w] += 1
            self.lane_issue[w] += len(grp)
            if self.max_steps is not None and self.issue[w] > self.max_steps:
                raise PTXError('warp%d 超过 max_steps=%d，疑似死循环' % (w, self.max_steps))
            if self._step(w, grp, self.prog.instrs[tgt]) == 'blocked':
                return 'blocked'

    def run(self, validate=True):
        if validate:
            self.validate()
        pending = set(range(self.nwarp))
        while pending:
            progressed = False
            for w in sorted(pending):
                if w in self.blocked:
                    continue
                progressed = True
                if self._run_warp(w) == 'done':
                    pending.discard(w)
            if not pending:
                break
            if not progressed:
                raise PTXError('死锁：剩余 warp 全部阻塞在 barrier 上')
        return self

    # ------------------------------------------------------------------
    # 4.7 观察接口
    # ------------------------------------------------------------------
    def smem_i32(self, word):
        return s32(self.smem[word])

    def warp_vals_i32(self, w, reg):
        return [s32(self.regs[w][l][reg]) for l in range(32)]

    def totals(self):
        """模型值三元组。★ 不是硬件上的发射数/活跃 lane 数（见模块头限定语 1）。"""
        return {
            'issue': sum(self.issue),
            'lane_issue': sum(self.lane_issue),
            'divergent_steps': sum(self.divergent),
        }


# ===========================================================================
# 5. ★★ 覆盖范围（= 上面两张分派表的键的并集）
# ===========================================================================

SUPPORTED_OPS = frozenset(_SCALAR) | frozenset(_WARP)
"""本解释器**实现**的 opcode 全集。

★ 它**由分派表导出**，不是手抄的第二份清单 ⇒「加了实现却没进表」不可能发生；
  反方向（「进了 `specs.PTX_SUBSET` 却没实现」）由 `01_virtual_isa.py` 的 `A.3`
  做**双向集合相等**断言钉住。
"""

SCALAR_OPS = frozenset(_SCALAR)
WARP_OPS = frozenset(_WARP)


# ===========================================================================
# 6. bank 冲突计数
# ===========================================================================

def max_bank_conflict(byte_addrs, nbank=32, word_bytes=4):
    """返回最大 bank 冲突路数。

    ★ 口径（`03` 篇 §1、探针 E8 第 5 条）：`bank = (字节地址 ÷ 4) mod 32`；
      **同一 word 的多线程访问是广播，只算 1 路** ⇒ 统计单位是
      「落在同一个 bank 上的**不同 word** 数」，因此这里用 `set` 而不是 `list`。
      把 `set` 换成 `list`，广播立刻会被算成冲突（`meta_falsify.py` 的 `D4` 就是这个缺陷）。
    """
    banks = {}
    for a in byte_addrs:
        word = a // word_bytes
        banks.setdefault(word % nbank, set()).add(word)
    return max((len(v) for v in banks.values()), default=0)
