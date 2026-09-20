#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态契约检查器：检查那些**编译器不会替你检查**的 CUDA 契约。

配套文档：`01-thread-hierarchy-and-sync.md`
运行：`make run`（或 `make 01_sync_contract`）

===============================================================================
★ 本程序是什么
===============================================================================

一个**词法 / 结构层面**的 CUDA 源码检查器。它不编译、不执行、不做完整数据流分析。
每条判据都能对着源码逐字复核，判据的真值来源分三档（`specs.RULE_STRENGTH`）：

  C1  `__syncthreads()` 落在依赖 `threadIdx` 的条件分支里        [A 官方原文]
  C2  静态共享内存的维度 / 总量与声明的 block 大小不自洽          [A 官方原文]
  C3  分块循环里的协作式装填漏掉分块号（每趟装同一块数据）        [B 成对文件]
  C4  归约的幺元与算子不匹配                                     [B 成对文件]
  C5  launch 覆盖范围远超 kernel 的边界假设（靠守卫兜住）         [B 真实文件]
  C6  —— **设计了但没实现**，本程序**不声称覆盖**（`specs.RULE_STRENGTH["C6"]`）

★★ 限定语（**不是硬件**）

  > **本程序没有执行任何一个 thread。** 本机是 Apple M1 Pro，无 GPU、无 `nvcc`。
  > 它读的是**文本**，产出的是**文本的结论**——一个 kernel 在硬件上会不会真的
  > 挂起 / 算错 / 慢下来，本程序**一概没有验证**，也**不声称**。
  > 凡本程序说「会」的地方，请读作「按 `<出处>` 的规定会」。

★★ 限定语（**不是完整数据流分析**）

  > C3 的依赖判定是**词法级 + 一层中间量传递**（计划 §0.7 G8-5）。
  > 具体说：它把循环体内「由归纳变量算出的局部量」也算作依赖（`tainted` 集合，
  > 传递 3 轮），但**不建 SSA、不做支配分析、不跟踪数组下标里的别名**。
  > ⇒ 它能**漏报**：一个经由 `float* p = A + i*ds;` 这类指针运算传递的依赖它看不出。

★ 三类断言（本程序内部，失败即非零退出；`CONTRIBUTING` 第 121 行第二种情形）

  A 算法正确性 —— 参照实现走**不同的代码路径**（§A）
  B 代价模型   —— 超发倍率与线程槽浪费（§B）
  C 边界与反例 —— §4 声称"会出问题"的输入必须**真的跑出问题**（§C）

★★ 本程序的可证伪断言（逐条给出"改哪一处、断言会怎么失败"）

  A.2  把 `_ev` 的 C 截断除法改回 `int(a/b)`（真除法）⇒
       `((1<<60)+1)/1` 会算成 `2**60` 而不是 `2**60+1`，A.2 立刻失败。
  A.4  把 C3 的 `TAINT_ROUNDS` 从 3 改成 0 ⇒ hw2 的两处仍报，但
       gemm.cu 的误报回归（A.4 的第二半）会失败。
  A.7  把任一 `ANCHORS[i]["quote"]` 里的一个字改掉 ⇒ A.7 的逐字子串检查失败。
  B.1  把 `HW13_THREADS` 从 512 改成 256 ⇒ 977 变成 1954，B.1 的期望值失败。
  C.3  把 hw2 起始代码第 40 行的 `A[idy*ds+idx]` 改成 `A[idy*ds+idx*i]`（人为造出 i 依赖）⇒
       C.3 要求「命中 2 处」的断言失败（只剩 1 处）。
  C.7  把 hw13 的 `blocks` 行**在内存里**替换成正确写法 ⇒ C.7 断言"WARN 消失"失败。
  C.9  把 `if (idx < N)` 守卫**在内存里**删掉 ⇒ C.9 断言"WARN 消失"失败。
  C.11 把嵌入夹具的 `tile[97][128]` 改成 `tile[96][128]`（正好 48 KB）⇒
       C.11 要求「必须报 48KB 违规」的断言失败。
  A.11f 把任一锚点的 `covered_by` 里的一个标签改掉 —— **哪怕改成另一个真实存在的标签**
       （`C.3` → `C.4`）⇒ 该锚点的 `note` 里出现的标签集合与 `covered_by` 不再相等，
       A.11g 立刻失败。★ 这一条是「`note` 是**索引**、不是**唯一载体**」的机械载体。
  A.11j 把 `TRAIN_COMMIT_DATE` 改成另一个**合法**日期（如 `2021-10-14`）⇒ 与
       `git show -s --format=%cs <TRAIN_COMMIT>` 不再一致，A.11j 立刻失败。
       ★ 订正轮补：此前这条只被 A.11c 的**格式**检查覆盖（只查 `YYYY-MM-DD`），
       改成另一个合法日期不会被任何断言抓到（实施者自报 + 控制者反向测试确认）。
  C.21  改 `LEARN_SHA16` 里任一哈希、或换一份 `cuda-learning` ⇒ C.21 失败。
       ★ 订正轮补：`make mutate` 的漏网清单显示这个常量**6 个叶子全部漏网** ——
       它是个**死常量**（三个程序一次都没引用它），而主流程还在打印「版本只能用文件哈希」。
       那是一句**没有实现的声明**（P6）。⇒ 补上断言，并把硬编码路径改走 `specs.LEARN_*_PATH`。

★★ 本程序**从不修改磁盘上的任何文件**。所有反例都在**内存里的字符串**上构造
   （见 §C.7/C.9 的 `replace_once`）。⇒ 不存在"超时被 kill 时污染被测对象"的风险
   （`45-kernel-lifecycle` 的变异测试踩过那个坑，本程序用架构规避，而不是靠 `try/finally`）。
   ★ 目录里的 `mutate.py`（`make mutate`）是**另一个**独立的诊断工具：它在**副本**上变异，
     并有 `try/finally` + 跑完校验哈希两道保证 —— 与本程序的反例构造方式**互不影响**。

★★ 数字口径：本程序的全部数字都是**推导值**（本仓库程序算出，可复核），
   **不是本机实测值**。外部工件的数字（N=500000、threads=512、行号、逐字行内容）
   来自 `specs.py`，是**规格值 / 引用值**。本程序**不产出任何性能数字**。
"""

import ast
import hashlib
import os
import re
import subprocess
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 外部工件的路径探测（CONTRIBUTING「本域的代码形态」：不得写死构造者目录布局）
# ===========================================================================


def find_artifact_root(env_name, rel_marker, candidates):
    """在候选列表里找外部工件根目录。

    ★ 规则（CONTRIBUTING 第 167–189 行）：
      · **不得写死构造者本机的目录布局**；
      · 找到后**打印它选中的路径**；
      · 找不到时**以可操作的错误信息退出**，不静默回落。
    """
    seen = []
    envv = os.environ.get(env_name)
    if envv:
        candidates = [envv] + list(candidates)
    for c in candidates:
        if not c:
            continue
        c = os.path.expanduser(c)
        seen.append(c)
        if os.path.exists(os.path.join(c, rel_marker)):
            return c, seen
    return None, seen


def artifact_roots():
    here = os.path.dirname(os.path.abspath(__file__))
    train, train_tried = find_artifact_root(
        specs.TRAIN_ROOT_ENV, "exercises/hw13",
        ["../cuda-training-series", "../../cuda-training-series",
         "../../../cuda-training-series",
         os.path.join(here, "..", "..", "..", "..", "cuda-training-series")],
    )
    learn, learn_tried = find_artifact_root(
        specs.LEARN_ROOT_ENV, "02_gemm/gemm.cu",
        ["../cuda-learning", "../../cuda-learning", "../../../cuda-learning",
         os.path.join(here, "..", "..", "..", "..", "cuda-learning")],
    )
    return (train, train_tried), (learn, learn_tried)


def die_missing_artifacts(train_info, learn_info):
    (train, tried), (learn, tried2) = train_info, learn_info
    print("=" * 78)
    print("!! 找不到外部工件，无法检查 —— 本程序的真值不在本目录里")
    print("=" * 78)
    if train is None:
        print("%s 未找到。探测过的路径：" % specs.TRAIN_ROOT_ENV)
        for t in tried:
            print("    %s" % t)
        print("  期望在它下面看到 exercises/hw13/ 。")
    if learn is None:
        print("%s 未找到（用于 C2/C3 的**误报回归**）。探测过的路径：" % specs.LEARN_ROOT_ENV)
        for t in tried2:
            print("    %s" % t)
        print("  期望在它下面看到 02_gemm/gemm.cu 。")
    print()
    print("怎么修：把外部工件放到本目录的兄弟目录，或显式指定：")
    print("    make run %s=/path/to/cuda-training-series \\" % specs.TRAIN_ROOT_ENV)
    print("             %s=/path/to/cuda-learning" % specs.LEARN_ROOT_ENV)
    print("注：`~` 由 shell 展开没问题，但**不要写进 make 变量的引号里**——")
    print("    `make ... X=~/y` 的值是字面量 `~/y`，请改用 `$HOME/y`。")
    print()
    print("本程序的断言**没有**跑，退出码非零。")
    return 2


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def replace_every(text, old, new):
    """在**内存里**把 old 的**每一处**都换成 new。一处都找不到就抛异常。

    ★ 与 `replace_once` 成对存在：C.8 只换第一处（`blocks` 只声明一次），
      C.9 必须换每一处（4 个 kernel 各有一个 `if (idx < N)` 守卫）。
      **这两个语义差别本身就是一个易错点** —— 第一版 C.9 用了 `replace_once`，
      结果 3 个 WARN 还在，失败的是**我的期望值**，不是检查器。
    """
    n = text.count(old)
    if n == 0:
        raise AssertionError("反例构造失败：在源码里找不到 %r" % old[:60])
    return text.replace(old, new), n


def replace_once(text, old, new):
    """在**内存里**把 old 换成 new（只换第一处）。找不到 old 就抛异常。

    ★ 本程序唯一的"造反例"手段。它**不碰磁盘**——见模块 docstring 末段。
    """
    if old not in text:
        raise AssertionError("反例构造失败：在源码里找不到 %r" % old[:60])
    return text.replace(old, new, 1)


# ===========================================================================
# 1. 预处理：去注释 / 字符串，但**保持字节偏移不变**（便于报行号）
# ===========================================================================

def clean(src):
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            while i < n and src[i] != '\n':
                out[i] = ' '
                i += 1
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            out[i] = out[i + 1] = ' '
            i += 2
            while i < n and not (src[i] == '*' and i + 1 < n and src[i + 1] == '/'):
                if src[i] != '\n':
                    out[i] = ' '
                i += 1
            if i < n:
                out[i] = ' '
                if i + 1 < n:
                    out[i + 1] = ' '
                i += 2
        elif c in '"\'':
            q = c
            out[i] = ' '
            i += 1
            while i < n and src[i] != q:
                if src[i] == '\\':
                    out[i] = ' '
                    i += 1
                    if i < n:
                        out[i] = ' '
                elif src[i] != '\n':
                    out[i] = ' '
                i += 1
            if i < n:
                out[i] = ' '
                i += 1
        else:
            i += 1
    return ''.join(out)


def line_of(src, off):
    """偏移量 -> 行号（1 起）。★ A.1 用一条**独立的路径**复核它。"""
    return src.count('\n', 0, off) + 1


def line_of_text(src, off):
    """★ A.1 的**参照实现**：把文本切成行、再逐行累加字节长度来定位。

    与 `line_of`（`str.count('\\n', 0, off) + 1`）是**两条完全不同的算法**：
      · `line_of`      —— 数 off 之前有多少个换行符；
      · `line_of_text` —— 累加每行长度，找到第一个"行尾越过 off"的行。
    两者不一致 ⇒ A.1 失败。
    """
    pos = 0
    for idx, ln in enumerate(src.split('\n')):
        end = pos + len(ln)
        if off <= end:
            return idx + 1
        pos = end + 1        # 跳过那个 '\n'
    return src.count('\n') + 1


# ===========================================================================
# 2. 符号表：#define / const 标量 / dim3
# ===========================================================================
C_TYPE_SIZE = {'float': 4, 'double': 8, '__half': 2, 'half': 2, 'int': 4,
               'unsigned int': 4, 'unsigned': 4, 'size_t': 8, 'short': 2, 'char': 1}

DECL_RE = re.compile(r'\b(?:const\s+)?(size_t|int|long|unsigned(?:\s+int)?|float|double)\s+'
                     r'([A-Za-z_]\w*)\s*=\s*([^;]+);')
DIM3_HEAD_RE = re.compile(r'\bdim3\s+([A-Za-z_]\w*)\s*\(')


class NotInt(Exception):
    """表达式的值不是「C 的整型常量」——_ev 借此返回 None。"""


def _c_int_binop(op, a, b):
    """★ C 语义的整型二元运算。**截断除法**（toward zero）不是 floor 除法。

    ★★ 这是本程序**刻意与"随手用 Python `/`"不同**的地方（A.2 钉住它）：
       Python 的 `int(a / b)` 在 |a/b| > 2**53 时会**丢精度**，
       而 C 的 `a / b` 对整型是精确的截断除法。
       `(2**60 + 1) / 1` 就是那个可判定的分界点（见 A.2）。
    """
    if isinstance(op, ast.Add):
        return a + b
    if isinstance(op, ast.Sub):
        return a - b
    if isinstance(op, ast.Mult):
        return a * b
    if isinstance(op, (ast.Div, ast.FloorDiv)):
        if b == 0:
            raise NotInt("division by zero")
        q = abs(a) // abs(b)
        return -q if (a < 0) != (b < 0) else q
    if isinstance(op, ast.Mod):
        if b == 0:
            raise NotInt("modulo by zero")
        r = abs(a) % abs(b)
        return -r if a < 0 else r
    if isinstance(op, ast.Pow):
        return a ** b
    if isinstance(op, ast.LShift):
        return a << b
    if isinstance(op, ast.RShift):
        return a >> b
    if isinstance(op, ast.BitAnd):
        return a & b
    if isinstance(op, ast.BitOr):
        return a | b
    if isinstance(op, ast.BitXor):
        return a ^ b
    raise NotInt("unsupported binop %s" % type(op).__name__)


def _c_int_eval(node):
    if isinstance(node, ast.Expression):
        return _c_int_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise NotInt("non-int literal %r" % (node.value,))
        return node.value
    if isinstance(node, ast.UnaryOp):
        v = _c_int_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise NotInt("unsupported unaryop")
    if isinstance(node, ast.BinOp):
        return _c_int_binop(node.op, _c_int_eval(node.left), _c_int_eval(node.right))
    raise NotInt("unsupported node %s" % type(node).__name__)


def _ev(expr, sym, dim3=None):
    """求值一个只含整型常量与已知符号的表达式，用 **C 的整型语义**。求不出来返回 None。"""
    e = expr
    # 去掉整型/浮点字面量后缀: 8ULL / 1024ULL / 5.0f / 640u / 3L
    # 注意不能写 `\d[\d\w\.]*[uUlLfF]+\b` —— 那个字符类会把后缀字母也吃掉。
    e = re.sub(r'(\d)(?:ULL|ull|LL|ll|UL|ul|LU|lu|[uUlLfF])\b', r'\1', e)
    if dim3:
        def rep(m):
            d = dim3.get(m.group(1))
            if not d:
                return m.group(0)
            i = 'xyz'.index(m.group(2))
            v = d[i] if i < len(d) else None
            return str(v) if isinstance(v, int) else m.group(0)
        e = re.sub(r'\b([A-Za-z_]\w*)\.([xyz])\b', rep, e)
    names = set(re.findall(r'[A-Za-z_]\w*', e))
    for nm in names:
        if nm in sym:
            e = re.sub(r'\b%s\b' % re.escape(nm), '(%r)' % sym[nm], e)
        else:
            return None
    try:
        return _c_int_eval(ast.parse(e, mode='eval'))
    except (NotInt, SyntaxError, ValueError):
        pass
    # ★ 浮点回退：C 里 `0.0f` / `1.5` 这类字面量**不是整型常量**，
    #   但判据要用它们（例如归约的初值 `sdata[tid] = 0.0f;`）。
    #   ⇒ 走 Python 的浮点求值，只用来读**值**，不参与 launch 配置的整型算术
    #     （那一条已经在上面的严格路径里成功返回了，不会被这里覆盖）。
    try:
        v = eval(e, {"__builtins__": {}}, {})
    except Exception:
        return None
    return v if isinstance(v, (int, float)) else None


def find_dim3_decls(src):
    """返回 [(名字, 参数文本)]。手写括号配平扫描，因为 dim3 的参数里含括号。"""
    out = []
    for m in DIM3_HEAD_RE.finditer(src):
        j = m.end() - 1
        d, k = 0, j
        while k < len(src):
            if src[k] == '(':
                d += 1
            elif src[k] == ')':
                d -= 1
                if d == 0:
                    break
            k += 1
        inner = src[j + 1:k]
        parts, d2, last = [], 0, 0
        for idx, ch in enumerate(inner):
            if ch == '(':
                d2 += 1
            elif ch == ')':
                d2 -= 1
            elif ch == ',' and d2 == 0:
                parts.append(inner[last:idx])
                last = idx + 1
        parts.append(inner[last:])
        out.append((m.group(1), parts))
    return out


def _dim3_sofar(src, sym):
    out = {}
    for nm, parts in find_dim3_decls(src):
        out[nm] = [_ev(a, sym) if _ev(a, sym) is not None else '?' for a in parts]
    return out


def build_symtab(src):
    sym, decl_text = {}, {}
    for m in re.finditer(r'^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*$', src, re.M):
        v = _ev(m.group(2), sym)
        if v is not None:
            sym[m.group(1)] = v
    for _ in range(3):     # 多趟，处理 A = B*C 这类链式定义
        for m in DECL_RE.finditer(src):
            nm, ex = m.group(2), m.group(3)
            if nm in sym:
                continue
            v = _ev(ex, sym, _dim3_sofar(src, sym))
            if v is not None:
                sym[nm] = v
                decl_text[nm] = ' '.join(ex.split())
    dim3 = {}
    for _ in range(2):
        for nm, parts in find_dim3_decls(src):
            dim3[nm] = [_ev(a, sym, dim3) if _ev(a, sym, dim3) is not None else '?'
                        for a in parts]
    return sym, dim3, decl_text


# ===========================================================================
# 3. 结构扫描
# ===========================================================================
def scan_constructs(body):
    """返回 [(open_off, close_off, kind, cond_text)]，用来判断某个偏移量在哪些分支内。"""
    toks = []
    for m in re.finditer(r'\b(if|for|while|else|do|switch)\b|[{}]', body):
        toks.append((m.start(), m.group(0)))
    out, stack, pending, depth = [], [], None, 0
    for off, tok in toks:
        if tok in ('if', 'for', 'while', 'switch'):
            j = body.find('(', off)
            k, d = j, 0
            while k < len(body):
                if body[k] == '(':
                    d += 1
                elif body[k] == ')':
                    d -= 1
                    if d == 0:
                        break
                k += 1
            pending = (tok, body[j + 1:k].strip())
        elif tok == 'else':
            pending = ('else', None)
        elif tok == 'do':
            pending = ('do', None)
        elif tok == '{':
            depth += 1
            if pending:
                stack.append([pending[0], pending[1], depth, off])
                pending = None
        elif tok == '}':
            while stack and stack[-1][2] == depth:
                c = stack.pop()
                out.append((c[3], off, c[0], c[1]))
            depth -= 1
    return out


class SharedArray:
    def __init__(self, name, dims, esize, off):
        self.name, self.dims, self.esize, self.off = name, dims, esize, off
        self.uses = []

    @property
    def count(self):
        n = 1
        for d in self.dims:
            n *= d if isinstance(d, int) else 1
        return n

    @property
    def bytes(self):
        return self.count * self.esize

    @property
    def dims_known(self):
        return all(isinstance(d, int) for d in self.dims)


SHARED_RE = re.compile(r'__shared__\s+([A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)?)\s+'
                       r'([A-Za-z_]\w*)\s*((?:\[[^\]]*\])+)\s*;')


class Kernel:
    def __init__(self, name, body, body_off):
        self.name, self.body, self.body_off = name, body, body_off
        self.shared, self.syncs, self.stores, self.loops = [], [], [], []


def find_kernels(csrc):
    kers = []
    for m in re.finditer(r'__global__\b', csrc):
        mm = re.search(r'([A-Za-z_]\w*)\s*\(', csrc[m.end():])
        if not mm:
            continue
        brace = csrc.find('{', m.end() + mm.end())
        semi = csrc.find(';', m.end() + mm.end())
        if brace < 0 or (0 <= semi < brace):
            continue          # 只有声明
        d, k = 0, brace
        while k < len(csrc):
            if csrc[k] == '{':
                d += 1
            elif csrc[k] == '}':
                d -= 1
                if d == 0:
                    break
            k += 1
        kers.append(Kernel(mm.group(1), csrc[brace + 1:k], brace + 1))
    return kers


def loop_var_of(cond_text):
    first = cond_text.split(';')[0]
    m = re.search(r'\b(?:int|unsigned(?:\s+int)?|size_t|long|float|double)\s+'
                  r'([A-Za-z_]\w*)', first)
    if m:
        return m.group(1)
    m = re.match(r'\s*([A-Za-z_]\w*)\s*=', first)
    return m.group(1) if m else None


def analyse_kernel(ker, sym):
    body, base = ker.body, ker.body_off
    cons = scan_constructs(body)

    def enclosing(off):
        return [c for c in cons if c[0] <= off <= c[1]
                and c[2] in ('if', 'for', 'while') and c[3] is not None]

    for m in SHARED_RE.finditer(body):
        ty, nm = m.group(1).strip(), m.group(2)
        dims = []
        for dm in re.finditer(r'\[([^\]]*)\]', m.group(3)):
            e = dm.group(1).strip()
            if e == '':
                dims.append('?')          # extern __shared__ float buf[];
            else:
                v = _ev(e, sym)
                dims.append(v if v is not None else '?')
        ker.shared.append(SharedArray(nm, dims, C_TYPE_SIZE.get(ty, 4), base + m.start()))

    for m in re.finditer(r'__syncthreads\s*\(\s*\)', body):
        off = base + m.start()
        ker.syncs.append((off, [(c[2], c[3], c[0] < off) for c in enclosing(m.start())]))

    for m in re.finditer(r'\bfor\s*\(([^;]*);([^;]*);([^)]*)\)', body):
        var = re.search(r'\b(?:int|unsigned(?:\s+int)?|size_t|long)\s+([A-Za-z_]\w*)',
                        m.group(1))
        if var:
            ker.loops.append((base + m.start(), var.group(1), m.group(2).strip()))

    for s in ker.shared:
        uses = []
        for um in re.finditer(r'\b%s\s*((?:\[[^\]]*\])+)' % re.escape(s.name), body):
            uses.append(re.findall(r'\[([^\]]*)\]', um.group(1)))
        s.uses = uses

    shared_names = {s.name for s in ker.shared}
    store_re = re.compile(r'([A-Za-z_]\w*)\s*((?:\[[^\]]*\])+)\s*(?:=|\+=|-=|\*=|/=)\s*([^;]+);')
    for m in store_re.finditer(body):
        arr = m.group(1)
        if arr not in shared_names:
            continue
        enc = enclosing(m.start())
        loopvars = []
        for c in enc:
            if c[2] != 'for':
                continue
            v = loop_var_of(c[3])
            if v:
                loopvars.append((v, c[0], c[1]))
        ker.stores.append({'off': base + m.start(), 'arr': arr,
                           'idx': re.findall(r'\[([^\]]*)\]', m.group(2)),
                           'rhs': m.group(3).strip(), 'loopvars': loopvars, 'enc': enc})
    return ker


# ===========================================================================
# 4. 宿主侧：launch 配置
# ===========================================================================
LAUNCH_RE = re.compile(r'([A-Za-z_]\w*)\s*<<<\s*([^,>]+)\s*,\s*([^,>]+)\s*'
                       r'(?:,\s*[^,>]*)?(?:,\s*[^>]*)?>>>')


def resolve_dims(expr, sym, dim3):
    expr = expr.strip()
    if expr in dim3:
        d = list(dim3[expr])
    else:
        v = _ev(expr, sym)
        d = [v, 1, 1] if v is not None else ['?', '?', '?']
    while len(d) < 3:
        d.append(1)
    return d


def find_launches(csrc, sym, dim3):
    out = []
    for m in LAUNCH_RE.finditer(csrc):
        out.append({
            'kernel': m.group(1),
            'grid': resolve_dims(m.group(2), sym, dim3),
            'block': resolve_dims(m.group(3), sym, dim3),
            'grid_raw': m.group(2).strip(),
            'block_raw': m.group(3).strip(),
            'off': m.start(),
        })
    return out


# ===========================================================================
# 5. 检查
# ===========================================================================
class Finding:
    def __init__(self, cid, severity, line, msg, evidence):
        self.cid, self.severity, self.line = cid, severity, line
        self.msg, self.evidence = msg, evidence

    def __repr__(self):
        return "Finding(%s/%s:%d)" % (self.cid, self.severity, self.line)


GUARD_RE = re.compile(r'if\s*\(\s*(?:idx|row|threadIdx\.x\s*\+\s*blockDim\.x'
                      r'\s*\*\s*blockIdx\.x)\s*<\s*([A-Za-z_]\w*)\s*\)')


def check_source(display_name, raw):
    """★ 唯一的入口，收的是**文本**（不读盘）——反例才能在内存里造。
    返回 (findings, kernels, launches, sym, decl_text)。"""
    csrc = clean(raw)
    sym, dim3, decl_text = build_symtab(csrc)
    kers = [analyse_kernel(k, sym) for k in find_kernels(csrc)]
    launches = find_launches(csrc, sym, dim3)
    findings = []

    for ker in kers:
        # ---- C1: __syncthreads() 落在发散分支里 --------------------------
        for off, encs in ker.syncs:
            for kind, cond, _ in encs:
                if cond is None:
                    continue
                if re.search(r'\bthreadIdx\b', cond):
                    findings.append(Finding(
                        "C1", specs.RULE_EXPECTED_SEVERITY["C1"], line_of(csrc, off),
                        "__syncthreads() 在依赖 threadIdx 的条件分支内 -> 分支发散",
                        "条件 = `%s`。\n"
                        "            契约（锚点 C1，%s §%s）: \"%s\"\n"
                        "            机制（锚点 C1b，%s §%s）: \"%s\""
                        % (cond, specs.ANCHORS['C1']['doc'], specs.ANCHORS['C1']['section'],
                           specs.ANCHORS['C1']['quote'],
                           specs.ANCHORS['C1b']['doc'], specs.ANCHORS['C1b']['section'],
                           specs.ANCHORS['C1b']['quote'])))

        for L in launches:
            if L['kernel'] != ker.name:
                continue
            bx, by, bz = L['block']
            # ---- C2a: 共享内存维度 vs 声明的 block 大小 --------------------
            for s in ker.shared:
                hits = []
                # 只有当下标里**真的出现** threadIdx 时，该维才有"必须 >= blockDim"的契约。
                # （flash_attention.cu 的 Ks[Bc][D] 用 r/j 走协作循环，不直接吃 threadIdx，
                #   所以那条维度契约不适用 —— 这正是 v1 的误报。）
                ti_last = any(len(u) == len(s.dims) and 'threadIdx' in u[-1] for u in s.uses)
                ti_first = any(len(u) >= 2 and 'threadIdx' in u[0] for u in s.uses)
                ti_only = any(len(u) == 1 and 'threadIdx' in u[0] for u in s.uses)
                if ti_only and len(s.dims) == 1 and isinstance(bx, int) and isinstance(s.dims[0], int):
                    if s.dims[0] < bx:
                        hits.append("%s[%s] 被 threadIdx.x 索引，但维度 < blockDim.x=%s"
                                    % (s.name, s.dims[0], bx))
                if len(s.dims) == 2:
                    d0, d1 = s.dims
                    if ti_first and isinstance(by, int) and isinstance(d0, int) and d0 < by:
                        hits.append("%s[%s][%s] 第一维被 threadIdx.y 索引，但 < blockDim.y=%s"
                                    % (s.name, d0, d1, by))
                    if ti_last and isinstance(bx, int) and isinstance(d1, int) and d1 < bx:
                        hits.append("%s[%s][%s] 第二维被 threadIdx.x 索引，但 < blockDim.x=%s"
                                    % (s.name, d0, d1, bx))
                if hits:
                    findings.append(Finding(
                        "C2", specs.RULE_EXPECTED_SEVERITY["C2"], line_of(csrc, s.off),
                        "共享内存维度与声明的 block 大小不自洽: " + "; ".join(hits),
                        "launch %s<<<grid=%r, block=%r>>>"
                        % (L['kernel'], L['grid'], L['block'])))

            # ---- C2b: 静态共享内存总量 > 48 KB -----------------------------
            tot = sum(s.bytes for s in ker.shared if s.dims_known)
            if tot > 48 * 1024:
                findings.append(Finding(
                    "C2", specs.RULE_EXPECTED_SEVERITY["C2"], line_of(csrc, ker.shared[0].off),
                    "静态共享内存合计 %d 字节 > 48KB -> 必须改用动态共享内存并 "
                    "cudaFuncSetAttribute() 显式 opt-in" % tot,
                    "出处 specs.ANCHORS['C2'] (%s §%s): \"%s\""
                    % (specs.ANCHORS['C2']['doc'], specs.ANCHORS['C2']['section'],
                       specs.ANCHORS['C2']['quote'])))

            # ---- C5: launch 覆盖范围 vs kernel 内的边界假设 ----------------
            g, b = L['grid'], L['block']
            if '?' in g or '?' in b:
                continue
            cover = (g[0] * b[0] * (g[1] if isinstance(g[1], int) else 1)
                     * (b[1] if isinstance(b[1], int) else 1))
            guard = GUARD_RE.search(ker.body)
            if not guard:
                continue
            nm = guard.group(1)
            N = sym.get(nm)
            if N is None:
                continue
            if cover < N:
                findings.append(Finding(
                    "C5", "ERROR", line_of(csrc, L['off']),
                    "launch 只覆盖 %d 个线程，而 kernel 内部按 %s=%d 做边界假设 -> 会少算"
                    % (cover, nm, N),
                    "launch<<<grid=%r, block=%r>>>, kernel 守卫 `if (idx < %s)`" % (g, b, nm)))
            elif cover > 4 * N:
                need = c_ceildiv(N, max(1, b[0]))
                findings.append(Finding(
                    "C5", specs.RULE_EXPECTED_SEVERITY["C5"], line_of(csrc, L['off']),
                    "launch 覆盖 %d 个线程，是 %s=%d 的 %.1f 倍（按 blockDim 只需 %d 个 block，"
                    "实际发了 %s 个）。结果仍然正确，但只因为 kernel 里有 `if (idx < %s)` 守卫。"
                    % (cover, nm, N, cover / N, need, g[0], nm),
                    "文件里写的是 `%s = %s`（= %s），而 blockDim=%s。"
                    "注意 (blockDim-1)/blockDim 在**C 的整型除法**下恒为 0，所以该表达式退化成 N 本身。"
                    "正确写法是 (N + blockDim - 1) / blockDim = %d。"
                    % (L.get('grid_raw', '?'), decl_text.get(L.get('grid_raw', ''), '?'),
                       g[0], b[0], need)))

        # ---- C3: 分块循环里的共享内存写入，RHS 不依赖归纳变量 --------------
        for st in ker.stores:
            if not any('threadIdx' in i for i in st['idx']):
                continue
            for lv, lo, hi in st['loopvars']:
                # 依赖感知：循环体内"由归纳变量算出来的中间量"也算依赖。
                # 例：gemm.cu 的 `int kA = t * TILE + threadIdx.x;` 之后用 `A[row*K+kA]`，
                # 文本上不含 t，语义上依赖 t。v1 只做文本匹配，对 gemm.cu 误报过。
                lbody = ker.body[lo:hi] if (lo is not None and hi is not None) else ker.body
                tainted = {lv}
                for _ in range(TAINT_ROUNDS):
                    for am in re.finditer(r'\b(?:int|unsigned(?:\s+int)?|size_t|long)\s+'
                                          r'([A-Za-z_]\w*)\s*=\s*([^;]+);', lbody):
                        if any(re.search(r'\b%s\b' % re.escape(t), am.group(2)) for t in tainted):
                            tainted.add(am.group(1))
                if not any(re.search(r'\b%s\b' % re.escape(t), st['rhs']) for t in tainted):
                    findings.append(Finding(
                        "C3", specs.RULE_EXPECTED_SEVERITY["C3"], line_of(csrc, st['off']),
                        "协作式装填 `%s[...threadIdx...] = %s;` 落在外层 for 循环 "
                        "(归纳变量 %s) 里，但右值既不直接含 %s、也不含任何由 %s 算出的中间量 "
                        "-> 每一趟装填的是同一块数据" % (st['arr'], st['rhs'], lv, lv, lv),
                        "契约: 分块(tiling)循环里装入共享内存的全局下标必须随分块号变化。"
                        "真值来源: hw2 的 matrix_mul_shared.cu(已知有问题) vs "
                        "matrix_mul_shared_solution.cu(已知正确)。"
                        "(注: 这是词法级依赖判定，不是完整数据流分析)"))

        # ---- C4: 归约的幺元与算子不匹配 ------------------------------------
        for s in ker.shared:
            init = re.search(r'\b%s\s*\[[^\]]*\]\s*=\s*([^;]+);' % re.escape(s.name), ker.body)
            if not init:
                continue
            initv = init.group(1).strip()
            body_after = ker.body[init.end():]
            op = None
            if re.search(r'\b(f?max|fmin|min)\s*\(', body_after):
                op = 'max/min'
            elif re.search(r'\b%s\s*\[[^\]]*\]\s*\+=' % re.escape(s.name), body_after):
                op = 'sum'
            v = _ev(initv, sym)
            if op == 'max/min' and v is not None and v >= 0:
                findings.append(Finding(
                    "C4", specs.RULE_EXPECTED_SEVERITY["C4"],
                    line_of(csrc, ker.body_off + init.start()),
                    "归约算子是 max/min，但累加器初值是 `%s` (= %s)。对全负数输入会错。"
                    % (initv, v),
                    "契约: 幺元必须与算子匹配（max 的幺元是 -inf）。"
                    "真值来源: hw5 的 max_reduction.cu(已知是 sum 版) vs "
                    "max_reduction_solution.cu(参考 max 版；它同样保留了 0.0f 初值)"))
            elif op == 'sum' and v is not None and v != 0:
                findings.append(Finding(
                    "C4", specs.RULE_EXPECTED_SEVERITY["C4"],
                    line_of(csrc, ker.body_off + init.start()),
                    "归约算子是 sum，但初值是 %s 而不是 0" % initv,
                    "契约: sum 的幺元是 0"))

    return findings, kers, launches, sym, decl_text


BANNED_IMPORT_ROOTS = ("torch", "triton", "numba", "cuda")
DYNAMIC_IMPORT_FUNCS = ("import_module", "__import__", "find_spec")


def ast_import_scan(directory):
    """★ 机械检查：本目录**全部** `.py`（**含 `specs.py` 与本文件自己**）里
    不得出现 `import torch` / `triton` / `numba` / `cuda`，
    也不得出现 `import_module("torch")` 这类**动态导入**。

    ★ 为什么用 `ast` 而不是 `grep`：注释与 docstring 里出现 "torch" 这个词不算违规
      —— 本目录的 docstring 里就写着「本程序不 import torch」，`grep` 必然假阳性。
      AST 里那只是一段字符串常量。
    ★ 判据按 CONTRIBUTING：动态导入（`import_module(...)`）**也算违规**。
    ★ 计划 §2 的 P6 高危点：声称「都不 import torch」却只查编号程序、漏了 `specs.py`
      ⇒ 本函数用 `glob("*.py")`，**不做任何筛选**。
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(directory, "*.py")))
    static, dynamic = [], []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            tree = ast.parse(f.read())
        base = os.path.basename(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in BANNED_IMPORT_ROOTS:
                        static.append((base, node.lineno, a.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BANNED_IMPORT_ROOTS:
                    static.append((base, node.lineno, node.module))
            elif isinstance(node, ast.Call):
                fn = node.func
                nm = (fn.attr if isinstance(fn, ast.Attribute)
                      else fn.id if isinstance(fn, ast.Name) else "")
                if (nm in DYNAMIC_IMPORT_FUNCS and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and node.args[0].value.split(".")[0] in BANNED_IMPORT_ROOTS):
                    dynamic.append((base, node.lineno, nm, node.args[0].value))
    return files, static, dynamic


def c_ceildiv(a, b):
    """★ C 语义的上取整除法（**声明用**，不是「正确写法」的展示）。"""
    return (a + b - 1) // b if a >= 0 else -((-a) // b)


# C3 的中间量传递轮数。★ 改成 0 会让 gemm.cu 的误报回归失败（见 docstring 的可证伪清单）。
TAINT_ROUNDS = 3


# ===========================================================================
# 6. 嵌入夹具（本仓库自己造的合成源码；**不来自外部工件**）
# ===========================================================================
FIXTURE_DIVERGENT = """\
__global__ void bad_sync(float* x, int n) {
    __shared__ float buf[128];
    int t = threadIdx.x;
    if (t < 128) {
        buf[t] = x[t];
    }
    if (threadIdx.x < 32) {
        __syncthreads();
        buf[t] += 1.0f;
    }
    __syncthreads();
    if (t < 128) {
        x[t] = buf[t];
    }
}
"""

# ★ C2b 的**边界对**：正好 48 KB 不违规，48 KB + 1 个 float 才违规。
#   96*128*4 = 49152 = 48*1024（不违规）；97*128*4 = 49664（违规）。
FIXTURE_SMEM_EXACT48K = """\
__global__ void big(float* d_x) {
    __shared__ float tile[96][128];
    int t = threadIdx.x;
    tile[0][t] = d_x[t];
    __syncthreads();
    d_x[t] = tile[0][t];
}
void launch(void) {
    big<<<64, 128>>>(0);
}
"""

FIXTURE_SMEM_OVER48K = FIXTURE_SMEM_EXACT48K.replace("tile[96][128]", "tile[97][128]", 1)


def expected_divergent_sync_lines(text):
    """★ A.5 的**参照实现**：从文本独立地算出「违规那一行」的行号。

    规则（与检查器实现完全不同——它只看"上一个非空行"，不看括号配平）：
      · 剥掉注释行；
      · `__syncthreads()` 那一行，若它**上一个非空行**含 `if` 且含 `threadIdx`，即为违规。
    """
    lines = text.split("\n")
    div, unif = [], []
    for i, l in enumerate(lines):
        if "__syncthreads()" not in l:
            continue
        if l.lstrip().startswith("//"):
            continue
        prev = i - 1
        while prev >= 0 and not lines[prev].strip():
            prev -= 1
        if prev >= 0 and "threadIdx" in lines[prev] and "if" in lines[prev]:
            div.append(i + 1)
        else:
            unif.append(i + 1)
    return div, unif


# ===========================================================================
# 7. 断言框架
# ===========================================================================
FAILS = []
STATS = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}
SKIPS = []

# ★★ 断言标签：`A.1` / `A.11f` / `C.20b` 这种**前缀**。它是 `specs.ANCHORS[*]['covered_by']`
#    里引用的那个东西 —— 所以必须由程序**在运行时**从断言名里提取，不能另立一份手写清单
#    （手写清单会与真实断言脱节，那正是 P6「声称覆盖了、其实没有」的载体）。
LABEL_RE = re.compile(r"[ABC]\.\d+[a-z]*")
SEEN_LABELS = set()          # 运行时**实际**出现过的标签


def label_of(name):
    """从断言名里取标签。**必须出现在开头** —— 名字写成 `A.1 …` 才算标签 `A.1`。"""
    m = LABEL_RE.match(name)
    return m.group(0) if m else None


def static_labels(path):
    """静态提取一个编号程序里**全部**断言标签（`check`/`eq` 的第 2 个实参）。

    ★ 用 `ast` 而非正则扫源码：注释与 docstring 里的 `A.1` 不算数。
    ★ 断言名可能是 `"…" % 值` 或 f-string —— 两种都能取到**字面量前缀**。
      取不到的（名字是变量）返回 `None` 并**跳过**；`A.11h` 会拿本程序的
      **运行时**集合去校验这个静态集，所以「取不到」不会静默变成「没有」。
    """
    def lit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return lit(node.left)
        if isinstance(node, ast.JoinedStr):
            return "".join(v.value for v in node.values if isinstance(v, ast.Constant))
        return None

    out = set()
    tree = ast.parse(read_text(path))
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ("check", "eq") and len(n.args) >= 2):
            s = lit(n.args[1])
            if s:
                lab = label_of(s)
                if lab:
                    out.add(lab)
    return out


def check(cls, name, cond, detail):
    STATS[cls][0] += 1
    lab = label_of(name)
    if lab:
        SEEN_LABELS.add(lab)
    ok = bool(cond)
    if ok:
        STATS[cls][1] += 1
    else:
        FAILS.append("%s/%s" % (cls, name))
    print("  [%s/%-7s] %s" % (cls, "PASS" if ok else "FAIL", name))
    print("        %s" % detail)
    return ok


def eq(cls, name, got, want, detail):
    return check(cls, name, got == want, "%s\n        实得=%r\n        期望=%r"
                 % (detail, got, want))


def cid_lines(findings, cid):
    return sorted(f.line for f in findings if f.cid == cid)


def cid_of(findings, cid):
    return [f for f in findings if f.cid == cid]


# ===========================================================================
# 8. 主流程
# ===========================================================================
def main():
    train_info, learn_info = artifact_roots()
    train, train_tried = train_info
    learn, learn_tried = learn_info
    if train is None or learn is None:
        return die_missing_artifacts(train_info, learn_info)

    import sys as _sys
    print("=" * 78)
    print("01_sync_contract —— 静态契约检查器（本程序不执行任何 CUDA）")
    print("=" * 78)
    print("  解释器      : %s" % _sys.executable)
    print("  python      : %s" % _sys.version.split()[0])
    print("  numpy       : %s" % np.__version__)
    print("                （§B 的代价模型用 numpy 的 **int64 数组**算，")
    print("                  因为 numpy 的整型 `//` 也是 C 语义的截断除法，与 `_ev` 同口径）")
    print("  ★ 本机无 GPU、无 nvcc。本程序读的是文本，产出的是**文本的结论**。")
    print("  外部工件 A  : %s（%s）" % (train, specs.TRAIN_ROOT_ENV))
    print("  外部工件 B  : %s（%s）" % (learn, specs.LEARN_ROOT_ENV))
    print("  工件版本    : cuda-training-series commit %s (%s)，无 release tag"
          % (specs.TRAIN_COMMIT_SHORT, specs.TRAIN_COMMIT_DATE))
    print("                cuda-learning **不是 git 仓库** ⇒ 版本只能用文件哈希")
    print()

    def tp(rel):
        return os.path.join(train, rel)

    def lp(rel):
        return os.path.join(learn, rel)

    # ---------------------------------------------------------------------
    print("-" * 78)
    print("§A 算法正确性（参照实现走**不同的代码路径**）")
    print("-" * 78)

    # A.1 行号换算的两条路径
    src_for_lines = read_text(tp(specs.BLOCKS_BUG_FILES[0][0]))
    mism = 0
    n_off = 0
    for m in re.finditer(r'[A-Za-z_]', src_for_lines):
        n_off += 1
        if line_of(src_for_lines, m.start()) != line_of_text(src_for_lines, m.start()):
            mism += 1
    eq("A", "A.1 行号换算：`str.count` 路径 vs 逐行累加路径（%d 个偏移量）" % n_off, mism, 0,
       "两条算法完全不同：`line_of` 用 `str.count('\\n',0,off)` 数换行符，`line_of_text` 逐行累加字节长度。"
       "改错任一条，全文件数以万计的偏移量里必有一处不一致。")

    # A.2 C 整型语义的求值器
    ev_tbl = [
        ("(512 - 1) / 512", {}, 0),
        ("(500000 + 512 - 1) / 512", {}, 977),
        ("(500000 + (512 - 1) / 512)", {}, 500000),
        ("(-7) / 2", {}, -3),            # ★ C 的截断除法（toward zero），不是 -4
        ("(-7) % 2", {}, -1),            # ★ 同上
        ("(1 << 60) + 1", {}, 2 ** 60 + 1),
        ("((1 << 60) + 1) / 1", {}, 2 ** 60 + 1),
        ("(N + (threads - 1) / threads)", {"N": 500000, "threads": 512}, 500000),
    ]
    bad = []
    for expr, sym, want in ev_tbl:
        got = _ev(expr, sym)
        if got != want:
            bad.append((expr, got, want))
    eq("A", "A.2 `_ev` 的 C 整型语义（%d 个表达式，含负数的截断除法）" % len(ev_tbl), bad, [],
       "★ 关键两行：`(-7)/2` 必须是 **-3**（C 截断）而不是 -4（Python `//` 的 floor）；"
       "`((1<<60)+1)/1` 必须是 **2**60+1** 而不是 `int(float)` 丢精度后的 2**60。"
       "把 `_c_int_binop` 的除法改回 `int(a/b)` 或 `a//b`，这一条立刻失败。")
    naive_floor = -7 // 2
    naive_float = int(((1 << 60) + 1) / 1)
    exact = _ev("((1 << 60) + 1) / 1", {})
    eq("A", "A.2b ⚠ 上一条的**反面**：两种错误写法给出的答案确实不同", 
       (naive_floor != _ev("-7 / 2", {}), naive_float != exact),
       (True, True),
       "`-7 // 2` = %d 而 C 的 `-7 / 2` = %d（floor vs 截断，**错**）；"
       "`int(((1<<60)+1)/1)` = %d 而精确值是 %d（float 只到 2**53，**丢精度**）。"
       "⇒ A.2 的期望值不是「随便挑一个数」——它挑的正是两种实现**分道扬镳**的点，"
       "所以它**可证伪**：把 `_c_int_binop` 的除法改回 `a//b` 或 `int(a/b)`，A.2 立刻失败。"
       % (naive_floor, _ev("-7 / 2", {}), naive_float, exact))
    eq("A", "A.2c `-7 / 2` 必须是 -3（截断）而不是 -4（floor）",
       _ev("-7 / 2", {}), -3, "C 的整型除法是**截断**（toward zero），Python 的 `//` 是 floor。"
       "两者在正数上一致、在负数上分道扬镳。")

    # A.3 七个文件逐字相同 + raw 计数（在**原始字节**上数，不在转写文本上数）
    occ = []
    for rel, ln in specs.BLOCKS_BUG_FILES:
        raw = open(tp(rel), "rb").read()
        cnt = raw.count(specs.BLOCKS_BUG_EXPR.encode("utf-8"))
        text = raw.decode("utf-8")
        line_txt = text.split("\n")[ln - 1].strip()
        occ.append((rel, ln, cnt, line_txt))
    eq("A", "A.3 `%s` 在 7 个文件里各出现 **1** 次，且在声明行号上逐字命中"
       % specs.BLOCKS_BUG_EXPR, [(o[2], o[3]) for o in occ],
       [(1, "int " + specs.BLOCKS_BUG_EXPR + ";")] * 7,
       "★ 在**原始字节**上计数（`raw.count(字面量)`）——不在「先转成文本再数」的转写层上数"
       "（计划 §2 本域特有 3）。7 个文件：\n        "
       + "\n        ".join("%-56s :%d" % (r, l) for r, l, _, _ in occ))

    # A.4 误报回归：C3 不能对 gemm.cu 的间接依赖开火
    # ★ 路径走 `specs.LEARN_*_PATH`，**不**在程序里再写一份字面量：
    #   两处写同一个路径 = 改一处漏一处的经典载体（P1）。订正轮统一。
    gemm_f, _, _, gemm_sym, _ = check_source("gemm.cu",
                                             read_text(lp(specs.LEARN_GEMM_PATH)))
    fa_f, _, _, _, _ = check_source("flash_attention.cu",
                                    read_text(lp(specs.LEARN_FLASH_PATH)))
    eq("A", "A.4 C3 对 cuda-learning/02_gemm/gemm.cu 的 `kA = t*TILE + threadIdx.x` 不误报",
       cid_of(gemm_f, "C3"), [],
       "gemm.cu 第 43-51 行：归纳变量 t，第 44 行 `int kA = t * TILE + threadIdx.x;`，"
       "第 49 行用 `A[row * K + kA]`。右值文本上不含 t，语义上依赖 t。"
       "把 `TAINT_ROUNDS` 改成 0 ⇒ 这一条失败（这正是它存在的理由）。")
    eq("A", "A.4b C2 对 cuda-learning/04_flash_attention/flash_attention.cu 的 Ks[Bc][D] 不误报",
       cid_of(fa_f, "C2"), [],
       "该文件第 28-29 行 `__shared__ __half Ks[Bc][D];`（Bc=32, D=64），"
       "launch 的 blockDim.x=Br=128。但装填走 `for (int r = threadIdx.x; r < Bc; r += blockDim.x)`"
       " + `Ks[r][j] = K[row*D+j]` —— 数组**从不被 threadIdx 直接索引**，"
       "所以「某维必须 >= blockDim」的条件不适用。无条件比对维度就是误报。")

    # A.5 C1 夹具：期望行号**现算**，不写死在注释里
    div_exp, unif_exp = expected_divergent_sync_lines(FIXTURE_DIVERGENT)
    fx_f, _, _, _, _ = check_source("<fixture:divergent>", FIXTURE_DIVERGENT)
    eq("A", "A.5 C1 在合成夹具上只报发散那一处（期望行号由独立路径现算）",
       (cid_lines(fx_f, "C1"), len(div_exp), len(unif_exp)),
       (div_exp, 1, 1),
       "违规处（现算）= %r；合法处（现算）= %r。参照实现只看「上一个非空行含 if 且含 threadIdx」，"
       "检查器走的是括号配平的 `scan_constructs` —— **两条不同的路径**。"
       % (div_exp, unif_exp))

    # A.6 三个程序的锚点覆盖：并集 == 全集，且两两不交
    here = os.path.dirname(os.path.abspath(__file__))
    others = {}
    for fn in ("02_stream_dag.py", "03_graph_capture.py"):
        p = os.path.join(here, fn)
        if not os.path.exists(p):
            SKIPS.append("A.6: 找不到 %s，跨程序锚点覆盖检查跳过" % fn)
            continue
        tree = ast.parse(read_text(p))
        got = None
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "COVERED_ANCHORS" for t in n.targets):
                got = tuple(ast.literal_eval(n.value))
        if got is None:
            SKIPS.append("A.6: %s 没有模块级 COVERED_ANCHORS，跳过" % fn)
        else:
            others[fn] = set(got)
    if others:
        mine = set(COVERED_ANCHORS)
        allc = set(specs.ANCHORS)
        union = mine | set().union(*others.values())
        overlaps = [(a, b) for a, sa in [("01", mine)] + list(others.items())
                    for b, sb in [("01", mine)] + list(others.items())
                    if a < b and (sa & sb)]
        eq("A", "A.6a 三个程序的 COVERED_ANCHORS 并集 == specs.ANCHORS 全集（%d 条）"
           % len(allc), sorted(union - allc) + sorted(allc - union), [],
           "缺的 = %r；多的 = %r。★ 在 specs.py 里加一条锚点而没人覆盖，这一条立刻失败"
           "（`45-kernel-lifecycle` 的同一手法）。" % (sorted(allc - union), sorted(union - allc)))
        eq("A", "A.6b 三个程序的 COVERED_ANCHORS 两两不交", overlaps, [],
           "⇒ 每条锚点有且只有一个「主人」，改一处不会漏改另一处（P1 的机械载体）。")

    # A.7 引文的逐字完整性（**在 specs.py 自身上做**，是规格的完整性检查）
    quote_bad = []
    for aid, a in sorted(specs.ANCHORS.items()):
        for frag in a["must_contain"]:
            if a["quote"].count(frag) != 1:
                quote_bad.append((aid, frag[:40], a["quote"].count(frag)))
    eq("A", "A.7 %d 条锚点的每条 `must_contain` 都在 `quote` 里**恰好出现 1 次**"
       % len(specs.ANCHORS),
       quote_bad, [],
       "★ 这是「引文与断言写在同一处」的机械执行（计划 §2 本域特有 3）："
       "改 `quote` 里任何一个字、或把 `must_contain` 抄歪，这一条失败。")

    # A.10 ★ 机械检查：本目录全部 .py 不得 import 装不上的东西
    py_files, static_bad, dynamic_bad = ast_import_scan(os.path.dirname(
        os.path.abspath(__file__)))
    eq("A", "A.10 ★ AST 扫描：本目录 %d 个 .py（**含 specs.py 与本文件自己**）里"
       "静态 import 违规 %d 处、动态导入违规 %d 处"
       % (len(py_files), len(static_bad), len(dynamic_bad)),
       (static_bad, dynamic_bad), ([], []),
       "扫描到的文件：%r。\n        ★ 用 `ast` 而非 `grep`：注释与 docstring 里出现 "
       "torch 这个词不算违规（本目录的 docstring 里就写着「本程序不 import torch」），"
       "AST 里那只是一段字符串常量。\n"
       "        ★ 含 `import_module('torch')` 这类**动态导入**（CONTRIBUTING 的判据）。\n"
       "        ★ 计划 §2 的 P6 高危点：声称「都不 import torch」却只查编号程序、"
       "漏了 `specs.py` ⇒ 这里 `glob('*.py')` **不做任何筛选**。"
       % ([os.path.basename(x) for x in py_files],))

    # A.11 ★ specs.py 自身的完整性（把「只被 print、没被断言」的字段收紧）
    struct_bad = []
    n_fields = {"doc", "section", "quote", "must_contain", "regex", "note", "covered_by"}
    for aid, a in sorted(specs.ANCHORS.items()):
        if set(a) != n_fields:
            struct_bad.append((aid, "字段集不符", sorted(set(a) ^ n_fields)))
        for k in ("quote", "regex", "note"):
            if not isinstance(a.get(k), str) or len(a.get(k, "")) < 12:
                struct_bad.append((aid, k, "过短或非字符串"))
        if not isinstance(a.get("must_contain"), tuple) or not a.get("must_contain"):
            struct_bad.append((aid, "must_contain", "非非空元组"))
    eq("A", "A.11a %d 条锚点的字段集与最小长度都合规（7 个字段，缺一不可）"
       % len(specs.ANCHORS), struct_bad, [],
       "★ 这一条挡的是「字段被删掉 / 被改成空串」——**它不保证字段内容正确**，"
       "后者由 A.7（quote 的逐字子串）、A.8（doc 的归属）、A.11f（covered_by 的形状"
       "与 note 的双向一致）、`make anchors`（regex 的在线再抽取）各自负责。")

    url_bad = []
    for key, url in sorted(specs.ANCHOR_DOC_URLS.items()):
        if not url.startswith("https://docs.nvidia.com/"):
            url_bad.append((key, url))
    eq("A", "A.11b 抽取器的 %d 个页面 URL 都是 docs.nvidia.com 的 https 地址"
       % len(specs.ANCHOR_DOC_URLS),
       (url_bad, sorted(specs.ANCHOR_DOC_URLS) ==
        sorted({a["doc"] for a in specs.ANCHORS.values()})), ([], True),
       "★ 后半段是**键集一致性**：`ANCHOR_DOC_URLS` 的键必须恰好是 `ANCHORS[*]['doc']` 的取值集合 "
       "⇒ 少一个页面、或某条锚点指向一个没有 URL 的 `doc`，这一条立刻失败。")

    meta_bad = []
    if not (len(specs.TRAIN_COMMIT) == 40
            and all(c in "0123456789abcdef" for c in specs.TRAIN_COMMIT)):
        meta_bad.append(("TRAIN_COMMIT", "不是 40 位十六进制"))
    if specs.TRAIN_COMMIT_SHORT != specs.TRAIN_COMMIT[:7]:
        meta_bad.append(("TRAIN_COMMIT_SHORT", "不是 TRAIN_COMMIT 的前 7 位"))
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", specs.TRAIN_COMMIT_DATE):
        meta_bad.append(("TRAIN_COMMIT_DATE", "不是 YYYY-MM-DD"))
    if specs.TRAIN_NO_TAG is not True:
        meta_bad.append(("TRAIN_NO_TAG", "必须为 True（该仓库无 release tag）"))
    if specs.LEARN_IS_GIT is not False:
        meta_bad.append(("LEARN_IS_GIT", "必须为 False（该仓库不是 git 仓库）"))
    eq("A", "A.11c 工件的版本标识内部自洽（提交号 40 位 + 短号是其前 7 位 + 日期格式 + 两个布尔）",
       meta_bad, [],
       "★ 这一条把「只被 print、没被断言」的版本字段收紧成可证伪的："
       "改 `TRAIN_COMMIT` 或 `TRAIN_COMMIT_SHORT` 中的任何一个，这一条立刻失败。"
       "★ 它对本条只查**格式**（`YYYY-MM-DD`）；日期的**值**由 A.11j 与提交号绑定核对"
       "（订正轮补上：此前把 `TRAIN_COMMIT_DATE` 改成另一个合法日期不会被任何断言抓到）。")

    un_bad = []
    if not isinstance(specs.UNSOURCED, tuple) or len(specs.UNSOURCED) < 5:
        un_bad.append(("UNSOURCED", "不是一个 ≥5 项的元组"))
    if not all(isinstance(x, tuple) and len(x) == 2 for x in specs.UNSOURCED):
        un_bad.append(("UNSOURCED", "含非 (项, 说明) 元素"))
    if not any("C6" in x[0] or "C6" in x[1] for x in specs.UNSOURCED):
        un_bad.append(("UNSOURCED", "缺 C6 那一条"))
    eq("A", "A.11d `UNSOURCED`（%d 条）结构合规，且**含 C6 那一条**"
       % len(specs.UNSOURCED), un_bad, [],
       "★ `UNSOURCED` 是**散文**，所以只做结构断言——它挡的是「整块被删掉」，"
       "不挡「某一条的措辞被改」。这一点在 §八的漏网清单里如实列出。")

    re_bad = []
    for aid, a in sorted(specs.ANCHORS.items()):
        # ★ 复刻 `make anchors` 的抽取逻辑：把正则里的**字面空格**放宽成 `\s+`，
        #   然后拿它去匹配**本模块里存的 quote**。⇒ 正则被改坏时离线也能发现。
        pat = re.sub(r"(?<!\\) ", r"\\s+", a["regex"])
        if not re.search(pat, a["quote"], re.S):
            re_bad.append(aid)
    eq("A", "A.11e 22 条锚点的 `regex` 都能匹配**本模块里存的那个 `quote`**"
       "（离线复刻 `make anchors` 的抽取逻辑）", re_bad, [],
       "★ 这一条的意义：`regex` 本来只在 `make anchors`（**联网**）里被用到，"
       "于是它在 `make run` 里是**不可证伪**的。这里把同一套变换搬到离线，"
       "让它对着 `quote` 自洽 ⇒ 改坏正则，`make run` 也能报错。"
       "★ 它**不等于**在线抽取：在线那一步额外证明「这条 quote 确实能在当前页面上抓到」。")

    # ---- A.11f/g/h ★★ 锚点的 `covered_by`：`note` 是**索引**，不是**唯一载体** ----
    #   ★ 为什么需要它：`note` 里的句子从前**没有任何断言消费**（改一个字，`make run`
    #     照样 rc=0）。一个只活在 `note` 里的论断是不可证伪的（P5）。
    #   ★ 三条子判据（订正轮裁定）：① 非空；② 每个标签**真实存在于某个编号程序的断言
    #     清单里**（不是编的）；③ 22 条锚点的并集覆盖全部 22 条。
    #   ★ 另加一条**双向**约束：`note` 里出现的标签集合必须**恰好等于** `covered_by`。
    #     有了它，「改 `covered_by` 里任何一个标签」都不再是**静默**的：note 里对不上，
    #     A.11f 立刻失败。
    here_lab = os.path.dirname(os.path.abspath(__file__))
    labels_by_file = {}
    for fn in ("01_sync_contract.py", "02_stream_dag.py", "03_graph_capture.py"):
        p = os.path.join(here_lab, fn)
        if os.path.exists(p):
            labels_by_file[fn] = static_labels(p)
        else:
            SKIPS.append("A.11h: 找不到 %s，该文件的标签池为空" % fn)
    label_pool = set()
    for s_ in labels_by_file.values():
        label_pool |= s_

    cb_shape, cb_note, cb_ghost = [], [], []
    for aid, a in sorted(specs.ANCHORS.items()):
        cbs = a.get("covered_by")
        if not isinstance(cbs, tuple) or not cbs:
            cb_shape.append((aid, "非非空元组", repr(cbs)))
            continue
        if not all(isinstance(x, str) and x for x in cbs):
            cb_shape.append((aid, "含非字符串/空元素", repr(cbs)))
        if len(set(cbs)) != len(cbs):
            cb_shape.append((aid, "有重复标签", repr(cbs)))
        note_tags = set(LABEL_RE.findall(a.get("note", "")))
        if note_tags != set(cbs):
            cb_note.append((aid, sorted(note_tags - set(cbs)),
                            sorted(set(cbs) - note_tags)))
        for lab in cbs:
            if lab not in label_pool:
                cb_ghost.append((aid, lab))
    eq("A", "A.11f ★ %d 条锚点的 `covered_by` 形状合规（非空元组、元素是非空字符串、无重复）"
       % len(specs.ANCHORS), cb_shape, [],
       "★ 它挡的是「整个字段被删掉 / 被清空 / 塞进重复项」——这是 `note` 索引化的**下限**。")
    eq("A", "A.11g ★★ `note` 里提到的标签集合与 `covered_by` **恰好相等**（%d 条锚点）"
       % len(specs.ANCHORS), cb_note, [],
       "格式：`(锚点, note 里有而 covered_by 没有的, covered_by 里有而 note 里没提的)`。\n"
       "        ★★ 这一条是「`note` 是**索引**、不是**唯一载体**」的机械执行：\n"
       "        ① `note` 里写了一个标签却不在 `covered_by` ⇒ 那句话**没有**被它自称的断言支撑；\n"
       "        ② `covered_by` 里列了标签而 `note` 只字不提 ⇒ 索引没写全。\n"
       "        ⇒ **改 `covered_by` 里任何一个标签，这一条立刻失败**（双向都堵死）。\n"
       "        ★ 边界（如实说出）：程序没有「标签 ↔ 锚点」的语义知识，所以把标签换成\n"
       "        **另一个真实存在且恰好也在 note 里出现**的标签，是它抓不到的 ——\n"
       "        这种情况必须由 `note` 的措辞来排除，机器只保证**两处一致**。")
    eq("A", "A.11h ★ 22 条锚点的 `covered_by` 并集覆盖全部 %d 条，且每个标签都在"
       "三个编号程序的**断言清单**里（标签池 %d 个）"
       % (len(specs.ANCHORS), len(label_pool)), cb_ghost, [],
       "标签池来源（`ast` 静态提取 `check`/`eq` 的第 2 个实参）：%s。\n"
       "        ★ 「幽灵标签」= `covered_by` 里指向了一个**不存在的断言** —— "
       "那正是「编一个看起来像的标签」的机械形态。\n"
       "        ★ 覆盖性：%d 条锚点全部有非空的 `covered_by`（形状那条已挡空值），"
       "并集 = %d 个标签。\n"
       "        ★★ 静态提取本身会不会漏？下一条 `A.11i` 拿**运行时**集合去自证它。"
       % (", ".join("%s:%d" % (k, len(v)) for k, v in sorted(labels_by_file.items())),
          len(specs.ANCHORS), len(label_pool)))

    # ★ A.11i 自证：静态提取不能漏掉运行时真出现过的标签（只对本程序可做）
    eq("A", "A.11i ★ 静态提取的自证：本程序**运行时**出现过的标签一个都不少地"
       "被`ast` 提取到（运行时 %d 个 / 静态 %d 个）"
       % (len(SEEN_LABELS), len(labels_by_file.get("01_sync_contract.py", set()))),
       sorted(SEEN_LABELS - labels_by_file.get("01_sync_contract.py", set())), [],
       "★ 静态提取是 A.11g/A.11h 判定 02/03 的依据，所以它自己必须**先被自证**："
       "若有人把断言名改成完全动态的（提取不到），这一条会失败 —— "
       "而不是让那两条静默地把该标签当作「不存在」。\n"
       "        ★ 静态集**多于**运行时集是允许的（例如某条断言只在某个分支里跑），"
       "反过来的差集必须为空。")

    # ---- A.11j ★★ 订正 2：`TRAIN_COMMIT_DATE` 的**值**与提交号绑定 ----
    #   ★ 原缺口（实施者自报、控制者反向测试确认）：A.11c 只断言了 `YYYY-MM-DD` **格式**，
    #     把日期换成另一个合法日期不会有任何断言失败。
    #   ★ 补法：日期**不是**一个可以独立写的常量，它是**那个提交的属性** ——
    #     于是它的真值来源只能是工件自己（`git show -s --format=%cs <commit>`）。
    #     这正是本域形态（真值来自外部真实工件），而不是在 `specs.py` 里再加一个副本常量
    #     （那样只是把同一句话写两遍，两处都错就都不报）。
    def git_call(repo, *args):
        try:
            p = subprocess.run(("git", "-C", repo) + args,
                               capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            return None, "%r" % (e,)
        if p.returncode != 0:
            return None, (p.stderr or p.stdout).strip().replace("\n", " ")[:160]
        return p.stdout.strip(), ""

    if not os.path.isdir(os.path.join(train, ".git")):
        SKIPS.append("A.11j: %s 里没有 .git ⇒ 「日期属于哪个提交」无法核对，"
                     "本轮的日期检查降级为 A.11c 的格式检查" % train)
    elif git_call(train, "--version")[0] is None:
        SKIPS.append("A.11j: 本机没有可用的 git ⇒ 日期与提交号的绑定无法核对")
    else:
        sha, sha_err = git_call(train, "rev-parse", "--verify",
                                specs.TRAIN_COMMIT + "^{commit}")
        dstr, d_err = git_call(train, "show", "-s", "--format=%cs", specs.TRAIN_COMMIT)
        eq("A", "A.11j ★★ `TRAIN_COMMIT_DATE` 与提交号**绑定**："
           "`git show -s --format=%%cs %s` == %s"
           % (specs.TRAIN_COMMIT_SHORT, specs.TRAIN_COMMIT_DATE),
           (sha, dstr), (specs.TRAIN_COMMIT, specs.TRAIN_COMMIT_DATE),
           "★ 实得 = (`rev-parse` 回的提交号 %r, `--format=%%cs` 回的日期 %r)；"
           "错误信息 = %r / %r。\n"
           "        ★★ 判据：**把 `TRAIN_COMMIT_DATE` 改成另一个合法日期**"
           "（如 `2021-10-14`），这一条立刻失败 —— 因为日期不是自由文本，"
           "是那个提交的属性。改 `TRAIN_COMMIT` 让它指向别的提交、而日期不改，同样失败。\n"
           "        ★ 覆盖边界（如实说出）：**没有 git 或工件不是 git 仓库时这条 SKIP**，"
           "此时日期只剩 A.11c 的格式检查 —— 这一点会出现在 `make run` 的 SKIP 清单里，"
           "不静默。"
           % (sha, dstr, sha_err, d_err))

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    print("§B 代价模型（推导值；**本程序不产出任何实测性能数字**）")
    print("-" * 78)

    # ★ 用 numpy 的 **int64 数组**算这一组量：numpy 的整型 `//` 是 C 语义的
    #   截断除法（对正数与 C 一致），与 `_ev` 的实现同口径 —— 两条路径互相独立。
    N = specs.HW13_N
    T = specs.HW13_THREADS
    occ_arr = np.array([c for _, _, c, _ in occ], dtype=np.int64)
    issued_blocks = N + (T - 1) // T if T > 0 else None
    # ★ 用 C 语义独立算一遍（不经检查器的求值器）
    # ★ 独立的一遍：直接把源码里的表达式喂给 `_ev`（不经任何手工改写）
    c_issued = _ev("(N + (threads - 1) / threads)", {"N": N, "threads": T})
    need_blocks = c_ceildiv(N, T)
    issued_threads = issued_blocks * T
    useful_threads = N
    check("B", "B.1 `blocks = (N + (threads-1)/threads)` 在 C 整型语义下退化成 N 本身",
          issued_blocks == N == 500000 and c_issued == 500000,
          "N=%d, threads=%d ⇒ (threads-1)/threads = %d/512 = **0** ⇒ blocks = N + 0 = **%d**。"
          "（Python 里 `(512-1)/512` = %.6f，但 C 的整型除法给 0 —— 这正是 `_ev` 必须实现"
          "C 语义而不是直接用 Python `/` 的原因。）"
          % (N, T, T - 1, issued_blocks, (T - 1) / T))
    check("B", "B.2 正确写法给出 %d 个 block（两条独立算路一致）" % need_blocks,
          need_blocks == 977 and c_ceildiv(N, T) == 977,
          "(N + threads - 1) / threads = (%d + %d) / %d = %d / %d = **%d**（C 截断除法）。"
          "另一条算路 `-(-N//T)` 给 %d。" % (N, T - 1, T, N + T - 1, T, need_blocks, -(-N // T)))
    ratio_np = np.array([issued_threads], dtype=np.int64) / np.array(
        [useful_threads], dtype=np.int64)
    blk_ratio_np = np.array([issued_blocks], dtype=np.int64) / np.array(
        [need_blocks], dtype=np.int64)
    check("B", "B.3 超发倍率：**线程口径 %.1f×**，**block 口径 %.2f×**（numpy float64 复核）"
          % (ratio_np[0], blk_ratio_np[0]),
          ratio_np[0] == 512.0 and round(float(blk_ratio_np[0]), 2) == 511.77,
          "线程：%d × %d = %d 个线程槽，是 N 的 **%.1f×**。"
          "block：发了 %d 个，只需 %d 个 ⇒ **%.2f×**。"
          "★ **两个口径不同、都正确**——写文档时必须写明是哪一个，"
          "否则「超发 512 倍」与「超发 511.8 倍」会被当成矛盾（P9 同名不同指）。"
          % (issued_blocks, T, issued_threads, issued_threads / useful_threads,
             issued_blocks, need_blocks, issued_blocks / need_blocks))
    passed_frac = useful_threads / issued_threads
    check("B", "B.4 通过 `if (idx < N)` 守卫的线程只占发射总数的 %.4f%%" % (passed_frac * 100),
          passed_frac == 1.0 / 512 and passed_frac * 100 == 0.1953125,
          "守卫让结果**正确**，但它挡不住**发射**：被丢弃的线程槽 = %d − %d = **%d**。"
          "★ 这是「结果对」与「代价对」分家的最小例子。本程序**不声称**这会让它慢 512 倍"
          "—— 发射代价、占用率、守卫分支的收敛行为，本机一个都没测。"
          % (issued_threads, useful_threads, issued_threads - useful_threads))
    lo, hi, iters = specs.HW13_TIMER_LOOP
    ks = specs.HW13_KERNELS_PER_ITER
    total_slots = iters * ks * issued_threads
    total_useful = iters * ks * useful_threads
    check("B", "B.5 定时循环 %d 次 × %d 个 kernel ⇒ 共发射 %.3g 个线程槽，其中 %.3g 个有用"
          % (iters, ks, total_slots, total_useful),
          total_slots == iters * ks * N * T and total_useful == iters * ks * N,
          "来源：axpy_stream_capture_timer.cu 第 %d–%d 行、`i < %d`；每趟 4 个 kernel"
          "（kernel_a/b/c/d，各一次 launch）。⇒ 总共 %.3g 个线程槽，有用的是 %.3g 个，"
          "比值恒为 512（**与迭代数无关**）。★ 这是**推导值**，不是实测。"
          % (lo, hi, iters, total_slots, total_useful))
    smem_48k = 48 * 1024
    check("B", "B.6 48 KB 共享内存门限的算术：%d 字节 = %d 个 float"
          % (smem_48k, smem_48k // 4),
          smem_48k == 49152 and smem_48k // 4 == 12288,
          "96 × 128 × 4 = %d = %d（**不违规**）；97 × 128 × 4 = %d（**违规**）。"
          "⇒ 判据的分界点落在**一个 float** 上（C.11 的边界对的依据）。"
          % (96 * 128 * 4, smem_48k, 97 * 128 * 4))
    occ_total = int(occ_arr.sum())
    check("B", "B.7 同错文件数 = %d，其中「参考解」= %d 份（出现次数用 int64 数组逐文件统计）"
          % (occ_total, len(specs.BLOCKS_BUG_SOLUTION_FILES)),
          occ_arr.shape == (7,) and bool((occ_arr == 1).all()) and occ_total == 7
          and len(specs.BLOCKS_BUG_SOLUTION_FILES) == 2,
          "★ 计划 v2 写「6/6」——那是**搜索范围就是手写的 6 条路径**造成的（P6），"
          "v2c 订正为 7。本程序用 `raw.count()` 在 7 条路径上逐个核实。"
          "包含两份参考解 ⇒ 是原作者的**写法习惯**，不是偶发笔误。")

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    print("§C 边界与反例（**C 类成立时程序应当失败**：下面每一条都在要求「真的跑出问题」）")
    print("-" * 78)

    # C.1 / C.2 合成夹具：只含统一控制流的版本**不报**（反例：判据不是永远开火）
    only_uniform = """\
__global__ void ok_sync(float* x, int n) {
    __shared__ float buf[128];
    int t = threadIdx.x;
    buf[t] = x[t];
    __syncthreads();
    x[t] = buf[t];
}
"""
    uf, _, _, _, _ = check_source("<fixture:uniform>", only_uniform)
    eq("C", "C.1 只含统一控制流的夹具上，C1 **不**报（判据不是永远开火）",
       cid_of(uf, "C1"), [],
       "★ 这一条与 C.2 是一对：只证明「能报」不构成证据，"
       "还要证明「该报的时候才报」。C 类的语义是「失败行为真的成立」，"
       "而「不发散 ⇒ 不报」是判据**边界**的成立。")
    eq("C", "C.2 发散夹具上 C1 **报 ERROR** 且落在现算出来的那一行",
       [(f.cid, f.severity, f.line) for f in cid_of(fx_f, "C1")],
       [("C1", "ERROR", div_exp[0])] if div_exp else [],
       "出处 specs.ANCHORS['C1'] = %s §%s。" % (specs.ANCHORS['C1']['doc'],
                                              specs.ANCHORS['C1']['section']))

    # C.3 / C.4 hw2
    bad_f, _, _, _, _ = check_source(specs.HW2_BAD_PATH, read_text(tp(specs.HW2_BAD_PATH)))
    good_f, _, _, _, _ = check_source(specs.HW2_GOOD_PATH, read_text(tp(specs.HW2_GOOD_PATH)))
    eq("C", "C.3 hw2 起始代码（已知有问题）⇒ C3 报 2 处，落在第 40、41 行",
       cid_lines(bad_f, "C3"), [40, 41],
       "文件对：%s（题面给的起始代码）与 %s（README 指路的参考解）。"
       "起始的右值**既不含归纳变量 i、也不含由 i 算出的中间量** ⇒ 每一趟装填同一块数据。"
       % (specs.HW2_BAD_PATH, specs.HW2_GOOD_PATH))
    eq("C", "C.4 hw2 参考解（已知正确）⇒ C3 **0** 命中（判据能区分两者）",
       cid_of(good_f, "C3"), [],
       "参考解第 40/41 行含 `i * block_size` ⇒ 随分块号变化。"
       "⇒ 判据不是「见 for 循环就报」，它区分了两个文件。")

    # C.5 / C.6 hw5
    sum_f, _, _, _, _ = check_source(specs.HW5_SUM_PATH, read_text(tp(specs.HW5_SUM_PATH)))
    max_f, _, _, _, _ = check_source(specs.HW5_MAX_PATH, read_text(tp(specs.HW5_MAX_PATH)))
    eq("C", "C.5 hw5 的 **max 参考版** ⇒ C4 报 1 处 WARN，落在第 %d 行"
       % specs.HW5_INIT_LINE,
       [(f.severity, f.line) for f in cid_of(max_f, "C4")], [("WARN", specs.HW5_INIT_LINE)],
       "真值来源是**两个文件的逐字差异**，不是我的判断：第 %d 行 `%s` 两版**完全相同**，"
       "而第 %d/%d 行的算子不同（`+=` vs `max(...)`）。⇒ 参考解在「max 的幺元」这条契约上"
       "自相矛盾：对全负数输入会返回 0.0f 而不是最大负数。"
       % (specs.HW5_INIT_LINE, specs.HW5_INIT_TEXT, specs.HW5_MAX_LINE, specs.HW5_MAX_LINE2))
    eq("C", "C.6 hw5 的 **sum 版** ⇒ C4 **0** 命中（0.0f 对 sum 是正确的幺元）",
       cid_of(sum_f, "C4"), [],
       "★ 这一条是 C.5 的对照：判据钉的是「**幺元与算子不匹配**」，"
       "不是「初值不是 0」。两个文件同一行初值，一个报一个不报 —— 这才叫判据。")

    # C.7 hw13 的 C5
    h13_rel = specs.BLOCKS_BUG_FILES[2][0]        # exercises/hw13/axpy_stream_capture_with_fixme.cu
    h13_ok, _, h13_l, _, h13_dt = check_source(h13_rel, read_text(tp(h13_rel)))
    h5f = cid_of(h13_ok, "C5")
    eq("C", "C.7 hw13 ⇒ C5 报 4 处 WARN，每处都指出是 512.0 倍超发",
       (len(h5f), sorted(set(f.severity for f in h5f)),
        all("512.0" in f.msg for f in h5f)),
       (4, ["WARN"], True),
       "4 个 kernel launch（kernel_a/b/c/d），逐个命中。"
       "期望值 4 是**数出来的**：文件里 `<<<blocks, threads...>>>` 出现 4 次。")

    # C.8 ★ 反向测试：在内存里把 blocks 行改成正确写法 ⇒ WARN 消失
    good_line = "int blocks = (N + threads - 1) / threads;"
    raw_bad = read_text(tp(specs.BLOCKS_BUG_FILES[2][0]))
    raw_fixed = replace_once(raw_bad, "int " + specs.BLOCKS_BUG_EXPR + ";", good_line)
    fx2, _, lx2, sym2, _ = check_source("<hw13:blocks-fixed>", raw_fixed)
    fixed_blocks = sym2.get("blocks")
    eq("C", "C.8 ★ 把 `blocks` 行**在内存里**换成正确写法 ⇒ C5 的 WARN **全部消失**",
       (cid_of(fx2, "C5"), fixed_blocks), ([], 977),
       "改前 %d 处 WARN，改后 0 处。正确写法 `%s` ⇒ 符号表算出 blocks = %r ⇒ "
       "cover = %d×%d = %d，**小于** 4×N=%d 的门限，所以不发火。"
       "★ 这是本程序最强的可证伪断言：它证明 C5 钉的**就是那个表达式**，不是别的。"
       % (len(h5f), good_line, fixed_blocks,
          fixed_blocks or 0, specs.HW13_THREADS,
          (fixed_blocks or 0) * specs.HW13_THREADS, 4 * specs.HW13_N))

    # C.9 ★ 反向测试：删掉守卫 ⇒ C5 也消失（说明它在诊断"靠守卫兜住"这件事）
    raw_noguard, n_guards = replace_every(
        raw_bad, "if (idx < N) y[idx] = 2.0*x[idx] + y[idx];",
        "y[idx] = 2.0*x[idx] + y[idx];")
    fx3, _, _, _, _ = check_source("<hw13:no-guard>", raw_noguard)
    eq("C", "C.9 ★ 把 %d 处 `if (idx < N)` 守卫**在内存里**全部删掉 ⇒ C5 的 WARN 也全部消失"
       % n_guards,
       cid_of(fx3, "C5"), [],
       "★ 这条初看反直觉：删掉一个**正确性**守卫反而让**代价**告警消失。"
       "原因是判据 C5 只在**同时**满足「超发」与「有守卫兜住」时才报——"
       "没有守卫版本是**错的**（会越界写），那是另一类问题，不在 C5 的判定范围内。"
       "⇒ 「C5 不发火」**不等于**「这个版本没问题」（P5：断言形式正确但语义相反）。"
       "★ 注意必须是**全部** %d 处都删——第一版用了只换第一处的 `replace_once`，"
       "剩下 3 个 kernel 的 WARN 还在，失败的是**我的期望值**（与探针 v1 同一类错误）。"
       % n_guards)

    # C.10 合成夹具：48 KB 边界对
    e48, _, _, _, _ = check_source("<fixture:smem-exact-48k>", FIXTURE_SMEM_EXACT48K)
    o48, _, _, _, _ = check_source("<fixture:smem-over-48k>", FIXTURE_SMEM_OVER48K)
    eq("C", "C.10 静态共享内存**正好 48 KB** ⇒ C2 **不**报",
       cid_of(e48, "C2"), [], "96×128×4 = %d 字节 = 48×1024，判据是 `> 48*1024`。" % (96 * 128 * 4))
    eq("C", "C.11 静态共享内存 **48 KB + 1 个 float** ⇒ C2 报 ERROR",
       [(f.cid, f.severity) for f in cid_of(o48, "C2")], [("C2", "ERROR")],
       "97×128×4 = %d 字节 > %d。出处 specs.ANCHORS['C2'] = %s §%s。"
       "★ 注意该锚点的**作用域**：抽取到的原句以 `Compute capability 7.x devices …` 开头 —— "
       "这条 48 KB 规则的语境是 **7.x 设备**，引用时不能省。"
       % (97 * 128 * 4, 48 * 1024, specs.ANCHORS['C2']['doc'], specs.ANCHORS['C2']['section']))

    # C.12 工件哈希：确认"我们读到的确实是这份工件"
    hash_bad = []
    for rel, want in sorted(specs.TRAIN_SHA16.items()):
        p = tp(rel)
        if not os.path.exists(p):
            hash_bad.append((rel, "缺失"))
            continue
        got = sha16(p)
        if got != want:
            hash_bad.append((rel, "got %s want %s" % (got, want)))
    eq("C", "C.12 外部工件 A 的 %d 个文件 sha256 前 16 位全部匹配" % len(specs.TRAIN_SHA16),
       hash_bad, [],
       "★ 哈希是**版本锚**：不匹配说明读到的不是 specs.py 记录的那一份，"
       "那么全部行号引用都失去意义。⇒ 这一条必须在读行号之前先跑（本程序按 §C 的顺序执行）。")

    # C.21 ★ 外部工件 B 的版本锚（订正轮补：`LEARN_SHA16` 此前是**死常量**）
    learn_hash_bad = []
    for rel, want in sorted(specs.LEARN_SHA16.items()):
        p = lp(rel)
        if not os.path.exists(p):
            learn_hash_bad.append((rel, "缺失"))
            continue
        got = sha16(p)
        if got != want:
            learn_hash_bad.append((rel, "got %s want %s" % (got, want)))
    eq("C", "C.21 ★ 外部工件 B（cuda-learning）的 %d 个文件 sha256 前 16 位全部匹配"
       % len(specs.LEARN_SHA16), learn_hash_bad, [],
       "★ 与 C.12 成对：`cuda-training-series` 有 commit 号当版本锚，而 "
       "`cuda-learning` **不是 git 仓库** ⇒ 它的版本锚**只能**是文件哈希。\n"
       "        ★★ 这一条是**订正轮补的**，来源是 `make mutate` 的漏网清单："
       "`specs.LEARN_SHA16` 的 %d 个叶子**全部漏网**，说明它是个**死常量** —— "
       "三个程序一次都没引用它，而主流程还在打印「版本只能用文件哈希」。"
       "**那是一句没有实现的声明**（P6：声称覆盖、实际没有）。\n"
       "        ★ 判据：改哈希、改文件名、换一份 cuda-learning，这一条立刻失败。"
       % len(specs.LEARN_SHA16))

    # C.16 ★ C1 的判据有**两条**锚点支撑：契约 + 机制
    c1_ev = [f.evidence for f in cid_of(fx_f, "C1")]
    eq("C", "C.16 ★ C1 的判据依据**两条**锚点：契约（C1）+ 机制（C1b）",
       (len(c1_ev) == 1
        and specs.ANCHORS["C1"]["quote"] in c1_ev[0]
        and specs.ANCHORS["C1b"]["quote"] in c1_ev[0]
        and "warp level" in specs.ANCHORS["C1b"]["quote"]
        and "enforced per thread" in specs.ANCHORS["C1b"]["quote"]),
       True,
       "★ 锚点 **C1** 给契约：条件必须「evaluates identically across the entire thread block」。"
       "锚点 **C1b**（%s §%s）给机制：Pascal 及更早在 **warp 级**强制、"
       "**Volta 起逐线程**强制 ⇒ 违反契约的代码「will likely deadlock」。"
       "⇒ 两条锚点合起来才解释得了「为什么旧卡上没事、新卡上死锁」。"
       "★ **P9 提示**：这里说的 `C1` 是**锚点 id**；`01` 的判据 id 也叫 `C1` —— 两套编号。"
       % (specs.ANCHORS["C1b"]["doc"], specs.ANCHORS["C1b"]["section"]))

    # C.13 逐字行内容：行号引用的最终依据
    line_bad = []
    for (rel, ln), want in [
            ((specs.HW13_N_PATH, specs.HW13_N_LINE), specs.HW13_N_TEXT),
            ((specs.HW13_N_PATH, specs.HW13_THREADS_LINE), specs.HW13_THREADS_TEXT),
            ((specs.HW13_N_PATH, specs.HW13_STREAM_FLAGS_LINE),
             specs.HW13_STREAM_FLAGS_TEXT),
            ((specs.HW2_BAD_PATH, specs.HW2_LOOP_LINE), specs.HW2_LOOP_TEXT),
            ((specs.HW10_PATH, specs.HW10_CHUNKS_LINE), specs.HW10_CHUNKS_TEXT),
            ((specs.HW10_PATH, specs.HW10_NUM_STREAMS_LINE), specs.HW10_NUM_STREAMS_TEXT),
            ((specs.HW10_PATH, specs.HW10_STREAM_CREATE_LINE), specs.HW10_STREAM_CREATE_TEXT),
            ((specs.HW10_PATH, specs.HW10_MEMSET_LINE), specs.HW10_MEMSET_TEXT),
            ((specs.HW10_PATH, specs.HW10_PRAGMA_LINE), specs.HW10_PRAGMA_TEXT),
            ((specs.HW10_PATH, specs.HW10_LOOP_FIRST_LINE), specs.HW10_LOOP_TEXT),
            ((specs.HW10_PATH, specs.HW10_SYNC_LINE), specs.HW10_SYNC_TEXT),
            ((specs.HW13_README_PATH, 15), specs.HW13_README_CITES[15]),
            ((specs.HW13_README_PATH, 74), specs.HW13_README_CITES[74]),
            ((specs.HW13_README_PATH, 81), specs.HW13_README_CITES[81]),
    ]:
        text = read_text(tp(rel))
        got = text.split("\n")[ln - 1].strip()
        if want.strip() not in got:
            line_bad.append((rel, ln, got[:70], want[:70]))
    eq("C", "C.13 14 处「行号 + 逐字内容」全部命中", line_bad, [],
       "★ 行号引用**只有配上逐字内容才有意义**（CONTRIBUTING：不写版本的源码引用视为无效引用；"
       "行号是同一个道理）。任一行号漂了，这一条立刻失败。")
    g45 = [
        (specs.HW2_BAD_PATH, 40, specs.HW2_BAD_LINES[40]),
        (specs.HW2_BAD_PATH, 41, specs.HW2_BAD_LINES[41]),
        (specs.HW2_GOOD_PATH, 40, specs.HW2_GOOD_LINES[40]),
        (specs.HW2_GOOD_PATH, 41, specs.HW2_GOOD_LINES[41]),
        (specs.HW5_SUM_PATH, specs.HW5_INIT_LINE, specs.HW5_INIT_TEXT),
        (specs.HW5_MAX_PATH, specs.HW5_INIT_LINE, specs.HW5_INIT_TEXT),
        (specs.HW5_SUM_PATH, specs.HW5_SUM_LINE, specs.HW5_SUM_TEXT),
        (specs.HW5_SUM_PATH, specs.HW5_SUM_LINE2, specs.HW5_SUM_TEXT2),
        (specs.HW5_MAX_PATH, specs.HW5_MAX_LINE, specs.HW5_MAX_TEXT),
        (specs.HW5_MAX_PATH, specs.HW5_MAX_LINE2, specs.HW5_MAX_TEXT2),
    ]
    g45_bad = []
    for rel, ln, want in g45:
        got = read_text(tp(rel)).split("\n")[ln - 1].strip()
        if want not in got:
            g45_bad.append((rel, ln, got[:70], want[:70]))
    eq("C", "C.13b G4/G5 的 %d 处逐字行内容全部命中" % len(g45), g45_bad, [],
       "★ 两版第 %d 行逐字相同（`%s`）⇒ 「max 的幺元是 0.0f」这条结论"
       "不是从注释或 README 推的，是两个文件的**逐字差异**里读出来的："
       "第 %d/%d 行的算子不同（`+=` vs `max(...)`）而初值行相同。"
       % (specs.HW5_INIT_LINE, specs.HW5_INIT_TEXT, specs.HW5_MAX_LINE, specs.HW5_MAX_LINE2))

    # C.14 C6：**不得声称覆盖**
    eq("C", "C.14 C6 的状态必须是「没实现」（分级 = None）——不得声称覆盖",
       specs.RULE_STRENGTH["C6"][0], None,
       "★ 这条断言的意义是**反向**的：它挡住的不是「漏实现」，而是「**把没实现的写成已实现**」。"
       "计划 §0.7 G8-6 明写 C6 设计了但没实现。")

    # C.17 ★ cuda-learning 的 T4 环境声明（引用它时**必须同时说明**）
    learn_bad = []
    for ln, frag in ((specs.LEARN_ENV_LINE, specs.LEARN_ENV_TEXT),
                     (specs.LEARN_ARCH_LINE, specs.LEARN_ARCH_TEXT),
                     (specs.LEARN_ABSENT_LINE, specs.LEARN_ABSENT_FRAG)):
        got = read_text(lp("README.md")).split("\n")[ln - 1]
        if frag not in got:
            learn_bad.append((ln, got.strip()[:70], frag[:70]))
    eq("C", "C.17 ★ cuda-learning/README.md 第 %d/%d/%d 行的 T4 / sm_75 声明逐字命中"
       % (specs.LEARN_ENV_LINE, specs.LEARN_ARCH_LINE, specs.LEARN_ABSENT_LINE),
       learn_bad, [],
       "★ 这条锚住的是「**引用它的代码时必须同时说明目标环境**」这句纪律的**依据**："
       "第 %d 行给出 Tesla T4 / sm_75（Turing），第 %d 行给出缺哪些特性"
       "（没有 bf16 / TF32 / cp.async / TMA / warp-specialization）。"
       "★ 改一个行号，这一条立刻失败。"
       % (specs.LEARN_ENV_LINE, specs.LEARN_ABSENT_LINE))

    # C.18 ★ hw5 的自带测试数据**全为正**（这正是它掩盖 max 幺元错误的原因）
    hw5_lines = read_text(tp(specs.HW5_MAX_PATH)).split("\n")
    hw5_bad, hw5_vals = [], []
    for ln, frag in specs.HW5_TEST_DATA_LINES:
        got = hw5_lines[ln - 1].strip()
        if frag not in got:
            hw5_bad.append((ln, got[:60], frag))
        else:
            hw5_vals.append(frag)
    eq("C", "C.18 ★ hw5 参考解自带的测试数据**全为正**（%r），且三个字面量逐行命中"
       % (hw5_vals,),
       (hw5_bad, specs.HW5_TEST_DATA_POSITIVE_ONLY, len(hw5_vals)),
       ([], True, 3),
       "★ 三个值分别是 **1.0f**（第 47 行）与 **5.0f**（第 45 行的 `float max_val = 5.0f;`，"
       "第 48 行再赋给 `h_A[100]`）—— **都是正数**。"
       "⇒ 判据 C4 之所以给 **WARN** 而不是 ERROR，就是因为**这段代码在它自带的输入上是正确的**。"
       "★ 把 `HW5_TEST_DATA_POSITIVE_ONLY` 改成 False、或把任一行改掉，这一条立刻失败。")

    # C.19 ★ hw13 里 `<<<` 的出现次数（C5 报 4 处的依据）
    eq("C", "C.19 ★ hw13 的 `<<<` 出现次数 = %d（C5 报 %d 处的依据）"
       % (read_text(tp(specs.HW13_TASK1_PATH)).count("<<<"), specs.HW13_LAUNCH_COUNT),
       (read_text(tp(specs.HW13_TASK1_PATH)).count("<<<"), True),
       (specs.HW13_LAUNCH_COUNT, True),
       "★ 把 `HW13_LAUNCH_COUNT` 从 4 改成 5，这一条立刻失败。")

    # C.20 ★ hw13 README 第 74 行（那条唯一的时间说法）
    r74 = read_text(tp(specs.HW13_README_PATH)).split("\n")[74 - 1]
    eq("C", "C.20 ★ hw13 README 第 74 行含那句唯一的定性时间说法（逐字）",
       (specs.HW13_README_TIME_FRAG in r74, specs.HW13_README_CITES[74][:40] in r74),
       (True, True),
       "★ 逐字（含原文的拼写错误 `preformance`）：`The overlapped version ... should still be "
       "able to see a small preformance increase`。**它没有给出倍数** —— 这是本篇"
       "「收益是宿主动作数、不是时间」这条结论的**唯一**外部依据。")
    eq("C", "C.20b ★ hw10 readme 里唯一的时间说法在第 %d 行、逐字命中"
       % specs.HW10_README_TIME_LINE,
       specs.HW10_README_TIME_FRAG in
       read_text(tp(specs.HW10_README_PATH)).split("\n")[specs.HW10_README_TIME_LINE - 1],
       True,
       "逐字：`%s`。⇒ 「about 2X」是**唯一的**时间说法，且**本机没有验证它**。"
       % specs.HW10_README_TIME_FRAG)

    # C.15 强度分级表本身
    strength_bad = []
    for cid, (lvl, why) in sorted(specs.RULE_STRENGTH.items()):
        if lvl not in specs.STRENGTH_LEVELS:
            strength_bad.append((cid, lvl))
    eq("C", "C.15 判据强度分级表的取值合法性（%d 条判据）" % len(specs.RULE_STRENGTH),
       strength_bad, [],
       "允许的等级只有 %r。C1/C2 是 A 级（官方原文），C3/C4/C5 是 B 级（成对文件 / 真实文件），"
       "C6 是 None（未实现）。" % (specs.STRENGTH_LEVELS,))

    # ---------------------------------------------------------------------
    print()
    print("-" * 78)
    total = sum(v[0] for v in STATS.values())
    passed = sum(v[1] for v in STATS.values())
    print("汇总: A %d/%d · B %d/%d · C %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (STATS['A'][1], STATS['A'][0], STATS['B'][1], STATS['B'][0],
             STATS['C'][1], STATS['C'][0], total, passed, len(FAILS)))
    if SKIPS:
        print("跳过 %d 项（**不计入通过数**）：" % len(SKIPS))
        for s in SKIPS:
            print("  [SKIP] %s" % s)
    for f in FAILS:
        print("  FAILED: %s" % f)
    print("★ 本程序的全部数字都是**推导值**（本仓库程序算出，可复核），不是本机实测值。")
    print("  外部工件：cuda-training-series (B 级，commit %s) 与 cuda-learning (B 级，非 git)"
          % specs.TRAIN_COMMIT_SHORT)
    print("  ★ cuda-learning 的目标环境是 **Tesla T4 / sm_75（Turing）**："
          "**没有** bf16 / TF32 / cp.async / TMA。")
    print("=" * 78)
    return 1 if FAILS else 0


# ★ 本程序覆盖的语义锚点（见 specs.ANCHORS）。`01` 自己用 ast 读另外两个程序的同名常量，
#   断言三个集合的**并集 == specs.ANCHORS 全集**且**两两不交**。
COVERED_ANCHORS = ("C1", "C1b", "C2")


def lint_main():
    """`make lint` 的入口：只做 AST 导入扫描，跑得快，不读外部工件。"""
    files, static_bad, dynamic_bad = ast_import_scan(
        os.path.dirname(os.path.abspath(__file__)))
    print("AST 检查：本目录 .py 共 %d 个（**含 specs.py**），"
          "静态 import 违规 %d 处，动态导入违规 %d 处"
          % (len(files), len(static_bad), len(dynamic_bad)))
    for x in static_bad + dynamic_bad:
        print("   !! %r" % (x,))
    print("  ★ 注释与 docstring 里出现 torch 这个词不算违规 —— AST 里那只是一段字符串。")
    return 1 if (static_bad or dynamic_bad) else 0


if __name__ == "__main__":
    if "--lint" in sys.argv:
        sys.exit(lint_main())
    sys.exit(main())
