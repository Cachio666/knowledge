"""04 动态形状：guard、重编译，以及「自动动态形状省掉了什么」。

对应文档：60-compilers/inductor/04-dynamic-shapes.md

★ 本程序跑的是**本仓库的迷你编译缓存**，不是 torch._dynamo 的 cache / guards。
  默认值（`dynamic=None`、`automatic_dynamic_shapes=True`、`recompile_limit=8`、
  `accumulated_recompile_limit=256`）逐条抄自固定版本的 torch 源码，
  见 specs.py 与 specs.RECOMPILE_LIMITS（**每一条都带版本**）。
  输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。

★ **本文件明写一条与真实默认不符的对照**：
  `automatic_dynamic_shapes=False` 的那组数据**不是**真实默认行为，
  官方默认是 `True`（specs.AUTOMATIC_DYNAMIC_SHAPES_DEFAULT）。
  那组数据是**反例专用**，用来把「自动动态形状省掉了什么」量化出来。

用法：python3 04_recompile_count.py
"""

import numpy as np

import minigraph as M
import specs as SPECS
from minigraph import Checks, eval_graph

checks = Checks()


# ---- 迷你编译缓存：guard + 重编译计数 ----------------------------------------------------


class Entry:
    """一次编译的产物：一张图 + 一组 guard + 本次编译的统计。"""

    def __init__(self, graph, shapes, dynamic_dims, consts, n_ops):
        self.graph = graph
        self.shapes = shapes              # tuple[tuple[int,...]] 编译时的静态形状
        self.dynamic_dims = dynamic_dims  # frozenset[(arg_i, dim)]
        self.consts = dict(consts)
        self.n_ops = n_ops
        self.hits = 0

    def guard_ok(self, shapes, consts):
        """guard 的内容（官方措辞）：**constant values, types, and object IDs**。

        本仓库只实现其中的两类：形状（含哪些维已标为 dynamic）与常量的精确值。
        """
        if len(shapes) != len(self.shapes):
            return False, "参数个数不同"
        for i, (got, want) in enumerate(zip(shapes, self.shapes)):
            if len(got) != len(want):
                return False, "arg%d 的秩变了（%d → %d）" % (i, len(want), len(got))
            for j, (g, w) in enumerate(zip(got, want)):
                if (i, j) in self.dynamic_dims:
                    continue
                if g != w:
                    return False, "arg%d 的第 %d 维变了（%d → %d）" % (i, j, w, g)
        for k, v in self.consts.items():
            if consts.get(k) != v:
                return False, "常量 %s 的值变了（%r → %r）" % (k, v, consts.get(k))
        return True, "guard 通过"

    def describe(self):
        dd = sorted(self.dynamic_dims)
        return "图 %d 个算子，静态形状 %s，dynamic 维 %s" % (
            self.n_ops, self.shapes, dd or "无")


class MiniCompileCache:
    """迷你版的「编译 → 缓存 → guard 失败 → 重编译 → 触顶降级」。"""

    def __init__(self, automatic_dynamic_shapes=True, recompile_limit=8,
                 accumulated_recompile_limit=256, name="f"):
        self.auto_dynamic = automatic_dynamic_shapes
        self.recompile_limit = recompile_limit
        self.accumulated_recompile_limit = accumulated_recompile_limit
        self.name = name
        self.entries = []
        self.compiles = 0
        self.hits = 0
        self.eager_calls = 0
        self.log = []

    # ---- 一次调用 ----
    def call(self, fn, args, consts=None, eager_fn=None):
        """`fn` 是「被编译的那段」（用迷你代理写）；`eager_fn` 是它的纯 numpy 版本，
        触顶降级时跑的就是它——即真实系统里「这一帧直接跑普通 Python」。"""
        consts = consts or {}
        eager_fn = eager_fn or fn
        shapes = tuple(tuple(np.asarray(a).shape) for a in args)
        for e in self.entries:
            ok, why = e.guard_ok(shapes, consts)
            if ok:
                e.hits += 1
                self.hits += 1
                self.log.append("命中缓存：%s（%s）" % (shapes, why))
                return self._run(e, args)
            last_why = why
        if self.compiles >= self.recompile_limit:
            # 触顶：**不再尝试编译**，这一帧直接跑 eager
            self.eager_calls += 1
            self.log.append("已达 recompile_limit=%d → 不再编译，直接跑 eager（%s）"
                            % (self.recompile_limit, shapes))
            return [np.asarray(eager_fn(*args, **consts))]   # 真的跑普通 Python 版
        return self._compile(fn, args, shapes, consts,
                             last_why if self.entries else "首次调用")

    def _compile(self, fn, args, shapes, consts, why):
        # 非张量参数（这里是 lr）走闭包进捕获：它会被写进节点的常量属性，
        # **这正是改 lr 必须重编译的原因**。
        def wrapped(*a, _fn=fn, _consts=dict(consts)):
            return _fn(*a, **_consts)

        d = M.MiniDynamo()
        d.register(wrapped)
        res = d.capture(wrapped, args)
        graph = res.subgraphs[-1]
        dd = set()
        if self.auto_dynamic and self.entries:
            # 默认路径：guard 失败后**把变化的维度自动标为 dynamic** 再重编译
            base = self.entries[0].shapes
            for i, (got, want) in enumerate(zip(shapes, base)):
                if len(got) != len(want):
                    continue
                for j, (g, w) in enumerate(zip(got, want)):
                    if g != w:
                        dd.add((i, j))
        e = Entry(graph, shapes, frozenset(dd), consts,
                  len([n for n in graph.nodes if n.kind not in ("placeholder", "effect")]))
        self.entries.append(e)
        self.compiles += 1
        self.log.append("编译第 %d 次：%s  [触发原因：%s]"
                        % (self.compiles, e.describe(), why))
        return self._run(e, args)

    def _run(self, entry, args):
        """执行一次编译产物。

        ★ 模型选择：**编译产物是形状特化的**——真实 Inductor 生成的 kernel 里
          `xnumel` 之类是常量，喂进不同元素数的输入不会「自动适配」。
          本仓库把它建模成「按编译时的形状 reshape」，对不上就直接报错。
          （真机上更可能是算错或越界访问，而不是一句 Python 异常。）
          `guard` 的作用是在**走到这一步之前**就挑对版本。
        """
        env = {}
        for i, a in enumerate(args):
            want = entry.shapes[i]
            arr = np.asarray(a)
            if entry.dynamic_dims:
                # 这张图里已经有维度被标成 dynamic：生成物按运行期形状取尺寸
                env["arg%d" % i] = arr
                continue
            if arr.size != int(np.prod(want)):
                raise ValueError(
                    "编译产物是形状特化的：这个核按 %s（%d 个元素）编出来，"
                    "喂进来的是 %s（%d 个元素）" % (want, int(np.prod(want)),
                                                 arr.shape, arr.size))
            env["arg%d" % i] = arr.reshape(want)
        out = eval_graph(entry.graph, env)
        return [out[o.buf] for o in entry.graph.outputs]

    def summary(self):
        return "编译 %d 次 · 命中缓存 %d 次 · eager 兜底 %d 次" % (
            self.compiles, self.hits, self.eager_calls)


# ---- 地形 ----------------------------------------------------------------------


def model(x, w, m, lr):
    """`lr` 是 `float`——它会被 guard 成**常量**的精确值（LR scheduler 的经典情形）。

    这里特意放一个 `matmul`：它让编译出的图**对形状敏感**——
    契约维一变，缓存的图就不再适用（`b` 那种纯广播的写法反而「碰巧还能算」）。
    """
    a = (x * w).relu()
    return a.matmul(m) * lr


def ref_model(x, w, m, lr):
    a = np.maximum(x * w, 0.0)
    return (a @ m) * lr


def make_inputs(n, d):
    x = (np.arange(n * d, dtype=np.float32).reshape(n, d) % 7 - 3.0) / 2.0
    w = np.full((n, d), 1.5, dtype=np.float32)
    m = np.full((d, 2), 0.25, dtype=np.float32)
    return [x, w, m]


SHAPES = [(4, 3), (8, 3), (16, 3), (32, 3), (64, 3), (128, 3)]   # N = 6，只变第 0 维
REPEATS = 5

M.rule("1. 默认值（逐条带版本，见 specs.py）")
print("%-40s %s" % ("项", "值"))
print("-" * 78)
print("%-40s %s" % ("torch.compile 的 dynamic 参数默认值",
                    "None（specs.DYNAMIC_ARG_DEFAULT）"))
print("%-40s %s" % ("automatic_dynamic_shapes 默认值",
                    SPECS.AUTOMATIC_DYNAMIC_SHAPES_DEFAULT))
for ver, d in sorted(SPECS.RECOMPILE_LIMITS.items()):
    print("%-40s %s" % ("recompile_limit / accumulated（torch %s）" % ver, d))
print("%-40s %s" % ("dynamic=True 的官方态度", SPECS.DYNAMIC_TRUE_IS_NOT_RECOMMENDED))
print("%-40s %s" % ("graph break 编号", "%s = %s"
                    % (SPECS.GB_RECOMPILE_LIMIT, SPECS.GB_RECOMPILE_LIMIT_TYPE)))
print()
print("guard 的内容（官方措辞）：**constant values, types, and object IDs**")
print("重编译语义（官方措辞）：「如果 guard 对**所有**已编译实例都失败，才必须重编译」。")

# ---- 2a：「同一个 shape 重复 M 次」与「N 个不同 shape」 -------------------------------------

M.rule("2. 真实默认（automatic_dynamic_shapes=True）下的重编译次数")

cache_auto = MiniCompileCache(name="auto")
for _ in range(REPEATS):
    cache_auto.call(model, make_inputs(4, 3), {"lr": 0.01})
same_shape_compiles = cache_auto.compiles
for _ in range(REPEATS * 3):
    cache_auto.call(model, make_inputs(4, 3), {"lr": 0.01})
same_shape_after = cache_auto.compiles
print("同一个 shape 调用 %d 次后再调用 %d 次：编译次数 %d → %d"
      % (REPEATS, REPEATS * 3, same_shape_compiles, same_shape_after))

before = cache_auto.compiles
for n, d in SHAPES:
    cache_auto.call(model, make_inputs(n, d), {"lr": 0.01})
after = cache_auto.compiles
print()
print("接着喂 %d 个互不相同的 shape：编译次数 %d → %d（增量 %d）"
      % (len(SHAPES), before, after, after - before))
print("缓存条目：")
for i, e in enumerate(cache_auto.entries):
    print("   #%d %s，命中 %d 次" % (i + 1, e.describe(), e.hits))
print("%s" % cache_auto.summary())

checks.C(same_shape_after == same_shape_compiles,
         "C2(b)：同一个 shape 重复调用，重编译次数**完全不增**",
         "%d → %d" % (same_shape_compiles, same_shape_after))
delta_auto = after - before
checks.C(delta_auto < len(SHAPES),
         "C2(b)：N 个不同 shape 的重编译**增量远小于 N**",
         "N=%d，增量=%d" % (len(SHAPES), delta_auto))
print("★ 后续断言里用的就是这两个本机实测值：同 shape 增量 0、不同 shape 增量 %d"
      % delta_auto)

# ---- 2b：关掉 automatic_dynamic_shapes 的对照（**不是真实默认**） --------------------------

M.rule("3. ★ 对照：把 automatic_dynamic_shapes 关掉（**这与真实默认不符**）")

print("★ 官方默认是 %s（specs.AUTOMATIC_DYNAMIC_SHAPES_DEFAULT）。"
      % SPECS.AUTOMATIC_DYNAMIC_SHAPES_DEFAULT)
print("  下面这组数据是**反例专用**：它演示的是「关掉之后会怎样」，不是默认行为。")

cache_static = MiniCompileCache(automatic_dynamic_shapes=False, name="static")
for n, d in SHAPES:
    cache_static.call(model, make_inputs(n, d), {"lr": 0.01})
print()
print("关掉之后，同样 %d 个不同 shape：%s" % (len(SHAPES), cache_static.summary()))
print("缓存条目：")
for i, e in enumerate(cache_static.entries):
    print("   #%d %s" % (i + 1, e.describe()))

checks.C(cache_static.compiles == len(SHAPES),
         "C2(a)：关掉自动动态形状后，N 个不同 shape → 重编译次数随 N 线性增长",
         "编译 %d 次 / N=%d" % (cache_static.compiles, len(SHAPES)))
checks.A(delta_auto < cache_static.compiles,
         "A：自动动态形状省下的编译次数 = %d − %d" % (cache_static.compiles, delta_auto),
         "开 %d 次 vs 关 %d 次" % (delta_auto, cache_static.compiles))

# ---- 4. 触顶：静默降级 --------------------------------------------------------------

M.rule("4. §4 失败模式 ①：触顶之后静默降级，不再尝试编译")

# 为了让调用真的把上限打满，这里关掉自动动态形状——否则新形状会被并进同一张动态图，
# 上限机制本身与这个开关无关。
cache_cap = MiniCompileCache(
    automatic_dynamic_shapes=False,
    recompile_limit=SPECS.RECOMPILE_LIMITS["2.13.0"]["recompile_limit"])
shapes_many = [(4 + 4 * k, 3) for k in range(14)]
for sh in shapes_many:
    cache_cap.call(model, make_inputs(*sh), {"lr": 0.01}, eager_fn=ref_model)
print("喂 %d 个不同 shape（limit=%d）：%s"
      % (len(shapes_many), cache_cap.recompile_limit, cache_cap.summary()))
print("日志末尾：")
for line in cache_cap.log[-4:]:
    print("   %s" % line)
print("源码依据：%s" % SPECS.RECOMPILE_LIMIT_HIT_BEHAVIOR)

checks.C(cache_cap.compiles == cache_cap.recompile_limit,
         "C：触顶后编译次数**不再增长**（上限就是 recompile_limit=8）",
         "编译 %d 次" % cache_cap.compiles)
checks.C(cache_cap.eager_calls > 0,
         "C：触顶后的调用改为直接跑 eager（静默降级，不报错）",
         "eager 兜底 %d 次" % cache_cap.eager_calls)
checks.B(cache_cap.compiles + cache_cap.eager_calls == len(shapes_many),
         "B：每一次调用要么编译、要么走 eager 兜底（没有第三种去向）",
         "%d + %d = %d" % (cache_cap.compiles, cache_cap.eager_calls,
                           len(shapes_many)))

# ---- 5. int / float 被当作常量 guard 精确值 ------------------------------------------

M.rule("5. §4 失败模式 ②：改学习率就重编译——它与形状无关")

cache_lr = MiniCompileCache()
cache_lr.call(model, make_inputs(4, 3), {"lr": 0.01})
for lr in (0.005, 0.0025, 0.001):
    cache_lr.call(model, make_inputs(4, 3), {"lr": lr})
print("形状一直是 %s，只改 lr：%s" % ((4, 3), cache_lr.summary()))
print("日志：")
for line in cache_lr.log:
    print("   %s" % line)

checks.C(cache_lr.compiles == 4,
         "C：形状完全没变，只改 lr 也触发了 4 次编译（float 被 guard 成精确值）",
         "编译 %d 次" % cache_lr.compiles)
same_shapes = len({e.shapes for e in cache_lr.entries}) == 1
checks.A(same_shapes and cache_lr.entries[0].shapes[0] == (4, 3),
         "A：这 4 次编译的输入形状全都一样（重编译与形状无关）",
         "不同的形状组数 %d 组" % len({e.shapes for e in cache_lr.entries}))

# ---- 6. A 类：编译出来的图在它声明的形状上必须算对 -----------------------------------------------

M.rule("6. A 算法正确性：每次编译出的图都与纯 numpy 参考实现一致")

errs = []
for n, d in [(8, 3), (16, 5)]:
    args = make_inputs(n, d)
    c = MiniCompileCache()
    got = c.call(model, args, {"lr": 0.01})[0]
    want = ref_model(*[np.asarray(a) for a in args], 0.01)
    errs.append(M.max_abs_err(got, want))
    print("shape %s：编译 %d 次，最大绝对误差 %.3e" % ((n, d), c.compiles, errs[-1]))
checks.A(max(errs) == 0.0, "A：编译图的执行结果 == 纯 numpy 参考实现",
         "最大绝对误差 %.3e" % max(errs))

# —— 可证伪性反证：把 shape guard 换成「永远通过」，用错形状的图去算 ——
print()
print("  —— 可证伪性反证：把 guard 换成「永远通过」的版本 ——")


class BlindCache(MiniCompileCache):
    def _compile(self, fn, args, shapes, consts, why):
        return MiniCompileCache._compile(self, fn, args, shapes, consts, why)


_blind_calls = {"n": 0}
orig_guard = Entry.guard_ok


def blind_guard(self, shapes, consts):
    _blind_calls["n"] += 1
    return True, "（故意写坏的 guard：永远通过）"


Entry.guard_ok = blind_guard
try:
    blind = MiniCompileCache()
    blind.call(model, make_inputs(4, 3), {"lr": 0.01})
    blind.call(model, make_inputs(4, 3), {"lr": 0.01})
    blind_err = "没有报错（形状被静默广播）"
    try:
        # 契约维从 3 变成 5 —— 缓存的图是在 d=3 上编译的
        blind_got = blind.call(model, make_inputs(8, 5), {"lr": 0.01})[0]
        want_b = ref_model(*[np.asarray(a) for a in make_inputs(8, 5)], 0.01)
        blind_err = "最大绝对误差 %.3e" % M.max_abs_err(blind_got, want_b)
    except Exception as e:                                    # noqa: BLE001
        blind_err = "抛了 %s: %s" % (type(e).__name__, e)
finally:
    Entry.guard_ok = orig_guard

print("坏 guard 下，用 (4,3) 编译出来的图去算 (8,5)：%s" % blind_err)
checks.C(_blind_calls["n"] > 0 and ("抛了" in blind_err or "inf" in blind_err),
         "反证：guard 失效时，缓存里那张形状特化的图会被拿去算错形状 → 真的出问题",
         blind_err)

M.rule("6b. B 代价模型：与文档 §5 的表格逐行核对")

b_rows = [
    ("同一个 shape 重复 20 次的编译增量（默认）",
     same_shape_after - same_shape_compiles, 0),
    ("N=6 个不同 shape 的编译增量（默认）", delta_auto, 1),
    ("N=6 个不同 shape 的编译次数（关掉自动动态形状）",
     cache_static.compiles, 6),
    ("自动动态形状省下的编译次数",
     cache_static.compiles - delta_auto, 5),
    ("喂 14 个 shape 时的编译次数（limit=8）", cache_cap.compiles, 8),
    ("喂 14 个 shape 时的 eager 兜底次数", cache_cap.eager_calls, 6),
    ("只改 lr 触发的编译次数", cache_lr.compiles, 4),
]
print("%-46s %8s %8s" % ("项", "本机值", "文档值"))
print("-" * 66)
for name, got_v, want_v in b_rows:
    okrow = (got_v == want_v)
    print("%-46s %8s %8s  %s" % (name, got_v, want_v, "ok" if okrow else "MISMATCH"))
    checks.B(okrow, "文档表：%s == %s" % (name, want_v), "本机 %s" % got_v)

M.rule("7. 断言汇总")
checks.summary()
