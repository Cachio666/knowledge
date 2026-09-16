"""03 codegen：后端按设备注册、默认 autotune 的边界、以及**生成物真的跑**。

对应文档：60-compilers/inductor/03-codegen.md

★ 本程序跑的是**本仓库的迷你 codegen**（minifuse.emit_python + exec），
  不是 Inductor 的 Triton/C++ 后端。设备注册表、默认值、路径选择的**边界**
  逐条抄自固定版本的 torch 源码（见 specs.py，每条带来源），
  **但「哪个节点走哪条路」的判定是在本仓库的迷你模型上跑的**——
  它演示的是判据的形状，不是 Inductor 的实际决策。
  输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。

用法：python3 03_codegen_and_exec.py
"""

import numpy as np

import minifuse as F
import minigraph as M
import specs as SPECS
from minigraph import Checks

checks = Checks()


class UnregisteredDevice(Exception):
    pass


class BackendDeviceMismatch(Exception):
    pass


# ---------------------------------------------------------------------------
# 迷你设备注册表：形状照抄 register_backend_for_device（device -> codegen）
# ---------------------------------------------------------------------------


class DeviceCodegens:
    """device 字符串 → codegen 名。真实系统里注册的是**类**，这里只留名字。"""

    def __init__(self):
        self.table = {}
        self.by_name = {}

    def register(self, device, codegen_name):
        self.table[device] = codegen_name
        self.by_name.setdefault(codegen_name, set()).add(device)

    def resolve(self, device):
        if device not in self.table:
            raise UnregisteredDevice(
                "device %r 没有注册后端（真实系统里 %s 只认已注册的 device）"
                % (device, SPECS.REGISTER_BACKEND_API))
        return self.table[device]

    def require(self, codegen_name, device):
        """断言某个 codegen **属于**某个 device。用来演示 §4 第 ① 条误读。"""
        where = self.by_name.get(codegen_name)
        if where is None:
            raise BackendDeviceMismatch("codegen %r 根本没注册" % codegen_name)
        if device not in where:
            raise BackendDeviceMismatch(
                "codegen %r 注册在 %s 上，**不在** %s 上"
                % (codegen_name, sorted(where), device))
        return True


DEFAULT_CODEGEN = {"cpu": "cpp", "cuda": "triton", "tpu": "pallas",
                   "xpu": "xpu_triton", "mps": "metal", "mtia": "triton"}


def build_registry():
    reg = DeviceCodegens()
    for device, _desc in SPECS.DEVICE_BACKENDS:
        reg.register(device, DEFAULT_CODEGEN[device])
    return reg


# ---------------------------------------------------------------------------
# 迷你 codegen 路径选择：默认 autotune 的边界
# ---------------------------------------------------------------------------

EXTERN_KINDS = ("mm",)          # 这一版里唯一的外部库算子：GEMM
EXTERN_PATH = "extern_atomlib"  # cuBLAS/cuDNN 这一类外部核
SEARCH_PATH = "template_search"  # Triton / C++ 模板——**默认就在搜**的那条路


def choose_path(node, max_autotune):
    """返回该节点走哪条路。

    ★ 边界（逐条来自 specs，不要按常识写）：
      · GEMM/conv（extern 那一类）：**默认不搜**，走外部库核；
        开 `max_autotune` 才把它纳入模板/CUTLASS 候选。
      · **pointwise / reduction：默认就在搜**
        （`triton.autotune_pointwise = True`，注释写着
        "should only be disabled for debugging/testing"）。
    """
    if node.kind == "extern" and node.op in EXTERN_KINDS:
        return SEARCH_PATH if max_autotune else EXTERN_PATH
    return SEARCH_PATH


# ---------------------------------------------------------------------------
# 地形
# ---------------------------------------------------------------------------


def model(x, w, m):
    a = (x * w).relu()          # pointwise ×2
    s = a.sum(axis=1)           # reduction
    e = a.matmul(m)             # extern（GEMM）
    c = s + 1.0                 # pointwise
    return [e, c]


def ref_model(x, w, m):
    a = np.maximum(x * w, 0.0)
    s = a.sum(axis=1)
    return [a @ m, s + 1.0]


X = np.array([[0.5, -1.0, 2.0, 0.25],
              [1.5, 0.25, -0.5, 2.0]], dtype=np.float32)
Wm = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
MM = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -1.0]], dtype=np.float32)


def capture(fn, inputs):
    d = M.MiniDynamo()
    d.register(fn)
    return d.capture(fn, inputs).subgraphs[0]


# ---------------------------------------------------------------------------
# 1. 设备注册表
# ---------------------------------------------------------------------------

M.rule("1. 后端按 **device** 注册（%s）" % SPECS.REGISTER_BACKEND_API)
print("%-8s %-28s %s" % ("device", "注册的 codegen", "来源"))
print("-" * 78)
reg = build_registry()
for device, desc in SPECS.DEVICE_BACKENDS:
    print("%-8s %-28s %s" % (device, desc, "codegen/common.py:519-586（2.13.0）"))
print()
print("★ 注册表里**没有**任何一个 device 注册到 GPU 的 C++ 模板：")
print("  `max_autotune_gemm_backends` 默认值 = %s，其中"
      % SPECS.MAX_AUTOTUNE_GEMM_BACKENDS_DEFAULT)
print("  同一段源码注释逐字写着：「%s」" % SPECS.CPP_COMMENT)
print("  GPU 侧的 C++ 模板是 CUTLASS —— 与 CPP 不是一回事。")

checks.A(reg.resolve("cuda") == "triton" and reg.resolve("cpu") == "cpp",
         "A：cuda 默认注册到 triton、cpu 默认注册到 cpp",
         "cuda=%s cpu=%s" % (reg.resolve("cuda"), reg.resolve("cpu")))
checks.A(reg.resolve("mps") == "metal",
         "A：mps 注册到 metal（本机模型里如实照抄源码的注册表）",
         "mps=%s" % reg.resolve("mps"))

# ---------------------------------------------------------------------------
# 2. 默认 autotune 的边界
# ---------------------------------------------------------------------------

M.rule("2. 默认 autotune 的准确边界（**不要写成「默认不搜，开 max_autotune 才搜」**）")

g = capture(model, [X, Wm, MM])
plan = F.schedule(g)
print("图：%s" % [(n.buf, n.op, n.kind) for n in g.nodes])
print("调度结果：%s" % [[n.op for n in grp] for grp in plan.groups])
print()
print("%-34s %-18s %s" % ("项", "默认值", "来源"))
print("-" * 78)
print("%-34s %-18s %s" % ("autotune_pointwise", SPECS.AUTOTUNE_POINTWISE_DEFAULT,
                          "2.13.0 config.py:1915-1917"))
print("%-34s %-18s %s" % ("max_autotune_gemm_backends",
                          SPECS.MAX_AUTOTUNE_GEMM_BACKENDS_DEFAULT,
                          "2.13.0 config.py:617-620"))
print("  注释原文：「%s」" % SPECS.AUTOTUNE_POINTWISE_COMMENT)
print()

rows = []
for max_autotune in (False, True):
    paths = {}
    for grp in plan.groups:
        for n in grp:
            if n.kind in ("placeholder", "effect"):
                continue
            p = choose_path(n, max_autotune)
            paths.setdefault(p, []).append("%s/%s" % (n.op, n.kind))
    extern_n = len(paths.get(EXTERN_PATH, []))
    search_n = len(paths.get(SEARCH_PATH, []))
    print("max_autotune=%-5s → 走外部库核 %d 个 %s；走搜索路径 %d 个 %s"
          % (max_autotune, extern_n,
             paths.get(EXTERN_PATH, []), search_n, paths.get(SEARCH_PATH, [])))
    rows.append((max_autotune, extern_n, search_n))

checks.B(rows[0][1] == 1 and rows[0][2] > 0,
         "B：默认下 GEMM 走外部库核（1 个），**而 pointwise/reduction 已在搜索路径上**",
         "默认 extern=%d search=%d" % (rows[0][1], rows[0][2]))
checks.B(rows[1][1] == 0 and rows[1][2] == rows[0][2] + 1,
         "B：开 max_autotune 后 GEMM 转入搜索路径（extern 归零、search +1）",
         "extern=%d search=%d" % (rows[1][1], rows[1][2]))

print()
print("官方调优工作流的入口（本仓库没有跑过它们，只给可定位的源码位置）：")
for name, where in SPECS.DEBUG_ENTRY_POINTS:
    print("   %-42s %s" % (name, where))

# ---------------------------------------------------------------------------
# 3. codegen：生成物真的 exec
# ---------------------------------------------------------------------------

M.rule("3. 生成的代码真的跑：exec 后与 numpy 参考实现对拍")

src, meta = F.emit_python(plan)
print(src)
print()
env = {n.buf: n.value for n in g.nodes if n.kind == "placeholder"}
got_env = F.exec_kernels(src, meta, env)
want = ref_model(X, Wm, MM)
errs = [M.max_abs_err(got_env[o.buf], w) for o, w in zip(g.outputs, want)]
for o, e in zip(g.outputs, errs):
    print("输出 %-6s 最大绝对误差 %.3e" % (o.buf, e))
checks.A(max(errs) == 0.0,
         "A（即计划的 C4）：生成代码 exec 的结果 == 纯 numpy 参考实现",
         "最大绝对误差 %.3e" % max(errs))

# —— 可证伪性反证：把 codegen 的一条表达式故意写错，A 类必须立刻失败 ——
_orig = F._EXPR["mul"]
F._EXPR["mul"] = "({a} / {b})"
try:
    src_bad, meta_bad = F.emit_python(plan)
    bad_env = F.exec_kernels(src_bad, meta_bad, env)
    bad_err = max(M.max_abs_err(bad_env[o.buf], w)
                  for o, w in zip(g.outputs, want))
finally:
    F._EXPR["mul"] = _orig
print()
print("  —— 可证伪性反证：把 codegen 的 mul 故意写成 div ——")
print("  写错的 codegen 给出的最大绝对误差：%.3e" % bad_err)
checks.C(bad_err > 0.0,
         "反证：codegen 写错时 A 类判据立刻报不一致（这条 A 不是自证）",
         "错误 codegen 的最大绝对误差 %.3e" % bad_err)

# ---------------------------------------------------------------------------
# 4. §4 的失败模式
# ---------------------------------------------------------------------------

M.rule("4. §4 失败模式：把 CPP 当 GPU 后端、以及 extern 不参与模板融合")

print("① `max_autotune_gemm_backends` 默认值是 %s。"
      % SPECS.MAX_AUTOTUNE_GEMM_BACKENDS_DEFAULT)
print("   逐字读源码注释会看到：「%s」" % SPECS.CPP_COMMENT)
print("   本机模型里，cpp 只注册在 cpu 上：%s"
      % {d: reg.table[d] for d in sorted(reg.table)})
checks.expect_raises(
    lambda: reg.require("cpp", "cuda"),
    BackendDeviceMismatch,
    "C：把 CPP 当成 CUDA 的 codegen → 真的报错（它注册在 cpu 上）")
checks.expect_raises(
    lambda: reg.resolve("npu"),
    UnregisteredDevice,
    "C：请求一个未注册的 device → 真的报错，不会静默退化")
checks.expect_ok(lambda: reg.require("cpp", "cpu"),
                 "对照：CPP 用在 CPU 上就通过",
                 "cpp ∈ %s" % sorted(reg.by_name["cpp"]))

print()
mm_node = [n for n in g.nodes if n.kind == "extern"][0]
mm_group = [grp for grp in plan.groups if mm_node in grp][0]
print("② extern 核 %s 所在组：%s；调度器对它的判定：" % (mm_node.op,
                                                [n.op for n in mm_group]))
for d in plan.decisions:
    if d.n1 is mm_node or d.n2 is mm_node:
        print("   [%s] %s  %s" % ("融" if d.ok else "拒", d.kind, d.why))
print("   手工尝试把 extern 核与某个模板核分到同一组：")
ok, why = F.can_fuse_horizontal(g, [mm_node], [g.by_buf("buf1")], F.FuseConfig())
print("   can_fuse_horizontal(extern, 模板) → %s：%s" % ("放行" if ok else "拒", why))
checks.C(not ok, "C：extern 核（默认不搜的那一类）不参与模板融合",
         why)

# ---------------------------------------------------------------------------
# 5. B 类：与文档 §5 的表格逐行核对
# ---------------------------------------------------------------------------

M.rule("5. B 代价模型：与文档 §5 的表格逐行核对")

traffic = F.plan_traffic(g, plan.groups)
b_rows = [
    ("图里的算子数（不含占位符）",
     len([n for n in g.nodes if n.kind not in ("placeholder", "effect")]), 5),
    ("融合组数（= 生成几个核）", len(plan.groups), 2),
    ("外部库核的个数", rows[0][1], 1),
    ("搜索路径上的算子个数（默认）", rows[0][2], 4),
    ("开 max_autotune 后搜索路径算子数", rows[1][2], 5),
    ("A 类对拍的最大绝对误差标志", 0 if max(errs) == 0.0 else 1, 0),
    ("被物化的缓冲个数", len(traffic["materialized"]), 3),
]
print("%-34s %8s %8s" % ("项", "本机值", "文档值"))
print("-" * 54)
for name, got_v, want_v in b_rows:
    ok_b = (got_v == want_v)
    print("%-34s %8s %8s  %s" % (name, got_v, want_v, "ok" if ok_b else "MISMATCH"))
    checks.B(ok_b, "文档表：%s == %s" % (name, want_v), "本机 %s" % got_v)

print()
print("两个核的字节流量：读 %d / 写 %d / 合计 %d"
      % (traffic["read_bytes"], traffic["write_bytes"], traffic["total"]))
checks.A(traffic["kernels"] == len(plan.groups),
         "A：代价模型里的核数 == 融合组数（同一件事的两个说法）",
         "%d vs %d" % (traffic["kernels"], len(plan.groups)))

M.rule("6. 断言汇总")
checks.summary()
