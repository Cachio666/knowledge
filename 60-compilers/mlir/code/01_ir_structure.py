"""01 IR 与方言：Op / Region / Block / Value 的四层递归结构。

对应文档：60-compilers/mlir/01-ir-and-dialects.md

★ 本程序跑的是本仓库的**迷你 IR 实现**（miniir.py），不是 MLIR。
  输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。

用法：python3 01_ir_structure.py
"""

import re
import sys

import miniir as M
import specs as SPECS
from miniir import (IRBuilder, IRVerificationError, Module, Op, SSAError, Checks,
                    count_ops, max_depth, print_ir, rule)

# ---------------------------------------------------------------------------
# 构造一棵递归的 IR
# ---------------------------------------------------------------------------

def build_module():
    """一棵把四层递归都用上的 IR：

    builtin.module { func.func { ... scf.for { scf.if { scf.yield } { scf.yield } } } }
    另外放一个**只有一个空 region** 的 op（region 里零个 block）。
    """
    m = Module()
    outer = IRBuilder(m.body)
    # func.func 自己**没有 SSA 结果**（它的签名是属性，不是 result），故 result_types 为空
    fop, b = outer.func("main", ["tensor<2x3xf32>", "tensor<2x3xf32>"])
    args = fop.regions[0].blocks[0].args

    c0 = b.emit("arith.constant", (), ["f32"], attrs={"value": 0.0})
    t = b.emit("tensor.empty", (), ["tensor<2x3xf32>"])
    fill = b.emit("linalg.fill", [c0.results[0], t.results[0]], ["tensor<2x3xf32>"])
    add = b.emit("linalg.add", [args[0], args[1], fill.results[0]],
                 ["tensor<2x3xf32>"])

    lb = b.emit("arith.constant", (), ["index"], attrs={"value": 0})
    ub = b.emit("arith.constant", (), ["index"], attrs={"value": 3})
    st = b.emit("arith.constant", (), ["index"], attrs={"value": 1})

    forop = b.emit("scf.for", [lb.results[0], ub.results[0], st.results[0]],
                   num_regions=1)
    body = forop.regions[0].new_block("^bb_iv")
    iv = body.add_arg("index", "%iv")

    # 一个 op 带**两个** region：operation 可以有零个到多个 region
    ifop = Op("scf.if", [iv], num_regions=2)
    body.append(ifop)
    ifop.regions[0].new_block("^then").append(Op("scf.yield"))
    ifop.regions[1].new_block("^else").append(Op("scf.yield"))
    body.append(Op("scf.yield"))

    # 一个 op 带**一个空的** region：region 可以是空的
    b.emit("test.empty_region", (), num_regions=1)

    b.terminate("func.return", [add.results[0]])
    return m


# ---------------------------------------------------------------------------
# 独立参考实现：从**打印出来的文本**重新数一遍
#
# 这条路径与 miniir 的结构化遍历完全解耦——它只认文本，不认 Op 对象树。
# 把 miniir 的 count_ops 故意改错，下面的对照就会失败（见 C5）。
# ---------------------------------------------------------------------------

_OP_LINE = re.compile(r"^\s*(?:%[\w]+(?::\s*[^=]*)?\s*=\s*)?([A-Za-z_][\w.]*)")


def text_op_names(text):
    """从打印文本里把 op 名字逐个抠出来（独立路径，不看 Op 对象）。"""
    names = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in ("{", "}") or s.endswith(":") or s.endswith("{"):
            continue
        mt = _OP_LINE.match(line)
        if mt:
            names.append(mt.group(1))
    return names


def text_region_count(text):
    """文本里独占一行的 `{` 恰好对应一个「属于某个 op 的 region」。

    module 自己的那个 region 不打印花括号（它就是整个文件的顶层花括号），
    所以文本计数 + 1 == 全部 region 数。
    """
    return sum(1 for line in text.splitlines() if line.strip() == "{")


def text_values(text):
    """文本里出现过的所有 `%xxx` 记号。每个 operand 也必须先定义，故这正好是值的全集。"""
    return set(re.findall(r"%[A-Za-z_0-9]+", text))


def text_max_op_depth(text):
    """用缩进推断 op 嵌套深度。

    打印格式：module 头在缩进 0，它 block 里的 op 在缩进 4（深度 1）；op 的 region 里
    的 block 再进 2 格、它的 op 再进 2 格——**每下一层 op 加 4 格**，故
    op 深度 = 缩进 / 4。module 自己那层不打印成 op 行，正是深度 0。
    """
    best = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s in ("{", "}") or s.endswith(":") or s.endswith("{"):
            continue
        if _OP_LINE.match(line):
            best = max(best, len(line) - len(line.lstrip()))
    return best // 4


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------

checks = Checks()
m = build_module()
text = print_ir(m)

rule("1. 结构：Operation → Region → Block → Operation 四层递归")
print(text)
print()
print("官方原文（来源见 code/specs.py）：")
print("  「%s」" % SPECS.LANGREF_HIERARCHY)
print("  「%s」" % SPECS.LANGREF_VALUE_DEF)

problems = M.verify(m)
print()
print("verify() 问题数：%d" % len(problems))

rule("2. 边界与计数")

struct_counts = count_ops(m.region)          # 从 module 的 region 起算，不含 module op 本身
text_counts = {}
for n in text_op_names(text):
    text_counts[n] = text_counts.get(n, 0) + 1

n_regions = len(list(M.all_regions(m)))
n_blocks = len(list(M.all_blocks(m)))
n_values = len(M.all_values(m))
n_uses = sum(v.num_uses for v in M.all_values(m))
n_empty_regions = sum(1 for r in M.all_regions(m) if r.num_blocks == 0)
n_op_no_region = sum(1 for op in M.walk_ops(m) if op.num_regions == 0)
n_op_two_region = sum(1 for op in M.walk_ops(m) if op.num_regions == 2)

rows = [
    ("op 总数", sum(struct_counts.values()), sum(text_counts.values())),
    ("region 总数", n_regions, text_region_count(text) + 1),
    ("block 总数", n_blocks, None),
    ("value 总数", n_values, len(text_values(text))),
    ("嵌套深度（op 层）", max_depth(m), text_max_op_depth(text)),
]
print("%-22s %10s %14s" % ("项", "结构遍历", "文本重解析"))
print("-" * 50)
for name, a, b in rows:
    print("%-22s %10s %14s" % (name, a, "-" if b is None else b))

# ---------------------------------------------------------------------------
# A 类：与独立参考实现（文本重解析）逐项比对
# ---------------------------------------------------------------------------

rule("3. A 算法正确性：结构遍历 vs 文本重解析")


def names_match(struct, txt):
    return struct == txt


checks.A(names_match(struct_counts, text_counts),
         "op 名字分布：结构遍历 == 文本重解析",
         "%d 种 op，共 %d 个" % (len(struct_counts), sum(struct_counts.values())))
checks.A(n_regions == text_region_count(text) + 1,
         "region 数：结构遍历 == 文本 `{` 数 + module 自己那一个",
         "%d vs %d+1" % (n_regions, text_region_count(text)))
checks.A(n_values == len(text_values(text)),
         "value 数：all_values() == 文本里出现的不同 %% 记号数",
         "%d vs %d" % (n_values, len(text_values(text))))
checks.A(max_depth(m) == text_max_op_depth(text),
         "最大 op 嵌套深度：递归走 == 按缩进推",
         "%d vs %d" % (max_depth(m), text_max_op_depth(text)))

# ---------------------------------------------------------------------------
# B 类：代价模型（数字须与文档 §2 的表格逐行一致）
# ---------------------------------------------------------------------------

rule("4. B 代价模型：与文档表格逐行核对")

b_rows = [
    ("op 总数", sum(struct_counts.values()), 15),
    ("region 总数", n_regions, 6),
    ("block 总数", n_blocks, 5),
    ("value 总数", n_values, 10),
    ("value 的总 use 数", n_uses, 10),
    ("嵌套深度（op 层）", max_depth(m), 4),
    ("空 region 数", n_empty_regions, 1),
    ("零 region 的 op 数", n_op_no_region, 11),
    ("两 region 的 op 数", n_op_two_region, 1),
]
print("%-22s %8s %8s" % ("项", "本机值", "文档值"))
print("-" * 42)
for name, got, want in b_rows:
    ok = (got == want)
    print("%-22s %8s %8s  %s" % (name, got, want, "ok" if ok else "MISMATCH"))
    checks.B(ok, "文档表：%s == %d" % (name, want), "本机 %d" % got)

# ---------------------------------------------------------------------------
# C 类：边界与反例——这些构造**必须被拒绝**
# ---------------------------------------------------------------------------

rule("5. C 边界与反例：四种非法 IR 都必须被拒")


def bad_empty_block():
    """Block 为空：官方语法 block ::= block-label operation+ 要求至少一个 op。"""
    mm = Module()
    mm.body.ops.clear()
    M.verify_or_raise(mm)


def bad_no_terminator():
    """SSACFG region 里的 Block 非空但不以 terminator 结尾。

    注意这里**不能**拿 module 的 body 当反例：module 带 NoTerminator，它的单块
    region 本来就免于这条要求（见 miniir.Module 的注释）。这正是「先跑再写」
    推翻了第一版写法的地方。
    """
    mm = Module()
    outer = IRBuilder(mm.body)
    _, fb = outer.func("f", ["f32"])
    fb.emit("arith.constant", (), ["f32"], attrs={"value": 1.0})   # 没有 func.return
    M.verify_or_raise(mm)


def bad_double_definition():
    """同一个 Value 被定义两次——直接违反 SSA。"""
    v = M.Value("f32", "%dup")
    Op("arith.constant", (), attrs={"value": 1.0}, result_values=[v])
    Op("arith.constant", (), attrs={"value": 2.0}, result_values=[v])


def bad_undefined_operand():
    """operand 是一个从未被定义过的 Value。"""
    mm = Module()
    ghost = M.Value("f32", "%ghost")            # 不给它 owner
    IRBuilder(mm.body).emit("arith.addf", [ghost, ghost], ["f32"])
    M.verify_or_raise(mm)


checks.expect_raises(bad_empty_block, IRVerificationError, "Block 为空 → 拒绝")
checks.expect_raises(bad_no_terminator, IRVerificationError,
                     "Block 无 terminator → 拒绝")
checks.expect_raises(bad_double_definition, SSAError,
                     "同一个 Value 定义两次 → 拒绝（plan 的 C2）")
checks.expect_raises(bad_undefined_operand, IRVerificationError,
                     "operand 未定义 → 拒绝")

# ---- C 类的对照：把被检验的那一方换成正确的，同一构造就必须通过 ----
print()
print("  —— C 类的对照：换成合法构造后，同一条判据必须放行 ——")


def good_block_with_terminator():
    mm = Module()
    outer = IRBuilder(mm.body)
    _, fb = outer.func("f", ["f32"])
    fb.emit("arith.constant", (), ["f32"], attrs={"value": 1.0})
    fb.emit("func.return")                     # 补上 terminator
    M.verify_or_raise(mm)


def good_two_distinct_definitions():
    v1 = M.Value("f32", "%d1")
    v2 = M.Value("f32", "%d2")
    Op("arith.constant", (), attrs={"value": 1.0}, result_values=[v1])
    Op("arith.constant", (), attrs={"value": 2.0}, result_values=[v2])


checks.expect_ok(good_block_with_terminator,
                 "对照：加一个 terminator 后，同一个 Block 通过校验")
checks.expect_ok(good_two_distinct_definitions,
                 "对照：两个**不同**的 Value 各自定义一次，通过")


# ---- C 类的可证伪性反证：把 A 类的被测实现故意改错，A 类判据必须失败 ----
print()
print("  —— 可证伪性反证：把被测实现故意改错，A 类判据必须报错 ——")


def broken_count_ops(node, name=None):
    """故意写错的计数器：漏数 scf.* 的 op。"""
    out = {}
    for op in M.walk_ops(node):
        if op.name.startswith("scf."):
            continue
        out[op.name] = out.get(op.name, 0) + 1
    return out


broken = broken_count_ops(m)
checks.C(not names_match(broken, text_counts),
         "反证：漏数 scf.* 的计数器会被 A 类对照判为不一致",
         "改名后分布 %d 种 vs 文本 %d 种" % (len(broken), len(text_counts)))

# ---- 空 region 是合法的（这是**成功**行为，故计入 A 类而非 C 类）----
checks.A(n_empty_regions == 1 and len(M.verify(m)) == 0,
         "空 region（零个 block）合法，verify 不报错",
         "整个 module 有 %d 个空 region，verify 问题数 %d" % (n_empty_regions,
                                                              len(M.verify(m))))
checks.A(n_op_no_region == 11,
         "op 可以零个 region（region-list 的 `*` 允许零次）",
         "%d 个 op 没有 region" % n_op_no_region)

rule("6. 断言汇总")
checks.summary()
