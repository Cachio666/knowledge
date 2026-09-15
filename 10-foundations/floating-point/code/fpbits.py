"""低精度浮点格式的位级模拟。

numpy 只提供 float16 / float32 / float64，没有 bfloat16 与 float8。
本模块用整数位运算实现这些格式的**舍入行为**。

这是模拟，不是真实硬件：它复现的是"一个实数落到该格式的网格上会
变成哪个数"，不涉及该格式的吞吐、指令集或累加器实现。
"""

import numpy as np

# LLM 里实际使用的浮点格式。tf32 未在此列——它只在矩阵乘的输入端生效，
# 没有对应的存储格式，本模块的通路模拟不了。
#
# variant 区分两套规则：
#   "ieee" —— 标准 IEEE754。全 1 指数域保留给 inf/NaN，最大指数 emax = bias。
#   "e4m3" —— OCP FP8 的变体。全 1 指数域是**有效**指数（emax = bias + 1），
#             没有无穷大；代价是全 1 尾数被征用为 NaN，故最大有限数比公式
#             算出的少一档。E5M2 不受影响，它沿用 IEEE 规则。
FORMATS = {
    "fp8_e4m3": (4, 3, "e4m3"),
    "fp8_e5m2": (5, 2, "ieee"),
    "fp16": (5, 10, "ieee"),
    "bf16": (8, 7, "ieee"),
    "fp32": (8, 23, "ieee"),
    "fp64": (11, 52, "ieee"),
}


def format_params(ebits, mbits, variant="ieee"):
    """由指数位宽与尾数位宽推出该格式的全部参数。

    IEEE754 的约定：bias = 2^(k-1) - 1（k 为指数位宽），规格化指数
    范围 [1-bias, emax]，尾数另有 1 个隐藏位，故有效精度 p = mbits + 1。
    """
    bias = (1 << (ebits - 1)) - 1
    emin = 1 - bias
    if variant == "e4m3":
        emax = bias + 1                             # 全 1 指数域是有效指数
        top = 2.0 ** (1 - mbits)                    # 全 1 尾数被征用为 NaN
    else:
        emax = bias
        top = 2.0 ** (-mbits)
    p = mbits + 1
    return {
        "ebits": ebits,
        "mbits": mbits,
        "variant": variant,
        "p": p,
        "bias": bias,
        "emin": emin,
        "emax": emax,
        "has_inf": variant != "e4m3",
        "eps": 2.0 ** (-mbits),                     # 相邻 1.0 之间的距离
        "u": 2.0 ** (-p),                           # 单位舍入误差 = eps / 2
        "tiny": 2.0 ** emin,                        # 最小规格化数
        "max": (2.0 - top) * 2.0 ** emax,           # 最大有限数
        "min_sub": 2.0 ** (emin - mbits),           # 最小次正规数
    }


def quantize(x, ebits, mbits, variant="ieee"):
    """把数组舍入到 (ebits, mbits) 格式，返回 float32。

    舍入模式为 round-to-nearest-even，与硬件默认一致。返回值用 float32
    承载是精确的——舍入后的值本身就落在目标格式的网格上。
    """
    par = format_params(ebits, mbits, variant)
    xd = np.asarray(x, dtype=np.float64)
    out = np.where(np.isfinite(xd), 0.0, xd)        # inf / nan 原样穿过
    finite = np.isfinite(xd)
    ax = np.abs(xd[finite])
    q = np.zeros_like(ax)
    nz = ax > 0
    if np.any(nz):
        v = ax[nz]
        m, e = np.frexp(v)                          # v = m * 2^e, m ∈ [0.5, 1)
        exp0 = e - 1                                # 进位前的规格化指数
        s = float(1 << (mbits + 1))
        t = np.rint(m * s)                          # 尾数整数 ∈ [2^mbits, 2^(mbits+1)]
        carry = t >= s                              # 舍入进位则指数加一
        t = np.where(carry, t * 0.5, t)
        exp = exp0 + carry

        # 两条通路都必须从进位前的 exp0 出发。用 exp 判次正规会让刚进位
        # 的值被当成规格化数，两条通路各自算一遍，结果差一个 ulp。
        normal = exp0 >= par["emin"]
        qn = t * np.exp2(exp - mbits)
        if par["has_inf"]:
            qn = np.where(exp > par["emax"], np.inf, qn)    # 超出最大指数即溢出
        else:
            qn = np.minimum(qn, par["max"])                 # 无 inf 的格式改为饱和
        # 次正规：指数被钉在 emin，尾数只剩 mbits - shift 个有效位，且必须
        # 直接从 m 舍到次正规网格——先舍到规格化网格再右移是两次舍入。
        qs = np.rint(m * np.exp2(exp0 + 1 - par["emin"] + mbits)) \
            * np.exp2(par["emin"] - mbits)
        q[nz] = np.where(normal, qn, qs)            # 下溢自然归零
    out[finite] = np.copysign(q, xd[finite])
    return out.astype(np.float32)


def to_bf16(x):
    """bfloat16 的快捷通路：float32 的高 16 位。"""
    return quantize(x, 8, 7)


def bits(x, ebits, mbits, variant="ieee"):
    """把一个数打印成 (符号, 指数域, 尾数域) 三元组的二进制串。"""
    par = format_params(ebits, mbits, variant)
    v = float(quantize(np.asarray([x], dtype=np.float32), ebits, mbits, variant)[0])
    if v == 0:
        return "0 " + "0" * ebits + " " + "0" * mbits
    m, e = np.frexp(abs(v))
    exp = e - 1
    if exp < par["emin"]:                           # 次正规：指数域全 0
        frac = int(round(v / par["min_sub"]))
        efield = 0
        mfield = frac
    else:
        efield = exp + par["bias"]
        mfield = int(round((m * 2 - 1.0) * (1 << mbits)))
    sign = "1" if np.signbit(v) else "0"
    return f"{sign} {efield:0{ebits}b} {mfield:0{mbits}b}"
