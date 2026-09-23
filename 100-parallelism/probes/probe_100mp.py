"""
100-parallelism 探测 MP：**本机多进程 loopback 的形态可行性**（计划 §0.7-D 的 D13）。

★★★ **本脚本探的是「形态可行性」，不是本域的知识。**
   它回答的是「这台机器上**能不能**跑起两个真的进程、经 `socket` 收发」——
   用来判断「多进程 loopback」这条路线**值不值得**作为编号程序的形态 ——
   结论：**形态可行，但不作为编号程序的形态**（理由与口径见 `probes/README.md` 的「不可用的数」一栏）。
   它**不回答**任何关于并行策略的问题。

★★★ **它的输出（loopback 带宽 / 延迟）是「不可用的数」。**
   · loopback **不代表任何真实链路**（没有网卡、没有 NVLink、没有交换机、没有协议栈的完整路径）；
   · 这里的数字**受 Python 层 `recv` 循环支配**，量的是**解释器的搬运速度**，不是链路；
   · ⇒ 它们**不得**被引用、**不得**当作 `110-communication` 或任何带宽讨论的依据。
   ★ 这也是本域**不把多进程当作编号程序形态**的理由（计划 §1.1 的方案甲被否掉）。

★ **只打印、不断言**，且**不以非零退出码报告失败**（它是形态探针）：
  失败会打印成一行 `形态：失败 —— …`，脚本仍然退出 0。
★ 两个子进程之间的收发**不经过任何本域的知识**：它是 `socket`，不是集合通信。
★ 不是 NCCL，也不是真机。
"""
import multiprocessing as mp
import socket
import time

HOST = "127.0.0.1"
NBYTES = 8 * 1024 * 1024        # 一次大块搬运的字节数
PING_ROUNDS = 200               # 小包往返次数
CHUNK = 1 << 20


def show(tag, msg):
    print(f"[{tag}] {msg}")


def _recv_exact(conn, n):
    """收满 n 个字节（`recv` 不保证一次收完 —— 这本身就是 loopback 的一个坑）。"""
    got = 0
    while got < n:
        b = conn.recv(min(CHUNK, n - got))
        if not b:
            break
        got += len(b)
    return got


def _child(port, nbytes, rounds):
    """子进程：连上父进程的监听端口 → 发一大块 → 再打 `rounds` 次小包往返。"""
    payload = b"\x5a" * nbytes                  # 子进程自己造，避免把 8 MB pickle 过去
    s = socket.create_connection((HOST, port), timeout=60)
    s.sendall(payload)
    for _ in range(rounds):
        s.sendall(b"\x00" * 8)                  # 探针：8 字节
        if _recv_exact(s, 8) != 8:
            break
    s.close()


def run_once(method):
    """跑一轮（`fork` 或 `spawn`）。返回 (成功否, 说明, 大块秒数, 中位 RTT)。"""
    ctx = mp.get_context(method)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, 0))                          # ★ 端口 0 = 让内核挑一个空闲端口
    port = srv.getsockname()[1]
    srv.listen(1)
    proc = ctx.Process(target=_child, args=(port, NBYTES, PING_ROUNDS))
    proc.start()
    srv.settimeout(60)
    conn, _ = srv.accept()
    conn.settimeout(60)

    t0 = time.perf_counter()
    got = _recv_exact(conn, NBYTES)
    t1 = time.perf_counter()

    rtts = []
    for _ in range(PING_ROUNDS):
        t2 = time.perf_counter()
        if _recv_exact(conn, 8) != 8:
            break
        conn.sendall(b"\x01" * 8)
        rtts.append(time.perf_counter() - t2)

    conn.close()
    srv.close()
    proc.join(timeout=30)
    exitcode = proc.exitcode

    rtts.sort()
    med = rtts[len(rtts) // 2] if rtts else float("nan")
    ok = (got == NBYTES) and exitcode == 0 and len(rtts) == PING_ROUNDS
    note = f"收到 {got}/{NBYTES} B；子进程退出码={exitcode}；往返 {len(rtts)}/{PING_ROUNDS} 次"
    return ok, note, (t1 - t0), med


def main():
    show("MP0", "★★★ 本脚本是**形态探针**，不是知识探针 —— 它探的是「本机能不能跑多进程」。")
    show("MP0", "★★★ 它的输出（带宽 / 延迟）是**不可用的数**：loopback 不代表任何真实链路。")
    show("MP0", f"参数：一次性搬运 {NBYTES / 1024 / 1024:.0f} MiB；小包往返 {PING_ROUNDS} 次"
                f"（每次 8 B）；本机 CPU 数 = {mp.cpu_count()}")
    print()
    print("  %-8s %-8s %14s %16s %12s" % ("方法", "形态", "大块(MiB/s)", "中位 RTT(µs)", "说明"))
    results = {}
    for method in ("fork", "spawn"):
        try:
            ok, note, secs, med = run_once(method)
            good = ok
            bw = NBYTES / secs / 1024 / 1024 if secs > 0 else float("nan")
            print("  %-8s %-8s %14.1f %16.1f %12s"
                  % (method, "成功" if ok else "失败", bw, med * 1e6, ""))
            show("MP" + method[0], f"{method}：{note}")
            results[method] = good
        except Exception as e:                   # ★ 形态探针：失败只打印，不改变退出码
            print("  %-8s %-8s %14s %16s %12s" % (method, "失败", "-", "-", ""))
            show("MP" + method[0], f"{method}：失败 —— {type(e).__name__}: {e}")
            results[method] = False

    print()
    for method, ok in results.items():
        show("MP1", f"{method}：形态{'**可行**' if ok else '**不可行**'}")
    show("MP1", "★ 结论（计划 §1.1 的方案甲）：**形态本身可行**，但**不作为编号程序的形态** ——")
    show("MP1", "  ① 不可控（端口 / 超时 / `fork` 与 `spawn` 的语义差别）；")
    show("MP1", "  ② loopback 的带宽与延迟**不代表任何真实链路** ⇒ 结论要带一长串限定语，证据强度反而低。")
    show("MP1", "  ⇒ 本域编号程序一律用**纯 numpy 单进程 + 显式的张量搬运**（域 README 的「代码约定」）。")
    show("MP9", "★ 上表两个时间数字**不可引用**：它们是「Python 解释器搬字节的速度」，不是链路指标。")
    show("MP9", "★ 不是 NCCL，也不是真机。")


if __name__ == "__main__":
    main()
