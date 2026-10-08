"""110-communication · probes/probe_110a —— ★ 真 socket 多进程 ring allreduce。

★★ 本脚本的用途：**证明「本域的算法在【真通信】下是对的」**（数据等价、轮数、顺序）。
★★★ 硬边界（P36）：本机走的是 **loopback** —— 它的**任何数字都不得**用作
   带宽 / 延迟 / 吞吐的依据（见 `100-parallelism/probes/README.md:78` 的逐字禁令）。
    本脚本**只打印、不断言**（域级 probes 的形态）；断言版在 `collectives/code/` 里。

★ 本脚本记下的一条**实测教训**（规划 §D 的 `D1` / `D1b`）：
  ① 第一版把 `buf` 写成「只有每卡自己的那一块」⇒ `reduce-scatter` 的累加目标越界；
  ② 第一版**在同一个 socket 上重试 `connect`** ⇒ **macOS/BSD 上失败后该 socket 不能复用**
     ⇒ 249 次重试**全部失败**（现象：只有最后启动的那个 rank 连得上）。
     ★★ **修法：每次重试新建 socket**（实测把 20 s 的超时降到 ~30 ms）。
"""
import socket, multiprocessing as mp, time, sys
import numpy as np

N, PER = 4, 3
S = N * PER


def recv_all(sock, n):
    out = b''
    while len(out) < n:
        c = sock.recv(n - len(out))
        if not c:
            raise RuntimeError("连接提前关闭")
        out += c
    return out


def worker(rank, ports):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', ports[rank])); srv.listen(1); srv.settimeout(20)
    # ★★ connect 重试：**每次新建 socket**（macOS 上失败后的 socket 不能复用）
    cli = None
    for _ in range(500):
        cli = socket.socket(); cli.settimeout(20)
        try:
            cli.connect(('127.0.0.1', ports[(rank + 1) % N])); break
        except OSError:
            cli.close(); cli = None; time.sleep(0.02)
    if cli is None:
        raise RuntimeError("rank %d 连不上 %d" % (rank, ports[(rank + 1) % N]))
    prev, _ = srv.accept()
    B = PER * 8
    # ★ 完整数组：只有自己那一块有值（`D1b` 的教训）
    buf = np.zeros(S)
    buf[rank*PER:(rank+1)*PER] = np.arange(rank*PER, (rank+1)*PER, dtype=np.float64) + 1.0
    for k in range(N - 1):                                    # reduce-scatter
        cid = (rank - k) % N
        cli.sendall(buf[cid*PER:(cid+1)*PER].tobytes())
        got = np.frombuffer(recv_all(prev, B), dtype=np.float64)
        j = (rank - k - 1) % N
        buf[j*PER:(j+1)*PER] += got
    for k in range(N - 1):                                    # all-gather
        cid = (rank - k + 1) % N
        cli.sendall(buf[cid*PER:(cid+1)*PER].tobytes())
        got = np.frombuffer(recv_all(prev, B), dtype=np.float64)
        j = (rank - k) % N
        buf[j*PER:(j+1)*PER] = got
    for s in (prev, cli, srv):
        s.close()
    return list(buf)


def main():
    ports = []
    for _ in range(N):
        s = socket.socket(); s.bind(('127.0.0.1', 0)); ports.append(s.getsockname()[1]); s.close()
    ctx = mp.get_context('fork')
    q = ctx.Queue()

    def run(r):
        q.put((r, worker(r, ports)))

    t0 = time.perf_counter()
    ps = [ctx.Process(target=run, args=(r,)) for r in range(N)]
    for p in ps: p.start()
    for p in ps: p.join(timeout=40)
    dt = (time.perf_counter() - t0) * 1000
    res = {}
    while not q.empty():
        r, v = q.get(); res[r] = v
    expect = list(np.arange(S, dtype=np.float64) + 1.0)
    print("probe_110a：真 socket 多进程 ring allreduce（N=%d, S=%d, %d 轮通信）" % (N, S, 2 * (N - 1)))
    for r in sorted(res):
        print("   rank%d → %s" % (r, res[r]))
    print("   期望（单进程 sum）→ %s" % expect)
    print("   全部 rank 与单进程逐位相同:", len(res) == N and all(res[r] == expect for r in res))
    print("   墙钟 %.0f ms" % dt)
    print("   ★ 限定语：**不是 NCCL，也不是真机** —— 走的是 loopback，本行的任何数字都不得当性能依据。")


if __name__ == '__main__':
    main()
