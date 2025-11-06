#!/usr/bin/env python3
import os
import time
import numpy as np

N = 2048
if __name__ == "__main__":
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    flop = 2 * N * N * N  # total floating-point ops

    for i in range(4):
        st = time.monotonic()
        C = A @ B.T
        et = time.monotonic()
        s = et - st
        gflops = flop / s * 1e-9
        tflops = flop / s * 1e-12
        print(f"{gflops:8.2f} GFLOP/S  |  {tflops:6.3f} TFLOP/S  |  {s*1e3:8.2f} ms")

    with open("/tmp/matmul", "wb") as f:
        f.write(A.data)
        f.write(B.data)
        f.write(C.data)
