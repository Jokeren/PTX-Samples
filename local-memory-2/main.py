import torch

import triton
import triton.language as tl


@triton.jit
def matmul_core(
    pid_m, pid_n, pid_z,
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr
):
    # do matrix multiplication
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, TILE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, TILE_N)
    rk = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), TILE_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), TILE_N)
    # pointers
    A0 = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B0 = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((TILE_M, TILE_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * (BLOCK_SIZE_K)
        a = tl.load(A0, mask=rk[None, :] < k_remaining, other=0.)
        b = tl.load(B0, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        A0 += BLOCK_SIZE_K * stride_ak
        B0 += BLOCK_SIZE_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, TILE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, TILE_N)
    C0 = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C0, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    fixed: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # re-order program ID for better L2 performance
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    if fixed:
        matmul_core(pid_m, pid_n, pid_z, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                    TILE_M=BLOCK_SIZE_M, TILE_N=BLOCK_SIZE_N)
    else:
        if (M - pid_m * BLOCK_SIZE_M < BLOCK_SIZE_M) and (N - pid_n * BLOCK_SIZE_N < BLOCK_SIZE_N):
            matmul_core(pid_m, pid_n, pid_z, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                        TILE_M=TILE_M, TILE_N=TILE_N)
        elif (M - pid_m * BLOCK_SIZE_M < BLOCK_SIZE_M) and (N - pid_n * BLOCK_SIZE_N >= BLOCK_SIZE_N):
            matmul_core(pid_m, pid_n, pid_z, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                        TILE_M=TILE_M, TILE_N=BLOCK_SIZE_N)
        elif (M - pid_m * BLOCK_SIZE_M >= BLOCK_SIZE_M) and (N - pid_n * BLOCK_SIZE_N < BLOCK_SIZE_N):
            matmul_core(pid_m, pid_n, pid_z, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                        TILE_M=BLOCK_SIZE_M, TILE_N=TILE_N)
        else:
            matmul_core(pid_m, pid_n, pid_z, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                        TILE_M=BLOCK_SIZE_M, TILE_N=BLOCK_SIZE_N)


def matmul(a, b, fixed=False):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        fixed=fixed,
        GROUP_SIZE_M=8,
        TILE_M=16,
        TILE_N=16,
    )
    return c


M = 257
K = 4096

torch.manual_seed(0)
a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, K), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print('max abs', torch.max(torch.abs(triton_output - torch_output)))
    print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # argument names to use as an x-axis for the plot
        x_vals=[
            M
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton', 'irregular'],
        # label name for the lines
        line_names=["cuBLAS", "Triton", 'Irregular'],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-')],
        ylabel="TFLOPS",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, K), device='cuda', dtype=torch.float16)
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), rep=100)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, fixed=True), rep=100)
    if provider == 'irregular':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), rep=100)

    def perf(ms): return 2 * M * K * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
