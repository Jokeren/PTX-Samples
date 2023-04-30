import torch

import batch_matmul
import irregular_matmul
import triton


def matmul(a, b, fixed=True):
    # checks constraints
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    Z, M, K = a.shape
    Z, K, N = b.shape
    # allocates output
    c = torch.empty((Z, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    def grid(META):
        return (Z, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    if fixed:
        batch_matmul.batch_matmul_kernel[grid](
            a, b, c,
            Z, M, N, K,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            GROUP_SIZE_M=8
        )
    else:
        irregular_matmul.batch_matmul_kernel[grid](
            a, b, c,
            Z, M, N, K,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            TILE_SIZE_M=16, TILE_SIZE_N=16
        )

    return c


Z = 100
M = 130
K = 4096

torch.manual_seed(0)
a = torch.randn((Z, M, K), device='cuda', dtype=torch.float16)
b = torch.randn((Z, K, K), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.bmm(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print('max abs', torch.max(torch.abs(triton_output - torch_output)))
    print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['K'],  # argument names to use as an x-axis for the plot
        x_vals=[
            512, 1024, 2048, 4096
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton', 'irregular'],
        # label name for the lines
        line_names=['cuBLAS', 'Triton', 'Irregular'],
        # line styles
        styles=[('green', '-'), ('red', '--'), ('blue', '--')],
        ylabel="TFLOPS",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(K, provider):
    a = torch.randn((Z, M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((Z, K, K), device='cuda', dtype=torch.float16)
    if provider == 'cublas':
        ms = triton.testing.do_bench(lambda: torch.bmm(a, b), rep=100)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: matmul(a, b), rep=100)
    if provider == 'irregular':
        ms = triton.testing.do_bench(lambda: matmul(a, b, fixed=False), rep=100)

    def perf(ms): return Z * (2 * M * K * K * 1e-12 / (ms * 1e-3))
    return perf(ms)


benchmark.run(show_plots=True, print_data=True)
