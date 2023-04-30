import triton
import triton.language as tl


@triton.jit(noinline=True)
def matmul_core(
    pid_m, pid_n,
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
    TILE_SIZE_M: tl.constexpr, TILE_SIZE_N: tl.constexpr
):
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, TILE_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, TILE_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), TILE_SIZE_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), TILE_SIZE_N)
    A0 = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B0 = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((TILE_SIZE_M, TILE_SIZE_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * (BLOCK_SIZE_K)
        a = tl.load(A0, mask=rk[None, :] < k_remaining, other=0.)
        b = tl.load(B0, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        A0 += BLOCK_SIZE_K * stride_ak
        B0 += BLOCK_SIZE_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, TILE_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, TILE_SIZE_N)
    C0 = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C0, acc, mask=mask)


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
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
    key=['Z', 'M', 'N', 'K'],
)
@triton.jit
def batch_matmul_kernel(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    Z, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_az, stride_am, stride_ak,
    stride_bz, stride_bk, stride_bn,
    stride_cz, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_M: tl.constexpr, TILE_SIZE_N: tl.constexpr,
):
    # matrix multiplication
    pid_z = tl.program_id(0)
    pid = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    A = A + pid_z * stride_az
    B = B + pid_z * stride_bz
    C = C + pid_z * stride_cz
    if (M - pid_m * BLOCK_SIZE_M < BLOCK_SIZE_M) and (N - pid_n * BLOCK_SIZE_N < BLOCK_SIZE_N):
        matmul_core(pid_m, pid_n, A, B, C, M, N, K, stride_am, stride_ak,
                    stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                    TILE_SIZE_M=TILE_SIZE_M, TILE_SIZE_N=TILE_SIZE_N)
    elif (M - pid_m * BLOCK_SIZE_M < BLOCK_SIZE_M) and (N - pid_n * BLOCK_SIZE_N >= BLOCK_SIZE_N):
        matmul_core(pid_m, pid_n, A, B, C, M, N, K, stride_am, stride_ak,
                    stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                    TILE_SIZE_M=TILE_SIZE_M, TILE_SIZE_N=BLOCK_SIZE_N)
    elif (M - pid_m * BLOCK_SIZE_M >= BLOCK_SIZE_M) and (N - pid_n * BLOCK_SIZE_N < BLOCK_SIZE_N):
        matmul_core(pid_m, pid_n, A, B, C, M, N, K, stride_am, stride_ak,
                    stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                    TILE_SIZE_M=BLOCK_SIZE_M, TILE_SIZE_N=TILE_SIZE_N)
    else:
        matmul_core(pid_m, pid_n, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                    TILE_SIZE_M=BLOCK_SIZE_M, TILE_SIZE_N=BLOCK_SIZE_N)
