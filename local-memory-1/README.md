# Steps to reproduce the IMA problem (potentially) caused by register spilling

## Package Installation

### Torch

```bash
pip3 install --force-reinstall --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118
pip3 uninstall -y pytorch-triton
```

### Triton

```bash
git clone https://github.com/openai/triton.git
git checkout ld.v2-repro
cd triton/python
pip install cmake
pip install -e .
```

### Hacky changes required

May fail to automate the process

```
file_path="$(pip show torch | grep Location | cut -d ':' -f 2 | tr -d ' ')/torch/_inductor/triton_heuristics.py"
sed -i '/num_warps = max(num_warps, 4) if conditional_product(x, y, z) >= 128 else num_warps/s/^/# /' "$file_path"
```

## Reproduce

```bash
compute-sanitizer --tool memcheck python main.py
```

Error messages are attached. I tried to increase the local memory limit for this kernel but didn't help.

```bash
========= Invalid __global__ read of size 8 bytes
=========     at 0x41b0 in triton__0d1d2d3d4d56d7d89d1011d1213d1415d1617d1819d2021d2223d2425d2627d2829d3031d3233d3435d3637d3839d4041d42d
=========     by thread (27,0,0) in block (54,0,0)
=========     Address 0x4c5e9cd21bda4 is misaligned
=========     and is 1203143593540261 bytes after the nearest allocation at 0x7fa91aa00100 of size 512 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
```

Suspicous SASS code is also attached

```
LDG.E.64 R28, [R28.64] ;
IADD3 R28, R2, 0x258, RZ ;
IMAD.WIDE R28, R28, R29, c[0x0][0x160] ;
LDG.E.64 R28, [R28.64] 
```
