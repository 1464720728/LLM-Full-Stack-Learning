# Kernel labs

All commands run from repository root. CPU core uses Python + NumPy; offline tests do not use network, API, or GPU. No model weights are downloaded.

```bash
python -m labs.kernels.run --all
python -m labs.kernels.run --lab 13
python -m labs.kernels.workbook all
python -m labs.kernels.capstone --output build/kernel-labs/correctness.json
python -m unittest tests.test_kernels_reference tests.test_kernels_workbooks -v
```

| Labs | Topic | Expected property |
|---|---|---|
| 01–02 | shape/stride/view | offset 23; transposed view shares storage |
| 03 | grid-stride coverage | every logical index appears exactly once |
| 04–05 | sectors/banks | ideal counts 4/32/5; conflict model 32/1/1 |
| 06 | floating reduction order | results depend on order and dtype |
| 07–08 | stable masked softmax | row sum one; fully masked row zero |
| 09–10 | RMSNorm and gradient | different from LayerNorm; finite-difference match |
| 11–12 | tiled GEMM | nonmultiple tiles equal mathematical reference |
| 13–14 | online attention | m/l/u trace; causal dense equivalence |
| 15–16 | block mask / softmax gradient | valid counts 8/2; gradient check |
| 17–20 | quantization | scale bound, all INT4 codes roundtrip, FP4 distinction |
| 21 | grouped experts | duplicate routing contributions are scatter-added |
| 22–24 | pipeline/roofline/ring | ideal models, never hardware measurements |
| 25 | collective semantics | ReduceScatter shards reconstruct AllReduce by AllGather |
| 26–27 | attention properties | block invariance and future noninterference |
| 28 | attention portfolio | 32 NumPy cases, machine-readable report |

Additional executable traces: `python -m labs.kernels.workbook softmax`, `gemm`, or `attention`. Each includes a deliberately wrong algorithm and confirms the selected test catches it; incorrect implementations are confined to the explicitly named fault demonstration.

## Source map

- `reference.py`: readable NumPy mathematical and layout implementations.
- `run.py`: 28 bounded experiments.
- `workbook.py`: trace and fault examples.
- `capstone.py`: reproducible correctness report; no GPU claims.
- `cuda/layout.cpp`: C++ address/view experiment, compiled and run during authoring.
- `cuda/basics.cu`: complete CUDA add, block reduction, tiled GEMM; GPU validation required.
- `cuda/numerical.cu`: stable row softmax and scalar-thread online attention; GPU validation required.
- `triton_ops.py`: optional add, sum, softmax, RMSNorm, GEMM, online attention.
- `gpu_check.py`: explicit small single-GPU correctness matrix.
- `custom_autograd.py`: optional PyTorch CPU gradient/registration checks.
- `benchmark.py`: explicit bounded GPU timing, correctness checked first.
- `nccl_check.py`: explicit exactly-two-rank correctness check, not a benchmark.
- `inspect_environment.py`: read-only package/tool availability; GPU query only with `--gpu`.

## Optional GPU commands

Select a GPU that has been allocated to you. All indices below are examples.

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.gpu_check --run
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op matmul --size 256 --repeats 20
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -m labs.kernels.nccl_check --run
```

CUDA build example for H100 (change the target for the actual GPU and compatible Toolkit):

```bash
mkdir -p build/kernel-labs
nvcc -O2 -lineinfo -arch=sm_90 labs/kernels/cuda/basics.cu -o build/kernel-labs/cuda_basics
nvcc -O2 -lineinfo -arch=sm_90 labs/kernels/cuda/numerical.cu -o build/kernel-labs/cuda_numerical
CUDA_VISIBLE_DEVICES=0 ./build/kernel-labs/cuda_basics
CUDA_VISIBLE_DEVICES=0 ./build/kernel-labs/cuda_numerical
```

The teaching kernels support restricted dtypes/layouts/shapes. Read their wrappers and book contracts. Do not infer full FlashAttention/CUTLASS performance or support from these examples.
