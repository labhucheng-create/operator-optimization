# 算子优化

这是一个用于学习和练习 CUDA 算子实现与性能优化的源码仓库。仓库整理了常见 GPU 并行计算算子的 CUDA 实现，包括基础示例、elementwise、reduce、SGEMM、矩阵转置、GEMV 和 LayerNorm 等内容，适合用于理解 CUDA 编程模型、优化思路和面试常见算子题。

## 项目内容

| 目录 | 说明 | 主要内容 |
| --- | --- | --- |
| `example` | CUDA 入门示例 | hello cuda、设备信息、cuBLAS 示例、矩阵拷贝 |
| `elementwise` | 逐元素计算 | 向量加法等基础 elementwise kernel |
| `gemv` | 矩阵向量乘法 | SGEMV kernel 示例 |
| `reduce` | 归约类算子 | sum、max、softmax、softmax matrix |
| `sgemm` | 矩阵乘法优化 | naive、分块、线程级 tile、double buffer 等优化版本 |
| `transpose` | 矩阵转置 | shared memory 与访存优化示例 |
| `layernorm` | 归一化算子 | LayerNorm CUDA 实现 |

## 学习重点

- CUDA thread、block、grid 的组织方式。
- 全局内存、共享内存和寄存器的使用差异。
- 合并访存、向量化访存、shared memory bank conflict 等常见优化点。
- reduce、softmax、SGEMM 等高频算子的并行化设计。
- naive 实现到优化实现之间的性能和代码结构差异。

## 环境要求

- NVIDIA GPU
- CUDA Toolkit
- 支持 CUDA 的 C++ 编译环境
- CMake，部分示例使用

不同目录的构建方式略有差异。带有 `CMakeLists.txt` 的目录可以使用 CMake 构建；单文件 `.cu` 示例也可以直接使用 `nvcc` 编译。

## 快速开始

克隆仓库后进入某个算子目录，例如：

```bash
cd example/hello_cuda
mkdir build
cd build
cmake ..
cmake --build .
```

对于单个 `.cu` 文件，也可以按需使用 `nvcc`：

```bash
nvcc add.cu -o add
./add
```

在 Windows 上生成的 `.exe`、`.pdb` 等编译产物不会提交到仓库。

## 目录说明

### `elementwise`

包含逐元素计算的 CUDA kernel，例如向量加法。该类算子结构简单，适合作为 CUDA 并行索引、边界判断、grid/block 配置和向量化访存的入门练习。

### `reduce`

包含 sum、max、softmax 等归约类算子。重点关注线程块内归约、warp shuffle、shared memory 使用、同步开销和全局写回策略。

### `sgemm`

包含单精度矩阵乘法的多个版本，从基础实现逐步引入分块、线程级 tile、访存优化和 double buffer 等方法，用于理解高性能 GEMM kernel 的优化路径。

### `transpose`

包含矩阵转置实现，重点展示如何使用 shared memory 改善访存模式，并减少或避免 bank conflict。

### `example`

包含 CUDA 基础示例和 cuBLAS 调用示例，便于验证本机 CUDA 环境和理解基础 API 用法。

## 仓库说明

本仓库只保留源码、文档、图片和构建配置文件，不提交 IDE 配置和编译生成文件。忽略规则主要包括：

- `.vscode/`
- `*build*/`
- `*.exe`
- `*.pdb`
- `*.obj`
- `*.dll`
- `*.lib`

## License

本仓库保留原项目中的 `LICENSE` 文件。使用和分发代码前，请先阅读许可证内容。
