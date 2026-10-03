# Graph Neural Network Inference on GPUs

High-performance full-batch Graph Convolutional Network (GCN) inference implemented in C++17, OpenMP, and CUDA. The project compares a sequential sparse baseline with multiple CPU and GPU parallelization strategies across real-world and synthetic graphs.

This repository contains the final integration branch of a team project developed for **System and Device Programming** at Politecnico di Torino (2025/2026).

## Highlights

- Sequential C++ reference implementation for correctness and performance baselines
- Three parallelization strategies: vertex parallelism, edge parallelism, and message batching
- Multicore CPU implementations with OpenMP
- CUDA basic and optimized implementations
- Sparse CSR graph storage and dense-adjacency comparison
- FP16 feature compression with FP32 accumulation
- Reproducible benchmarks, checksum validation, and numerical output comparison
- NVIDIA Nsight Compute profiling of occupancy, memory throughput, and kernel behavior
- Tests on Cora, PubMed, ogbn-arxiv, and reproducible synthetic graphs

## Representative results

The following measurements are taken from the benchmark logs committed to this repository. They use a hidden dimension of 64 and two GCN layers. Data loading and host-device transfers are excluded from the measured inference region.

| Dataset | Nodes | Edges | Sequential | CUDA edge improved | Speedup |
|---|---:|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 285.38 ms | 2.24 ms | 127.4x |
| Barabasi-Albert | 100,000 | 999,950 | 895.84 ms | 14.40 ms | 62.2x |
| ogbn-arxiv | 169,343 | 1,166,243 | 1,647.26 ms | 20.70 ms | 79.6x |

These figures are representative of the recorded test environment and should not be interpreted as hardware-independent results. Every implementation is checked against the sequential baseline using probability and prediction checksums, with numerical tolerances for parallel floating-point reductions.

## Architecture

```text
Graph + node features + fixed weights
                  |
                  v
          CSR graph representation
                  |
       +----------+-----------+
       |          |           |
       v          v           v
   Sequential   OpenMP       CUDA
                CPU          GPU
       |          |           |
       +----------+-----------+
                  |
                  v
       GCN aggregation and update
                  |
                  v
          ReLU + final softmax
                  |
                  v
   Predictions, metrics, and checksums
```

### Implemented strategies

| Platform | Strategy | Description |
|---|---|---|
| CPU | Sequential sparse | Reference full-batch GCN inference on CSR graphs |
| CPU | Vertex parallel | Distributes destination vertices across OpenMP threads |
| CPU | Edge parallel | Parallelizes message processing across graph edges |
| CPU | Message batching | Processes messages in parallel batches |
| CPU | Dense parallel | Compares CSR with a dense adjacency representation |
| GPU | Vertex parallel | Assigns graph vertices to CUDA threads |
| GPU | Edge parallel | Parallelizes aggregation over edges |
| GPU | Message batching | Groups message operations for GPU execution |
| GPU | FP16 compression | Stores activations and weights in FP16 with FP32 accumulation |

## Repository structure

```text
.
├── dataset/converted/       # Versioned sample dataset in project format
├── src/
│   ├── GCN/
│   │   ├── sequential/      # Sequential sparse baseline
│   │   ├── sequential_dense/
│   │   ├── CPU_parallelization/
│   │   ├── CUDA/
│   │   └── utilities/       # Graph loading, inference, and benchmarking
│   └── scripts/             # Conversion, generation, validation, and plots
├── weights/                 # Fixed model weights used across implementations
├── benchmark_config_cpu.json
├── benchmark_config_cuda.json
└── benchmark_config.example.json
```

Large raw datasets, generated executables, plots, and benchmark output directories are intentionally excluded from version control.

## Requirements

### Core implementations

- A C++17-compatible compiler
- OpenMP for multicore CPU implementations
- NVIDIA CUDA Toolkit for GPU implementations
- An NVIDIA GPU for CUDA execution

### Python utilities

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/scripts/requirements.txt
```

## Dataset preparation

Converters generate a common representation under `dataset/converted/<name>/`, containing CSR topology, dense node features, labels, and metadata.

```bash
python src/scripts/convert_dataset.py planetoid Cora
python src/scripts/convert_dataset.py planetoid PubMed
python src/scripts/convert_dataset.py ogb ogbn-arxiv
```

Reproducible synthetic graphs are also supported:

```bash
python src/scripts/generate_synthetic.py erdos-renyi --nodes 10000 --p 0.001 --features 128
python src/scripts/generate_synthetic.py barabasi-albert --nodes 10000 --m 4 --features 128
python src/scripts/generate_synthetic.py watts-strogatz --nodes 10000 --k 8 --p 0.1 --features 128
```

Use `--seed` to reproduce topology, features, and labels exactly.

## Fixed model weights

The project benchmarks inference rather than training. Generate one fixed model per dataset and reuse the same parameters across all implementations:

```bash
python src/scripts/generate_weights.py Cora PubMed ogbn-arxiv \
  --hidden-dim 16 --num-layers 2 --seed 42
```

Weight loading takes place before the measured inference region.

## Build and run

### Sequential baseline

Run from `src/GCN/sequential`:

```bash
g++ -O3 -std=c++17 main.cpp ../utilities/graph.cpp ../utilities/inference.cpp -o sequential
./sequential Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
```

### OpenMP vertex parallelization

Run from `src/GCN/CPU_parallelization/vertex_parallelization`:

```bash
g++ -O3 -std=c++17 -fopenmp main.cpp ../../utilities/graph.cpp ../../utilities/inference.cpp -o vertex_cpu
OMP_NUM_THREADS=8 ./vertex_cpu Cora 16 7 2 ../../../../weights/Cora/h16_l2_seed42 8
```

### CUDA implementation

Run from an implementation directory such as `src/GCN/CUDA/edge_parallelization/improved_version`:

```bash
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42
```

## Validation

All executables print a machine-readable `RESULT` line with timing, throughput, memory estimates, and checksums. To compare complete probability outputs:

```bash
GCN_OUTPUT_FILE=/tmp/sequential.txt ./sequential Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
GCN_OUTPUT_FILE=/tmp/candidate.txt ./candidate Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
python src/scripts/compare_outputs.py /tmp/sequential.txt /tmp/candidate.txt
```

The comparator uses numerical tolerances because parallel reductions may change the order of floating-point operations.

## Reproducible benchmarking

The benchmark runner performs warm-up iterations, repeated measurements, checksum validation, and statistical aggregation:

```bash
python src/scripts/run_benchmarks.py --config benchmark_config_cpu.json
python src/scripts/run_benchmarks.py --config benchmark_config_cuda.json
```

The default protocol uses three warm-up runs followed by ten measured runs. Reports include the Git commit, host, platform, inference latency, graph and message throughput, and estimated memory use.

Plots can be generated with:

```bash
python src/scripts/generate_plots.py
python src/scripts/generate_plots.py --cuda-only
```

## My contributions

My work focused on the sequential reference pipeline, CPU vertex parallelization, CPU message batching, common benchmarking and output-validation utilities, dataset conversion, reproducible synthetic graph generation, fixed-weight integration, benchmark alignment across implementations, and CUDA-specific profiling visualizations.

See the Git history for the detailed contribution record.

## Team

- Francesca De Pascale
- Davide Maugeri
- Luisa Crivo

This was developed as a collaborative university project. This fork preserves the complete commit history and authorship of the original repository while exposing the final integrated version used for benchmarking and analysis.
