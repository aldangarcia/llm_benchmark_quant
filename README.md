# Benchmark: LLM Quantization

A benchmark study comparing **FP16**, **INT8**, and **INT4** quantization across three small language models, measuring inference speed, throughput, and RAM usage on Apple Silicon (MPS/CPU).

---

## Motivation

Modern LLMs are large. Quantization reduces the number of bits used to represent each weight, trading a small amount of numerical precision for major reductions in memory and (on compatible hardware) faster inference.

This project answers: **how does quantization actually affect performance in practice?**

---

## Quantization Background

### Floating-point baseline (FP16)

Each weight $w$ is stored as a 16-bit float. The representable range is approximately:

$$w \in [-65504,\ 65504], \quad \text{precision} \approx 3 \times 10^{-3}$$

### INT8 quantization

Weights are linearly mapped to 8-bit integers. Given a tensor with range $[w_{\min},\ w_{\max}]$:

$$S = \frac{w_{\max} - w_{\min}}{2^8 - 1}, \qquad z = \text{round}\!\left(-\frac{w_{\min}}{S}\right)$$

$$w_q = \text{clamp}\!\left(\text{round}\!\left(\frac{w}{S}\right) + z,\ 0,\ 255\right)$$

The compression ratio relative to FP16 is:

$$\text{CR}_{\text{INT8}} = \frac{16}{8} = 2\times$$

### INT4 quantization

The same affine mapping, but only 4 bits are used:

$$S = \frac{w_{\max} - w_{\min}}{2^4 - 1}, \qquad w_q \in [0,\ 15]$$

$$\text{CR}_{\text{INT4}} = \frac{16}{4} = 4\times$$

Dequantization at inference time recovers an approximation:

$$\hat{w} = S \cdot (w_q - z)$$

The reconstruction error $\varepsilon = w - \hat{w}$ is bounded but non-zero, and grows as bit-width decreases.

---

## Metrics

### Throughput

$$\text{TPS} = \frac{N_{\text{tokens}}}{t_{\text{inference}}} \quad [\text{tokens/sec}]$$

where $N_{\text{tokens}}$ is the total number of generated tokens and $t_{\text{inference}}$ is wall-clock time in seconds.

### RAM delta

$$\Delta R = R_{\text{final}} - R_{\text{initial}} \quad [\text{GB}]$$

Measured from the process RSS before and after generation. A negative value indicates the OS reclaimed memory during generation.

---

## Models

| Model | Parameters | Source |
|---|---|---|
| [Phi-2](https://huggingface.co/microsoft/phi-2) | ~2.7B | Microsoft |
| [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | ~1.1B | TinyLlama |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | ~1.5B | Alibaba |

---

## Results

All runs used the same prompt (150 max new tokens, temperature=0.7, top_p=0.9). FP16 runs on MPS; INT8/INT4 run on CPU via `bitsandbytes`.

| Model | Precision | Inference Time (s) | Tokens/sec | RAM Δ (GB) |
|---|---|---|---|---|
| Phi-2 | FP16 | 14.16 | 12.86 | 0.21 |
| Phi-2 | INT8 | 8.91 | **20.43** | 0.40 |
| Phi-2 | INT4 | 291.75 | 0.29 | 1.08 |
| TinyLlama-1.1B | FP16 | 8.82 | 20.97 | 0.56 |
| TinyLlama-1.1B | INT8 | **5.07** | **36.48** | 0.01 |
| TinyLlama-1.1B | INT4 | 304.80 | 0.61 | -0.13 |
| Qwen2.5-1.5B | FP16 | **3.40** | **52.00** | 0.34 |
| Qwen2.5-1.5B | INT8 | 9.77 | 18.12 | 0.44 |
| Qwen2.5-1.5B | INT4 | 317.56 | 0.45 | -1.08 |

### Key observations

- **INT8 outperforms FP16** for Phi-2 and TinyLlama on CPU, because `bitsandbytes` uses optimized CPU INT8 kernels that are faster than naive FP16 computation on CPU.
- **INT4 is extremely slow on CPU**: `bitsandbytes` INT4 (NF4/FP4) is designed for CUDA GPUs. On CPU, dequantization overhead dominates and throughput collapses by ~40–100×.
- **Qwen2.5-1.5B FP16 on MPS** is the fastest overall at 52 tokens/sec, highlighting the advantage of the Apple Silicon GPU for half-precision inference.

---

## Project Structure

```
benchmark_quant_llm/
├── notebooks/
│   ├── 01_baseline_phi2.ipynb          # Phi-2 FP16 baseline
│   ├── 02_phi2_int8_quantization.ipynb # Phi-2 INT8
│   ├── 03_phi2_int4_quantization.ipynb # Phi-2 INT4
│   ├── 04_visualization.ipynb          # Phi-2 plots
│   ├── 05_tinyllama_benchmark.ipynb    # TinyLlama all precisions
│   ├── 06_qwen2.5-1.5B_benchmark.ipynb # Qwen2.5 all precisions
│   └── 07_visualization_models.ipynb   # Cross-model comparison plots
├── src/
│   ├── benchmarking.py   # run_benchmark() — times generation, saves CSV
│   ├── config.py         # Shared prompt and token limit
│   ├── loader.py         # Loads model in FP16 / INT8 / INT4
│   ├── metrics.py        # RAM usage via psutil
│   ├── utils.py          # MPS cache cleanup
│   └── visualization.py  # BenchmarkVisualizer class
├── results/              # Per-run CSV files
├── figures/              # Saved PNG plots
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch with MPS support (Apple Silicon) for FP16 runs. INT8/INT4 require `bitsandbytes`.

---

## Running a benchmark

From the `src/` directory:

```python
from benchmarking import run_benchmark

df = run_benchmark("Qwen/Qwen2.5-1.5B", "fp16")
print(df)
```

Or open the corresponding notebook in `notebooks/`.

---

## Visualization

```bash
cd src
python visualization.py
```

The script lists available CSVs, prompts you to select them by index, then renders and saves three plots: RAM usage, throughput, and inference latency — grouped by model and precision.
