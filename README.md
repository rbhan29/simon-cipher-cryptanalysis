# Simon Cipher Cryptanalysis

Cryptanalysis of the SIMON lightweight block cipher using Meet-in-the-Middle (MitM) and Brute Force attacks across multiple cipher scales (8/16, 16/32, 32/64).

## Attack Techniques
- Correlated Sequence Meet-in-the-Middle (MitM) attack (CPU & GPU)
- 4-Round MitM chunked search
- GPU-accelerated Brute Force with CPU vs GPU benchmarking

## Tech Stack
Python, CuPy, Numba (CUDA)

> GPU scripts in Folders 3 and 4 use a CuPy Anaconda environment. Newer additions in Folders 4 and 5 use Numba instead of CuPy.

---

## Repository Structure

### Folder 1 — `1.SIMON_32-16-Testvector`
SIMON32-64 block cipher algorithm with test vector generation.

| File | Description |
|------|-------------|
| `simon_enc.py` | Simple encryption engine. Run with standard Python. |
| `simon_trail.py` | Generates PT-CT pairs with dynamic round trails. Run with standard Python. |

---

### Folder 2 — `Simon8_16`
MitM attack scripts for the 8/16-bit cipher scale.

| File | Description |
|------|-------------|
| `simon_816-mitm-PTCT.py` | Generates PT-CT pairs with a user-chosen key. Run with standard Python. |
| `simon_816_correlated_mitm.py` | Correlated sequence MitM attack on reduced Simon8/16. Runs entirely on CPU. |

---

### Folder 3 — `Simon16_32`
MitM attack scripts for the 16/32-bit cipher scale.

| File | Description |
|------|-------------|
| `simon1632_testvectors.py` | Generates PT-CT pairs with a user-chosen key. Run with standard Python. |
| `simon_mitm_1632_correlated_gpu.py` | Correlated sequence MitM attack on reduced Simon16/32, targeting GPU. Requires CuPy (Anaconda). |

```bash
python simon_mitm_1632_correlated_gpu.py
```

---

### Folder 4 — `Simon32_64`
MitM attack scripts for the full 32/64-bit cipher scale.

| File | Description |
|------|-------------|
| `simon_mitm_3264_correlated_gpu.py` | Correlated sequence MitM on reduced Simon32/64 targeting GPU. Uses fixed key bits to reduce search space. Requires CuPy (Anaconda). |
| `simon_mitm_4r_normal.py` | 4-round MitM chunked search using Python dictionaries. Requires NumPy. |
| `simon_mitm_4r_normal_gpu.py` | GPU port of the 4-round script using Numba (`@cuda.jit`), bypassing CuPy. |

```bash
# Correlated GPU MitM (fix 28 bits to reduce 64-bit search space)
python simon_mitm_3264_correlated_gpu.py --fixed-bits 28 --fixed-value 0x020021001

# 4-round MitM (CPU)
python simon_mitm_4r_normal.py --pt-side-bits 16 --ct-side-bits 16

# 4-round MitM (GPU via Numba)
python simon_mitm_4r_normal_gpu.py --pt-side-bits 16 --ct-side-bits 16
```

> For larger bit sizes in the GPU script, careful host memory management is required.

---

### Folder 5 — `Simon_BF`
Brute force enumeration scripts for SIMON32_64.

| File | Description |
|------|-------------|
| `simon_bruteforce.py` | Benchmarks CPU brute force vs parallel GPU CUDA sweeps for key sizes from 8 to 40 bits. Requires NumPy, Numba, and Matplotlib. No CuPy required. |
| *(older scripts)* | Legacy brute force via standard Python and CuPy (Anaconda). |

```bash
python simon_bruteforce.py
```

---

## Requirements

```bash
pip install numpy numba matplotlib
# For CuPy (GPU scripts in Folders 3 & 4):
conda install cupy  # inside an Anaconda environment
```

---

## Results

Benchmark results and performance graphs are included in the root directory:
- `Brute Force Results (log scale).png`
- `Mean Chunk Time [Classical MitM].png`
- `Memory Usage [Classical MitM].png`
- `Projected Total FT-Sweep Time [Classical MitM].png`
- `Simon Cryptoanalysis report.pdf` — Full project report
