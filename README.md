# Simon Cipher Cryptanalysis

Cryptanalysis of the SIMON lightweight block cipher using Meet-in-the-Middle (MitM) 
and Brute Force attacks across multiple cipher scales (8/16, 16/32, 32/64).

## Attack Techniques
- Correlated Sequence MitM attack (CPU & GPU via CuPy / Numba)
- 4-Round MitM chunked search
- GPU-accelerated Brute Force with benchmarking

## Structure
- `Simon8_16/` – MitM on SIMON 8/16
- `Simon16_32/` – GPU MitM on SIMON 16/32
- `Simon32_64/` – Full-scale 32/64 MitM + 4-round attacks
- `Simon_BF/` – Brute Force benchmarks (CPU vs GPU)
- `1.SIMON_32_16_Testvector/` – Encryption engine & test vectors

## Requirements
- Python, NumPy, Numba (for GPU scripts), CuPy (for older GPU scripts via Anaconda)
