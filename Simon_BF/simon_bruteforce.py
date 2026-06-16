import numpy as np
from numba import cuda, uint16, uint32, uint64, njit
import time
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# ENVIRONMENT SAFEGUARDS
# ---------------------------------------------------------
# Forcing Numba to look in the default Anaconda base environment path 
# just in case Windows loses track of the DLLs again.
try:
    os.environ.setdefault('NUMBA_NVVM', r'C:\ProgramData\anaconda3\Library\bin\nvvm.dll')
    os.environ.setdefault('NUMBA_LIBDEVICE', r'C:\ProgramData\anaconda3\Library\lib\nvvm\libdevice')
except Exception:
    pass

# SIMON 32/64 Constants
C_CONST = 0xFFFC
Z0_SEQ = np.array([1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,
                   1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0], dtype=np.uint16)

# Updated Target Pairs
PT1, CT1 = 0xaaaaffff, 0x92ccf4ae
PT2, CT2 = 0x12345678, 0xf82af9d3

# ---------------------------------------------------------
# DEVICE (GPU) FUNCTIONS
# ---------------------------------------------------------
@cuda.jit(device=True)
def rotate_left_gpu(x, n):
    return ((x << n) & 0xFFFF) | (x >> (16 - n))

@cuda.jit(device=True)
def rotate_right_gpu(x, n):
    return (x >> n) | ((x << (16 - n)) & 0xFFFF)

@cuda.jit(device=True)
def simon_encrypt_core_gpu(pt, subkeys):
    x = uint16((pt >> 16) & 0xFFFF)
    y = uint16(pt & 0xFFFF)
    for i in range(32):
        f_x = (rotate_left_gpu(x, 1) & rotate_left_gpu(x, 8)) ^ rotate_left_gpu(x, 2)
        new_x = uint16(y ^ f_x ^ subkeys[i])
        y, x = x, new_x
    return (uint32(x) << 16) | uint32(y)

@cuda.jit
def brute_force_kernel(ct1, pt1, ct2, pt2, known_part, brute_bits, z_gpu, offset, found_key_out):
    idx = cuda.grid(1)
    current_val = uint64(idx) + uint64(offset)
    
    candidate_key = uint64((uint64(known_part) << brute_bits) | current_val)
    
    k = cuda.local.array(32, dtype=uint16)
    for i in range(4):
        k[i] = uint16((candidate_key >> (16 * i)) & 0xFFFF)
    
    for i in range(4, 32):
        tmp = rotate_right_gpu(k[i-1], 3) ^ k[i-3]
        tmp ^= rotate_right_gpu(tmp, 1)
        k[i] = uint16(k[i-4] ^ tmp ^ z_gpu[(i-4) % 62] ^ C_CONST)

    if simon_encrypt_core_gpu(pt1, k) == ct1:
        if simon_encrypt_core_gpu(pt2, k) == ct2:
            found_key_out[0] = candidate_key

# ---------------------------------------------------------
# HOST (CPU) FUNCTIONS
# ---------------------------------------------------------
@njit
def rotate_left_cpu(x, n):
    return ((x << n) & 0xFFFF) | (x >> (16 - n))

@njit
def rotate_right_cpu(x, n):
    return (x >> n) | ((x << (16 - n)) & 0xFFFF)

@njit
def simon_encrypt_core_cpu(pt, subkeys):
    x = np.uint16((pt >> 16) & 0xFFFF)
    y = np.uint16(pt & 0xFFFF)
    for i in range(32):
        f_x = (rotate_left_cpu(x, 1) & rotate_left_cpu(x, 8)) ^ rotate_left_cpu(x, 2)
        new_x = np.uint16(y ^ f_x ^ subkeys[i])
        y, x = x, new_x
    return (np.uint32(x) << 16) | np.uint32(y)

@njit
def cpu_brute_force(ct1, pt1, ct2, pt2, known_part, brute_bits, z_seq):
    total_iter = 1 << brute_bits
    k = np.zeros(32, dtype=np.uint16)
    
    for current_val in range(total_iter):
        candidate_key = np.uint64((np.uint64(known_part) << brute_bits) | current_val)
        
        for i in range(4):
            k[i] = np.uint16((candidate_key >> (16 * i)) & 0xFFFF)
        
        for i in range(4, 32):
            tmp = rotate_right_cpu(k[i-1], 3) ^ k[i-3]
            tmp ^= rotate_right_cpu(tmp, 1)
            k[i] = np.uint16(k[i-4] ^ tmp ^ z_seq[(i-4) % 62] ^ C_CONST)

        if simon_encrypt_core_cpu(pt1, k) == ct1:
            if simon_encrypt_core_cpu(pt2, k) == ct2:
                return candidate_key
    return np.uint64(0)

# ---------------------------------------------------------
# RUNNER & PLOTTER
# ---------------------------------------------------------
def run_benchmark():
    # Configurations: (Fixed Bits, Known Hex, Unknown Bits)
    # Master Key is 0x1918111009080100
    configs = [
        (56, 0x19181110090801, 8),   
        (52, 0x1918111009080, 12),   
        (48, 0x191811100908, 16),
        (44, 0x19181110090, 20),     
        (40, 0x1918111009, 24),
        (32, 0x19181110, 32),
        (28, 0x1918111, 36),     # NEW: 36 brute bits
        (24, 0x191811, 40)       # NEW: 40 brute bits
    ]
    
    cpu_times = []
    gpu_times = []
    labels = []

    z_gpu = cuda.to_device(Z0_SEQ)

    for fixed_bits, known_hex, brute_bits in configs:
        print(f"\n--- Testing Configuration: {fixed_bits} bits fixed ({brute_bits} bits brute-forced) ---")
        labels.append(f"{fixed_bits} Fixed")
        total_iter = 1 << brute_bits

        # 1. CPU RUN
        print(f"Starting CPU brute force... (Checking 2^{brute_bits} keys)")
        if brute_bits >= 36:
            print(f"[*] WARNING: CPU brute forcing {brute_bits} bits may take several minutes/hours!")
            
        start_cpu = time.perf_counter()
        cpu_res = cpu_brute_force(uint32(CT1), uint32(PT1), uint32(CT2), uint32(PT2), 
                                  uint64(known_hex), brute_bits, Z0_SEQ)
        end_cpu = time.perf_counter()
        
        # Safeguard: Minimum time threshold set to 100 nanoseconds to prevent log scale errors
        cpu_t = max(end_cpu - start_cpu, 1e-7) 
        cpu_times.append(cpu_t)
        print(f"[CPU] Key Found: {hex(cpu_res) if cpu_res else 'Not Found'} | Time: {cpu_t:.6f} seconds")

        # 2. GPU RUN
        print("Starting GPU brute force...")
        gpu_batch_size = 1 << 26 
        if total_iter < gpu_batch_size:
            gpu_batch_size = total_iter
            
        tp_block = 256
        bp_grid = max(1, gpu_batch_size // tp_block)
        
        found_key_out = cuda.to_device(np.zeros(1, dtype=np.uint64))
        
        # We use CUDA events for highly accurate GPU timing
        start_event = cuda.event()
        end_event = cuda.event()
        
        start_event.record()
        for offset in range(0, total_iter, gpu_batch_size):
            brute_force_kernel[bp_grid, tp_block](
                uint32(CT1), uint32(PT1), uint32(CT2), uint32(PT2), 
                uint64(known_hex), brute_bits, z_gpu, 
                uint64(offset), found_key_out
            )
            cuda.synchronize()
            res = found_key_out.copy_to_host()
            if res[0] != 0:
                break
        
        end_event.record()
        end_event.synchronize()
        
        # time given in milliseconds, convert to seconds
        raw_gpu_t = cuda.event_elapsed_time(start_event, end_event) / 1000.0
        gpu_t = max(raw_gpu_t, 1e-7)
        gpu_times.append(gpu_t)
        print(f"[GPU] Key Found: {hex(res[0]) if res[0] else 'Not Found'} | Time: {gpu_t:.6f} seconds")

    # 3. PLOT THE GRAPH
    print("\nBenchmarking complete. Generating graph...")
    x = np.arange(len(labels))

    # Widened the graph slightly to fit all 8 labels nicely
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotting lines with markers ('o' for circle, 's' for square)
    ax.plot(x, cpu_times, marker='o', linestyle='-', linewidth=2, label='CPU Time (s)', color='dodgerblue', markersize=8)
    ax.plot(x, gpu_times, marker='s', linestyle='-', linewidth=2, label='GPU Time (s)', color='darkorange', markersize=8)

    ax.set_ylabel('Time (Seconds) - Logarithmic Scale')
    ax.set_xlabel('Key Configuration')
    ax.set_title('Simon32/64 Brute Force: CPU vs GPU')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # --- LOG SCALE ACTIVATED ---
    ax.set_yscale('log')
    # ---------------------------

    # Add text labels next to the exact points (Formatted to 6 decimal places)
    for i, v in enumerate(cpu_times):
        # Multiply by 1.3 to push the text slightly above the point
        ax.text(i, v * 1.3, f"{v:.6f}s", ha='center', va='bottom', fontsize=9, fontweight='bold', color='midnightblue')
        
    for i, v in enumerate(gpu_times):
        # Multiply by 0.75 to push the text slightly below the point
        ax.text(i, v * 0.75, f"{v:.6f}s", ha='center', va='top', fontsize=9, fontweight='bold', color='saddlebrown')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Warm-up compile for Numba JIT to ensure compile times aren't added to the benchmark
    print("Compiling JIT functions...")
    cpu_brute_force(uint32(CT1), uint32(PT1), uint32(CT2), uint32(PT2), uint64(0x191811100908), 16, Z0_SEQ)
    print("Compilation finished. Starting Benchmarks...\n")
    run_benchmark()